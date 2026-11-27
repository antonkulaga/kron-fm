"""Contact maps as a training target: the stores ``scripts/kron_dna/prep_hic.py`` writes, read per window, a head
that predicts them from the core's states, and its loss.

A contact map says which stretches of a chromosome touch in the nucleus. Its value for a pair of cells cannot be
read from either cell's own sequence: the loss on it is one that needs both ends, at any distance up to the
store's reach (1 Mb), which the per-position track losses never ask of the core.

The grid is the store's (cells of 5,000 bases on chromosome coordinates), not the model's bins: ``ContactStores.
window`` gives every bin of a window the cell it lies in, and the head pools the core's states by that.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContactStores:
    """Stores of one cell size and reach, one per biosample (the head's tracks, in the order given), on a prepared
    genome's layout (``contigs``: chrom, start, end, offset as ``genome-prepare`` writes them)."""

    def __init__(self, paths: list[Path], contigs: pl.DataFrame) -> None:
        if not paths:
            raise ValueError("no contact store given")
        infos = [json.loads((Path(p) / "hic.json").read_text()) for p in paths]
        self.cell, self.diagonals = int(infos[0]["bin"]), int(infos[0]["diagonals"])
        tables = [pl.read_parquet(Path(p) / "chroms.parquet") for p in paths]
        for p, i, t in zip(paths, infos, tables):
            same = t.select("chrom", "cells", "offset").equals(tables[0].select("chrom", "cells", "offset"))
            if (int(i["bin"]), int(i["diagonals"])) != (self.cell, self.diagonals) or not same:
                raise ValueError(f"{p}: another cell size, reach or chromosome layout than {paths[0]}")
        self.names = [Path(p).name for p in paths]
        self.chroms = {r["chrom"]: (int(r["offset"]), int(r["cells"])) for r in tables[0].iter_rows(named=True)}
        total = sum(n for _, n in self.chroms.values())
        self.maps = [np.memmap(Path(p) / "contacts.f16", dtype=np.float16, mode="r", shape=(total, self.diagonals))
                     for p in paths]
        c = contigs.sort("offset")
        self._offset, self._start = c["offset"].to_numpy(), c["start"].to_numpy()
        self._end, self._chrom = c["end"].to_numpy(), c["chrom"].to_list()

    def window(self, at: int, take: int, bin: int, flip: bool = False) -> tuple[np.ndarray, np.ndarray] | None:
        """For ``take`` bases at layout offset ``at`` read in bins of ``bin`` bases: ``cells`` ``[take // bin]`` (each
        bin's cell within the window, by the bin's middle base, from 0) and ``target`` ``[C, diagonals, tracks]``
        float32, observed / expected for the pairs (i, i + d) of the window's C cells, NaN where the store has none
        or the second cell lies outside the window. ``flip``: the window is read as its reverse complement, so the
        cells run the other way. None when the window is not inside one contig of a chromosome the stores hold."""
        k = int(np.searchsorted(self._offset, at, side="right")) - 1
        if k < 0 or at + take > self._offset[k] + (self._end[k] - self._start[k]) or self._chrom[k] not in self.chroms:
            return None
        pos = int(self._start[k]) + (at - int(self._offset[k]))
        first, n_chrom = self.chroms[self._chrom[k]]
        cells = (pos + np.arange(take // bin, dtype=np.int64) * bin + bin // 2) // self.cell
        lo, n = int(cells[0]), int(cells[-1] - cells[0]) + 1
        if lo + n > n_chrom:
            return None
        target = np.stack([np.asarray(m[first + lo:first + lo + n]) for m in self.maps], -1).astype(np.float32)
        i, d = np.indices((n, self.diagonals))
        target[i + d >= n] = np.nan
        cells = cells - lo
        if flip:                                                      # the pair (i, i + d) is (n - 1 - i - d, .. + d)
            src = np.clip(n - 1 - i - d, 0, n - 1)
            target = np.where((i + d < n)[..., None], target[src, d], np.nan).astype(np.float32)
            cells = (n - 1 - cells)[::-1].copy()
        return cells, target


class ContactHead(nn.Module):
    """Contacts between the cells of a window from the core's states. The states of the bins of a cell are averaged;
    each cell gets two vectors, and a pair's features are their products taken both ways (so a pair reads the same
    from either end) plus an embedding of its distance in cells; a small MLP gives one value per track, read as
    ``log1p(observed / expected)``. Only pairs up to ``reach`` cells apart are computed."""

    def __init__(self, d_model: int, width: int, reach: int, tracks: int) -> None:
        super().__init__()
        self.reach = int(reach)
        self.inp = nn.Sequential(nn.Linear(d_model, 2 * width), nn.GELU())
        self.left, self.right = nn.Linear(2 * width, width), nn.Linear(2 * width, width)
        self.distance = nn.Embedding(self.reach, width)
        self.out = nn.Sequential(nn.RMSNorm(width, eps=1e-5), nn.Linear(width, width), nn.GELU(), nn.Linear(width, tracks))

    def forward(self, core: torch.Tensor, cells: torch.Tensor, n_cells: int | None = None) -> torch.Tensor:
        """``core`` ``[B, bins, d_model]``, ``cells`` ``[B, bins]`` long (each bin's cell, 0 .. C - 1; -1 for a bin
        that belongs to none) -> ``[B, C, reach, tracks]``, entry (i, d) for the pair (i, i + d). A pair whose
        second cell is past C, or a cell with no bin, gets a value all the same: mask it with the target."""
        b, bins, d = core.shape
        c = int(n_cells if n_cells is not None else cells.max()) + (0 if n_cells is not None else 1)
        ok = cells >= 0
        at = (torch.arange(b, device=core.device).unsqueeze(1) * c + cells.clamp(min=0))[ok]
        total = torch.zeros(b * c, d, device=core.device, dtype=core.dtype).index_add_(0, at, core[ok])
        count = torch.zeros(b * c, device=core.device, dtype=core.dtype).index_add_(
            0, at, torch.ones(at.shape[0], device=core.device, dtype=core.dtype))
        u = self.inp((total / count.clamp(min=1).unsqueeze(1)).reshape(b, c, d))
        left, right = self.left(u), self.right(u)                      # [B, C, w]

        def ahead(x: torch.Tensor) -> torch.Tensor:                    # [B, C, reach, w]: cell i + d beside cell i
            return F.pad(x, (0, 0, 0, self.reach - 1)).unfold(1, self.reach, 1).permute(0, 1, 3, 2)

        pair = left.unsqueeze(2) * ahead(right) + ahead(left) * right.unsqueeze(2)
        return self.out(pair + self.distance.weight.to(pair.dtype))


def contact_target(observed_over_expected: torch.Tensor) -> torch.Tensor:
    """What the head predicts: ``log1p`` of observed / expected (0 for a pair with no read, ln 2 at the expected
    contact); NaN stays NaN."""
    return torch.log1p(observed_over_expected.float().clamp(min=0))


def contact_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error between ``pred`` and ``contact_target(target)`` over the pairs the target has (not NaN),
    same shapes; zero with no such pair."""
    ok = ~torch.isnan(target)
    if not bool(ok.any()):
        return pred.sum() * 0.0
    return F.mse_loss(pred.float()[ok], contact_target(target)[ok])


def contact_correlation(pred: torch.Tensor, target: torch.Tensor, min_cells: int = 2) -> torch.Tensor:
    """Pearson r per track between ``pred`` and ``contact_target(target)`` over the pairs at least ``min_cells``
    apart that the target has (``[..., reach, tracks]`` each); higher is better. The values are already divided by
    the expected contact at each distance, so this is not the decay with distance."""
    p, t = pred.float()[..., min_cells:, :], contact_target(target)[..., min_cells:, :]
    p, t = p.reshape(-1, p.shape[-1]), t.reshape(-1, t.shape[-1])
    ok = ~torch.isnan(t)
    n = ok.sum(0).clamp(min=1)
    pm = torch.where(ok, p, torch.zeros_like(p)).sum(0) / n
    tm = torch.where(ok, t, torch.zeros_like(t)).sum(0) / n
    dp, dt = torch.where(ok, p - pm, torch.zeros_like(p)), torch.where(ok, t - tm, torch.zeros_like(t))
    return (dp * dt).sum(0) / ((dp.pow(2).sum(0) * dt.pow(2).sum(0)).sqrt() + 1e-12)
