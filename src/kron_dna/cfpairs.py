"""Measured counterfactual pairs as a training loss (genome-train --cf-pairs; JEPA step 2c).

The builder's tables (``interim/train/cf-pairs-v1/*.parquet``): one row per measured pair, columns ``chrom``, ``pos``
(0-based), ``ref``, ``alt`` (a substitution block of any length, an indel, or both empty for a silenced element
given by ``del_start``/``del_end``), ``read_pos`` (where the effect is read: the variant, or the gene's TSS),
``head`` (a further head of the model), ``channel_ids`` (the head's channels the measurement maps to; empty: the
gene strand's channels when the row has ``strand``, else all), ``effect`` (signed, in the assay's own units),
``weight``, ``split`` (``train`` | ``pair_test``). Only ``train`` rows outside the run's held-out chromosomes are
used; everything else is refused at load, so the scored subsets stay unseen.

The loss: for a pair the model reads a window around the variant and the read position twice, reference and
alternative, and its head's prediction (log1p signal, as the benchmarks score it) is summed over the read bins and
averaged over the row's channels; ``delta = alt - ref``. Effects are in each assay's own units, so the loss is on the
sign only: ``weight * softplus(-sign(effect) * delta / temp)``; a row with effect 0 (a measured absence) pulls
``delta`` towards 0 with ``weight * (delta / temp) ** 2 / 2``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

CODE = {"A": 0, "C": 1, "G": 2, "T": 3}
BIN = 128
NEED = ("chrom", "pos", "ref", "alt", "read_pos", "head", "channel_ids", "effect", "weight", "split")


@dataclass
class Pair:
    chrom: str
    pos: int
    ref: str
    alt: str
    read_pos: int
    head: str
    channels: np.ndarray            # indices into the head's outputs
    sign: float                     # -1, 0, +1
    weight: float
    del_span: tuple[int, int] | None
    table: str = ""


def channels_for(ids: list[str], channel_ids, strand) -> np.ndarray:
    """Indices of ``channel_ids`` in ``ids``; empty: the strand's channels (ids ending _P for +, _M for -, plus the
    unstranded ones) when ``strand`` is given, else every channel."""
    at = {i: k for k, i in enumerate(ids)}
    if channel_ids is not None and len(channel_ids):
        return np.array([at[c] for c in channel_ids if c in at], dtype=np.int64)
    if strand in ("+", "-"):
        drop = "_M" if strand == "+" else "_P"
        return np.array([k for k, i in enumerate(ids) if not i.endswith(drop)], dtype=np.int64)
    return np.arange(len(ids), dtype=np.int64)


class CfPairs:
    """The training rows of one or more pair tables, for the heads ``heads`` (name -> channel ids)."""

    def __init__(self, paths: list[Path], heads: dict[str, list[str]], held_out: set[str],
                 window: int, edge: int | None = None) -> None:
        frames = []
        for p in paths:
            t = pl.read_parquet(p)
            missing = [c for c in NEED if c not in t.columns]
            if missing:
                raise ValueError(f"{p}: no column(s) {missing}")
            frames.append(t.with_columns(pl.lit(str(Path(p).stem)).alias("_table")))
        t = pl.concat(frames, how="diagonal_relaxed")
        n_all = t.height
        bad = t.filter(pl.col("chrom").is_in(sorted(held_out)) & (pl.col("split") == "train"))
        if bad.height:
            raise ValueError(f"{bad.height} train rows lie on the held-out chromosomes {sorted(held_out)}: refusing")
        t = t.filter((pl.col("split") == "train") & ~pl.col("chrom").is_in(sorted(held_out)))
        t = t.filter(pl.col("head").is_in(list(heads)))
        edge = min(4096, window // 8) if edge is None else edge      # the window's margins, as the benches keep
        span = (pl.max_horizontal("pos", "read_pos") - pl.min_horizontal("pos", "read_pos"))
        t = t.filter(span < window - 2 * edge - 2 * BIN)
        self.window, self.edge = window, edge
        self.counts = dict(rows_in_tables=n_all, used=t.height,
                           by_table={k: int(v) for k, v in t.group_by("_table").len().iter_rows()})
        cols = set(t.columns)
        self.pairs: list[Pair] = []
        for r in t.iter_rows(named=True):
            ch = channels_for(heads[r["head"]], r["channel_ids"], r.get("strand") if "strand" in cols else None)
            if not len(ch):
                continue
            ds = (int(r["del_start"]), int(r["del_end"])) if ("del_start" in cols and r.get("del_start") is not None
                                                              and not (r["ref"] or r["alt"])) else None
            e = float(r["effect"]) if r["effect"] is not None else 0.0
            if "regulated" in cols and r.get("regulated") is False:    # a measured absence (CRISPRi): no sign
                e = 0.0
            self.pairs.append(Pair(r["chrom"], int(r["pos"]), r["ref"] or "", r["alt"] or "", int(r["read_pos"]),
                                   r["head"], ch, float(np.sign(e)), float(r["weight"] or 0.0), ds, r["_table"]))
        self.pairs = [p for p in self.pairs if p.weight > 0]
        self.counts["pairs"] = len(self.pairs)
        self.by_table: dict[str, list[int]] = {}
        for k, p in enumerate(self.pairs):
            self.by_table.setdefault(p.table, []).append(k)

    def sample(self, rng: np.random.Generator, k: int) -> list[Pair]:
        """``k`` pairs, each from a table drawn uniformly (allelic binding alone has ~90 % of the rows, so a
        uniform draw over rows would be that table only), then a row of it uniformly."""
        names = sorted(self.by_table)
        out = []
        for _ in range(k):
            rows = self.by_table[names[int(rng.integers(0, len(names)))]]
            out.append(self.pairs[rows[int(rng.integers(0, len(rows)))]])
        return out


def codes_of(contigs: pl.DataFrame, read, chrom: str, lo: int, hi: int) -> np.ndarray:
    """Base codes of chrom[lo:hi) (N outside the contigs). ``read(a, b)``: bases.u8 rows a:b."""
    out = np.full(hi - lo, 4, dtype=np.int64)
    for r in contigs.filter((pl.col("chrom") == chrom) & (pl.col("start") < hi) & (pl.col("end") > lo)).iter_rows(named=True):
        a, b = max(lo, r["start"]), min(hi, r["end"])
        out[a - lo:b - lo] = read(r["offset"] + a - r["start"], r["offset"] + b - r["start"])
    return out


def alt_of(seq: np.ndarray, pair: Pair, lo: int) -> np.ndarray | None:
    """The alternative window: the substitution or indel applied (kept at the window's length), or the silenced
    element set to N; None when the reference does not hold ``ref``."""
    if pair.del_span is not None:
        out = seq.copy()
        a, b = max(pair.del_span[0] - lo, 0), min(pair.del_span[1] - lo, len(seq))
        out[a:b] = 4
        return out
    at = pair.pos - lo
    r = np.array([CODE.get(c, 4) for c in pair.ref.upper()], dtype=seq.dtype)
    if at < 0 or at + len(r) > len(seq) or not np.array_equal(seq[at:at + len(r)], r):
        return None
    a = np.array([CODE.get(c, 4) for c in pair.alt.upper()], dtype=seq.dtype)
    new = np.concatenate([seq[:at], a, seq[at + len(r):]])
    if len(new) < len(seq):
        new = np.concatenate([new, np.full(len(seq) - len(new), 4, dtype=seq.dtype)])
    return new[:len(seq)]


def pair_loss(model, pairs: list[Pair], contigs: pl.DataFrame, read, consts: dict, signal_of, window: int,
              temp: float, dev: torch.device, autocast) -> tuple[torch.Tensor | None, dict]:
    """The sign loss over ``pairs`` (grouped by head, one forward of 2 x n windows per head), with gradients.
    ``consts[head]``: the head's store constants (nonzero_mean, is_rna as ``signal_of`` takes them), or None for a
    binary head (then the raw logit is the signal). Returns (mean loss or None, stats)."""
    unit = 2 ** model.config.n_down
    by_head: dict[str, list] = {}
    for p in pairs:
        mid = (p.pos + p.read_pos) // 2
        lo = max((mid // BIN) * BIN - window // 2, 0)
        ref = codes_of(contigs, read, p.chrom, lo, lo + window)
        alt = alt_of(ref, p, lo)
        if alt is None:
            continue
        rel = p.read_pos - lo
        by_head.setdefault(p.head, []).append((p, ref, alt, rel))
    terms, agree, n = [], 0, 0
    old_active, old_span = model.x_active, model.x_span
    # the pair windows are not the training batch: no hidden blocks (the batch's mask has another shape) and the
    # decoder reads the core as it is
    old_hidden, old_dec = getattr(model, "jepa_hidden", None), getattr(model, "jepa_decode_pred", False)
    model.jepa_hidden, model.jepa_decode_pred = None, False
    try:
        for head, items in by_head.items():
            span_lo = min(((it[3] - BIN) // BIN) * BIN for it in items)
            span_hi = max(((it[3] + 2 * BIN) // BIN) * BIN for it in items)
            span_lo, span_hi = max(span_lo, 0), min(span_hi, window)
            seqs = torch.from_numpy(np.stack([s for it in items for s in (it[1], it[2])])).to(dev)
            seg = torch.zeros(len(seqs), window // unit, dtype=torch.long, device=dev)
            ctx = torch.zeros(len(seqs), dtype=torch.long, device=dev) if model.config.n_contexts > 0 else None
            model.x_active, model.x_span = {head}, (span_lo, span_hi)
            with autocast:
                model(seqs, seg, ctx)
            raw = model.last_x[head].float()                       # [2n, rows of span, outputs]
            c = consts.get(head)
            sig = raw if c is None else signal_of(F.softplus(raw), c[0], c[1])
            res = (span_hi - span_lo) // sig.shape[1]
            for j, (p, _, _, rel) in enumerate(items):
                a = max((rel - span_lo) // res, 0)
                b = max(a + 1, (rel + BIN - span_lo) // res)
                ch = torch.from_numpy(p.channels).to(dev)
                r_ = sig[2 * j, a:b][:, ch].sum(0).mean()
                a_ = sig[2 * j + 1, a:b][:, ch].sum(0).mean()
                delta = (a_ - r_) / temp
                if p.sign == 0:
                    terms.append(p.weight * delta.pow(2) / 2)
                else:
                    terms.append(p.weight * F.softplus(-p.sign * delta))
                    agree += int(float(delta.detach()) * p.sign > 0)
                    n += 1
    finally:
        model.x_active, model.x_span = old_active, old_span
        model.jepa_hidden, model.jepa_decode_pred = old_hidden, old_dec
    if not terms:
        return None, dict(pairs=0, signed=0, agree=float("nan"))
    loss = torch.stack(terms).mean()
    return loss, dict(pairs=len(terms), signed=n, agree=agree / n if n else float("nan"), loss=float(loss.detach()))
