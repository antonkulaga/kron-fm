"""Contact maps near the diagonal from a ``.hic`` file (Juicer, format versions 8 and 9), read with no outside code.

    uv run python scripts/kron_dna/prep_hic.py FILE.hic OUT_DIR [--bin 5000] [--reach 1000000] [--norm SCALE]
        [--chroms chr1,chr2,...] [--check]

Per chromosome the contacts between cells of ``--bin`` bases (a resolution the file holds) up to ``--reach`` bases
apart, normalised with the file's own vector ``--norm`` (count / (v[i] * v[j]); a cell whose vector entry is missing
or not positive is masked), then divided by the expected contact at that distance: the mean normalised contact over
all unmasked pairs of cells that far apart, over the autosomes together (the sex chromosomes and chrM use the same
curve). OUT_DIR:

- ``contacts.f16`` ``[cells, reach / bin + 1]`` float16: row i of a chromosome holds observed / expected for the
  pairs (i, i), (i, i + 1), ...; NaN where either cell is masked or the pair runs past the chromosome's end. A
  measured zero is 0 (take ``log1p`` or clip before a logarithm). Chromosomes laid end to end as ``chroms.parquet``
  says (chrom, length, cells, offset, masked);
- ``expected.f32`` ``[reach / bin + 1]``: the curve the values were divided by (normalised counts);
- ``hic.json``: the file's genome and software, what was read, and with ``--check`` the comparison below.

``--check`` reads every block of each chromosome (not only those near the diagonal) and tests the reading against
what the file says about itself: the sum of raw counts against the matrix record's ``sumCounts`` (which counts a
pair off the diagonal twice, as the symmetric matrix has it; the ratio must be 1), and the normalisation vector
against the counts: a balancing vector makes every unmasked row of the whole normalised matrix sum to the same
number, so the spread of the row sums (their standard deviation over their mean) must be small; a block read
wrong or a vector laid on the wrong cells breaks it. Juicer balances the whole genome's matrix at once, contacts
between chromosomes included, so within one chromosome a few percent remain (raw rows spread by tens of percent). The curve computed here is also set beside the file's own
expected vector, for information: Juicer computes that one over all possible pairs, masked or not, and scales it
per chromosome, so the two differ by a factor.

The format, as read here (little endian): header (magic ``HIC``, version, position of the master index, genome,
in version 9 the position and length of the normalisation index, attributes, chromosomes, resolutions); the master
index (matrix records by ``"i_j"``); version 8 keeps the expected vectors and the normalisation index after it;
a matrix record lists per resolution the blocks (number, position, size); a block is zlib-compressed lists of
(cell x, cell y, count) in one of two layouts. In version 9 a block of an intra-chromosomal matrix covers a
stretch along the diagonal at one depth away from it (``depth = floor(log2(1 + |x - y| / sqrt(2) / blockBinCount))``,
number ``depth * blockColumnCount + (x + y) / 2 / blockBinCount``), so only the first depths are read.
"""
from __future__ import annotations

import json
import math
import mmap
import struct
import zlib
from pathlib import Path

import numpy as np
import polars as pl
import typer


class Cursor:
    """Sequential little-endian reads from a buffer."""

    def __init__(self, buf, pos: int = 0) -> None:
        self.buf, self.pos = buf, pos

    def take(self, fmt: str):
        out = struct.unpack_from("<" + fmt, self.buf, self.pos)
        self.pos += struct.calcsize("<" + fmt)
        return out[0] if len(out) == 1 else out

    def text(self) -> str:
        end = self.buf.find(b"\0", self.pos)
        out = bytes(self.buf[self.pos:end]).decode()
        self.pos = end + 1
        return out

    def array(self, dtype: str, n: int) -> np.ndarray:
        out = np.frombuffer(self.buf, dtype=np.dtype(dtype).newbyteorder("<"), count=n, offset=self.pos)
        self.pos += out.nbytes
        return out


class Hic:
    """A ``.hic`` file's header, master index and normalisation index."""

    def __init__(self, path: Path) -> None:
        self.fh = open(path, "rb")
        self.buf = mmap.mmap(self.fh.fileno(), 0, access=mmap.ACCESS_READ)
        c = Cursor(self.buf)
        if c.text() != "HIC":
            raise ValueError(f"{path} is not a .hic file")
        self.version = v = c.take("i")
        if v not in (8, 9):
            raise ValueError(f".hic version {v}: versions 8 and 9 are read")
        master = c.take("q")
        self.genome = c.text()
        nvi = c.take("qq") if v >= 9 else None
        self.attributes = {}
        for _ in range(c.take("i")):
            k = c.text()
            self.attributes[k] = c.text()
        self.chroms = []                                               # (name, length), by the file's index
        for _ in range(c.take("i")):
            name = c.text()
            self.chroms.append((name, c.take("q" if v >= 9 else "i")))
        self.resolutions = list(c.array("i4", c.take("i")))
        c = Cursor(self.buf, master)
        c.take("q" if v >= 9 else "i")                                 # bytes of the index
        self.matrices = {}
        for _ in range(c.take("i")):
            key = c.text()
            self.matrices[key] = c.take("qi")[0]
        self.expected = {}                                             # (norm, unit, bin) -> (values, {chrom index: factor})
        for normed in (False, True):
            for _ in range(c.take("i")):
                norm = c.text() if normed else "NONE"
                unit, size = c.text(), c.take("i")
                values = c.array("f4" if v >= 9 else "f8", c.take("q" if v >= 9 else "i")).astype(np.float64)
                factors = {}
                for _ in range(c.take("i")):
                    i = c.take("i")
                    factors[i] = float(c.take("f" if v >= 9 else "d"))
                self.expected[(norm, unit, size)] = (values, factors)
        if nvi is not None:
            c = Cursor(self.buf, nvi[0])
        self.norms = {}                                                # (norm, chrom index, unit, bin) -> position
        for _ in range(c.take("i")):
            norm, i, unit, size = c.text(), c.take("i"), c.text(), c.take("i")
            self.norms[(norm, i, unit, size)] = c.take("qq" if v >= 9 else "qi")[0]

    def index_of(self, chrom: str) -> int:
        names = [n for n, _ in self.chroms]
        for cand in (chrom, chrom.removeprefix("chr"), "chr" + chrom):
            if cand in names:
                return names.index(cand)
        raise KeyError(chrom)

    def norm_vector(self, norm: str, i: int, size: int) -> np.ndarray:
        c = Cursor(self.buf, self.norms[(norm, i, "BP", size)])
        n = c.take("q" if self.version >= 9 else "i")
        return c.array("f4" if self.version >= 9 else "f8", n).astype(np.float64)

    def blocks(self, i: int, size: int) -> dict:
        """The matrix of chromosome ``i`` with itself at one resolution: ``sum`` (raw counts), ``block_bins``,
        ``columns`` and ``blocks`` ``[(number, position, bytes)]``."""
        c = Cursor(self.buf, self.matrices[f"{i}_{i}"])
        c.take("ii")
        for _ in range(c.take("i")):
            unit = c.text()
            c.take("i")
            total = c.take("f")
            c.take("fff")
            bin_size, block_bins, columns = c.take("iii")
            entries = np.frombuffer(self.buf, dtype=np.dtype([("n", "<i4"), ("pos", "<i8"), ("size", "<i4")]),
                                    count=c.take("i"), offset=c.pos)
            c.pos += entries.nbytes
            if unit == "BP" and bin_size == size:
                return dict(sum=float(total), block_bins=block_bins, columns=columns, blocks=entries)
        raise KeyError(f"no {size}-base resolution for chromosome index {i}")

    def records(self, pos: int, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One block as ``(x, y, count)`` arrays."""
        c = Cursor(zlib.decompress(self.buf[pos:pos + size]))
        n = c.take("i")
        if self.version < 7:
            rec = c.array(np.dtype([("x", "<i4"), ("y", "<i4"), ("c", "<f4")]), n)
            return rec["x"].astype(np.int64), rec["y"].astype(np.int64), rec["c"].astype(np.float64)
        x0, y0 = c.take("ii")
        short_c = c.take("b") == 0
        short_x, short_y = (c.take("b") == 0, c.take("b") == 0) if self.version >= 9 else (True, True)
        kind = c.take("b")
        if kind == 1:
            cell = np.dtype([("x", "<i2" if short_x else "<i4"), ("c", "<i2" if short_c else "<f4")])
            xs, ys, cs = [], [], []
            for _ in range(c.take("h" if short_y else "i")):
                y = c.take("h" if short_y else "i")
                row = c.array(cell, c.take("h" if short_x else "i"))
                xs.append(row["x"].astype(np.int64))
                cs.append(row["c"].astype(np.float64))
                ys.append(np.full(len(row), y, dtype=np.int64))
            if not xs:
                return (np.zeros(0, np.int64),) * 2 + (np.zeros(0),)
            return np.concatenate(xs) + x0, np.concatenate(ys) + y0, np.concatenate(cs)
        if kind == 2:
            n_points, width = c.take("i"), c.take("h")
            v = c.array("i2" if short_c else "f4", n_points)
            keep = (v != -32768) if short_c else ~np.isnan(v)
            at = np.flatnonzero(keep)
            return at % width + x0, at // width + y0, v[keep].astype(np.float64)
        raise ValueError(f"block layout {kind}")


def band(hic: Hic, i: int, size: int, diags: int, vector: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Raw counts of chromosome ``i`` as ``[cells, diags]`` (row c, column d: the pair (c, c + d)). With ``vector``
    (the check) every block is read, also those far from the diagonal, and the second value holds ``sum`` (the
    counts read, a pair off the diagonal twice), ``file`` (the file's own sum) and ``spread`` (of the row sums of
    the whole normalised matrix over its unmasked rows)."""
    cells = -(-hic.chroms[i][1] // size)
    m = hic.blocks(i, size)
    out = np.zeros((cells, diags), dtype=np.float64)
    blocks = m["blocks"]
    everything = vector is not None
    if everything:
        v = np.full(cells, np.nan)
        v[:min(cells, len(vector))] = vector[:cells]
        v[~(v > 0)] = np.nan
        sums = np.zeros(cells)
    if hic.version >= 9 and not everything:
        depth = int(math.log2(1 + (diags - 1) / math.sqrt(2) / m["block_bins"]))
        blocks = blocks[blocks["n"] // m["columns"] <= depth]
    total = 0.0
    for b in blocks:
        if b["size"] == 0:
            continue
        x, y, c = hic.records(int(b["pos"]), int(b["size"]))
        lo, d = np.minimum(x, y), np.abs(x - y)
        if everything:
            total += float(c.sum() + c[d > 0].sum())
            ok = (x < cells) & (y < cells)
            w = np.nan_to_num(c[ok] / (v[x[ok]] * v[y[ok]]))
            np.add.at(sums, x[ok], w)
            np.add.at(sums, y[ok], np.where(d[ok] > 0, w, 0.0))
        keep = (d < diags) & (lo >= 0) & (lo + d < cells)
        np.add.at(out, (lo[keep], d[keep]), c[keep])
    if not everything:
        return out, {}
    rows = sums[~np.isnan(v)]
    return out, dict(sum=total, file=m["sum"], spread=float(rows.std() / rows.mean()) if len(rows) else float("nan"))


def normalise(counts: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """``counts / (v[i] * v[j])``; NaN where a cell's vector entry is missing or not positive or the pair runs past
    the end."""
    cells, diags = counts.shape
    v = np.full(cells, np.nan)
    v[:min(cells, len(vector))] = vector[:cells]
    v[~(v > 0)] = np.nan
    out = np.full(counts.shape, np.nan)
    for d in range(diags):
        if d < cells:
            out[:cells - d, d] = counts[:cells - d, d] / (v[:cells - d] * v[d:])
    return out


def main(hic_path: Path, out: Path, bin: int = 5000, reach: int = 1_000_000, norm: str = "SCALE",
         chroms: str = typer.Option("", help="comma separated; default chr1..chr22, chrX"),
         check: bool = typer.Option(False, help="read every block and compare with the file's own sums and curve")) -> None:
    hic = Hic(hic_path)
    if bin not in hic.resolutions:
        raise typer.BadParameter(f"the file holds the resolutions {hic.resolutions}")
    names = [c for c in chroms.split(",") if c] or [f"chr{k}" for k in range(1, 23)] + ["chrX"]
    diags = reach // bin + 1
    maps, rows, sums = {}, [], {}
    num, den = np.zeros(diags), np.zeros(diags)
    for name in names:
        i = hic.index_of(name)
        if (norm, i, "BP", bin) not in hic.norms:
            raise typer.BadParameter(f"no {norm} vector for {name} at {bin}; the file has "
                                     f"{sorted({k[0] for k in hic.norms})}")
        vector = hic.norm_vector(norm, i, bin)
        counts, sums[name] = band(hic, i, bin, diags, vector if check else None)
        m = normalise(counts, vector)
        maps[name] = m
        if name.removeprefix("chr").isdigit():                         # the curve comes from the autosomes
            num += np.nansum(m, 0)
            den += (~np.isnan(m)).sum(0)
        typer.echo(f"{name}: {m.shape[0]:,} cells, {int(np.isnan(m[:, 0]).sum()):,} masked, counts near the diagonal "
                   f"{counts.sum():.4g}" + (f"; all counts over the file's sum {sums[name]['sum'] / sums[name]['file']:.6f}, "
                                            f"spread of normalised row sums {sums[name]['spread']:.2e}" if check else ""))
    if not den.all():
        raise typer.BadParameter("no autosome among the chromosomes, or a distance with no unmasked pair: no expected curve")
    expected = num / den
    out.mkdir(parents=True, exist_ok=True)
    offset = 0
    with open(out / "contacts.f16", "wb") as fh:
        for name in names:
            m = maps[name] / expected
            np.minimum(m, np.finfo(np.float16).max).astype(np.float16).tofile(fh)
            rows.append(dict(chrom=name, length=hic.chroms[hic.index_of(name)][1], cells=m.shape[0], offset=offset,
                             masked=int(np.isnan(maps[name][:, 0]).sum())))
            offset += m.shape[0]
    expected.astype(np.float32).tofile(out / "expected.f32")
    pl.DataFrame(rows).write_parquet(out / "chroms.parquet")
    info = dict(file=hic_path.name, version=hic.version, genome=hic.genome, software=hic.attributes.get("software", ""),
                bin=bin, reach=reach, diagonals=diags, norm=norm, cells=offset, chroms=names,
                values="observed / expected of normalised counts, float16; NaN = masked cell or past the end; row i, "
                       "column d = the pair (i, i + d)",
                expected="mean normalised contact per distance over unmasked pairs of the autosomes read")
    if check:
        info["sum_read_over_file"] = {k: round(v["sum"] / v["file"], 6) for k, v in sums.items() if v["file"]}
        info["row_sum_spread"] = {k: float(f"{v['spread']:.3g}") for k, v in sums.items()}
        key = (norm, "BP", bin)
        if key in hic.expected:
            theirs = hic.expected[key][0][:diags]
            n = min(len(theirs), diags)
            ratio = expected[:n] / theirs[:n]
            info["expected_over_file"] = dict(median=round(float(np.median(ratio)), 4), min=round(float(ratio.min()), 4),
                                              max=round(float(ratio.max()), 4))
    (out / "hic.json").write_text(json.dumps(info, indent=1))
    typer.echo(json.dumps(info))


if __name__ == "__main__":
    typer.run(main)
