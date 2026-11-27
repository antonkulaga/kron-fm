"""prep_hic.py: a version-9 .hic file written here (both block layouts, short and wide fields) is read back."""
import importlib.util
import json
import struct
import zlib
from pathlib import Path

import numpy as np
import polars as pl

spec = importlib.util.spec_from_file_location(
    "prep_hic", Path(__file__).parents[1] / "scripts" / "kron_dna" / "prep_hic.py")
ph = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ph)

BIN, BLOCK = 100, 4                                                    # 4 cells per block along the diagonal


def text(s: str) -> bytes:
    return s.encode() + b"\0"


def block_rows(recs, wide: bool) -> bytes:
    """Layout 1 (rows of cells); ``wide``: int32 cells and float counts instead of int16."""
    x0, y0 = min(r[0] for r in recs), min(r[1] for r in recs)
    rows: dict[int, list] = {}
    for x, y, c in recs:
        rows.setdefault(y - y0, []).append((x - x0, c))
    i, cf = ("i", "f") if wide else ("h", "h")
    out = struct.pack("<iiibbbb", len(recs), x0, y0, int(wide), int(wide), int(wide), 1) + struct.pack("<" + i, len(rows))
    for y, cells in sorted(rows.items()):
        out += struct.pack("<" + i + i, y, len(cells)) + b"".join(struct.pack("<" + i + cf, x, c) for x, c in cells)
    return zlib.compress(out)


def block_dense(recs) -> bytes:
    """Layout 2 (a dense rectangle of int16 counts, -32768 for no record)."""
    x0, y0 = min(r[0] for r in recs), min(r[1] for r in recs)
    w, h = max(r[0] for r in recs) - x0 + 1, max(r[1] for r in recs) - y0 + 1
    grid = np.full((h, w), -32768, dtype="<i2")
    for x, y, c in recs:
        grid[y - y0, x - x0] = c
    return zlib.compress(struct.pack("<iiibbbbih", len(recs), x0, y0, 0, 0, 0, 2, w * h, w) + grid.tobytes())


def write_hic(path: Path, lengths: dict[str, int], counts: dict[str, np.ndarray], vectors: dict[str, np.ndarray]) -> None:
    names = ["All"] + list(lengths)
    head = text("HIC") + struct.pack("<iq", 9, 0) + text("hg38") + struct.pack("<qq", 0, 0)
    head += struct.pack("<i", 1) + text("software") + text("test")
    head += struct.pack("<i", len(names)) + b"".join(text(n) + struct.pack("<q", lengths.get(n, 0)) for n in names)
    head += struct.pack("<ii", 1, BIN) + struct.pack("<i", 0)
    body, master, norm_at = bytearray(head), {}, {}
    for k, name in enumerate(lengths, start=1):
        m = counts[name]
        cells = len(m)
        columns = cells // BLOCK + 1
        blocks: dict[int, list] = {}
        for x in range(cells):
            for y in range(x, cells):
                if m[x, y]:
                    depth = int(np.log2(1 + (y - x) / np.sqrt(2) / BLOCK))
                    blocks.setdefault(depth * columns + (x + y) // 2 // BLOCK, []).append((x, y, int(m[x, y])))
        index = []
        for j, (number, recs) in enumerate(sorted(blocks.items())):
            data = block_dense(recs) if j % 3 == 0 else block_rows(recs, wide=j % 3 == 1)
            index.append((number, len(body), len(data)))
            body += data
        master[f"{k}_{k}"] = len(body)
        total = float(m.sum() + np.triu(m, 1).sum())
        body += struct.pack("<iii", k, k, 1) + text("BP") + struct.pack("<iffffiiii", 0, total, 0, 0, 0, BIN, BLOCK, columns,
                                                                       len(index))
        body += b"".join(struct.pack("<iqi", *e) for e in index)
        norm_at[k] = len(body)
        body += struct.pack("<q", cells) + vectors[name].astype("<f4").tobytes()
    master_at = len(body)
    body += struct.pack("<qi", 0, len(master)) + b"".join(text(k) + struct.pack("<qi", v, 0) for k, v in master.items())
    body += struct.pack("<i", 0) + struct.pack("<i", 0)                # no expected vectors
    nvi_at = len(body)
    body += struct.pack("<i", len(norm_at)) + b"".join(
        text("SCALE") + struct.pack("<i", k) + text("BP") + struct.pack("<iqq", BIN, at, 0) for k, at in norm_at.items())
    struct.pack_into("<q", body, 8, master_at)
    struct.pack_into("<qq", body, 8 + 8 + len(text("hg38")), nvi_at, len(body) - nvi_at)
    path.write_bytes(bytes(body))


def test_a_written_file_is_read_back(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    lengths = {"chr1": 3950, "chr2": 2000}                             # 40 cells (the last one short) and 20
    counts, vectors = {}, {}
    for name, n in (("chr1", 40), ("chr2", 20)):
        i, j = np.indices((n, n))
        m = np.triu(rng.poisson(200.0 / (1 + np.abs(i - j)))).astype(np.int64)
        m[5, :] = m[:, 5] = 0                                          # an empty cell: no vector entry
        counts[name] = m
        v = rng.uniform(0.5, 2.0, n)
        v[5] = np.nan
        vectors[name] = v
    write_hic(tmp_path / "x.hic", lengths, counts, vectors)
    hic = ph.Hic(tmp_path / "x.hic")
    assert hic.version == 9 and hic.genome == "hg38" and hic.chroms[1] == ("chr1", 3950) and hic.resolutions == [BIN]
    out = tmp_path / "out"
    ph.main(tmp_path / "x.hic", out, bin=BIN, reach=1000, norm="SCALE", chroms="chr1,chr2", check=True)
    info = json.loads((out / "hic.json").read_text())
    assert info["diagonals"] == 11 and info["sum_read_over_file"] == {"chr1": 1.0, "chr2": 1.0}
    table = pl.read_parquet(out / "chroms.parquet")
    assert table["cells"].to_list() == [40, 20] and table["offset"].to_list() == [0, 40] and table["masked"].to_list() == [1, 1]
    got = np.fromfile(out / "contacts.f16", dtype=np.float16).reshape(60, 11).astype(np.float64)
    expected = np.fromfile(out / "expected.f32", dtype=np.float32).astype(np.float64)
    want, num, den = {}, np.zeros(11), np.zeros(11)
    for name in ("chr1", "chr2"):
        m, v = counts[name], vectors[name]
        n = len(m)
        w = np.full((n, 11), np.nan)
        for d in range(11):
            for a in range(n - d):
                w[a, d] = m[a, a + d] / (v[a] * v[a + d])
        want[name] = w
        num, den = num + np.nansum(w, 0), den + (~np.isnan(w)).sum(0)
    assert np.allclose(expected, num / den, rtol=1e-5)
    whole = np.concatenate([want["chr1"], want["chr2"]]) / (num / den)
    assert np.array_equal(np.isnan(got), np.isnan(whole)) and np.isnan(got[5]).all() and np.isnan(got[39, 1:]).all()
    assert np.allclose(got[~np.isnan(got)], whole[~np.isnan(whole)], rtol=2e-3)       # float16
    near = tmp_path / "near"                                           # without the check only the near blocks are read
    ph.main(tmp_path / "x.hic", near, bin=BIN, reach=1000, norm="SCALE", chroms="chr1,chr2", check=False)
    assert (near / "contacts.f16").read_bytes() == (out / "contacts.f16").read_bytes()
