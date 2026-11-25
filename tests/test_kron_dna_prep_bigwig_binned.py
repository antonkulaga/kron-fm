"""scripts/kron_dna/prep_bigwig_binned.py: bigWig straight to 128-bp cells, the same quantities as the per-base route."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from test_kron_dna_multigenome import genomes  # noqa: F401  (a fixture)

pyBigWig = pytest.importorskip("pyBigWig")
ROOT = Path(__file__).resolve().parents[1] / "scripts" / "kron_dna"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = m                       # the worker pool finds its function by module name
    spec.loader.exec_module(m)
    return m


def test_direct_cells_equal_the_per_base_route(genomes, tmp_path, monkeypatch):
    hgt, root = genomes["hgt"], genomes["root"]
    idx = pl.read_parquet(hgt / "contigs.parquet").sort("offset")
    total = int((idx["end"] - idx["start"]).sum())
    (hgt / "ntv3").mkdir(exist_ok=True)
    np.ones(total, dtype=np.uint8).tofile(hgt / "ntv3" / "split.u8")
    bwd = tmp_path / "bw"
    bwd.mkdir()
    rng = np.random.default_rng(0)
    for fid in ("ENCFFA", "ENCFFB"):
        bw = pyBigWig.open(str(bwd / f"{fid}.bigWig"), "w")
        sizes = [(r["chrom"], int(r["end"])) for r in idx.iter_rows(named=True)]
        bw.addHeader(sizes)
        for c, n in sizes:
            v = np.where(rng.random(n) < 0.3, rng.gamma(1.0, 3.0, n), 0.0)
            bw.addEntries(c, 0, values=v.astype(np.float64), span=1, step=1)
        bw.close()
    sel = tmp_path / "select.tsv"
    sel.write_text("experiment\tfile\tassay\tbiosample\ttarget\toutput_type\n"
                   "E1\tENCFFA\tChIP-seq\tK562\tH3K4me3\tfold change over control\n"
                   "E2\tENCFFB\tChIP-seq\tHepG2\tH3K27ac\tfold change over control\n")
    BB = load("prep_bigwig_binned")
    monkeypatch.setattr(BB, "interim_dir", lambda *p: root.joinpath(*p))
    out = tmp_path / "direct"
    BB.main(sel, bwd, out, name="hgt", bin=16, workers=1, scale_percentile=99.99, convert_only=False,
            provenance=None, check_cells=200, split=None)
    meta = json.loads((out / "binned.json").read_text())
    # the per-base route's encoding, built here (its own main checks 131 kb windows the test genome lacks)
    PB = load("prep_bigwig_tracks")
    rows = [(r["chrom"], r["start"], r["end"], r["offset"]) for r in idx.iter_rows(named=True)]
    scales = []
    dense = np.zeros((total, 2), dtype=np.uint8)
    for k, fid in enumerate(("ENCFFA", "ENCFFB")):
        bw = pyBigWig.open(str(bwd / f"{fid}.bigWig"))
        sc = PB.scale_of(bw, rows, 99.99)
        scales.append(sc)
        for c, s0, e0, off in rows:
            v = np.nan_to_num(bw.values(c, s0, e0, numpy=True))
            dense[off:off + e0 - s0, k] = np.clip(np.log1p(np.maximum(v, 0)) / sc * 255.0, 0, 255).astype(np.uint8)
        bw.close()
    assert np.allclose(meta["scales"], scales)                             # the same encoding
    dj = dict(scales=scales)
    cells = meta["cells"]
    lg = np.memmap(out / "log1p.f16", dtype=np.float16, mode="r", shape=(cells, 2)).astype(np.float32)
    tg = np.memmap(out / "target.f16", dtype=np.float16, mode="r", shape=(cells, 2)).astype(np.float32)
    n_full = total // 16
    want_lg = (np.asarray(dense[:n_full * 16]).astype(np.float32) / 255 * np.array(dj["scales"])).reshape(n_full, 16, 2).mean(1)
    assert np.allclose(lg[:n_full], want_lg, atol=2e-3, rtol=2e-3)
    BT = load("bin_track_store")
    x = np.asarray(dense[:n_full * 16])
    want_t = BT.targets_np(x, np.array(dj["scales"], dtype=np.float32), np.array(meta["nonzero_mean"], dtype=np.float32),
                           np.zeros(2, dtype=bool)).reshape(n_full, 16, 2).mean(1)
    assert np.allclose(tg[:n_full], want_t, atol=5e-3, rtol=5e-3)
    # the check reads the raw signal, unclipped: here the scale falls back to log1p(1) (no 100 kb window to sample in
    # the test genome), so much of the signal is clipped and r stays near 0.9; real tracks clip at their 99.99th
    assert all(t["check_pearson_cells_vs_bigwig"] > 0.5 for t in meta["tracks"])


def test_a_negative_minus_strand_file_gives_the_same_cells(genomes, tmp_path, monkeypatch):
    """ENCODE's minus-strand PRO-cap and eCLIP files hold their signal as negative numbers: read as they are, the
    track would be all zeros. The same signal written positive and negative gives identical cells, the negative one
    marked sign -1; a file with both signs is left as it is."""
    hgt, root = genomes["hgt"], genomes["root"]
    idx = pl.read_parquet(hgt / "contigs.parquet").sort("offset")
    total = int((idx["end"] - idx["start"]).sum())
    (hgt / "ntv3").mkdir(exist_ok=True)
    np.ones(total, dtype=np.uint8).tofile(hgt / "ntv3" / "split.u8")
    bwd = tmp_path / "bw"
    bwd.mkdir()
    sizes = [(r["chrom"], int(r["end"])) for r in idx.iter_rows(named=True)]
    for fid, sign in (("POS", 1.0), ("NEG", -1.0)):
        rng = np.random.default_rng(0)
        bw = pyBigWig.open(str(bwd / f"{fid}.bigWig"), "w")
        bw.addHeader(sizes)
        for c, n in sizes:
            v = np.where(rng.random(n) < 0.3, rng.gamma(1.0, 3.0, n), 0.0)
            bw.addEntries(c, 0, values=(sign * v).astype(np.float64), span=1, step=1)
        bw.close()
    sel = tmp_path / "select.tsv"
    sel.write_text("experiment\tfile\tassay\tbiosample\ttarget\toutput_type\tchannel\n"
                   "E1\tPOS\tPRO-cap\tK562\t\tplus strand signal of unique reads\tE1_P\n"
                   "E1\tNEG\tPRO-cap\tK562\t\tminus strand signal of unique reads\tE1_M\n")
    BB = load("prep_bigwig_binned")
    monkeypatch.setattr(BB, "interim_dir", lambda *p: root.joinpath(*p))
    out = tmp_path / "store"
    BB.main(sel, bwd, out, name="hgt", bin=16, workers=1, scale_percentile=99.99, convert_only=False,
            provenance=None, check_cells=200, split=None)
    meta = json.loads((out / "binned.json").read_text())
    assert meta["ids"] == ["E1_P", "E1_M"] and [t["sign"] for t in meta["tracks"]] == [1.0, -1.0]
    tg = np.memmap(out / "target.f16", dtype=np.float16, mode="r", shape=(meta["cells"], 2))
    assert float(np.abs(np.asarray(tg[:, 1])).max()) > 0 and np.array_equal(np.asarray(tg[:, 0]), np.asarray(tg[:, 1]))
    assert all(t["check_pearson_cells_vs_bigwig"] > 0.5 and t["negative_values"] == 0 for t in meta["tracks"])
    from kron_dna.prep.bigwig import Signed

    class Mixed:                                                     # both signs: not a strand convention
        def header(self):
            return dict(minVal=-2.0, maxVal=3.0)
    assert Signed(Mixed()).sign == 1.0
