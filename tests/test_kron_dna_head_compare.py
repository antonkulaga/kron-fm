"""scripts/kron_dna/head_compare.py: head variants trained side by side on a frozen trunk's decoder states."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from kron_dna import genome as G
from test_kron_dna_multigenome import genomes, make_ntv3  # noqa: F401  (genomes is a fixture)
from test_kron_dna_split import small
from test_kron_dna_xheads_cli import make_store

spec = importlib.util.spec_from_file_location(
    "head_compare", Path(__file__).resolve().parents[1] / "scripts" / "kron_dna" / "head_compare.py")
HC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(HC)


def test_every_variant_maps_states_to_outputs_and_counts_its_parameters():
    x = torch.randn(2, 16, 64)
    sizes = {}
    for v in HC.VARIANTS:
        h = HC.make_head(v, 16, 5)
        assert h(x).shape == (2, 5, 64), v
        sizes[v] = HC.n_params(h)
    assert sizes["linear"] == 16 * 5 + 5 and sizes["mlp512"] > sizes["mlp128"] > sizes["linear"]
    edge = HC.make_head("conv9", 16, 5)(torch.randn(1, 16, 9))      # the conv head keeps the length
    assert edge.shape == (1, 5, 9)


def test_pooled_scores_match_a_direct_computation():
    g = torch.Generator().manual_seed(0)
    pred, want = torch.rand(3, 512, 2, generator=g), torch.rand(3, 512, 2, generator=g)
    stats = torch.stack([HC.pearson_stats(pred[w], want[w], bin=128) for w in range(3)])
    r = HC.pearson_from(stats.sum(0))
    xb, yb = (v.reshape(3, 4, 128, 2).mean(2).reshape(-1, 2).numpy() for v in (pred, want))
    assert r[0] == pytest.approx(np.corrcoef(xb[:, 0], yb[:, 0])[0, 1], abs=1e-6)
    y = torch.rand(3, 2000, 2, generator=g) < 0.03
    x = torch.randn(3, 2000, 2, generator=g) + 2.0 * y.float()
    pieces = [HC.ap_piece(x[w], y[w].to(torch.uint8), keep=1.0) for w in range(3)]
    order = np.argsort(-x[..., 0].reshape(-1).numpy(), kind="stable")
    hit = y[..., 0].reshape(-1).numpy()[order]
    brute = float((np.cumsum(hit) / np.arange(1, len(hit) + 1))[hit].mean())
    assert HC.ap_pooled(pieces)[0] == pytest.approx(brute, abs=1e-9)


def test_run_trains_every_variant_on_the_same_frozen_features(genomes, tmp_path, monkeypatch, device):
    if device.type != "cpu":
        pytest.skip("the smoke run is the CPU one")
    root = genomes["root"]
    monkeypatch.setattr(HC, "interim_dir", lambda *parts: root.joinpath(*parts))
    nd = make_ntv3(root, genomes["hgt"], genomes["gA"])
    import polars as pl
    idx = pl.read_parquet(genomes["hgt"] / "contigs.parquet")
    split = np.fromfile(nd / "split.u8", dtype=np.uint8)
    for r in idx.filter(pl.col("chrom").is_in(["chr8", "chr9"])).iter_rows(named=True):
        split[r["offset"]:r["offset"] + r["end"] - r["start"]] = 4
    split.tofile(nd / "split.u8")
    store = make_store(tmp_path, genomes["hgt"])
    total = int(np.fromfile(genomes["hgt"] / "bases.u8", dtype=np.uint8).size)
    lab_dir = tmp_path / "labels"
    lab_dir.mkdir()
    rng = np.random.default_rng(4)
    lab = np.zeros(total, dtype=np.uint8)
    bits = {"donor_plus": 0, "acceptor_plus": 1, "donor_minus": 2, "acceptor_minus": 3}
    for b in bits.values():
        lab[rng.random(total) < 0.02] |= 1 << b
    lab.tofile(lab_dir / "splice.u8")
    (lab_dir / "splice.json").write_text(json.dumps(dict(
        genome="hgt", total_bases=total, bits=bits, definition="one base per site, per strand", splice_sha256="x",
        check={k: dict(sites=int(((lab >> b) & 1).sum())) for k, b in bits.items()})))
    m = small("v2")
    ckpt = tmp_path / "trunk.pt"
    torch.save({"model": m.state_dict(), "config": m.config.to_dict(), "step": 7}, ckpt)
    before = {k: v.clone() for k, v in m.state_dict().items()}
    res = HC.run(ckpt, tmp_path / "out", store, lab_dir, name="hgt", split=nd / "split.u8", window=1024, batch=2,
                 steps=6, eval_windows=4, eval_every=3, device="cpu", n_boot=20, n_boot_ap=5, warmup=2)
    assert res["steps"] == 6 and res["trunk_step"] == 7 and set(res["variants"]) == set(HC.VARIANTS)
    ref = res["variants"]["mlp128"]
    assert ref["dnase_minus_reference"] == 0.0 and ref["dnase_minus_reference_ci"] == [0.0, 0.0]
    for v, r in res["variants"].items():
        assert len(r["dnase_pearson_per_track"]) == 3 and len(r["splice_ap_per_channel"]) == 4
        assert np.isfinite(r["dnase_pearson_mean"]) and np.isfinite(r["splice_ap_mean"]) and r["parameters"]["dnase"] > 0
    rows = (tmp_path / "out" / "progress.csv").read_text().strip().split("\n")
    assert len(rows) == 1 + 2 * len(HC.VARIANTS)
    again, _ = G.load_checkpoint(ckpt)                           # the trunk file is only read
    assert all(torch.equal(before[k], v) for k, v in again.state_dict().items())
    assert (tmp_path / "out" / "heads.pt").exists() and json.loads((tmp_path / "out" / "results.json").read_text())
