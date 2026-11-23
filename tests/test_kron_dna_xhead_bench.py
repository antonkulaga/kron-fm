"""scripts/kron_dna/xhead_bench.py: the scoring pieces of the further heads' benchmark."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from test_kron_dna_multigenome import genomes  # noqa: F401  (a fixture)

spec = importlib.util.spec_from_file_location(
    "xhead_bench", Path(__file__).resolve().parents[1] / "scripts" / "kron_dna" / "xhead_bench.py")
XB = importlib.util.module_from_spec(spec)
spec.loader.exec_module(XB)


def direct_ap(score: np.ndarray, label: np.ndarray, weight: np.ndarray) -> float:
    """Average precision with integer base weights, by repeating each base ``weight`` times (the definition)."""
    s, y = np.repeat(score, weight), np.repeat(label, weight)
    order = np.argsort(-s, kind="stable")
    y = y[order]
    tp = np.cumsum(y)
    return float((tp[y > 0] / (np.flatnonzero(y) + 1)).sum() / y.sum())


def test_ranked_ap_equals_the_definition_with_and_without_window_weights():
    rng = np.random.default_rng(0)
    n, length = 6, 50
    label = (rng.random((n, length)) < 0.1).astype(np.uint8)
    score = (rng.normal(size=(n, length)) + 1.5 * label).astype(np.float32)   # distinct values: no ties
    r = XB.RankedSplice(score, label)
    assert np.isclose(r.ap(), direct_ap(score.reshape(-1), label.reshape(-1), np.ones(n * length, int)))
    w = np.array([0, 2, 1, 3, 0, 1])
    assert np.isclose(r.ap(w.astype(float)), direct_ap(score.reshape(-1), label.reshape(-1), np.repeat(w, length)))


def test_pooled_pearson_from_window_statistics():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4, 1024, 3)).astype(np.float32)
    y = (x + rng.normal(size=x.shape)).astype(np.float32)
    s = XB.pearson_rows(x, y)
    assert s.shape == (4, 6, 3)
    r = XB.pearson_of(s.sum(0))
    want = [np.corrcoef(x[:, :, k].reshape(-1), y[:, :, k].reshape(-1))[0, 1] for k in range(3)]
    assert np.allclose(r, want, atol=1e-6)


def test_targets_score_and_report_run_end_to_end(genomes, tmp_path, monkeypatch, device):
    """A small KronGenome with KronLSTM heads, saved as a checkpoint, scored on two windows of the test genome with
    context; a second source (a copy with noise) gives the paired comparison; the report writes every table."""
    import json
    import shutil

    import polars as pl
    import torch
    from typer.testing import CliRunner

    from kron_dna import genome as G
    from test_kron_dna_split import small
    from test_kron_dna_xheads_cli import make_store

    if device.type != "cpu":
        import pytest
        pytest.skip("the end-to-end run is the CPU one, with the stand-in for the KronLSTM layer")
    root, hgt = genomes["root"], genomes["hgt"]
    monkeypatch.setattr(XB, "interim_dir", lambda *parts: root.joinpath(*parts))
    monkeypatch.setattr(XB, "WIN", 256)
    store = make_store(tmp_path, hgt, n=3)
    total = int(np.fromfile(hgt / "bases.u8", dtype=np.uint8).size)
    lab_dir = tmp_path / "splice"
    lab_dir.mkdir()
    rng = np.random.default_rng(4)
    lab = np.zeros(total, dtype=np.uint8)
    bits = {"donor_plus": 0, "acceptor_plus": 1, "donor_minus": 2, "acceptor_minus": 3}
    for b in bits.values():
        lab[rng.random(total) < 0.05] |= 1 << b
    lab.tofile(lab_dir / "splice.u8")
    (lab_dir / "splice.json").write_text(json.dumps(dict(
        genome="hgt", total_bases=total, bits=bits, definition="one base per site, per strand", splice_sha256="x",
        check={k: dict(sites=int(((lab >> b) & 1).sum())) for k, b in bits.items()})))
    off = int(pl.read_parquet(hgt / "contigs.parquet").filter(pl.col("chrom") == "chr8")["offset"][0])
    rows = [dict(window_id=f"w{k}", chrom="chr8", start=s, end=s + 256, bases_offset=off + s, context_start=s - 256,
                 context_end=s + 512) for k, s in enumerate((600, 1400))]
    pl.DataFrame(rows).write_csv(tmp_path / "windows.csv")
    bench = tmp_path / "bench"
    run = lambda *a: CliRunner().invoke(XB.app, [str(x) for x in a])
    res = run("targets", bench, "--windows", tmp_path / "windows.csv", "--dnase", store, "--splice", lab_dir, "--name", "hgt")
    assert res.exit_code == 0, res.output + repr(res.exception)
    cfg = G.KronGenomeConfig(**{**small("v2").config.to_dict(), "xheads": [["dnase", 3], ["splice", 4]],
                                "xhead_kind": "kron", "xhead_bin": 4, "xhead_width": 64})
    model = G.KronGenome(cfg)
    torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "step": 7}, tmp_path / "ck.pt")
    res = run("score", bench, f"tiny={tmp_path / 'ck.pt'}", "--context", 768, "--device", "cpu", "--name", "hgt")
    assert res.exit_code == 0, res.output + repr(res.exception)
    d = bench / "pred" / "tiny"
    assert np.load(d / "dnase_128.npy").shape == (2, 2, 3) and np.load(d / "splice.npy").shape == (2, 256, 4)
    for i in (1, 0):                                     # two shards, merged: the unsharded arrays, window by window
        res = run("score", bench, f"sharded={tmp_path / 'ck.pt'}", "--context", 768, "--device", "cpu", "--name", "hgt",
                  "--shard", f"{i}/2", "--dtype", "fp32")
        assert res.exit_code == 0, res.output + repr(res.exception)
    assert not (bench / "pred" / "sharded").exists()
    res = run("merge", bench, "sharded", "--shards", 2)
    assert res.exit_code == 0, res.output + repr(res.exception)
    m = bench / "pred" / "sharded"
    assert np.allclose(np.load(m / "dnase_128.npy"), np.load(d / "dnase_128.npy"), atol=1e-5)
    assert np.array_equal(np.load(m / "splice.npy"), np.load(d / "splice.npy"))
    assert json.loads((m / "index.json").read_text())["windows"] == [0, 1]
    res = run("score", bench, f"capped={tmp_path / 'ck.pt'}", "--context", 768, "--device", "cpu", "--name", "hgt",
              "--chunk", 256)                            # the memory cap: the same predictions
    assert res.exit_code == 0, res.output + repr(res.exception)
    c = bench / "pred" / "capped"
    assert np.allclose(np.load(c / "dnase_128.npy"), np.load(d / "dnase_128.npy"), rtol=1e-4, atol=1e-4)
    assert np.allclose(np.load(c / "splice.npy").astype(np.float32), np.load(d / "splice.npy").astype(np.float32),
                       rtol=1e-2, atol=1e-2)
    shutil.rmtree(c)
    shutil.rmtree(m)                                     # the report below compares two sources
    other = bench / "pred" / "noisy"                     # a second source on the same windows
    other.mkdir()
    np.save(other / "dnase_128.npy", np.load(d / "dnase_128.npy") + rng.normal(size=(2, 2, 3)).astype(np.float32))
    np.save(other / "splice.npy", (np.load(d / "splice.npy").astype(np.float32) + rng.normal(size=(2, 256, 4))).astype(np.float16))
    (other / "index.json").write_text(json.dumps(dict(label="noisy", windows=[0, 1])))
    res = run("report", bench, "--n-boot", 20, "--n-boot-ap", 5, "--min-shared", 2)
    assert res.exit_code == 0, res.output + repr(res.exception)
    r = pl.read_csv(bench / "results.csv")
    assert r.height == 2 * (3 + 4) and set(r["source"]) == {"tiny", "noisy"}
    assert r.filter(pl.col("task") == "splice")["value"].is_between(0, 1).all()
    assert pl.read_csv(bench / "pairs.csv").height == 1 and (bench / "chart_2windows.png").exists()
