"""genome-train --xhead end to end on the CPU with a stand-in model: a continuation from a parent checkpoint trains
the new head alone first (the inherited weights do not move), then both, each group at its own rate under the run's
schedule; a stopped and resumed run is at the same phase, counts and rates as one that ran through."""
from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest
import torch

from kron_dna import cli
from kron_dna import genome as G
from kron_dna.cli import genome_lr
import test_kron_dna_multigenome as MGT
from test_kron_dna_multigenome import SMALL, cpu_only, genomes, make_ntv3, run  # noqa: F401  (genomes is a fixture)


class StubWithHeads(MGT.StubGenome):
    """The stand-in model with further heads: each a linear map of the input's embedding."""

    def __init__(self, config):
        super().__init__(config)
        self.xheads = torch.nn.ModuleDict({n: torch.nn.Linear(4, k) for n, k in config.xheads})
        self.x_active, self.last_x = None, {}

    def forward(self, seq, segments=None, context=None):
        h = self.emb(seq.long())
        self.last_x = {n: m(h) for n, m in self.xheads.items() if self.x_active is None or n in self.x_active}
        return h, None, None


def make_store(root, primary, n=3):
    total = int(np.fromfile(primary / "bases.u8", dtype=np.uint8).size)
    d = root / "store"
    d.mkdir()
    rng = np.random.default_rng(3)
    dense = rng.integers(0, 255, (total, n), dtype=np.uint8)
    dense[rng.random((total, n)) < 0.5] = 0
    dense.tofile(d / "dense.u8")
    (d / "dense.json").write_text(json.dumps(dict(
        ids=[f"F{k}" for k in range(n)], total_bases=total, n_tracks=n, scales=[3.0] * n, assays=["DNase-seq"] * n,
        nonzero_mean=[1.0] * n, mean_stride=64, encoding="uint8 = log1p(signal)/log1p_scale*255, clipped",
        genome="hgt", tracks=[{} for _ in range(n)])))
    return d


LR, STEPS = 1e-2, 16
STAGE = ["--xhead-only", "3", "--xhead-warmup", "2", "--inherit-warmup", "2", "--xhead-lr", "2e-2", "--inherit-lr",
         "1e-3"]


def setup(genomes, tmp_path, monkeypatch):
    cpu_only(monkeypatch)
    monkeypatch.setattr(G, "KronGenome", StubWithHeads)
    root = genomes["root"]
    nd = make_ntv3(root, genomes["hgt"], genomes["gA"])
    idx = pl.read_parquet(genomes["hgt"] / "contigs.parquet")       # as the real split: 4 on the held-out chromosomes
    split = np.fromfile(nd / "split.u8", dtype=np.uint8)
    for r in idx.filter(pl.col("chrom").is_in(["chr8", "chr9"])).iter_rows(named=True):
        split[r["offset"]:r["offset"] + r["end"] - r["start"]] = 4
    split.tofile(nd / "split.u8")
    store = make_store(tmp_path, genomes["hgt"])
    common = SMALL + ["--eval-every", "1", "--lr", str(LR), "--ntv3-dir", str(nd), "--ntv3-assays", "all"]
    parent = tmp_path / "parent"
    res, _ = run(cli, monkeypatch, root, parent, common + ["--curriculum", "1024:4"])
    assert res.exit_code == 0, res.output + repr(res.exception)
    child = common + ["--curriculum", f"1024:{STEPS}", "--init", str(parent / "checkpoint.pt"), "--start-step", "4",
                      "--xhead", f"dnase:{store}", "--xhead", "splice:2"] + STAGE
    return root, parent, store, child


def expected_inherited(step_before_update: int, joint_updates_done: int) -> float:
    factor = genome_lr(step_before_update, LR, 1, STEPS) / LR
    return 1e-3 * factor * min(1.0, (joint_updates_done + 1) / 2)


def replay(d, prev=0, joint_done=0, phase="head-only"):
    """Check a head's xheads.csv rows against the stage replayed from the logged counts: a step is labelled when the
    head's count moved (a window can hold no supervised position), and each row carries the rates of the update that
    led to it, at step - 1 on the run's schedule. Returns the final count and the step the joint phase began."""
    switch = None
    for row in d.iter_rows(named=True):
        t = row["step"] - 1
        factor = genome_lr(t, LR, 1, STEPS) / LR
        labelled = row["updates"] - prev
        assert labelled in (0, 1)
        head = 2e-2 * factor * min(1.0, (prev + 1) / 2) if labelled else 0.0
        inh = expected_inherited(t, joint_done) if phase == "joint" else 0.0
        assert row["lr"] == pytest.approx(head, rel=2e-3, abs=1e-12), row
        assert row["lr_inherited"] == pytest.approx(inh, rel=2e-3, abs=1e-12), row
        if phase == "joint":
            joint_done += 1
        elif row["updates"] >= 3:
            phase, switch = "joint", row["step"]
        assert row["phase"] == phase, row
        prev = row["updates"]
    return prev, switch


def test_head_only_then_joint_at_scheduled_rates(genomes, tmp_path, monkeypatch):
    root, parent, store, child = setup(genomes, tmp_path, monkeypatch)
    out = tmp_path / "child"
    res, _ = run(cli, monkeypatch, root, out, child)
    assert res.exit_code == 0, res.output + repr(res.exception)
    text = MGT.flat(res.output)
    assert "the new parameters are the heads' only" in text
    x = pl.read_csv(out / "xheads.csv")
    d = x.filter(pl.col("head") == "dnase").sort("step")
    assert d["step"].to_list() == list(range(5, STEPS + 1))
    s = x.filter(pl.col("head") == "splice")
    assert set(s["updates"].to_list()) == {0} and set(s["lr"].to_list()) == {0.0}   # registered, no data: untouched
    prev, switch = replay(d)
    assert switch is not None and f"head-only phase done after step {switch}" in text and prev >= 6
    m = pl.read_csv(out / "metrics.csv")                                   # its lr column is the inherited rate
    assert m["lr"].to_list() == pytest.approx(d["lr_inherited"].to_list(), rel=2e-3, abs=1e-12)
    # the inherited weights stood still through the head-only phase, the head moved; then both moved
    p = torch.load(parent / "checkpoint.pt", map_location="cpu", weights_only=False)
    h = torch.load(out / f"head-only-step{switch}.pt", map_location="cpu", weights_only=False)
    f = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert torch.equal(h["model"]["emb.weight"], p["model"]["emb.weight"])
    assert not torch.equal(f["model"]["emb.weight"], p["model"]["emb.weight"])
    assert not torch.equal(f["model"]["xheads.dnase.weight"], h["model"]["xheads.dnase.weight"])
    assert f["config"]["xheads"] == [["dnase", 3], ["splice", 2]]
    st = f["xstage"]["state"]
    assert st["phase"] == "joint" and st["updates"] == {"dnase": prev, "splice": 0} and st["active"] == {"dnase": 0}
    assert st["stores"]["dnase"]["ids"] == ["F0", "F1", "F2"] and st["joint_step"] == STEPS - switch
    assert st["stage_step"] == STEPS - 4
    assert np.isfinite(d["heldout_mean"].cast(pl.Float64).to_numpy()).all() and (d["heldout_n"] > 0).all()
    assert set(d["heldout_metric"].to_list()) == {"Pearson"}


def test_a_resumed_stage_goes_on_from_the_checkpoint(genomes, tmp_path, monkeypatch):
    """A resumed run draws its windows from step-keyed streams (audit TR04), so its labelled steps may differ from an
    uninterrupted run's; what must hold is that it goes on from the saved phase and counts at the scheduled rates."""
    root, parent, store, child = setup(genomes, tmp_path, monkeypatch)
    cut = tmp_path / "cut"
    res, _ = run(cli, monkeypatch, root, cut, child + ["--stop-step", "6"])        # stops inside the head-only phase
    assert res.exit_code == 0, res.output + repr(res.exception)
    mid = torch.load(cut / "checkpoint.pt", map_location="cpu", weights_only=False)["xstage"]["state"]
    assert mid["phase"] == "head-only" and mid["stage_step"] == 2
    res, _ = run(cli, monkeypatch, root, cut, [a for a in child if a not in ("--init", str(parent / "checkpoint.pt"),
                                                                             "--start-step", "4")])
    assert res.exit_code == 0, res.output + repr(res.exception)
    assert f"phase head-only, stage step 2, updates {mid['updates']}" in MGT.flat(res.output)
    d = pl.read_csv(cut / "xheads.csv").filter(pl.col("head") == "dnase").sort("step")
    assert d["step"].to_list() == list(range(5, STEPS + 1))
    prev, switch = replay(d)                                   # one schedule across the stop, as if never stopped
    end = torch.load(cut / "checkpoint.pt", map_location="cpu", weights_only=False)["xstage"]["state"]
    assert end["stage_step"] == STEPS - 4 and end["updates"]["dnase"] == prev and end["joint_step"] == STEPS - switch
    assert end["active"] == mid["active"] and end["stores"] == mid["stores"]


def test_resume_refuses_another_store_or_other_settings(genomes, tmp_path, monkeypatch):
    root, parent, store, child = setup(genomes, tmp_path, monkeypatch)
    out = tmp_path / "child"
    res, _ = run(cli, monkeypatch, root, out, child + ["--stop-step", "6"])
    assert res.exit_code == 0, res.output + repr(res.exception)
    again = [a for a in child if a not in ("--init", str(parent / "checkpoint.pt"), "--start-step", "4")]
    j = json.loads((store / "dense.json").read_text())
    (store / "dense.json").write_text(json.dumps(j | {"ids": ["F0", "F2", "F1"]}))  # the channels reordered
    res, _ = run(cli, monkeypatch, root, out, again)
    assert res.exit_code != 0 and "another store" in MGT.flat(res.output + repr(res.exception))
    (store / "dense.json").write_text(json.dumps(j))
    res, _ = run(cli, monkeypatch, root, out, again + ["--xhead-only", "5"])
    assert res.exit_code != 0 and "other settings" in MGT.flat(res.output + repr(res.exception))
    res, _ = run(cli, monkeypatch, root, out, [a for a in again if not a.startswith(("--xhead", "dnase:", "splice:"))
                                               and a not in STAGE])
    assert res.exit_code != 0


def test_without_xhead_nothing_changes(genomes, tmp_path, monkeypatch):
    cpu_only(monkeypatch)
    res, _ = run(cli, monkeypatch, genomes["root"], tmp_path / "r", SMALL + ["--curriculum", "1024:3", "--eval-every", "1"])
    assert res.exit_code == 0, res.output + repr(res.exception)
    state = torch.load(tmp_path / "r" / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert "xstage" not in state and state["config"]["xheads"] == [] and state["config"]["xhead_hidden"] == 0 and not (tmp_path / "r" / "xheads.csv").exists()
    assert len(state["optimizer"]["param_groups"]) == 2


def test_a_label_store_trains_a_binary_head_and_reports_average_precision(genomes, tmp_path, monkeypatch):
    root, parent, store, child = setup(genomes, tmp_path, monkeypatch)
    total = int(np.fromfile(genomes["hgt"] / "bases.u8", dtype=np.uint8).size)
    d = tmp_path / "labels"
    d.mkdir()
    rng = np.random.default_rng(4)
    lab = np.zeros(total, dtype=np.uint8)
    bits = {"donor_plus": 0, "acceptor_plus": 1, "donor_minus": 2, "acceptor_minus": 3}
    for b in bits.values():
        lab[rng.random(total) < 0.02] |= 1 << b
    lab.tofile(d / "splice.u8")
    (d / "splice.json").write_text(json.dumps(dict(
        genome="hgt", total_bases=total, bits=bits, definition="one base per site, per strand", splice_sha256="x",
        check={k: dict(sites=int(((lab >> b) & 1).sum())) for k, b in bits.items()})))
    args = [a if a != "splice:2" else f"splice:{d}" for a in child] + ["--rc"]
    out = tmp_path / "child"
    res, _ = run(cli, monkeypatch, root, out, args)
    assert res.exit_code == 0, res.output + repr(res.exception)
    x = pl.read_csv(out / "xheads.csv")
    s = x.filter(pl.col("head") == "splice").sort("step")
    assert set(s["heldout_metric"].to_list()) == {"average_precision"} and s["updates"][-1] >= 6
    assert np.isfinite(s["heldout_mean"].cast(pl.Float64).to_numpy()).all() and (s["heldout_n"] > 0).all()
    moved = np.diff(np.concatenate([[0], s["updates"].to_numpy()])) > 0        # rows after a labelled step
    loss = s["train_loss"].cast(pl.Float64).to_numpy()
    assert np.isfinite(loss[moved]).all() and np.isnan(loss[~moved]).all() and loss[moved][-1] < loss[moved][0]
    f = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=False)
    p = torch.load(parent / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert f["config"]["xheads"] == [["dnase", 3], ["splice", 4]]
    assert f["xstage"]["state"]["stores"]["splice"]["definition"] == "one base per site, per strand"
    assert "xheads.splice.weight" in f["model"] and "xheads.splice.weight" not in p["model"]
