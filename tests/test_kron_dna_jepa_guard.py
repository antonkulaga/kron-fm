"""JEPA collapse guard (``genome-train --jepa-guard off|monitor|on``): the measures, the trigger rule, the off path
against b52e2b9, the loss while not triggered, a forced trigger (an injected collapsed core) through weight doubling,
the cap, the checkpoint and exit code 42, and resume of the guard state.

CLI tests run genome-train on the CPU with the stub model of ``test_kron_dna_jepa_block.py``."""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from kron_dna import genome as G

from test_kron_dna_multigenome import metric_rows, run
from test_kron_dna_jepa_block import JEPA, SMALLJ, JepaStub, jgenomes, same_events, stub_jepa  # noqa: F401  (jgenomes: a fixture)

BASE = "b52e2b9"         # block JEPA without the guard
REAL_METRICS = G.jepa_guard_metrics
GUARD = ["--jepa-guard-windows", "2", "--jepa-guard-window", "4096"]


def cli_at_base(monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    if os.environ.get("KRON_DNA_BASE_CLI_B52E2B9"):
        src = Path(os.environ["KRON_DNA_BASE_CLI_B52E2B9"]).read_text()
    else:
        try:
            src = subprocess.run(["git", "-C", str(repo), "show", f"{BASE}:src/kron_dna/cli.py"], capture_output=True,
                                 check=True, text=True).stdout
        except (OSError, subprocess.CalledProcessError):
            pytest.skip(f"commit {BASE} is not reachable from {repo} (set KRON_DNA_BASE_CLI_B52E2B9 to its cli.py)")
    mod = types.ModuleType("kron_dna_cli_b52e2b9")
    mod.__file__ = f"{BASE}:src/kron_dna/cli.py"
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod


def jepa_rows(path: Path) -> list[dict]:
    lines = path.read_text().strip().split("\n")
    head = lines[0].split(",")
    return [dict(zip(head, line.split(","))) for line in lines[1:]]


def guard_rows(path: Path) -> list[dict]:
    return [r for r in jepa_rows(path) if r.get("row") == "guard"]


def train_events(ev):
    return [e for e in ev if e[0] == "forward" and e[1]]


def same_tensors(a, b):
    return same_events(a, b)


# ----------------------------------------------------------------------------------------------- pure functions

def test_guard_rule_doubles_caps_and_stops():
    st = G.jepa_guard_new()
    seq = [(100.0, 0.5, "ok", 1.0), (40.0, 0.5, "triggered", 1.0), (40.0, 0.5, "raise", 2.0),
           (90.0, 0.5, "ok", 2.0),                                   # a healthy evaluation resets the run of hits
           (100.0, 0.01, "triggered", 2.0), (100.0, 0.01, "raise", 4.0), (100.0, 0.01, "raise", 8.0),
           (100.0, 0.01, "stop", 8.0)]
    for rank, sd, action, mult in seq:
        info = G.jepa_guard_update(st, rank, sd, act=True)
        assert (info["action"], st["mult"]) == (action, mult), (rank, sd, info)
    assert st["stopped"] and st["evals"] == 8
    # triggered evaluations do not enter the references: max stays 100, minima are the 2 healthy ones
    assert st["max_rank"] == 100.0 and st["std_mins"] == [0.5, 0.5]
    # thresholds: 0.5 x max and 0.05 x median, strict
    st = G.jepa_guard_new()
    G.jepa_guard_update(st, 80.0, 0.5, act=True)
    assert G.jepa_guard_update(st, 40.0, 0.025, act=True)["action"] == "ok"          # exactly at both bars
    assert G.jepa_guard_update(st, 39.9, 0.4, act=True)["trigger_rank"]
    st = G.jepa_guard_new()
    G.jepa_guard_update(st, 80.0, 0.5, act=True)
    assert G.jepa_guard_update(st, 80.0, 0.0249, act=True)["trigger_std"]
    # not finite -> triggered; monitor never acts
    st = G.jepa_guard_new()
    G.jepa_guard_update(st, 80.0, 0.4, act=False)
    assert G.jepa_guard_update(st, float("nan"), 0.4, act=False)["action"] == "triggered"
    for _ in range(5):
        assert G.jepa_guard_update(st, 1.0, 0.0, act=False)["action"] == "would-raise"
    assert st["mult"] == 1.0 and not st["stopped"]


def test_guard_measures_healthy_and_collapsed():
    g = torch.Generator().manual_seed(0)
    w, b, d = 3, 64, 32
    healthy = torch.randn(w, b, d, generator=g)
    hidden = torch.zeros(w, b, dtype=torch.bool)
    hidden[:, 10:20] = True
    m = G.jepa_guard_metrics(healthy, healthy, healthy, healthy, hidden)
    assert m["entropy_rank"] > 0.8 * d and m["participation_ratio"] > 0.5 * d
    assert m["jepa_hidden"] == pytest.approx(0, abs=1e-6) and m["top1_hidden"] == 1.0 and m["top1_chance"] == 1 / b
    collapsed = torch.randn(d, generator=g).expand(w, b, d).clone()
    collapsed[..., 0] += 1e-3 * torch.randn(w, b, generator=g)
    mc = G.jepa_guard_metrics(collapsed, healthy, collapsed, collapsed, hidden)
    assert mc["entropy_rank"] < 1.1 and mc["std_min"] < 1e-5 and mc["top1_hidden"] < 0.2
    assert mc["jepa_hidden"] > 0.5 and mc["copy_hidden"] == pytest.approx(mc["jepa_hidden"])
    # a copy that is right on visible bins and wrong on hidden ones
    other = torch.randn(w, b, d, generator=g)
    online = torch.where(hidden.unsqueeze(-1), other, healthy)
    mo = G.jepa_guard_metrics(healthy, healthy, online, online, hidden)
    assert mo["copy_visible"] == pytest.approx(0, abs=1e-6) and mo["copy_hidden"] > 0.5


# ------------------------------------------------------------------------------------------------ CLI on the CPU

@pytest.mark.parametrize("case", ["jepa", "blocks-log"])
def test_guard_off_is_b52e2b9(jgenomes, tmp_path, monkeypatch, case):
    """Without --jepa-guard everything is as at b52e2b9: every model input, metrics rows, the checkpoint keys and
    jepa.csv (bar the timing columns)."""
    root = jgenomes["root"]
    args = SMALLJ + JEPA + ["--bert-mask", "--steps", "6", "--eval-every", "3", "--batch-size", "3"]
    if case == "blocks-log":
        args += ["--jepa-block-share", "0.25", "--jepa-block-len", "256:1024", "--jepa-log-every", "2"]
    stub_jepa(monkeypatch)
    from kron_dna import cli as new
    old = cli_at_base(monkeypatch)
    res_o, ev_o = run(old, monkeypatch, root, tmp_path / "old", args)
    assert res_o.exit_code == 0, res_o.output + repr(res_o.exception)
    res_n, ev_n = run(new, monkeypatch, root, tmp_path / "new", args)
    assert res_n.exit_code == 0, res_n.output + repr(res_n.exception)
    assert same_tensors(ev_n, ev_o)
    assert metric_rows(tmp_path / "new" / "metrics.csv") == metric_rows(tmp_path / "old" / "metrics.csv")
    ck_o = torch.load(tmp_path / "old" / "checkpoint.pt", map_location="cpu", weights_only=False)
    ck_n = torch.load(tmp_path / "new" / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert "jepa_guard" not in ck_n and set(ck_n) == set(ck_o)
    assert all(torch.equal(ck_n["model"][k], ck_o["model"][k]) for k in ck_o["model"])
    assert "jepa guard" not in res_n.output
    if case == "blocks-log":
        ro, rn = jepa_rows(tmp_path / "old" / "jepa.csv"), jepa_rows(tmp_path / "new" / "jepa.csv")
        assert list(ro[0]) == list(rn[0]) and "row" not in rn[0]
        drop = ("sec_per_step", "peak_gib")
        assert [{k: v for k, v in r.items() if k not in drop} for r in ro] == \
               [{k: v for k, v in r.items() if k not in drop} for r in rn]
    else:
        assert not (tmp_path / "new" / "jepa.csv").exists()


def test_guard_on_untriggered_leaves_the_loss_alone(jgenomes, tmp_path, monkeypatch):
    """--jepa-guard on that never triggers (and monitor): the same training inputs, losses, held-out bits and
    weights as without the guard; one guard row per evaluation with finite measures; the state in the checkpoint."""
    root = jgenomes["root"]
    args = SMALLJ + JEPA + ["--bert-mask", "--steps", "6", "--eval-every", "2", "--batch-size", "3",
                            "--jepa-block-share", "0.25", "--jepa-block-len", "256:1024"]
    stub_jepa(monkeypatch)
    from kron_dna import cli as new
    res0, ev0 = run(new, monkeypatch, root, tmp_path / "off", args)
    assert res0.exit_code == 0, res0.output + repr(res0.exception)
    for mode in ("on", "monitor"):
        res1, ev1 = run(new, monkeypatch, root, tmp_path / mode, args + GUARD + [
            "--jepa-guard", mode, "--jepa-guard-rank", "0.001", "--jepa-guard-std", "0.0001"])
        assert res1.exit_code == 0, res1.output + repr(res1.exception)
        assert same_tensors(train_events(ev1), train_events(ev0))
        assert metric_rows(tmp_path / mode / "metrics.csv") == metric_rows(tmp_path / "off" / "metrics.csv")
        a = torch.load(tmp_path / "off" / "checkpoint.pt", map_location="cpu", weights_only=False)
        b = torch.load(tmp_path / mode / "checkpoint.pt", map_location="cpu", weights_only=False)
        assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
        assert b["jepa_guard"]["mult"] == 1.0 and b["jepa_guard"]["evals"] == 3 and not b["jepa_guard"]["stopped"]
        rows = guard_rows(tmp_path / mode / "jepa.csv")
        assert [int(r["step"]) for r in rows] == [2, 4, 6] and all(r["g_action"] == "ok" for r in rows)
        for r in rows:
            for k in ("g_entropy_rank", "g_participation_ratio", "g_std_min", "g_std_mean", "g_jepa_hidden",
                      "g_jepa_visible", "g_copy_hidden", "g_top1_hidden"):
                assert math.isfinite(float(r[k])), (k, r)
            assert float(r["g_reg_weight"]) == 1.0 and float(r["g_top1_chance"]) == 1 / 32 and r["g_windows"] == "2"
        assert "jepa guard (" + mode + "): 2 fixed held-out windows of 4,096 bases" in " ".join(res1.output.split())


def inject_collapse(monkeypatch, from_call: int):
    """From the from_call-th measurement on, the clean core states the guard watches are a collapsed set: one
    vector plus noise of 1e-6 in a single dimension (an injected collapsed core)."""
    calls = [0]
    real = REAL_METRICS

    def fake(clean, target, online, pred, hidden):
        calls[0] += 1
        if calls[0] >= from_call:
            g = torch.Generator().manual_seed(calls[0])
            c = clean.mean((0, 1), keepdim=True).expand_as(clean).clone()
            c[..., 0] += 1e-6 * torch.randn(c.shape[:2], generator=g)
            clean = c
        return real(clean, target, online, pred, hidden)

    monkeypatch.setattr(G, "jepa_guard_metrics", fake)
    return calls


def test_forced_trigger_doubles_caps_saves_and_exits_42(jgenomes, tmp_path, monkeypatch):
    root = jgenomes["root"]
    args = SMALLJ + JEPA + ["--bert-mask", "--steps", "10", "--eval-every", "1", "--batch-size", "3"] + GUARD
    stub_jepa(monkeypatch)
    from kron_dna import cli as new
    inject_collapse(monkeypatch, 2)
    res, ev = run(new, monkeypatch, root, tmp_path / "on", args + ["--jepa-guard", "on"])
    assert res.exit_code == 42, res.output + repr(res.exception)
    rows = guard_rows(tmp_path / "on" / "jepa.csv")
    assert [r["g_action"] for r in rows] == ["ok", "triggered", "raise", "raise", "raise", "stop"]
    assert [float(r["g_mult"]) for r in rows] == [1, 1, 2, 4, 8, 8]
    assert [float(r["g_reg_weight"]) for r in rows] == [1, 1, 2, 4, 8, 8]
    assert all(float(r["g_entropy_rank"]) < 1.5 for r in rows[1:]) and float(rows[0]["g_entropy_rank"]) > 1.5
    text = " ".join(res.output.split())
    assert text.count("WARNING: JEPA collapse guard") == 5 and "saving and stopping" in text
    assert "JEPA collapse guard: stopped at step 6 (exit code 42)" in text
    assert (tmp_path / "on" / "JEPA_GUARD_STOP").exists()
    ck = torch.load(tmp_path / "on" / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert ck["step"] == 6 and ck["jepa_guard"]["mult"] == 8.0 and ck["jepa_guard"]["stopped"]
    assert ck["jepa_guard"]["hits"] == 5 and len(ck["jepa_guard"]["std_mins"]) == 1
    assert [r[0] for r in metric_rows(tmp_path / "on" / "metrics.csv")][1:] == [str(i) for i in range(1, 7)]
    # the raised weight is the weight in the loss: against monitor (same injection, nothing raised) the losses agree
    # up to step 3 (the first raise follows evaluation 3) and differ from step 4
    inject_collapse(monkeypatch, 2)
    res_m, ev_m = run(new, monkeypatch, root, tmp_path / "mon", args + ["--jepa-guard", "monitor"])
    assert res_m.exit_code == 0, res_m.output + repr(res_m.exception)
    assert [r["g_action"] for r in guard_rows(tmp_path / "mon" / "jepa.csv")] == \
           ["ok", "triggered"] + ["would-raise"] * 8
    mo, mm = metric_rows(tmp_path / "on" / "metrics.csv")[1:], metric_rows(tmp_path / "mon" / "metrics.csv")[1:]
    assert [r[2] for r in mo[:3]] == [r[2] for r in mm[:3]]
    assert all(a[2] != b[2] for a, b in zip(mo[3:6], mm[3:6]))
    assert same_tensors(train_events(ev)[:4], train_events(ev_m)[:4])       # same inputs: only the weight differs


def test_resume_restores_the_guard_state(jgenomes, tmp_path, monkeypatch):
    root = jgenomes["root"]
    args = SMALLJ + JEPA + ["--bert-mask", "--steps", "10", "--eval-every", "1", "--batch-size", "3"] + GUARD + [
        "--jepa-guard", "on"]
    stub_jepa(monkeypatch)
    from kron_dna import cli as new
    out = tmp_path / "run"
    inject_collapse(monkeypatch, 2)
    res, _ = run(new, monkeypatch, root, out, args + ["--stop-step", "4"])
    assert res.exit_code == 0, res.output + repr(res.exception)
    ck = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert ck["step"] == 4 and ck["jepa_guard"]["mult"] == 4.0 and ck["jepa_guard"]["hits"] == 3
    saved = dict(ck["jepa_guard"])
    inject_collapse(monkeypatch, 1)                                   # still collapsed after the restart
    res, _ = run(new, monkeypatch, root, out, args)
    assert res.exit_code == 42, res.output + repr(res.exception)
    text = " ".join(res.output.split())
    assert "resumed at step 4" in text
    assert (f"jepa guard: state restored from the checkpoint: --jepa-reg weight x4, 3 triggered evaluation(s) in a "
            f"row, running max entropy rank {round(saved['max_rank'], 2)}, 1 reference minima") in text
    rows = guard_rows(out / "jepa.csv")
    assert [(int(r["step"]), r["g_action"], float(r["g_mult"])) for r in rows] == [
        (1, "ok", 1.0), (2, "triggered", 1.0), (3, "raise", 2.0), (4, "raise", 4.0), (5, "raise", 8.0),
        (6, "stop", 8.0)]
    assert float(rows[4]["g_ref_max_rank"]) == pytest.approx(saved["max_rank"], rel=1e-5)
    ck = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert ck["step"] == 6 and ck["jepa_guard"]["mult"] == 8.0 and ck["jepa_guard"]["evals"] == 6
    # resumed with the guard off: the saved factor is reported and not applied
    res, _ = run(new, monkeypatch, root, out, [a for a in args if a not in ("--jepa-guard", "on")] + ["--steps", "12"])
    assert res.exit_code == 0, res.output + repr(res.exception)
    assert "not applied with --jepa-guard off" in " ".join(res.output.split())


def test_old_runs_resume_with_the_guard(jgenomes, tmp_path, monkeypatch):
    """A b52e2b9 checkpoint (no guard state) with a narrow jepa.csv resumes with --jepa-guard on: the file is widened
    (earlier rows marked train) and guard rows follow."""
    root = jgenomes["root"]
    stub_jepa(monkeypatch)
    from kron_dna import cli as new
    old = cli_at_base(monkeypatch)
    out = tmp_path / "run"
    args = SMALLJ + JEPA + ["--bert-mask", "--batch-size", "3", "--eval-every", "2", "--steps", "6",
                            "--jepa-block-share", "0.2", "--jepa-block-len", "512:1024", "--jepa-log-every", "1"]
    res, _ = run(old, monkeypatch, root, out, args + ["--stop-step", "2"])
    assert res.exit_code == 0, res.output + repr(res.exception)
    res, _ = run(new, monkeypatch, root, out, args + GUARD + ["--jepa-guard", "on"])
    assert res.exit_code == 0, res.output + repr(res.exception)
    rows = jepa_rows(out / "jepa.csv")
    assert [(r["step"], r["row"]) for r in rows] == [("1", "train"), ("2", "train"), ("3", "train"), ("4", "train"),
                                                    ("4", "guard"), ("5", "train"), ("6", "train"), ("6", "guard")]
    assert torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=False)["jepa_guard"]["evals"] == 2


@pytest.mark.parametrize("more,words", [
    (["--jepa-guard", "maybe"], "must be off, monitor or on"),
    (["--jepa-guard", "on", "--jepa-reg", "none"], "needs --jepa-reg sigreg or vicreg"),
    (["--jepa-guard", "on", "--jepa-guard-window", "1000"], "positive multiple of 128"),
    (["--jepa-guard", "on", "--jepa-guard-cap", "1"], "--jepa-guard-cap >= 2"),
    (["--jepa-guard", "monitor", "--jepa-guard-window", "65536"], "no held-out contig is that long"),
])
def test_guard_option_checks(jgenomes, tmp_path, monkeypatch, more, words):
    root = jgenomes["root"]
    stub_jepa(monkeypatch)
    from kron_dna import cli as new
    args = SMALLJ + JEPA + ["--steps", "2", "--batch-size", "2"] + more
    if "--jepa-reg" in more:
        args = [a for i, a in enumerate(args) if not (a == "--jepa-reg" or (i > 0 and args[i - 1] == "--jepa-reg"))] \
            + ["--jepa-reg", "none"]
    res, _ = run(new, monkeypatch, root, tmp_path / "x", args)
    flat = " ".join(re.sub("[\u2502\u256d\u256e\u256f\u2570\u2500]", " ", res.output).split())
    assert res.exit_code != 0 and words in flat, res.output


# ---------------------------------------------------------------------------------------- two data-parallel ranks

class DdpJepaStub(JepaStub):
    """The stub with every parameter in each step's graph (its integer- and float-input paths both touched, at
    weight 0), as plain DistributedDataParallel needs."""

    def forward(self, seq, segments=None, context=None, extra=None):
        x, core, t = super().forward(seq, segments, context, extra)
        return x + 0.0 * sum(p_.sum() for p_ in (self.emb.weight, self.lin.weight, self.lin.bias)), core, t


def guard_ddp_worker(root: str, out: str, result: str) -> None:
    """One rank of a two-process run on the CPU (gloo instead of nccl, DistributedDataParallel around the stub). Only
    rank 0 measures (an injected collapsed core from its second evaluation); every rank records the factor on the
    --jepa-reg weight in effect at each optimizer step, and its exit code."""
    import torch.distributed as dist
    mp = pytest.MonkeyPatch()
    stub_jepa(mp)
    real_init, real_ddp = dist.init_process_group, torch.nn.parallel.DistributedDataParallel
    mp.setattr(dist, "init_process_group", lambda backend, **k: real_init("gloo", **k))
    mp.setattr(torch.cuda, "set_device", lambda *a, **k: None)
    mp.setattr(torch.nn.parallel, "DistributedDataParallel", lambda m, device_ids=None: real_ddp(m))
    mp.setattr(G, "KronGenome", DdpJepaStub)
    states = []
    real_new = G.jepa_guard_new
    mp.setattr(G, "jepa_guard_new", lambda: states.append(real_new()) or states[-1])
    mults = []
    real_clip = torch.nn.utils.clip_grad_norm_

    def clip(params, max_norm, *a, **k):
        mults.append(states[-1]["mult"])
        return real_clip(params, max_norm, *a, **k)

    mp.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    calls = inject_collapse(mp, 2)
    from kron_dna import cli as new
    args = SMALLJ + JEPA + ["--bert-mask", "--steps", "10", "--eval-every", "1", "--batch-size", "2",
                            "--jepa-block-share", "0.25", "--jepa-block-len", "256:1024"] + GUARD + ["--jepa-guard", "on"]
    res, _ = run(new, mp, Path(root), Path(out), args)
    Path(result).write_text(json.dumps(dict(exit=res.exit_code, mults=mults, measured=calls[0],
                                            stopped=states[-1]["stopped"], output=res.output,
                                            exc=repr(res.exception))))
    mp.undo()


def test_guard_on_two_ranks_decides_on_rank_0_and_broadcasts(jgenomes, tmp_path):
    import socket
    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0))
        port = sk.getsockname()[1]
    procs = []
    for rank in (0, 1):
        env = dict(os.environ, RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2", MASTER_ADDR="127.0.0.1",
                   MASTER_PORT=str(port), OMP_NUM_THREADS="1",
                   PYTHONPATH=os.pathsep.join([str(Path(__file__).parent), str(Path(G.__file__).parents[1]),
                                               os.environ.get("PYTHONPATH", "")]))
        code = (f"import test_kron_dna_jepa_guard as T; T.guard_ddp_worker({str(jgenomes['root'])!r}, "
                f"{str(tmp_path / 'run')!r}, {str(tmp_path / f'rank{rank}.json')!r})")
        procs.append(subprocess.Popen([sys.executable, "-c", code], env=env, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True))
    outs = [q.communicate(timeout=600)[0] for q in procs]
    assert all(q.returncode == 0 for q in procs), "\n".join(o[-3000:] for o in outs)
    r0, r1 = (json.loads((tmp_path / f"rank{k}.json").read_text()) for k in (0, 1))
    assert "data-parallel over 2 cards" in " ".join(r0["output"].split())
    assert r0["exit"] == r1["exit"] == 42, (r0["exc"], r1["exc"], r0["output"][-1500:])
    assert r0["measured"] == 6 and r1["measured"] == 0                  # only rank 0 measures
    assert r0["mults"] == r1["mults"] == [1.0, 1.0, 1.0, 2.0, 4.0, 8.0]  # the factor rank 0 chose, on both ranks
    assert r0["stopped"] and r1["stopped"]
    assert [r["g_action"] for r in guard_rows(tmp_path / "run" / "jepa.csv")] == \
           ["ok", "triggered", "raise", "raise", "raise", "stop"]
    ck = torch.load(tmp_path / "run" / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert ck["step"] == 6 and ck["jepa_guard"]["mult"] == 8.0
