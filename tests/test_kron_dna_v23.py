"""v2.3 = v2.2 (no-pad, block JEPA, collapse guard) + v2.3-pre (split-window, split + pack, balanced split plan).

What the merge has to get right, tested here on the CPU (two logical pieces on one device with real parameter
copies, max-pool gradients routed continuously, as test_kron_dna_split / test_kron_dna_v23pre):

- options off: v2.3 trains bit-identically to each parent's cli.py: to v2.2 (cd2f87b) without --split-cards (with
  and without block JEPA + guard + jepa.csv, pad and pack) and to v2.3-pre (945034b) without the block-JEPA / guard /
  jepa.csv options (split off and on);
- split + block JEPA + pack: the first step's total losses and gradients equal those of the same run unsplit within
  split-window's tolerances, and so does every loss term of every packed part (masked-base, epigenome, NTv3 track,
  the JEPA hidden-bin term, SIGReg on visible bins, the copy baselines); the same blocks are drawn; the online input
  forward_split receives is MASK inside blocks with the repeat channel zeroed there; the target pass goes through
  forward_split(core_only=True);
- the collapse guard in split mode: a forced trigger raises the --jepa-reg weight x2/x4/x8, then saves a checkpoint,
  writes JEPA_GUARD_STOP and exits 42 (also as a process exit code), with its measurement unsplit (no evaluation
  forward goes through forward_split).
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from kron_dna import genome as G
from kron_dna import packing as PK

import test_kron_dna_jepa_guard as GT
from test_kron_dna_multigenome import genomes, metric_rows, run  # noqa: F401  (genomes is a fixture)
from test_kron_dna_split import _restore_precision, make_recipe_data, real_model_cpu, recipe, smooth_routing  # noqa: F401
from test_kron_dna_v23pre import CASES, OFF_ARGS, assert_identical, parent_cli, spy_plans, trained, with_option

V22_PARENT, V23PRE_PARENT = "cd2f87b", "945034b"
BLOCK = ["--jepa-block-share", "0.25", "--jepa-block-len", "64:256"]            # n-down 4: 16-base bins, 4-16 bins
GUARD = ["--jepa-guard-windows", "2", "--jepa-guard-window", "1024"]             # held-out chr8 2600, chr9 1900
QUIET = ["--jepa-guard-rank", "0.001", "--jepa-guard-std", "0.0001"]            # a guard that never triggers
TIMING = ("sec_per_step", "peak_gib")


def jepa_csv_rows(path: Path) -> list[dict]:
    return [{k: v for k, v in r.items() if k not in TIMING} for r in GT.jepa_rows(path)]


def base_args(nd) -> list[str]:
    return with_option(recipe(nd, "v2"), "--mito-share", "0.5")


# ------------------------------------------------------------------------------------- (a) options off: parents
@pytest.mark.parametrize("short", ["pad", "pack"])
@pytest.mark.parametrize("jepa", ["plain", "block-guard-log"])
def test_split_off_trains_as_v22(genomes, tmp_path, monkeypatch, short, jepa):
    nd = make_recipe_data(genomes)
    real_model_cpu(monkeypatch)
    from kron_dna import cli
    base = parent_cli(monkeypatch, V22_PARENT, "KRON_V23_V22_CLI")
    assert "split_cards" not in base.genome_train.__code__.co_varnames
    assert "jepa_guard" in base.genome_train.__code__.co_varnames
    args = base_args(nd) + OFF_ARGS + ["--short-sources", short]
    if jepa != "plain":
        args += BLOCK + ["--jepa-log-every", "1", "--jepa-guard", "on"] + GUARD + QUIET
    assert_identical(*trained(base, monkeypatch, genomes, tmp_path / "parent", args),
                     *trained(cli, monkeypatch, genomes, tmp_path / "v23", args))
    if jepa != "plain":
        a, b = jepa_csv_rows(tmp_path / "parent" / "jepa.csv"), jepa_csv_rows(tmp_path / "v23" / "jepa.csv")
        assert a == b and len(a) == 4 and [r["row"] for r in a] == ["train"] * 3 + ["guard"]


@pytest.mark.parametrize("short", ["pad", "pack"])
@pytest.mark.parametrize("split", [[], ["--split-cards", "2"]])
def test_jepa_options_off_trains_as_v23pre(genomes, tmp_path, monkeypatch, short, split):
    nd = make_recipe_data(genomes)
    real_model_cpu(monkeypatch)
    from kron_dna import cli
    base = parent_cli(monkeypatch, V23PRE_PARENT, "KRON_V23_PRE_CLI")
    assert "jepa_block_share" not in base.genome_train.__code__.co_varnames
    assert "split_cards" in base.genome_train.__code__.co_varnames
    args = base_args(nd) + OFF_ARGS + ["--short-sources", short] + split
    assert_identical(*trained(base, monkeypatch, genomes, tmp_path / "parent", args),
                     *trained(cli, monkeypatch, genomes, tmp_path / "v23", args))
    assert not (tmp_path / "v23" / "jepa.csv").exists()


# -------------------------------------------------------------------- (b) split + block JEPA + pack = unsplit
def spy_split_calls(monkeypatch, sink: dict | None = None) -> list:
    """Record every forward_split call: (training, core_only, input, extra) on the host, into the returned list or
    into ``sink["calls"]`` (switchable between runs: a second spy would wrap the first)."""
    calls: list = []
    real = G.KronGenome.forward_split

    def wrap(self, seq, segments=None, context=None, extra=None, **kw):
        (calls if sink is None else sink["calls"]).append(
            (self.training, bool(kw.get("core_only")), seq.detach().cpu(), None if extra is None else extra.detach().cpu()))
        return real(self, seq, segments, context, extra, **kw)

    monkeypatch.setattr(G.KronGenome, "forward_split", wrap)
    return calls


def spy_blocks(monkeypatch, sink: dict) -> None:
    """Record every block_mask result into ``sink["blocks"]``."""
    real = G.block_mask

    def wrap(*a, **k):
        m = real(*a, **k)
        sink["blocks"].append(m.copy())
        return m

    monkeypatch.setattr(G, "block_mask", wrap)


TERMS = ("mlm", "epi", "ntv3", "jepa_term", "jepa_hidden", "jepa_visible", "copy_hidden", "copy_visible", "reg",
         "core_std", "core_std_visible", "core_std_hidden", "total")


@pytest.mark.parametrize("case", list(CASES))
def test_split_block_jepa_pack_first_step_equals_unsplit(genomes, tmp_path, monkeypatch, case):
    curr, tokens, accum = CASES[case]
    nd = make_recipe_data(genomes)
    real_model_cpu(monkeypatch)
    smooth_routing(monkeypatch)
    plans = spy_plans(monkeypatch)
    monkeypatch.setattr(PK, "TRACE", [])
    from kron_dna import cli
    args = base_args(nd) + ["--curriculum", curr, "--tokens-per-step", tokens, "--grad-accum", accum, "--bench",
                            "--checkpoint-core", "--short-sources", "pack", "--jepa-log-every", "1"] + BLOCK
    out, traces, terms, blocks, calls = {}, {}, {}, {}, {}
    sink: dict = {}
    spy_blocks(monkeypatch, sink)
    spy_split_calls(monkeypatch, sink)
    for tag, extra in (("one", []), ("split", ["--split-cards", "2"])):
        PK.TRACE.clear()
        monkeypatch.setattr(cli, "JEPA_TRACE", [])
        blocks[tag] = sink["blocks"] = []
        calls[tag] = sink["calls"] = []
        res, _ = run(cli, monkeypatch, genomes["root"], tmp_path / tag,
                     args + extra + ["--bench-grads", str(tmp_path / f"{tag}.pt")])
        assert res.exit_code == 0, res.output + repr(res.exception)
        assert "block JEPA: 0.25 of each row's bins hidden" in " ".join(res.output.split())
        out[tag], traces[tag], terms[tag] = torch.load(tmp_path / f"{tag}.pt"), list(PK.TRACE), list(cli.JEPA_TRACE)
        if extra:
            assert "split windows over 2 cards" in res.output and "split over 2 cards" in res.output
    assert traces["one"] == traces["split"]                         # the same rows drawn, split or not
    assert any(t[4] > 1 for t in traces["one"]), "no pack was drawn: the test would not test packing"
    # the same blocks, per row of every part, and some hidden
    assert len(blocks["one"]) == len(blocks["split"]) == len(terms["one"]) == len(terms["split"]) > int(accum)
    assert all(np.array_equal(a, b) for a, b in zip(blocks["one"], blocks["split"]))
    assert sum(int(m.sum()) for m in blocks["one"]) > 0
    # every loss term of every part (packed parts included) equals unsplit's
    for ta, tb in zip(terms["one"], terms["split"]):
        assert ta["hidden_share"] == tb["hidden_share"]
        for k in TERMS:
            a_, b_ = ta.get(k, float("nan")), tb.get(k, float("nan"))
            assert math.isnan(a_) == math.isnan(b_), k
            if not math.isnan(a_):
                assert b_ == pytest.approx(a_, rel=1e-5, abs=1e-7), (k, a_, b_)
    assert any(math.isfinite(t["jepa_hidden"]) and t["jepa_hidden"] > 0 for t in terms["one"])
    assert all(t["jepa_term"] == t["jepa_hidden"] for t in terms["one"] if math.isfinite(t["jepa_hidden"]))
    assert all(math.isfinite(t["reg"]) for t in terms["one"])
    assert any("ntv3" in t for t in terms["split"]) and any("epi" in t for t in terms["split"])
    # losses and gradients, as split-window's tolerances
    a, b = out["one"], out["split"]
    assert len(a["losses"]) == int(accum) and all(np.isfinite(a["losses"])) and all(np.isfinite(b["losses"]))
    assert b["losses"] == pytest.approx(a["losses"], rel=1e-5)
    n_grad = 0
    for n, g in a["grads"].items():
        assert (g is None) == (b["grads"][n] is None), n
        if g is not None:
            n_grad += 1
            assert torch.isfinite(b["grads"][n]).all(), n
            assert float((g - b["grads"][n]).abs().max()) <= 1e-4 * (float(g.abs().max()) + 1e-12) + 1e-9, n
    assert n_grad > 50
    assert all(by_ == "balanced" for *_, by_ in plans) and any(p_[1] < 2048 for p_ in plans)
    # what forward_split received: the online input of block JEPA, and the target pass core-only
    assert not calls["one"]
    online = [c_ for c_ in calls["split"] if c_[0] and not c_[1]]
    target = [c_ for c_ in calls["split"] if c_[1]]
    assert len(online) == len(target) == len(blocks["split"])
    for (_, _, x_, e_), m in zip(online, blocks["split"]):
        at = torch.from_numpy(m).repeat_interleave(16, dim=1)
        assert x_.dim() == 3 and tuple(x_.shape[::2]) == tuple(at.shape)       # [B, 6, L] (soft bases on)
        inside = x_.transpose(1, 2)[at]
        assert (inside[:, G.MASK_CHANNEL] == 1).all() and (inside.sum(-1) == 1).all()
        if e_ is not None:
            assert (e_.transpose(1, 2)[at] == 0).all()
    for (_, _, x_, _), m in zip(target, blocks["split"]):                       # the clean row: base codes
        assert x_.dim() == 2 and int(x_.max()) <= 4


# ----------------------------------------------------------------------------- (c) the guard in split mode
def guard_split_args(nd) -> list[str]:
    return base_args(nd) + ["--curriculum", "2048:10", "--tokens-per-step", "4096", "--eval-every", "1",
                            "--checkpoint-core", "--short-sources", "pack", "--split-cards", "2"] + BLOCK + GUARD


def test_guard_forced_trigger_in_split_mode_exits_42(genomes, tmp_path, monkeypatch):
    nd = make_recipe_data(genomes)
    real_model_cpu(monkeypatch)
    calls = spy_split_calls(monkeypatch)
    from kron_dna import cli
    GT.inject_collapse(monkeypatch, 2)
    res, _ = run(cli, monkeypatch, genomes["root"], tmp_path / "on", guard_split_args(nd) + ["--jepa-guard", "on"])
    assert res.exit_code == 42, res.output + repr(res.exception)
    text = " ".join(res.output.split())
    assert "split windows over 2 cards" in text
    rows = GT.guard_rows(tmp_path / "on" / "jepa.csv")
    assert [r["g_action"] for r in rows] == ["ok", "triggered", "raise", "raise", "raise", "stop"]
    assert [float(r["g_mult"]) for r in rows] == [1, 1, 2, 4, 8, 8]
    assert [float(r["g_reg_weight"]) for r in rows] == [1, 1, 2, 4, 8, 8]
    assert text.count("WARNING: JEPA collapse guard") == 5 and "saving and stopping" in text
    assert "JEPA collapse guard: stopped at step 6 (exit code 42)" in text
    assert (tmp_path / "on" / "JEPA_GUARD_STOP").exists()
    ck = torch.load(tmp_path / "on" / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert ck["step"] == 6 and ck["jepa_guard"]["mult"] == 8.0 and ck["jepa_guard"]["stopped"]
    # training went through forward_split (online + core-only target); evaluation and the guard's measurement did not
    assert all(tr_ or co_ for tr_, co_, _, _ in calls)
    assert sum(1 for tr_, co_, _, _ in calls if tr_ and not co_) >= 6 and sum(1 for _, co_, _, _ in calls if co_) >= 6


SUBPROCESS = r"""
import sys
from pathlib import Path
import pytest, torch
sys.path.insert(0, sys.argv[1])
mp = pytest.MonkeyPatch()
import test_kron_dna_split as ST
import test_kron_dna_jepa_guard as GT
ST.real_model_cpu(mp)
GT.inject_collapse(mp, 2)
from kron_dna import cli
root = Path(sys.argv[2])
mp.setattr(cli, "interim_dir", lambda *parts: root.joinpath(*parts))
torch.manual_seed(0)
cli.app(["genome-train", *sys.argv[3:]], prog_name="kron-dna")
"""


def test_guard_stop_exit_42_reaches_the_shell(genomes, tmp_path):
    """The exit code of a guard stop in split mode is the process's (what an sbatch script sees)."""
    nd = make_recipe_data(genomes)
    script = tmp_path / "run.py"
    script.write_text(SUBPROCESS)
    env = dict(os.environ, OMP_NUM_THREADS="2")
    args = [str(tmp_path / "out")] + guard_split_args(nd) + ["--jepa-guard", "on"]
    p = subprocess.run(["bash", "-c", 'exec "$@"', "bash", sys.executable, str(script),
                        str(Path(__file__).resolve().parent), str(genomes["root"]), *args],
                       capture_output=True, text=True, env=env, timeout=1200)
    assert p.returncode == 42, p.stdout[-3000:] + p.stderr[-3000:]
    assert (tmp_path / "out" / "JEPA_GUARD_STOP").exists()
