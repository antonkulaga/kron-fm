"""v2.3-pre = no-pad (205c3d3, ``--short-sources pack``) + split-window (6309824/21d0fcf, ``--split-cards N``).

Split + pack: with ``pack`` a training micro-batch is a list of parts, one per row length (packing.py), and
``loss_parts`` runs ``loss_on`` once per part under one backward. In split mode with ``pack`` every part goes
through ``KronGenome.forward_split`` with ``split_plan(..., by="balanced")``: whole rows per card when the part's row
count divides by the number of cards; otherwise uneven whole rows (no halo) or each row along its length (with the
halo), whichever puts fewer encoded bases on the busiest card. No part runs unsplit on the first card: at 1 Mb a
micro-batch is one window, so its pack is one part holding 7/8 to 1 window of bases, which one card cannot train.
Without ``pack`` the plan is split-window's (by=None). Evaluation stays unsplit, as before.

Tests (CPU, two logical pieces on one device with real parameter copies, max-pool gradients routed continuously
as in test_kron_dna_split):
- the balanced plan over arm F's real 1 Mb row ladder: every row and base owned once, the busiest card's load;
- uneven row pieces: outputs and gradients equal forward (model level, as test_kron_dna_split);
- split + pack gives the same first-step losses and gradients as pack unsplit (split-window's tolerances), finite,
  with packed parts split by even rows, uneven rows and length;
- options off: v2.3-pre trains bit-identically to 205c3d3's cli.py without --split-cards (pad and pack), and to
  6309824's cli.py with --short-sources pad (split off and on).
"""
from __future__ import annotations

import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from kron_dna import genome as G
from kron_dna import packing as PK

import test_kron_dna_split as ST
from test_kron_dna_multigenome import genomes, run  # noqa: F401  (genomes is a fixture)
from test_kron_dna_split import _restore_precision, make_recipe_data, real_model_cpu, recipe, smooth_routing  # noqa: F401

NOPAD_PARENT, SPLIT_PARENT = "205c3d3", "6309824"


def with_option(args: list[str], name: str, value: str) -> list[str]:
    i = args.index(name)
    return args[:i + 1] + [value] + args[i + 2:]


def spy_plans(monkeypatch) -> list:
    """Record (rows, row length, plan kind) of every split_plan call."""
    seen: list = []
    real = G.split_plan

    def plan(batch, length, n, unit, halo, by=None):
        out = real(batch, length, n, unit, halo, by)
        seen.append((batch, length, out[0][0] if out[0][1].stop - out[0][1].start == batch // n or out[0][0] == "length"
                     else "rows-uneven", by))
        return out

    monkeypatch.setattr(G, "split_plan", plan)
    return seen


# window 2048 (n-down 4): mito contigs 900-1233 bases make packs of 2 x 1024 / 2 x 896 rows (even rows), chr3 (700)
# 3 x 640 (uneven rows 1/2: cheaper than length pieces with the halo), gA's 2000-base contig 1 x 1792 (length).
CASES = {"batch1x4": ("2048:10", "8192", "4"), "batch4": ("2048:10", "8192", "1"), "batch2x2": ("2048:10", "8192", "2")}


@pytest.mark.parametrize("case", list(CASES))
def test_split_pack_first_step_equals_unsplit_pack(genomes, tmp_path, monkeypatch, case):
    curr, tokens, accum = CASES[case]
    nd = make_recipe_data(genomes)
    real_model_cpu(monkeypatch)
    smooth_routing(monkeypatch)
    plans = spy_plans(monkeypatch)
    monkeypatch.setattr(PK, "TRACE", [])
    from kron_dna import cli
    args = with_option(recipe(nd, "v2"), "--mito-share", "0.5") + [
        "--curriculum", curr, "--tokens-per-step", tokens, "--grad-accum", accum, "--bench", "--checkpoint-core",
        "--short-sources", "pack"]
    out, traces = {}, {}
    for tag, extra in (("one", []), ("split", ["--split-cards", "2"])):
        PK.TRACE.clear()
        res, _ = run(cli, monkeypatch, genomes["root"], tmp_path / tag,
                     args + extra + ["--bench-grads", str(tmp_path / f"{tag}.pt")])
        assert res.exit_code == 0, res.output + repr(res.exception)
        out[tag], traces[tag] = torch.load(tmp_path / f"{tag}.pt"), list(PK.TRACE)
        if extra:
            assert "split windows over 2 cards" in res.output and "split over 2 cards" in res.output
    assert traces["one"] == traces["split"]                     # the same rows drawn, split or not
    assert any(t[4] > 1 for t in traces["one"]), "no pack was drawn: the test would not test packing"
    a, b = out["one"], out["split"]
    assert len(a["losses"]) == int(accum) and all(np.isfinite(a["losses"])) and all(np.isfinite(b["losses"]))
    assert b["losses"] == pytest.approx(a["losses"], rel=1e-5)
    assert a["grads"].keys() == b["grads"].keys()
    n_grad = 0
    for n, g in a["grads"].items():
        assert (g is None) == (b["grads"][n] is None), n
        if g is not None:
            n_grad += 1
            assert torch.isfinite(b["grads"][n]).all(), n
            assert float((g - b["grads"][n]).abs().max()) <= 1e-4 * (float(g.abs().max()) + 1e-12) + 1e-9, n
    assert n_grad > 50
    assert all(by_ == "balanced" for *_, by_ in plans)
    packed = [p_ for p_ in plans if p_[1] < 2048]                # parts of packed rows (shorter than the window)
    assert packed, "no packed part went through forward_split"


def test_split_pack_takes_both_plans(genomes, tmp_path, monkeypatch):
    """Over a few training steps, packed parts are split by rows (even row counts) and by length (odd counts)."""
    nd = make_recipe_data(genomes)
    real_model_cpu(monkeypatch)
    plans = spy_plans(monkeypatch)
    from kron_dna import cli
    args = with_option(recipe(nd, "v2"), "--mito-share", "0.5") + [
        "--curriculum", "2048:40", "--tokens-per-step", "4096", "--eval-every", "40", "--checkpoint-core",
        "--short-sources", "pack", "--split-cards", "2"]
    res, _ = run(cli, monkeypatch, genomes["root"], tmp_path / "r", args)
    assert res.exit_code == 0, res.output + repr(res.exception)
    kinds = {(L < 2048, k) for _, L, k, _ in plans}
    assert {(True, "rows"), (True, "rows-uneven"), (True, "length")} <= kinds, kinds
    assert all(by_ == "balanced" for *_, by_ in plans)
    for b_, L, k, _ in plans:                                    # even batches: whole rows, as split-window
        if b_ % 2 == 0:
            assert k == "rows"
    assert "peak cpu 0.0, cpu 0.0 GiB" in res.output             # the log line reports every card's peak
    m = pl.read_csv(tmp_path / "r" / "metrics.csv")
    assert m.height == 1 and np.isfinite(m["train_loss"].to_numpy()).all()


# ------------------------------------------------------------------------------------------- the balanced plan

def test_balanced_plan_on_arm_f_ladder():
    """Arm F: 1,048,576-base windows, unit 128, halo 1,152 (v2), two cards. For every rung of the pack ladder the
    plan owns each row and base exactly once, matches split-window's plan when the row count divides, and keeps the
    busiest card within 1.035 x half a window (split-window's length split alone reaches 2.0 x for odd packs of
    short rows)."""
    W, unit, halo = 1 << 20, 128, 1152
    worst = 0.0
    for r, n in PK.row_ladder(W, unit):
        plan = G.split_plan(n, r, 2, unit, halo, "balanced")
        if n % 2 == 0:
            assert plan == G.split_plan(n, r, 2, unit, halo)
        own = np.zeros((n, r), dtype=int)
        for _, rows, s_, e_, lo, hi in plan:
            own[rows, s_:e_] += 1
            assert lo <= s_ < e_ <= hi
        assert (own == 1).all(), (r, n)
        load = max((hi - lo) * (rows.stop - rows.start) for _, rows, _, _, lo, hi in plan) / (W // 2)
        old = max((hi - lo) * (rows.stop - rows.start) for _, rows, _, _, lo, hi in G.split_plan(n, r, 2, unit, halo))
        assert load <= old / (W // 2) + 1e-12
        worst = max(worst, load)
    assert worst <= 1.035
    p_ = G.split_plan(73, 14336, 2, unit, halo, "balanced")       # mito genomes < 16,384 bases: 73 rows
    assert [x[0] for x in p_] == ["rows", "rows"] and [x[1].stop - x[1].start for x in p_] == [36, 37]
    assert G.split_plan(3, 327680, 2, unit, halo, "balanced")[0][0] == "length"     # 2 of 3 long rows would not be
    assert G.split_plan(1, W, 2, unit, halo, "balanced") == G.split_plan(1, W, 2, unit, halo)


def uneven_case(arch):
    m = ST.small(arch)
    b, length = 5, 1024
    plan = G.split_plan(b, length, 2, 2 ** m.config.n_down, G.split_halo(m.config), "balanced")
    assert plan[0][0] == "rows" and [x[1].stop - x[1].start for x in plan] == [2, 3], plan
    return b, length


@pytest.mark.parametrize("arch", ["v2", "v2l"])
def test_uneven_rows_equal_forward_eval(device, monkeypatch, arch):
    """As test_kron_dna_split.test_forward_split_equals_forward_eval, for uneven row pieces (2 and 3 rows)."""
    b, length = uneven_case(arch)
    ST.forced_replicas(monkeypatch)
    m = ST.small(arch).to(device).eval()
    seq, seg, ctx, extra = ST.batch_for(b, length, device, soft=True)
    with torch.no_grad():
        logits, core, _ = m(seq, seg, ctx, extra)
        ntv3, epi = m.last_ntv3, m.last_epi
        pieces, core_s = m.forward_split(seq, seg, ctx, extra, devices=[device, device], by="balanced")
        assert [p_.rows for p_ in pieces] == [slice(0, 2), slice(2, 5)]
        tol = dict(rtol=1e-4, atol=1e-5)
        assert torch.allclose(G.gather_split(pieces, "logits", device), logits, **tol)
        assert torch.allclose(G.gather_split(pieces, "ntv3", device), ntv3, **tol)
        assert torch.allclose(core_s, core, **tol) and torch.allclose(m.last_epi, epi, **tol)
        _, core_only = m.forward_split(seq, seg, ctx, extra, devices=[device, device], by="balanced", core_only=True)
        assert torch.allclose(core_only, core, **tol)


@pytest.mark.parametrize("arch", ["v2", "v2l"])
def test_uneven_rows_gradients_equal_forward(device, monkeypatch, arch):
    b, length = uneven_case(arch)
    ST.test_forward_split_gradients_equal_forward(device, monkeypatch, arch, b, length, "balanced")


# ------------------------------------------------------------------------------------------- options off: parents

AUDIT_MODEL = "    model = G.KronGenome(config).to(dev)\n"
AUDIT_START = "    if main and not bench:\n        record[\"started_at_step\"] = step\n"


def with_audit_rng(src: str) -> str:
    """A parent's cli.py with the audit's random-stream changes (2026-11-18, TR03/TR04) applied, so "option off
    trains as the parent" compares like with like: since the audit the initial weights come from --seed, the
    mask/dropout stream from (seed, rank, start step), and a run starting past step 0 draws its windows from
    step-keyed streams. Those change every run on purpose; what the parents are compared for is everything else."""
    if "audit 2026-11-18, TR03" in src:                         # already carries them
        return src
    assert src.count(AUDIT_MODEL) == 1 and src.count(AUDIT_START) == 1, "parent's anchors for the audit RNG patch"
    src = src.replace(AUDIT_MODEL, "    torch.manual_seed(seed)                       # audit 2026-11-18, TR03\n" + AUDIT_MODEL)
    return src.replace(AUDIT_START,
                       "    torch.manual_seed(int(np.random.SeedSequence([seed, rank, 2035, step]).generate_state(1)[0]))\n"
                       "    if step > 0:\n"
                       "        rng = np.random.default_rng([seed, rank, step])\n"
                       "        rep_rng = np.random.default_rng([seed, rank, 2027, step])\n" + AUDIT_START)


def parent_cli(monkeypatch, commit: str, env: str):
    """cli.py at ``commit`` (git show), or the file the environment variable ``env`` names."""
    repo = Path(__file__).resolve().parents[1]
    if os.environ.get(env):
        src = Path(os.environ[env]).read_text()
    else:
        try:
            src = subprocess.run(["git", "-C", str(repo), "show", f"{commit}:src/kron_dna/cli.py"], capture_output=True,
                                 check=True, text=True).stdout
        except (OSError, subprocess.CalledProcessError):
            pytest.skip(f"commit {commit} is not reachable from {repo} (set {env} to its cli.py)")
    src = with_audit_rng(src)
    mod = types.ModuleType(f"kron_dna_cli_{commit}")
    mod.__file__ = f"{commit}:src/kron_dna/cli.py"
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod


def trained(mod, monkeypatch, genomes, out: Path, args: list[str]):
    res, _ = run(mod, monkeypatch, genomes["root"], out, args)
    assert res.exit_code == 0, res.output + repr(res.exception)
    state = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=False)
    metrics = pl.read_csv(out / "metrics.csv").drop("hours")
    return state, metrics


def assert_identical(sa, ma, sb, mb):
    assert sa["step"] == sb["step"]
    assert ma.equals(mb), (ma, mb)
    assert sa["model"].keys() == sb["model"].keys()
    for k in sa["model"]:
        assert torch.equal(sa["model"][k], sb["model"][k]), k
    for (ka, va), (kb, vb) in zip(sa["optimizer"]["state"].items(), sb["optimizer"]["state"].items()):
        assert ka == kb and all(torch.equal(va[x], vb[x]) for x in va), ka


OFF_ARGS = ["--curriculum", "2048:3", "--tokens-per-step", "4096", "--eval-every", "3", "--checkpoint-core"]


@pytest.mark.parametrize("short", ["pad", "pack"])
def test_split_off_trains_as_no_pad_parent(genomes, tmp_path, monkeypatch, short):
    nd = make_recipe_data(genomes)
    real_model_cpu(monkeypatch)
    from kron_dna import cli
    base = parent_cli(monkeypatch, NOPAD_PARENT, "KRON_V23_NOPAD_CLI")
    assert "split_cards" not in base.genome_train.__code__.co_varnames
    args = with_option(recipe(nd, "v2"), "--mito-share", "0.5") + OFF_ARGS + ["--short-sources", short]
    assert_identical(*trained(base, monkeypatch, genomes, tmp_path / "parent", args),
                     *trained(cli, monkeypatch, genomes, tmp_path / "v23", args))


@pytest.mark.parametrize("split", [[], ["--split-cards", "2"]])
def test_pack_off_trains_as_split_parent(genomes, tmp_path, monkeypatch, split):
    nd = make_recipe_data(genomes)
    real_model_cpu(monkeypatch)
    from kron_dna import cli
    base = parent_cli(monkeypatch, SPLIT_PARENT, "KRON_V23_SPLIT_CLI")
    assert "short_sources" not in base.genome_train.__code__.co_varnames
    args = with_option(recipe(nd, "v2"), "--mito-share", "0.5") + OFF_ARGS + split
    assert_identical(*trained(base, monkeypatch, genomes, tmp_path / "parent", args),
                     *trained(cli, monkeypatch, genomes, tmp_path / "v23", args + ["--short-sources", "pad"]))
