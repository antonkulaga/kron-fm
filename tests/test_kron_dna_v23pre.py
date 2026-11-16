"""v2.3-pre = no-pad (205c3d3, ``--short-sources pack``) + split-window (6309824/21d0fcf, ``--split-cards N``).

Split + pack: with ``pack`` a training micro-batch is a list of parts, one per row length (packing.py), and
``loss_parts`` runs ``loss_on`` once per part under one backward. In split mode every part goes through
``KronGenome.forward_split`` with ``split_plan``'s rule: whole rows per card when the part's row count divides by the
number of cards, otherwise each row cut along its length with the halo. No part runs unsplit on the first card: at
1 Mb a micro-batch is one window, so its pack is one part holding 7/8 to 1 window of bases, which one card cannot
train. Evaluation stays unsplit, as before.

Tests (CPU, two logical pieces on one device with real parameter copies, max-pool gradients routed continuously
as in test_kron_dna_split):
- split + pack gives the same first-step losses and gradients as pack unsplit (split-window's tolerances), finite,
  with packed parts split both by rows and by length;
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
        seen.append((batch, length, out[0][0]))
        return out

    monkeypatch.setattr(G, "split_plan", plan)
    return seen


# window 2048 (n-down 4): mito contigs 900-1233 bases make packs of 2 x 1024 / 2 x 896 rows (split by rows), chr3
# (700) 3 x 640 (by length), gA's 2000-base contig 1 x 1792 (by length); full windows of 2048 come from the rest.
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
    packed = [p_ for p_ in plans if p_[1] < 2048]                # parts of packed rows (shorter than the window)
    assert packed, "no packed part went through forward_split"


def test_split_pack_takes_both_plans(genomes, tmp_path, monkeypatch):
    """Over a few training steps, packed parts are split by rows (even row counts) and by length (odd counts)."""
    nd = make_recipe_data(genomes)
    real_model_cpu(monkeypatch)
    plans = spy_plans(monkeypatch)
    from kron_dna import cli
    args = with_option(recipe(nd, "v2"), "--mito-share", "0.5") + [
        "--curriculum", "2048:12", "--tokens-per-step", "4096", "--eval-every", "12", "--checkpoint-core",
        "--short-sources", "pack", "--split-cards", "2"]
    res, _ = run(cli, monkeypatch, genomes["root"], tmp_path / "r", args)
    assert res.exit_code == 0, res.output + repr(res.exception)
    kinds = {(L < 2048, k) for _, L, k in plans}
    assert (True, "rows") in kinds and (True, "length") in kinds, kinds
    for b_, L, k in plans:                                       # the rule itself
        assert k == ("rows" if b_ % 2 == 0 else "length")
    assert "peak cpu 0.0, cpu 0.0 GiB" in res.output             # the log line reports every card's peak
    m = pl.read_csv(tmp_path / "r" / "metrics.csv")
    assert m.height == 1 and np.isfinite(m["train_loss"].to_numpy()).all()


# ------------------------------------------------------------------------------------------- options off: parents

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
