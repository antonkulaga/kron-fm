"""genome-train --eval-fixed: every evaluation scores the same held-out windows with the same masks."""
from __future__ import annotations

import torch

import test_kron_dna_jepa_block as JB
from test_kron_dna_jepa_block import jgenomes  # noqa: F401  (fixture)
from test_kron_dna_multigenome import run


def eval_inputs(ev):
    return [e[3] for e in ev if e[0] == "forward" and not e[1]]     # no JEPA target here: evaluation passes only


def test_fixed_windows_repeat_at_every_eval_and_moving_ones_do_not(jgenomes, tmp_path, monkeypatch):
    JB.stub_jepa(monkeypatch)
    from kron_dna import cli as new
    args = JB.SMALLJ + ["--steps", "4", "--eval-every", "2", "--batch-size", "2"]
    res0, ev0 = run(new, monkeypatch, jgenomes["root"], tmp_path / "moving", args)
    assert res0.exit_code == 0, res0.output + repr(res0.exception)
    res1, ev1 = run(new, monkeypatch, jgenomes["root"], tmp_path / "fixed", args + ["--eval-fixed", "3"])
    assert res1.exit_code == 0, res1.output + repr(res1.exception)
    mov, fix = eval_inputs(ev0), eval_inputs(ev1)
    assert len(mov) == 16 and len(fix) == 6                         # 8 windows per eval, or the 3 fixed ones
    assert all(torch.equal(a, b) for a, b in zip(fix[:3], fix[3:]))  # step 2's windows and masks again at step 4
    assert not all(torch.equal(a, b) for a, b in zip(mov[:8], mov[8:]))


def test_negative_is_refused(jgenomes, tmp_path, monkeypatch):
    JB.stub_jepa(monkeypatch)
    from kron_dna import cli as new
    res, _ = run(new, monkeypatch, jgenomes["root"], tmp_path / "bad",
                 JB.SMALLJ + ["--steps", "2", "--batch-size", "2", "--eval-fixed", "-1"])
    assert res.exit_code != 0 and "--eval-fixed must be" in " ".join(res.output.replace("│", " ").split())
