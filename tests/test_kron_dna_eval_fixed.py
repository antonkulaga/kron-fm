"""genome-train --eval-fixed: beside the 8 moving windows, the same held-out windows and masks scored at every eval
(masked-base loss only), to eval_fixed.csv; metrics.csv is unchanged."""
from __future__ import annotations

import torch

import test_kron_dna_jepa_block as JB
from test_kron_dna_jepa_block import jgenomes  # noqa: F401  (fixture)
from test_kron_dna_multigenome import run


def eval_inputs(ev):
    return [e[3] for e in ev if e[0] == "forward" and not e[1]]     # no JEPA target here: evaluation passes only


def test_fixed_windows_repeat_and_the_moving_ones_stay(jgenomes, tmp_path, monkeypatch):
    JB.stub_jepa(monkeypatch)
    from kron_dna import cli as new
    args = JB.SMALLJ + ["--steps", "4", "--eval-every", "2", "--batch-size", "2"]
    res0, ev0 = run(new, monkeypatch, jgenomes["root"], tmp_path / "moving", args)
    assert res0.exit_code == 0, res0.output + repr(res0.exception)
    res1, ev1 = run(new, monkeypatch, jgenomes["root"], tmp_path / "fixed", args + ["--eval-fixed", "3"])
    assert res1.exit_code == 0, res1.output + repr(res1.exception)
    mov, both = eval_inputs(ev0), eval_inputs(ev1)
    assert len(mov) == 16 and len(both) == 22                     # each eval: the 8 moving windows, then the 3 fixed
    assert all(torch.equal(a, b) for a, b in zip(mov[:8], both[:8]))      # the moving windows are untouched
    assert all(torch.equal(a, b) for a, b in zip(mov[8:], both[11:19]))
    fix2, fix4 = both[8:11], both[19:22]
    assert all(torch.equal(a, b) for a, b in zip(fix2, fix4))     # step 2's fixed windows and masks again at step 4
    assert (tmp_path / "fixed" / "metrics.csv").read_text().split("\n")[0] == \
        (tmp_path / "moving" / "metrics.csv").read_text().split("\n")[0]
    rows = (tmp_path / "fixed" / "eval_fixed.csv").read_text().strip().split("\n")
    assert rows[0] == "step,windows,val_bits_per_base_fixed,seconds" and [r.split(",")[0] for r in rows[1:]] == ["2", "4"]
    assert not (tmp_path / "moving" / "eval_fixed.csv").exists()


def test_negative_is_refused(jgenomes, tmp_path, monkeypatch):
    JB.stub_jepa(monkeypatch)
    from kron_dna import cli as new
    res, _ = run(new, monkeypatch, jgenomes["root"], tmp_path / "bad",
                 JB.SMALLJ + ["--steps", "2", "--batch-size", "2", "--eval-fixed", "-1"])
    assert res.exit_code != 0 and "--eval-fixed must be" in " ".join(res.output.replace("│", " ").split())
