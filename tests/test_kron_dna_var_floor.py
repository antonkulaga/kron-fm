"""--jepa-var-floor: a per-dimension variance floor beside SIGReg (kron_dna.losses.variance_floor) and its options."""
from __future__ import annotations

import pytest
import torch

from kron_dna.losses import variance_floor

import test_kron_dna_jepa_block as JB
from test_kron_dna_jepa_block import jgenomes  # noqa: F401  (fixture)
from test_kron_dna_multigenome import run


def test_floor_is_zero_for_even_spread_and_pushes_only_the_dying_dimension():
    assert variance_floor(torch.ones(8)) == 0
    std = torch.tensor([1.0, 1.0, 1.0, 0.05], requires_grad=True)
    loss = variance_floor(std, 0.5)
    assert loss.item() == pytest.approx((0.5 * 0.7625 - 0.05) / 4)
    loss.backward()
    assert std.grad.tolist() == pytest.approx([0, 0, 0, -0.25])     # only the low dimension, and upward


@pytest.mark.parametrize("more,words", [
    (["--jepa-var-floor", "1.0"], "needs --jepa-weight"),
    (JB.JEPA + ["--jepa-var-floor", "-1"], "must be >= 0"),
    (JB.JEPA + ["--jepa-var-floor", "1", "--jepa-var-floor-frac", "1.5"], "(0, 1]"),
])
def test_option_checks(jgenomes, tmp_path, monkeypatch, more, words):
    JB.stub_jepa(monkeypatch)
    from kron_dna import cli as new
    res, _ = run(new, monkeypatch, jgenomes["root"], tmp_path / "bad", JB.SMALLJ + ["--steps", "2", "--batch-size", "2"]
                 + more)
    assert res.exit_code != 0 and words in " ".join(res.output.replace("│", " ").split())


def test_floor_adds_to_the_training_loss(jgenomes, tmp_path, monkeypatch):
    JB.stub_jepa(monkeypatch)
    from kron_dna import cli as new
    args = JB.SMALLJ + JB.JEPA + JB.RECIPE + ["--steps", "4", "--eval-every", "4", "--batch-size", "4", "--jepa-log-every", "1"]
    totals = {}
    for name, extra in (("off", []), ("on", ["--jepa-var-floor", "100", "--jepa-var-floor-frac", "1.0"])):
        trace = []
        monkeypatch.setattr(new, "JEPA_TRACE", trace)
        res, _ = run(new, monkeypatch, jgenomes["root"], tmp_path / name, args + extra)
        assert res.exit_code == 0, res.output + repr(res.exception)
        totals[name] = [t["total"] for t in trace]
    assert totals["on"][0] > totals["off"][0]          # frac 1: any uneven spread is below the floor at step 1
