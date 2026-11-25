"""genome-train --focus-map / --focus-mask-tilt: masked bases lean toward weighted cells of the primary genome while
each row keeps its masking rate; tilt 0 is the old uniform masking."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from kron_dna import genome as G

import test_kron_dna_jepa_block as JB
from test_kron_dna_jepa_block import jgenomes  # noqa: F401  (fixture)
from test_kron_dna_multigenome import run


def make_focus(d, bin_=128):
    n = np.fromfile(d / "bases.u8", dtype=np.uint8).size
    cells = -(-n // bin_)
    f = np.zeros((cells, 2), dtype=np.uint8)
    f[::2, 0] = 255                                  # every other cell fully weighted in column a
    f[:, 1] = 25                                     # column b: low everywhere
    out = d / "focus-bin128"
    out.mkdir()
    f.tofile(out / "focus.u8")
    (out / "focus.json").write_text(json.dumps(dict(genome="hgj", bin=bin_, cells=cells, columns=["a", "b"])))
    return out


@pytest.fixture
def probs(monkeypatch):
    seen = []
    real = G.mask_bases

    def spy(seq, prob=0.15, generator=None, bert=False):
        seen.append(prob)
        return real(seq, prob, generator, bert)

    monkeypatch.setattr(G, "mask_bases", spy)
    return seen


def test_tilt_leans_on_weighted_cells_and_keeps_the_rate(jgenomes, tmp_path, monkeypatch, probs):
    JB.stub_jepa(monkeypatch)
    from kron_dna import cli as new
    fm = make_focus(jgenomes["hgj"])
    res, _ = run(new, monkeypatch, jgenomes["root"], tmp_path / "tilt",
                 JB.SMALLJ + ["--steps", "2", "--eval-every", "2", "--batch-size", "2", "--focus-map", str(fm),
                              "--focus-mask-tilt", "0.5"])
    assert res.exit_code == 0, res.output + repr(res.exception)
    tensors = [p for p in probs if torch.is_tensor(p)]
    assert tensors, "training masks should get a per-base probability"
    p = tensors[0]
    hi, lo = float(p.max()), float(p.min())
    assert hi > 0.15 > lo >= 0.15 * 0.5 - 1e-6          # weighted cells above the rate, the floor (1 - T) kept


def test_tilt_zero_masks_as_before(jgenomes, tmp_path, monkeypatch, probs):
    JB.stub_jepa(monkeypatch)
    from kron_dna import cli as new
    fm = make_focus(jgenomes["hgj"])
    res, _ = run(new, monkeypatch, jgenomes["root"], tmp_path / "flat",
                 JB.SMALLJ + ["--steps", "2", "--eval-every", "2", "--batch-size", "2", "--focus-map", str(fm)])
    assert res.exit_code == 0, res.output + repr(res.exception)
    assert probs and not any(torch.is_tensor(p) for p in probs)


@pytest.mark.parametrize("more,words", [
    (["--focus-mask-tilt", "0.5"], "needs --focus-map"),
    (["--focus-mask-tilt", "1.0"], "must lie in [0, 1)"),
])
def test_option_checks(jgenomes, tmp_path, monkeypatch, more, words):
    JB.stub_jepa(monkeypatch)
    from kron_dna import cli as new
    res, _ = run(new, monkeypatch, jgenomes["root"], tmp_path / "bad", JB.SMALLJ + ["--steps", "2", "--batch-size", "2"]
                 + more)
    assert res.exit_code != 0 and words in " ".join(res.output.replace("│", " ").split())
