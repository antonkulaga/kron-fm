"""The training loop's side of --jepa-track-dims: split pieces' binned head outputs joined as the unsplit forward
gives them, per-base outputs as bin means, and the option checks."""
from __future__ import annotations

import pytest
import torch

from kron_dna import cli as C
from kron_dna import genome as G

import test_kron_dna_jepa_block as JB
from test_kron_dna_jepa_block import jgenomes  # noqa: F401  (fixture)
from test_kron_dna_multigenome import run


@pytest.mark.parametrize("by", ["length", "rows"])
def test_split_pieces_join_into_the_whole(by):
    torch.manual_seed(0)
    whole = {"a": torch.randn(2, 16, 3), "b": torch.randn(2, 16, 5)}
    if by == "length":                         # two stretches of the window, both rows
        cuts = [(slice(0, 2), 0, 1024), (slice(0, 2), 1024, 2048)]
    else:                                      # one row each, the whole window
        cuts = [(slice(0, 1), 0, 2048), (slice(1, 2), 0, 2048)]
    pieces = [G.SplitPiece(rows=r, start=s, end=e, device=torch.device("cpu"),
                           x={n: t[r, s // 128:e // 128] for n, t in whole.items()}, x_at=(s, e))
              for r, s, e in cuts]
    got = C.gather_track_outputs(pieces, ["a", "b"], 128, 2, 2048, "cpu")
    assert all(torch.equal(got[n], whole[n]) for n in whole)


def test_bin_means():
    x = torch.arange(2 * 8 * 3, dtype=torch.float32).reshape(2, 8, 3)
    got = C.bin_means(x, 4)
    assert got.shape == (2, 2, 3) and torch.allclose(got, x.reshape(2, 2, 4, 3).mean(2))


@pytest.mark.parametrize("more,words", [
    (["--jepa-track-dims", "16"], "needs --jepa-weight"),
    (JB.JEPA + ["--jepa-track-dims", "16"], "needs --jepa-weight > 0 and --jepa-block-share"),
    (JB.JEPA + ["--jepa-block-share", "0.2", "--jepa-track-dims", "16"], "binned --xhead heads"),
])
def test_option_checks(jgenomes, tmp_path, monkeypatch, more, words):
    JB.stub_jepa(monkeypatch)
    from kron_dna import cli as new
    res, _ = run(new, monkeypatch, jgenomes["root"], tmp_path / "bad", JB.SMALLJ + ["--steps", "2", "--batch-size", "2"]
                 + more)
    assert res.exit_code != 0 and words in " ".join(res.output.replace("│", " ").split())
