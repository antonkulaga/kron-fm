"""kron_dna.jepa_eval: the decode and centred-distance checks of a JEPA predictor. The pure pieces against hand
cases; the whole evaluation on test_kron_dna_jepa_block.py's real model with the CUDA-only core replaced by a CPU
mixing, both predictor kinds."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import torch

import kron_core.blocks
from kron_core.testing import CausalMixer
from kron_dna import genome as G
from kron_dna import jepa_eval as J

import test_kron_dna_jepa_block as JB


def test_nearest_visible_picks_the_closer_side_and_the_left_on_a_tie():
    hidden = np.array([True, False, True, True, True, False, True])
    assert J.nearest_visible(hidden).tolist() == [1, 1, 1, 1, 5, 5, 5]
    assert J.nearest_visible(np.array([True, False, True])).tolist() == [1, 1, 1]
    with pytest.raises(ValueError):
        J.nearest_visible(np.ones(3, dtype=bool))


def test_interpolate_draws_lines_inside_blocks_and_holds_flat_at_edges():
    core = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1) * 10
    hidden = np.array([True, False, False, True, True, True, False, True])
    out = J.interpolate(core, hidden)[0, :, 0].tolist()
    assert out == pytest.approx([10, 10, 20, 30, 40, 50, 60, 60])
    assert torch.equal(J.interpolate(core, np.zeros(8, dtype=bool)), core)


def test_pearson_tracks_skips_flat_tracks():
    a = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    assert J.pearson_tracks(a, a) == pytest.approx(1.0)
    assert J.pearson_tracks(a, -a) == pytest.approx(-1.0)
    assert np.isnan(J.pearson_tracks(a[:, 1:], a[:, 1:]))


def test_centring_removes_a_common_mean_that_raw_cosine_rewards():
    torch.manual_seed(0)
    mean = torch.full((16,), 5.0)
    clean = mean + torch.randn(200, 16)
    guess = mean.expand(200, 16)                       # the mean alone, no per-bin information
    mu, sd = clean.mean(0), clean.std(0)
    assert J.cosine_distance(guess, clean).mean() < 0.05
    assert J.cosine_distance(guess, clean, mu, sd).mean() > 0.9


def test_within_window_share_tells_per_window_constants_from_fast_features():
    torch.manual_seed(0)
    cores = []
    for w in range(10):
        c = torch.randn(100, 2)
        c[:, 1] = float(w)                             # dimension 1: one value per window
        cores.append(c)
    share = J.within_window_share(cores)
    assert share[0] > 0.9 and share[1] < 1e-6


def test_sample_windows_is_seeded_and_keeps_acgt_windows():
    bases = np.zeros(10_000, dtype=np.uint8)
    bases[:3000] = 4                                   # an N stretch the windows must avoid
    contigs = pl.DataFrame(dict(offset=[0], len=[10_000]))
    a = J.sample_windows(bases, contigs, 5, 1000, seed=3)
    b = J.sample_windows(bases, contigs, 5, 1000, seed=3)
    assert [x[0] for x in a] == [x[0] for x in b]
    assert all(at >= 2950 for at, _ in a)
    with pytest.raises(ValueError, match="longer"):
        J.sample_windows(bases, contigs, 1, 20_000)


@pytest.mark.parametrize("kind", ["mlp", "context"])
def test_evaluate_scores_every_variant(monkeypatch, kind):
    monkeypatch.setattr(kron_core.blocks, "kronlstm_mixer", CausalMixer)
    torch.manual_seed(0)
    cfg = G.KronGenomeConfig(d_model=32, n_heads=2, n_layers=1, ff_units=64, stem_channels=8, n_down=3, arch="v2",
                             jepa=True, jepa_pred_kind=kind, jepa_pred_layers=1, n_ntv3=3)
    model = JB.CpuCoreGenome(cfg).eval()
    rng = np.random.default_rng(0)
    picks = [(0, rng.integers(0, 4, 2048).astype(np.int64)) for _ in range(3)]
    row = J.evaluate(model, picks, share=0.3, lo=64, hi=256)
    assert row["windows"] == 3
    for k in J.DECODE_VARIANTS:
        assert 0.0 <= row[f"base_acc_{k}"] <= 1.0
        assert np.isfinite(row[f"ntv3_r_vs_clean_{k}"])
    for tag in ("raw", "centred"):
        assert np.isfinite(row[f"{tag}_pred_gain"])
    if kind == "context":                              # zero-init readout: the predictor is the masked core
        assert row["raw_pred_hidden"] == pytest.approx(row["raw_masked_hidden"], abs=1e-5)
    assert model.jepa_hidden is None


def test_evaluate_refuses_a_model_without_a_predictor():
    model = JB.CpuCoreGenome(G.KronGenomeConfig(d_model=32, n_heads=2, n_layers=1, ff_units=64, stem_channels=8,
                                                n_down=3, arch="v2"))
    with pytest.raises(ValueError, match="no JEPA predictor"):
        J.evaluate(model, [(0, np.zeros(512, dtype=np.int64))])
