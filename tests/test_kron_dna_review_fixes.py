"""Faults an outside review found in the architecture changes (25 Nov 2026), each with the case that showed it:
lean masks outliving their forward, a split that left a window's tail unowned, a head finer than the core in the
richer JEPA target, an empty lean selection without gradients, and a track target that depended on the batch."""
from __future__ import annotations

import pytest
import torch

import kron_core.blocks
from kron_core.testing import CausalMixer
from kron_dna import genome as G


@pytest.fixture
def cpu_mixer(monkeypatch):
    monkeypatch.setattr(kron_core.blocks, "kronlstm_mixer", CausalMixer)


def small(**kw) -> G.KronGenome:
    torch.manual_seed(0)
    return G.KronGenome(G.KronGenomeConfig(**dict(dict(d_model=32, n_heads=2, n_layers=1, ff_units=64,
                                                       stem_channels=8, n_down=3, arch="v2"), **kw))).eval()


def test_a_lean_mask_holds_for_one_forward_only(cpu_mixer):
    m = small()
    seq = torch.randint(0, 4, (2, 64))
    dense = m(seq)[0]
    where = torch.zeros(2, 64, dtype=torch.bool)
    where[:, ::5] = True
    m.head_at = where
    lean = m(seq)[0]
    assert m.head_at is None                                         # used once
    assert torch.allclose(lean[where], dense[where], atol=1e-6) and float(lean[~where].abs().max()) == 0
    assert torch.allclose(m(seq)[0], dense, atol=1e-6)               # the next forward is the whole head again
    other = torch.randint(0, 4, (3, 32))                             # a later batch of another shape
    assert m(other)[0].shape == (3, 32, 4)
    m.head_at = where                                                # a mask set for another window: an error
    with pytest.raises(ValueError, match="head_at"):
        m(other)
    assert m.head_at is None


def test_an_empty_selection_gives_zero_gradients_not_none():
    head = G.MLPHead(6, 12, 4)
    x = torch.randn(2, 6, 10, requires_grad=True)
    out = head.at(x, torch.zeros(2, 10, dtype=torch.bool))
    assert float(out.abs().max()) == 0
    out.sum().backward()
    assert all(p.grad is not None and float(p.grad.abs().max()) == 0 for p in head.parameters())


def test_every_base_of_a_window_has_an_owner_in_every_split():
    for batch, length, n, by, align in [(3, 1152, 2, None, 512), (3, 1152, 2, "balanced", 512), (5, 1152, 4, None, 512),
                                        (3, 1024, 2, None, 512), (4, 1152, 2, None, 512), (3, 1152, 2, None, 0)]:
        owned = torch.zeros(batch, length, dtype=torch.int)
        for _, rows, start, end, lo, hi in G.split_plan(batch, length, n, 128, 128, by, align=align):
            owned[rows, start:end] += 1
            assert lo <= start < end <= hi
        assert bool((owned == 1).all()), (batch, length, n, by, align)
    with pytest.raises(ValueError):                                  # fewer rows than pieces and no cut by length
        G.split_plan(1, 1152, 2, 128, 128, None, align=512)


def test_a_head_finer_than_the_core_is_refused_by_name():
    with pytest.raises(ValueError, match="not finer"):
        G.pool_bins(torch.zeros(1, 8, 3), 0)


def test_track_target_ignores_padding_and_the_rest_of_the_batch():
    torch.manual_seed(1)
    out = {"dnase": torch.randn(2, 16, 20) + 2, "tf": torch.randn(2, 16, 30) * 3}
    f = G.jepa_track_features(out, 12)
    padded = {k: torch.cat([v, torch.zeros(2, 8, v.shape[-1])], 1) for k, v in out.items()}
    valid = torch.cat([torch.ones(2, 16, dtype=torch.bool), torch.zeros(2, 8, dtype=torch.bool)], 1)
    assert float((G.jepa_track_features(padded, 12)[:, :16] - f).abs().max()) > 0.05    # the fault: padding moved it
    assert torch.allclose(G.jepa_track_features(padded, 12, valid=valid)[:, :16], f, atol=1e-5)
    assert float(G.jepa_track_features(padded, 12, valid=torch.zeros_like(valid)).abs().max()) == 0
    # running statistics: the same window gets the same target whatever shares its batch
    stats = G.TrackFeatureStats({"dnase": 20, "tf": 30}, 12, momentum=0.1)
    first = G.jepa_track_features(out, 12, stats=stats)
    assert torch.allclose(first, f, atol=1e-2)                       # bias-corrected: the first batch's own statistics
    for _ in range(30):
        G.jepa_track_features({"dnase": torch.randn(2, 16, 20) + 2, "tf": torch.randn(2, 16, 30) * 3}, 12, stats=stats)
    alone = G.jepa_track_features({k: v[:1] for k, v in out.items()}, 12, stats=stats, update=False)
    mate = {k: torch.cat([v[:1], v[:1] * 0 + 50]) for k, v in out.items()}
    assert torch.equal(G.jepa_track_features(mate, 12, stats=stats, update=False)[:1], alone)
    assert abs(float(alone.std()) - 1) < 0.25
    again = G.TrackFeatureStats({"dnase": 20, "tf": 30}, 12, momentum=0.1)
    again.load_state_dict(stats.state_dict())                        # the statistics travel with a checkpoint
    assert torch.equal(G.jepa_track_features(mate, 12, stats=again, update=False)[:1], alone)
    with pytest.raises(ValueError):
        G.jepa_track_features({"dnase": out["dnase"]}, 12, stats=stats)


def test_coverage_counts_hidden_steps_the_track_loss_sees():
    h = torch.zeros(1, 16, dtype=torch.bool)
    h[0, 0:4] = True                                                 # one whole track bin of 4 core steps
    h[0, 6:9] = True                                                 # 3 hidden steps across two bins: none whole
    assert G.jepa_track_coverage(h, 1) == 1.0
    assert abs(G.jepa_track_coverage(h, 4) - 4 / 7) < 1e-9
    assert G.jepa_track_coverage(torch.zeros(1, 16, dtype=torch.bool), 4) == 0.0
