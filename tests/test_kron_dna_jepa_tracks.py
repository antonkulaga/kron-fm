"""The richer JEPA target (config ``jepa_track_dims``): the predictor's readout to track features, the fixed
compression of the target's head predictions, and its loss. CPU, with the stand-in for the KronLSTM layer."""
from __future__ import annotations

import pytest
import torch

import kron_core.blocks
from kron_core.testing import CausalMixer
from kron_dna import genome as G


@pytest.fixture
def cpu_mixer(monkeypatch):
    monkeypatch.setattr(kron_core.blocks, "kronlstm_mixer", CausalMixer)


def config(**kw) -> G.KronGenomeConfig:
    return G.KronGenomeConfig(**dict(dict(d_model=32, n_heads=2, n_layers=1, ff_units=64, stem_channels=8, n_down=3,
                                          arch="v2", jepa=True, jepa_pred_kind="context", jepa_pred_layers=1,
                                          jepa_track_dims=24, xheads=[["dnase", 5, 8], ["rna", 7, 8], ["splice", 2]],
                                          xhead_kind="mlp"), **kw))


def test_features_are_fixed_dense_and_weigh_heads_equally():
    torch.manual_seed(0)
    out = {"tf": torch.randn(2, 16, 300) * 5 + 3, "dnase": torch.randn(2, 16, 20) * 0.1}
    f = G.jepa_track_features(out, 24)
    assert f.shape == (2, 16, 24) and not f.requires_grad
    assert torch.equal(f, G.jepa_track_features({k: v.clone() for k, v in reversed(list(out.items()))}, 24))   # fixed
    a, b = f[..., :12], f[..., 12:]                                  # name order: dnase, then tf; 12 dims each
    assert abs(float(a.std()) - 1) < 1e-3 and abs(float(b.std()) - 1) < 1e-3           # equal weight, whatever the scale
    assert float((f.abs() < 1e-3).float().mean()) < 0.01             # dense: no block of zeros
    assert torch.equal(G.track_projection("tf", 300, 12), G.track_projection("tf", 300, 12))
    assert not torch.equal(G.track_projection("tf", 300, 12), G.track_projection("dnase", 300, 12))
    assert G.jepa_track_features(out, 25).shape[-1] == 25            # the first head takes the remainder
    with pytest.raises(ValueError):
        G.jepa_track_features(out, 1)


def test_pooling_to_the_heads_bins_and_the_loss():
    x = torch.arange(2 * 8 * 3, dtype=torch.float32).reshape(2, 8, 3)
    assert torch.equal(G.pool_bins(x, 1), x) and G.pool_bins(x, 4).shape == (2, 2, 3)
    assert torch.allclose(G.pool_bins(x, 4)[0, 0], x[0, :4].mean(0))
    hid = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 1], [0] * 8], dtype=torch.bool)
    assert G.pool_bins(hid, 4).tolist() == [[True, False], [False, False]]     # hidden only when every step is
    with pytest.raises(ValueError):
        G.pool_bins(x, 3)
    t = torch.randn(2, 8, 5)
    hidden = torch.zeros(2, 8, dtype=torch.bool)
    hidden[0, 2:6] = True
    assert float(G.jepa_track_loss(t.clone(), t, hidden)) == 0.0     # the predictor equals the target
    p = t.clone()
    p[1] += 3.0                                                      # a visible row is not scored
    assert float(G.jepa_track_loss(p, t, hidden)) == 0.0
    p[0, 3] += 2.0
    assert float(G.jepa_track_loss(p, t, hidden)) == pytest.approx(4.0 * 5 / (4 * 5))
    assert float(G.jepa_track_loss(p, t, torch.zeros_like(hidden))) == 0.0


def test_the_readout_trains_the_predictor_and_the_trunk_but_no_head(cpu_mixer):
    torch.manual_seed(0)
    m = G.KronGenome(config()).train()
    assert m.jepa_tracks is not None and m.jepa_tracks.out_features == 24
    seq = torch.randint(0, 4, (2, 256))
    hidden = torch.zeros(2, 32, dtype=torch.bool)
    hidden[:, 8:20] = True
    m.jepa_hidden = hidden
    torch.nn.init.normal_(m.jepa_pred.out.weight, std=0.1)           # the predictor starts as the identity: open it
    m(seq)
    assert m.last_jepa_tracks.shape == (2, 32, 24)
    with torch.no_grad():                                            # "the target": this model's own heads, clean
        target = G.jepa_track_features({k: v for k, v in m.last_x.items() if k != "splice"}, 24)
    assert target.shape == (2, 32, 24)                               # heads at 8 bp = the core's step here
    loss = G.jepa_track_loss(m.last_jepa_tracks, target, hidden)
    assert float(loss) > 0
    loss.backward()
    grad = lambda ps: sum(float(p.grad.abs().sum()) for p in ps if p.grad is not None)
    assert grad(m.jepa_tracks.parameters()) > 0 and grad(m.jepa_pred.parameters()) > 0
    assert grad(m.core.parameters()) > 0 and grad(m.down.parameters()) > 0          # through the gate into the trunk
    assert grad(m.xheads.parameters()) == 0 and grad(m.head.parameters()) == 0      # and into no head
    assert grad(m.up.parameters()) == 0                              # nor the decoder: the readout reads the predictor
    m.zero_grad()
    m.jepa_gate, m.jepa_skip_cut = 0.0, True                         # gate closed: the trunk gets nothing
    m(seq)
    G.jepa_track_loss(m.last_jepa_tracks, target, hidden).backward()
    assert grad(m.jepa_tracks.parameters()) > 0 and grad(m.core.parameters()) == 0


def test_off_by_default_and_old_checkpoints_load(cpu_mixer):
    plain = G.KronGenome(config(jepa_track_dims=0))
    assert plain.jepa_tracks is None
    plain.train()
    plain(torch.randint(0, 4, (1, 64)))
    assert plain.last_jepa_tracks is None
    rich = G.KronGenome(config())
    missing, unexpected = rich.load_state_dict(plain.state_dict(), strict=False)
    assert sorted(missing) == ["jepa_tracks.bias", "jepa_tracks.weight"] and not unexpected   # new parameters only
    with pytest.raises(ValueError):
        G.KronGenome(config(jepa=False))
