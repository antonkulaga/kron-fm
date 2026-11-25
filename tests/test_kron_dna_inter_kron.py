"""inter_levels: recurrent blocks inside the conv tower (one in the encoder, one in the decoder per level). They
start as the identity, restart every local_span bases, and a split window gives the same outputs. CPU, with the
stand-in for the KronLSTM layer."""
from __future__ import annotations

import pytest
import torch

import kron_core.blocks
from kron_core.testing import CausalMixer
from kron_dna import genome as G
from test_kron_dna_split import forced_replicas  # noqa: F401


@pytest.fixture
def cpu_mixer(monkeypatch):
    monkeypatch.setattr(kron_core.blocks, "kronlstm_mixer", CausalMixer)


def small(**kw) -> G.KronGenome:
    torch.manual_seed(0)
    return G.KronGenome(G.KronGenomeConfig(**dict(dict(d_model=32, n_heads=2, n_layers=1, ff_units=64, stem_channels=8,
                                                       n_down=3, arch="v2", local_span=128, inter_width=16), **kw))).eval()


def shake(m: G.KronGenome) -> None:
    g = torch.Generator().manual_seed(1)
    for b in m.inter.values():
        b.out.weight.data = torch.randn(b.out.weight.shape, generator=g) * 0.3


def test_they_start_as_the_identity_and_fit_an_older_checkpoint(cpu_mixer):
    old, new = small(), small(inter_levels=[1, 2])
    assert sorted(new.inter) == ["dec1", "dec2", "enc1", "enc2"]
    missing = new.load_state_dict(old.state_dict(), strict=False)
    assert missing.unexpected_keys == [] and all(k.startswith("inter.") for k in missing.missing_keys)
    seq = torch.randint(0, 4, (2, 512))
    with torch.no_grad():
        a = old(seq)[0]
        assert torch.allclose(new(seq)[0], a, atol=1e-6)
        shake(new)
        assert not torch.allclose(new(seq)[0], a, atol=1e-3)


def test_the_state_restarts_every_span(cpu_mixer):
    m = small(inter_levels=[2])
    shake(m)
    block, x = m.inter["enc2"], torch.randn(1, m.inter["enc2"].inp.in_channels, 96)
    with torch.no_grad():
        whole = block(x, 32)
        parts = torch.cat([block(x[:, :, a:a + 32], 32) for a in (0, 32, 64)], -1)
        assert torch.allclose(whole, parts, atol=1e-6)                 # three segments, each on its own
        assert not torch.allclose(block(x, 96), whole, atol=1e-4)      # one segment reads further


def test_a_split_window_gives_the_same_outputs(cpu_mixer, monkeypatch):
    forced_replicas(monkeypatch)
    m = small(inter_levels=[2])
    shake(m)
    assert G.split_reach(m.config) >= 2 * 128 + G.split_reach(small().config)
    seq = torch.randint(0, 4, (1, 4096))
    with torch.no_grad():
        whole = m(seq)[0]
        pieces, _ = m.forward_split(seq, devices=["cpu", "cpu"], by="length")
    assert [p.start % 128 for p in pieces] == [0, 0]                  # cut where the state restarts
    assert torch.allclose(G.gather_split(pieces, "logits", "cpu"), whole, atol=1e-5)
    with torch.no_grad():                                             # one segment of halo is not enough
        short, _ = m.forward_split(seq, devices=["cpu", "cpu"], by="length", halo=128)
    assert not torch.allclose(G.gather_split(short, "logits", "cpu"), whole, atol=1e-5)


def test_they_train_and_refuse_what_they_cannot_do(cpu_mixer):
    m = small(inter_levels=[1]).train()
    m(torch.randint(0, 4, (2, 256)))[0].float().pow(2).sum().backward()
    assert all(float(b.out.weight.grad.abs().sum()) > 0 for b in m.inter.values())
    for bad in (dict(inter_levels=[3]), dict(inter_levels=[0]), dict(inter_levels=[1], arch="v2l"),
                dict(inter_levels=[2], local_span=130)):
        with pytest.raises(ValueError):
            small(**bad)
