"""bin_convs: extra dilated convolutions at bin resolution on each side of the core. They start as the identity, add
reach that the split forward accounts for, and train. CPU, with the stand-in for the KronLSTM layer."""
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
                                                       n_down=3, arch="v2"), **kw))).eval()


def shake(m: G.KronGenome) -> None:
    g = torch.Generator().manual_seed(1)
    for blocks in (m.bin_pre, m.bin_post):
        for b in blocks:
            b.conv.weight.data = torch.randn(b.conv.weight.shape, generator=g) * 0.2
    m.bin_skip.weight.data = torch.randn(m.bin_skip.weight.shape, generator=g) * 0.2


def test_dilations_and_reach():
    assert G.bin_dilations(0) == [] and G.bin_dilations(6) == [1, 2, 4, 8, 8, 8]
    base = G.KronGenomeConfig(d_model=32, stem_channels=8, n_down=5, arch="v2")
    more = G.KronGenomeConfig(d_model=32, stem_channels=8, n_down=5, arch="v2", bin_convs=4)
    grown = G.split_reach(more) - G.split_reach(base)                         # 30 bins of 32 bp on the longer path
    assert 2 * 2 * (1 + 2 + 4 + 8) * 32 <= grown <= 2 * 2 * (1 + 2 + 4 + 8) * 32 + G.split_reach(base)


@pytest.mark.parametrize("arch", ["v2", "v2l"])
def test_they_start_as_the_identity_and_fit_an_older_checkpoint(cpu_mixer, arch):
    old, new = small(arch=arch), small(arch=arch, bin_convs=3)
    missing = new.load_state_dict(old.state_dict(), strict=False)
    assert missing.unexpected_keys == [] and all(k.startswith(("bin_pre.", "bin_post.", "bin_skip.")) for k in missing.missing_keys)
    seq = torch.randint(0, 4, (2, 256))
    with torch.no_grad():
        a, b = old(seq)[0], new(seq)[0]
    assert torch.allclose(a, b, atol=1e-6)                                    # the trained model is unchanged
    assert new.n_parameters() - old.n_parameters() == 2 * 3 * (32 * 32 * 5 + 32 + 32) + 32 * 32 + 32
    shake(new)
    with torch.no_grad():
        assert not torch.allclose(new(seq)[0], a, atol=1e-3)                  # and they do something once trained


def test_a_split_window_gives_the_same_outputs(cpu_mixer, monkeypatch):
    forced_replicas(monkeypatch)
    m = small(bin_convs=4)
    shake(m)
    seq = torch.randint(0, 4, (1, 4096))
    with torch.no_grad():
        whole = m(seq)[0]
        pieces, _ = m.forward_split(seq, devices=["cpu", "cpu"], by="length")
    assert torch.allclose(G.gather_split(pieces, "logits", "cpu"), whole, atol=1e-5)
    m2 = small(bin_convs=4)
    shake(m2)
    with torch.no_grad():                                                     # with too short a halo they differ:
        short, _ = m2.forward_split(seq, devices=["cpu", "cpu"], by="length", halo=G.split_halo(small().config))
    assert not torch.allclose(G.gather_split(short, "logits", "cpu"), whole, atol=1e-5)   # the reach is real


def test_they_train(cpu_mixer):
    m = small(bin_convs=2).train()
    out = m(torch.randint(0, 4, (2, 128)))[0]
    out.float().pow(2).sum().backward()
    for b in list(m.bin_pre) + list(m.bin_post):
        assert b.conv.weight.grad is not None and float(b.conv.weight.grad.abs().sum()) > 0
    assert float(m.bin_skip.weight.grad.abs().sum()) > 0


def test_the_skip_carries_the_bins_past_the_core(cpu_mixer):
    m = small(bin_convs=1)
    seq = torch.randint(0, 4, (1, 256))
    with torch.no_grad():
        before = m(seq)[0]
        torch.nn.init.eye_(m.bin_skip.weight[:, :, 0])
        after = m(seq)[0]
    assert not torch.allclose(before, after, atol=1e-3)               # the decoder now also reads the core's input
    with pytest.raises(ValueError):
        small(arch="v1", bin_convs=2)
