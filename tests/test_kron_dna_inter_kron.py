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


def test_every_piecewise_path_restarts_where_a_whole_forward_does(cpu_mixer):
    """The capped evaluation and the whole-contig forward cut a window into pieces: with in-tower blocks their
    pieces must start and end on the restart span, or the blocks restart elsewhere than in training."""
    from kron_dna import evalkit
    m = small(inter_levels=[2], bin_convs=2, xheads=[["a", 2, 8]], xhead_kind="mlp")
    shake(m)
    g = torch.Generator().manual_seed(2)
    for b in list(m.bin_pre) + list(m.bin_post):
        b.conv.weight.data = torch.randn(b.conv.weight.shape, generator=g) * 0.2
    assert G.piece_step(m.config) == 128 and G.piece_step(small().config) == 8
    assert G.piece_bounds(m.config, 1000, 1400, 8192, 100) == (896, 1536)
    seq = torch.randint(0, 4, (4096,))
    with torch.no_grad():
        logits, core, _ = m(seq[None])
        whole_x = {k: v.clone() for k, v in m.last_x.items()}
        capped = evalkit.core_capped(m, seq, None, 1000)              # chunks that do not sit on the span
        assert torch.allclose(capped, core, atol=1e-5)
        got = evalkit.decode_span(m, seq, core, (1024, 2048))
        lo, hi = m.last_x_at
        assert torch.allclose(got["a"], whole_x["a"][:, lo // 8:hi // 8], atol=1e-5)
        outs = m.forward_contig(seq, [1024, 3072], chunk=1024)
        for (lg, _), s in zip(outs, (1024, 3072)):
            assert torch.allclose(lg, logits[:, s:s + 1024], atol=1e-5)


def test_the_parallel_branch_starts_at_zero_has_the_same_weights_and_can_be_switched_off(cpu_mixer):
    """inter_parallel: the two blocks of a level as a side branch from the encoder's features into the decoder."""
    plain, serial, par = small(), small(inter_levels=[2]), small(inter_levels=[2], inter_parallel=True)
    assert {k: v.shape for k, v in par.state_dict().items()} == {k: v.shape for k, v in serial.state_dict().items()}
    par.load_state_dict(serial.state_dict())
    seq = torch.randint(0, 4, (2, 512))
    with torch.no_grad():
        a = plain(seq)
        assert torch.allclose(par(seq)[0], a[0], atol=1e-6)
        shake(par), shake(serial)
        out = par(seq)
        assert not torch.allclose(out[0], a[0], atol=1e-3)
        assert torch.allclose(out[1], a[1], atol=1e-6)                 # the tower and the core are left plain
        assert not torch.allclose(serial(seq)[1], a[1], atol=1e-4)     # the serial blocks sit before the core
        assert not torch.allclose(out[0], serial(seq)[0], atol=1e-3)
        par.inter_branch_off = True
        assert torch.allclose(par(seq)[0], a[0], atol=1e-6)
    par.inter_branch_off = False
    par.train()(seq)[0].float().pow(2).sum().backward()
    assert all(float(b.out.weight.grad.abs().sum()) > 0 for b in par.inter.values())


def test_the_parallel_branch_in_a_split_window_and_in_the_piecewise_paths(cpu_mixer, monkeypatch):
    from kron_dna import evalkit
    forced_replicas(monkeypatch)
    m = small(inter_levels=[2], inter_parallel=True, xheads=[["a", 2, 8]], xhead_kind="mlp")
    shake(m)
    seq = torch.randint(0, 4, (1, 4096))
    with torch.no_grad():
        logits, core, _ = m(seq)
        whole_x = {k: v.clone() for k, v in m.last_x.items()}
        pieces, _ = m.forward_split(seq, devices=["cpu", "cpu"], by="length")
        assert [p.start % 128 for p in pieces] == [0, 0]
        assert torch.allclose(G.gather_split(pieces, "logits", "cpu"), logits, atol=1e-5)
        got = evalkit.decode_span(m, seq[0], core, (1024, 2048))
        lo, hi = m.last_x_at
        assert torch.allclose(got["a"], whole_x["a"][:, lo // 8:hi // 8], atol=1e-5)
        outs = m.forward_contig(seq[0], [1024, 3072], chunk=1024)
        for (lg, _), s in zip(outs, (1024, 3072)):
            assert torch.allclose(lg, logits[:, s:s + 1024], atol=1e-5)
