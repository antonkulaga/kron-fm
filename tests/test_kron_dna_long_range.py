"""Two options for long-range use of the core: slow memory heads by initialisation (set_slow_heads) and a block
over the core's states pooled to coarse positions (coarse_ratio). CPU; the recurrent layer is the stand-in except
where the vendored layer's own decay parameters are tested."""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

import kron_core.blocks
from kron_core.testing import CausalMixer
from kron_dna import genome as G
from test_kron_dna_split import forced_replicas  # noqa: F401


@pytest.fixture
def cpu_mixer(monkeypatch):
    monkeypatch.setattr(kron_core.blocks, "kronlstm_mixer", CausalMixer)


def small(**kw) -> G.KronGenome:
    torch.manual_seed(0)
    return G.KronGenome(G.KronGenomeConfig(**dict(dict(d_model=32, n_heads=2, n_layers=2, ff_units=64, stem_channels=8,
                                                       n_down=3, arch="v2", coarse_width=16, coarse_heads=2,
                                                       coarse_after=1), **kw))).eval()


def shake(m: G.KronGenome) -> None:
    g = torch.Generator().manual_seed(1)
    m.coarse.out.weight.data = torch.randn(m.coarse.out.weight.shape, generator=g) * 0.3


def test_slow_heads_set_the_vendored_layers_decay():
    mixer = kron_core.blocks.kronlstm_mixer(64, 4, 0)                 # built on CPU; never run here
    before = (mixer.A_log.clone(), mixer.dt_bias.clone())
    G.set_slow_heads(mixer, 2, 234.0, 8192.0)                         # 30 kb to 1 Mb at 128-bp bins
    per = mixer.dt_bias.shape[0] // mixer.A_log.shape[0]
    x = torch.randn(5, 64)
    g = (mixer.A_log.exp().repeat_interleave(per) * F.softplus(mixer.f_proj(x) + mixer.dt_bias))[:, :2 * per]
    half = math.log(2) / g                                            # the layer's own formula, whatever the input
    assert torch.allclose(half, half[:1].expand_as(half), rtol=1e-4)
    assert abs(float(half[0, 0]) - 234) < 0.5 and abs(float(half[0, -1]) - 8192) < 20
    ratio = half[0, 1:] / half[0, :-1]
    assert torch.allclose(ratio, ratio[:1].expand_as(ratio), rtol=1e-3)            # log-uniform across the channels
    assert torch.equal(mixer.A_log[2:], before[0][2:]) and torch.equal(mixer.dt_bias[2 * per:], before[1][2 * per:])
    for bad in ((5, 10.0, 20.0), (0, 10.0, 20.0), (1, 1.0, 20.0), (1, 30.0, 20.0)):
        with pytest.raises(ValueError):
            G.set_slow_heads(mixer, *bad)


def test_slow_layers_reach_both_directions_and_are_refused_out_of_range(monkeypatch):
    seen = []
    monkeypatch.setattr(G, "set_slow_heads", lambda mixer, heads, lo, hi: seen.append((mixer, heads, lo, hi)))

    class WithDecay(CausalMixer):
        def __init__(self, *a):
            super().__init__(*a)
            self.dt_bias = torch.nn.Parameter(torch.zeros(4))

    monkeypatch.setattr(kron_core.blocks, "kronlstm_mixer", WithDecay)
    m = small(slow_layers=[1], slow_heads=1, slow_range=[800, 8000])
    assert [x[1:] for x in seen] == [(1, 100.0, 1000.0)] * 2          # bases over 8-bp bins
    assert {id(x[0]) for x in seen} == {id(m.core[1].forward_mixer), id(m.core[1].backward_mixer)}
    assert G.apply_slow_heads(m) == 2
    with pytest.raises(ValueError):
        small(slow_layers=[2])


@pytest.mark.parametrize("kind", ["attention", "kron"])
def test_the_coarse_block_starts_at_zero_and_fits_an_older_checkpoint(cpu_mixer, kind):
    old, new = small(), small(coarse_ratio=4, coarse_kind=kind)
    missing = new.load_state_dict(old.state_dict(), strict=False)
    assert missing.unexpected_keys == [] and all(k.startswith("coarse.") for k in missing.missing_keys)
    seq = torch.randint(0, 4, (2, 512))
    with torch.no_grad():
        a = old(seq)
        assert torch.allclose(new(seq)[0], a[0], atol=1e-6)
        shake(new)
        out = new(seq)
        assert not torch.allclose(out[1], a[1], atol=1e-3)
        new.coarse_off = True
        assert torch.allclose(new(seq)[0], a[0], atol=1e-6)
    new.coarse_off = False
    new.train()(seq)[0].float().pow(2).sum().backward()
    assert float(new.coarse.out.weight.grad.abs().sum()) > 0 and float(new.coarse.inp.weight.grad.abs().sum()) > 0


def test_attention_reads_the_far_end_of_a_row_and_rows_stay_apart(cpu_mixer):
    m = small(coarse_ratio=4, coarse_kind="attention")
    shake(m)
    block = m.coarse
    h = torch.randn(2, 64, 32)
    with torch.no_grad():
        base = block(h)
        far = h.clone()
        far[0, 60:] += 1.0                                            # the last coarse position of row 0
        moved = block(far)
    assert float((moved[0, :4] - base[0, :4]).abs().max()) > 1e-5     # row 0's first position sees it
    assert torch.allclose(moved[1], base[1], atol=1e-6)               # row 1 does not


@pytest.mark.parametrize("kind", ["attention", "kron"])
def test_a_row_that_is_not_whole_positions_ends_in_a_short_one(cpu_mixer, kind):
    """Short packed rows (the organelle slice gives 20 bins against a ratio of 16) are taken, not skipped."""
    m = small(coarse_ratio=16, coarse_kind=kind)
    shake(m)
    block = m.coarse
    h = torch.randn(2, 20, 32)
    with torch.no_grad():
        out = block(h)
        assert out.shape == h.shape and not torch.allclose(out, h, atol=1e-4)
        add = out - h
        assert torch.allclose(add[:, :16], add[:, :1].expand(-1, 16, -1), atol=1e-6)     # one value per position
        assert torch.allclose(add[:, 16:], add[:, 16:17].expand(-1, 4, -1), atol=1e-6)
        assert not torch.allclose(add[:, 15], add[:, 16], atol=1e-5)
        moved = h.clone()
        moved[0, 19] += 1.0                                           # the short tail's own bin moves its position
        assert not torch.allclose(block(moved)[0, 16:] - moved[0, 16:], add[0, 16:], atol=1e-5)
        short = block(torch.randn(1, 5, 32))                          # shorter than one position
        assert short.shape == (1, 5, 32)
        whole = torch.randn(1, 32, 32)                                # whole rows are as before
        assert torch.allclose(block(whole), block(torch.cat([whole, whole[:, :0]], 1)))
    seq = torch.randint(0, 4, (2, 160))                               # 20 bins of 8 bp through the whole model
    with torch.no_grad():
        assert m(seq)[0].shape[:2] == (2, 160)


@pytest.mark.parametrize("kind", ["attention", "kron"])
def test_a_split_window_gives_the_same_outputs_with_a_coarse_block(cpu_mixer, monkeypatch, kind):
    forced_replicas(monkeypatch)
    m = small(coarse_ratio=4, coarse_kind=kind)
    shake(m)
    seq = torch.randint(0, 4, (2, 2048))
    with torch.no_grad():
        whole = m(seq)[0]
        pieces, _ = m.forward_split(seq, devices=["cpu", "cpu"], by="length")
    assert torch.allclose(G.gather_split(pieces, "logits", "cpu"), whole, atol=1e-5)


def test_what_it_refuses(cpu_mixer):
    for bad in (dict(coarse_ratio=4, coarse_after=0), dict(coarse_ratio=4, coarse_after=3),
                dict(coarse_ratio=4, coarse_kind="conv")):
        with pytest.raises(ValueError):
            small(**bad)
