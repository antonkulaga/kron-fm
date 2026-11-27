"""The codon head (config ``codon_dims``): protein features of a codon from the decoder states of its three bases,
its targets from a protein model's sparse features, and its loss. CPU, with the stand-in for the KronLSTM layer."""
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
                                                       n_down=3, arch="v2", codon_dims=12), **kw))).eval()


def test_targets_are_fixed_unit_vectors_of_the_active_features():
    g = torch.Generator().manual_seed(0)
    feat = torch.stack([torch.randperm(G.PROTEIN_SPACE, generator=g)[:64] for _ in range(5)])
    val = torch.randint(1, 256, (5, 64), generator=g, dtype=torch.uint8)
    scale = torch.tensor([9.0, 10.0, 11.0, 9.5, 10.5])
    t = G.protein_targets(feat, val, scale, 16)
    assert t.shape == (5, 16) and not t.requires_grad
    assert torch.allclose(t.norm(dim=-1), torch.ones(5), atol=1e-5)
    assert torch.equal(t, G.protein_targets(feat.clone(), val.clone(), scale.clone(), 16))        # fixed
    perm = torch.randperm(64, generator=g)
    assert torch.allclose(t, G.protein_targets(feat[:, perm], val[:, perm], scale, 16), atol=1e-5)   # a set, not a list
    assert torch.allclose(t, G.protein_targets(feat, val, scale * 3, 16), atol=1e-5)              # the length is dropped
    same = G.protein_targets(feat[:1].repeat(2, 1), val[:1].repeat(2, 1), scale[:2], 16)
    assert torch.allclose(same[0], same[1], atol=1e-5)
    assert float((t @ t.t() - torch.eye(5)).abs().max()) < 0.9                                     # others differ
    assert float(G.codon_loss(t, t)) < 1e-6 and float(G.codon_loss(t, -t)) > 1.99
    assert float(G.codon_loss(t[:0].requires_grad_(), t[:0])) == 0.0


def test_head_reads_the_three_bases_in_the_order_given(cpu_mixer):
    m = small()
    assert small(codon_dims=0).codon_head is None
    seq = torch.randint(0, 4, (2, 64))
    rows = torch.tensor([0, 0, 1, 1])
    pos = torch.tensor([[3, 4, 5], [5, 4, 3], [10, 40, 41], [61, 62, 63]])     # plus, the same on minus, split, the end
    m.codon_at = (rows, pos)
    m(seq)
    out, ok = m.last_codon, m.last_codon_ok
    assert out.shape == (4, 12) and bool(ok.all()) and m.codon_at is None      # given for one forward
    assert not torch.allclose(out[0], out[1])                                  # the order of the bases matters
    m(seq)
    assert m.last_codon is None                                                # no codons given: nothing computed
    m.codon_at = (rows[:0], pos[:0])
    m(seq)
    assert m.last_codon.shape == (0, 12)
    m.train()
    m.codon_at = (rows, pos)
    m(seq)
    G.codon_loss(m.last_codon, torch.nn.functional.normalize(torch.randn(4, 12), dim=-1)).backward()
    assert all(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in m.codon_head.parameters())
    assert m.stem.weight.grad is not None and float(m.stem.weight.grad.abs().sum()) > 0   # it teaches the trunk


def test_split_gives_the_same_codons_and_marks_those_cut(cpu_mixer, monkeypatch):
    forced_replicas(monkeypatch)
    m = small()
    seq = torch.randint(0, 4, (2, 128))
    rows = torch.tensor([0, 1, 1, 0])
    pos = torch.tensor([[3, 4, 5], [100, 99, 98], [62, 63, 64], [10, 11, 120]])   # the last two straddle the cut at 64
    m.codon_at = (rows, pos)
    with torch.no_grad():
        m(seq)
        whole = m.last_codon
        m.codon_at = (rows, pos)
        m.forward_split(seq, devices=["cpu", "cpu"], by="length")
    assert m.last_codon_ok.tolist() == [True, True, False, False]
    assert torch.allclose(m.last_codon[:2], whole[:2], atol=1e-5) and float(m.last_codon[2:].abs().max()) == 0
    m.codon_at = (rows, pos)
    with torch.no_grad():
        m.forward_split(seq, devices=["cpu", "cpu"], by="rows")
    assert bool(m.last_codon_ok.all()) and torch.allclose(m.last_codon, whole, atol=1e-5)


def test_a_piece_without_codons_joins_one_with_codons_under_mixed_precision(cpu_mixer, monkeypatch):
    forced_replicas(monkeypatch)
    m = small()
    seq = torch.randint(0, 4, (1, 128))
    for pos in ([[3, 4, 5], [10, 11, 12]], [[100, 101, 102]]):       # all in the first piece; all in the second
        m.codon_at = (torch.zeros(len(pos), dtype=torch.long), torch.tensor(pos))
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            m.forward_split(seq, devices=["cpu", "cpu"], by="length")
        assert bool(m.last_codon_ok.all()) and m.last_codon.shape == (len(pos), 12)
        assert float(m.last_codon.float().abs().sum()) > 0


def test_centred_targets_lose_the_shared_component():
    g = torch.Generator().manual_seed(0)
    feat = torch.randint(0, G.PROTEIN_SPACE, (400, 64), generator=g)
    feat[:, :40] = feat[0, :40]                                       # 40 features every codon shares
    val = torch.randint(1, 255, (400, 64), generator=g).to(torch.uint8)
    scale = torch.ones(400)
    plain = G.protein_targets(feat, val, scale, 64)
    mean = torch.nn.functional.normalize(plain.mean(0), dim=0)
    assert float((plain @ mean).mean()) > 0.6                         # a constant output nearly meets the loss
    centre = G.protein_centre(feat, val, scale, 64)
    assert centre.shape == (64,)
    t = G.protein_targets(feat, val, scale, 64, centre)
    assert torch.allclose(t.norm(dim=-1), torch.ones(400), atol=1e-5)
    assert abs(float((t @ torch.nn.functional.normalize(t.mean(0), dim=0)).mean())) < 0.1
