"""Every module of the model is either run on each piece's own card in a split window (SPLIT_LOCAL) or on the first
card over the whole batch: a module added to the conv path and left out of SPLIT_LOCAL runs the second card's piece
with the first card's weights, which a one-device test cannot see (25 Nov 2026: five new modules were left out)."""
from __future__ import annotations

import pytest
import torch

import kron_core.blocks
from kron_core.testing import CausalMixer
from kron_dna import genome as G

#: what runs once, on the first card: the core over the whole batch and what reads its states
WHOLE_BATCH = {"core", "core_norm", "context", "jepa_pred", "jepa_tracks", "epi_in", "epi_film", "epi_out"}


@pytest.fixture
def cpu_mixer(monkeypatch):
    monkeypatch.setattr(kron_core.blocks, "kronlstm_mixer", CausalMixer)


def everything() -> G.KronGenome:
    return G.KronGenome(G.KronGenomeConfig(
        d_model=32, n_heads=2, n_layers=1, ff_units=64, stem_channels=8, n_down=3, arch="v2", bin_convs=1,
        inter_levels=[1], codon_dims=4, n_ntv3=3, n_tracks=2, xheads=[["a", 2, 8]], xhead_kind="mlp", jepa=True,
        jepa_track_dims=4, n_contexts=2, n_epigenomes=2, local_width=16))


def test_every_module_has_a_side_in_a_split(cpu_mixer):
    kids = {n for n, _ in everything().named_children()}
    undecided = kids - set(G.KronGenome.SPLIT_LOCAL) - WHOLE_BATCH
    assert not undecided, f"new modules {sorted(undecided)}: add them to SPLIT_LOCAL (conv path, per-base heads) or to WHOLE_BATCH here"
    assert not set(G.KronGenome.SPLIT_LOCAL) & WHOLE_BATCH


def test_empty_selections_work_on_a_replica(cpu_mixer):
    m = everything()
    r = m.split_replica(torch.device("cpu"), force=True)
    assert list(r.codon_head.parameters()) == []                     # a replica's weights are plain attributes
    x = torch.randn(1, 8, 16, requires_grad=True)
    none = torch.zeros(1, 16, dtype=torch.bool)
    out = r.head.at(x, none)                                         # the lean head with nothing selected
    cod = r.codon_head(x, torch.zeros(0, dtype=torch.long), torch.zeros(0, 3, dtype=torch.long))
    assert out.shape == (1, 4, 16) and cod.shape == (0, 4)
    (out.sum() + cod.sum()).backward()
    assert float(m.head.fc1.weight.grad.abs().sum()) == 0 and float(m.codon_head.fc1.weight.grad.abs().sum()) == 0
    for head in (m.head, m.codon_head):                              # every weight, the norm's too: zero, not None
        assert all(p.grad is not None and float(p.grad.abs().sum()) == 0 for p in head.parameters())
