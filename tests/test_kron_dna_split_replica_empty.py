"""A split replica's heads with nothing to compute: the empty branches name their weights, since a replica's
parameters() is empty (seen 25 Nov on two cards: a lean head with no selected position, and the codon head with no
codon in a piece, summed nothing into an int)."""
import torch

from kron_dna import genome as G


def _model(**kw):
    torch.manual_seed(0)
    return G.KronGenome(G.KronGenomeConfig(d_model=96, n_layers=1, stem_channels=16, n_down=3, arch="v2", **kw))


def test_lean_head_empty_selection_on_replica():
    model = _model()
    rep = model.split_replica("cpu", force=True)
    assert not list(rep.head.parameters())                     # the case: a replica registers no parameter
    x = torch.randn(1, model.config.stem_channels, 64)
    out = rep.head.at(x, torch.zeros(1, 64, dtype=torch.bool))
    assert out.shape[-1] == 64 and float(out.abs().sum()) == 0.0
    out.sum().backward()
    assert model.head.fc1.weight.grad is not None              # zero gradients reach the model, as unsplit


def test_codon_head_no_codon_on_replica():
    model = _model(codon_dims=8)
    rep = model.split_replica("cpu", force=True)
    x = torch.randn(1, model.config.stem_channels, 64)
    out = rep.codon_head(x, torch.zeros(0, dtype=torch.long), torch.zeros(0, 3, dtype=torch.long))
    assert out.shape == (0, 8)
    out.sum().backward()
    assert model.codon_head.fc1.weight.grad is not None


def test_new_modules_run_on_the_piece_card():
    assert {"inter", "bin_pre", "bin_post", "bin_skip", "codon_head"} <= set(G.KronGenome.SPLIT_LOCAL)
