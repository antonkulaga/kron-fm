"""KronGenome on CPU: masking leaves N alone, bins and segments line up (the core itself needs the CUDA kernels)."""
import pytest
import torch

from kron_dna import genome as G


def test_mask_bases_never_masks_n():
    torch.manual_seed(0)
    seq = torch.randint(0, 5, (2, 4096))
    inputs, labels = G.mask_bases(seq, prob=0.5)
    picked = labels != -100
    assert (seq[picked] < 4).all() and (inputs[picked] == G.MASK_CHANNEL).all()
    assert (inputs[~picked] == seq[~picked]).all() and 0.3 < picked.float().mean() < 0.5


def test_one_hot_channels():
    x = G.one_hot(torch.tensor([[0, 1, 2, 3, 4, 5]]))
    assert x.shape == (1, G.N_IN, 6) and torch.equal(x[0].argmax(0), torch.arange(6))


def test_length_must_fit_the_bins():
    model = G.KronGenome(G.KronGenomeConfig(d_model=32, n_heads=2, n_layers=1, ff_units=64, stem_channels=8, n_down=3))
    with pytest.raises(ValueError):
        model(torch.zeros(1, 20, dtype=torch.long))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the KronLSTM core needs CUDA")
def test_forward_shapes_cuda():
    c = G.KronGenomeConfig(d_model=64, n_heads=2, n_layers=2, ff_units=128, stem_channels=16, n_down=4)
    model = G.KronGenome(c).cuda()
    seq = torch.randint(0, 4, (2, 1024), device="cuda")
    seg = torch.zeros(2, 1024 // 16, dtype=torch.long, device="cuda")
    seg[:, 32:] = 1
    logits, core = model(seq, seg)
    assert logits.shape == (2, 1024, 4) and core.shape == (2, 64, 64)


def test_encode_and_split_contigs():
    seq = "ACGT" * 5 + "N" * 60 + "acgtR" * 4 + "N" * 10 + "TTTT"
    codes = G.encode_bases(seq)
    assert codes[:4].tolist() == [0, 1, 2, 3] and codes[84] == 4                      # R -> N
    assert G.contigs_of(codes, min_gap=50, min_len=5) == [(0, 20), (80, len(seq))]     # the 10-N run stays inside
