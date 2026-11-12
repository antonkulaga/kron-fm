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
    logits, core, _ = model(seq, seg)
    assert logits.shape == (2, 1024, 4) and core.shape == (2, 64, 64)


def test_encode_and_split_contigs():
    seq = "ACGT" * 5 + "N" * 60 + "acgtR" * 4 + "N" * 10 + "TTTT"
    codes = G.encode_bases(seq)
    assert codes[:4].tolist() == [0, 1, 2, 3] and codes[84] == 4                      # R -> N
    assert G.contigs_of(codes, min_gap=50, min_len=5) == [(0, 20), (80, len(seq))]     # the 10-N run stays inside


def test_paint_maps_chromosome_coordinates_into_contigs():
    import numpy as np
    import polars as pl
    index = pl.DataFrame({"chrom": ["chr1", "chr1"], "start": [100, 500], "end": [200, 600], "offset": [0, 100]})
    out = np.zeros(200, dtype=np.uint8)
    n = G.paint(index, "chr1", np.array([150, 190, 300]), np.array([160, 520, 310]), out, 1 << 2)
    assert n == 3                                            # the third interval falls in a gap; the second spans it
    assert out[50:60].tolist() == [4] * 10 and out[90:100].tolist() == [4] * 10 and out[100:120].tolist() == [4] * 20
    act = np.zeros(200, dtype=np.uint8)
    G.paint(index, "chr1", np.array([150]), np.array([152]), act, 7, mode="max")
    G.paint(index, "chr1", np.array([150]), np.array([152]), act, 3, mode="max")
    assert act[50:52].tolist() == [7, 7]


def test_splice_sites_by_strand():
    import polars as pl
    ex = pl.DataFrame({"chrom": ["chr1"] * 4, "start": [101, 201, 101, 201], "end": [150, 250, 150, 250],
                       "strand": ["+", "+", "-", "-"], "transcript_id": ["a", "a", "b", "b"]})
    donors, acceptors = G.splice_sites(ex)
    # intron 151..200 (1-based) = 0-based [150, 200): GT at [150,152) on +, AG at [198,200) on +; swapped on -
    d = {(r["strand"], r["start"], r["end"]) for r in donors.iter_rows(named=True)}
    a = {(r["strand"], r["start"], r["end"]) for r in acceptors.iter_rows(named=True)}
    assert d == {("+", 150, 152), ("-", 198, 200)} and a == {("+", 198, 200), ("-", 150, 152)}


def test_paint_takes_a_value_per_interval():
    import numpy as np
    import polars as pl
    index = pl.DataFrame({"chrom": ["chr1"], "start": [0], "end": [10], "offset": [0]})
    out = np.zeros(10, dtype=np.uint8)
    G.paint(index, "chr1", np.array([0, 2]), np.array([4, 6]), out, np.array([5, 9]), mode="max")
    assert out.tolist() == [5, 5, 9, 9, 9, 9, 0, 0, 0, 0]


def test_track_targets_unpack_bits_and_activity():
    import torch
    lab = torch.tensor([[0b10000001, 0]], dtype=torch.uint8)
    act = torch.tensor([[255, 0]], dtype=torch.uint8)
    t = G.track_targets(lab, act)
    assert t.shape == (1, 2, len(G.TRACKS))
    assert t[0, 0].tolist() == [1, 0, 0, 0, 0, 0, 0, 1, 1] and t[0, 1].sum() == 0


def test_ranking_scores_match_sklearn():
    import numpy as np
    import torch
    from sklearn.metrics import average_precision_score, roc_auc_score
    rng = np.random.default_rng(0)
    y = rng.random(500) < 0.1
    s = rng.random(500) + y * 0.5
    auroc, ap = G.ranking_scores(torch.tensor(s), torch.tensor(y))
    assert abs(auroc - roc_auc_score(y, s)) < 1e-6 and abs(ap - average_precision_score(y, s)) < 1e-6


def test_track_head_shape_on_cpu_free_parts():
    import torch
    c = G.KronGenomeConfig(d_model=32, n_heads=2, n_layers=1, ff_units=64, stem_channels=8, n_down=3, n_tracks=9)
    m = G.KronGenome(c)
    assert m.tracks.out_channels == 9 and m.tracks.in_channels == 8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the KronLSTM core needs CUDA")
def test_forward_contig_matches_forward():
    torch.manual_seed(0)
    c = G.KronGenomeConfig(d_model=32, n_heads=2, n_layers=2, ff_units=64, stem_channels=8, n_down=3, dropout=0.0,
                           n_tracks=2)
    model = G.KronGenome(c).cuda().eval()
    seq = torch.randint(0, 4, (1, 8192), device="cuda")
    with torch.no_grad():
        full, _, tracks = model(seq)
        one = model.forward_contig(seq[0], train_at=[0], chunk=8192, halo=0)          # one chunk: exact
        parts = model.forward_contig(seq[0], train_at=[2048, 6144], chunk=2048, halo=1024)
    assert torch.allclose(one[0][0], full, atol=1e-3) and torch.allclose(one[0][1], tracks, atol=1e-3)
    # several chunks: GroupNorm's statistics are per chunk, so close but not equal
    r = torch.corrcoef(torch.stack([parts[0][0].flatten(), full[:, 2048:4096].flatten()]))[0, 1]
    assert r > 0.99 and parts[1][1].shape == (1, 2048, 2)
