import numpy as np
import polars as pl
import pytest
import torch

from kron_dna import genome as G
from kron_dna import genome_variants as GV

INDEX = pl.DataFrame({"chrom": ["chr1", "chr1", "chr2"], "start": [100, 1000, 0], "end": [600, 1400, 300],
                      "offset": [0, 500, 900]})


def test_locate_maps_positions_and_gaps():
    chrom = np.array(["chr1", "chr1", "chr1", "chr2", "chr3", "chr1"])
    pos0 = np.array([100, 599, 1000, 299, 5, 700])
    g, off, length = GV.locate(INDEX, chrom, pos0)
    assert g.tolist() == [0, 499, 500, 1199, -1, -1]
    assert off.tolist() == [0, 0, 500, 900, -1, -1] and length.tolist() == [500, 500, 400, 300, -1, -1]


def test_region_order_and_near_splice():
    bit = lambda n: 1 << G.LABEL_BITS[n]
    labels = np.zeros(100, dtype=np.uint8)
    labels[10] = bit("CDS") | bit("exon")
    labels[20] = bit("donor")
    labels[40] = bit("exon")
    labels[60] = bit("enhancer")
    g = np.array([10, 20, 25, 40, 60, 80])
    assert GV.region_of(labels, g, near=10).tolist() == [
        "coding", "splice site", "near splice site", "UTR exon", "regulatory element", "intron or intergenic"]


def test_splice_distance():
    labels = np.zeros(100, dtype=np.uint8)
    labels[[20, 21]] = 1 << G.LABEL_BITS["donor"]
    labels[70] = 1 << G.LABEL_BITS["acceptor"]
    assert GV.splice_distance(labels, np.array([0, 20, 30, 50, 99]), block=16).tolist() == [20, 0, 9, 20, 29]


def test_tile_keeps_a_quarter_window_of_context():
    window = 64
    c_off, c_len = np.full(300, 1000), np.full(300, 300)
    g = c_off + np.arange(300)
    start, length = GV.tile_of(g, c_off, c_len, window)
    local = g - start
    assert (length == window).all() and (start >= c_off).all() and (start + length <= c_off + c_len).all()
    inner = (g - c_off >= window // 4) & (c_off + c_len - g > window // 4)
    assert ((local[inner] >= window // 4) & (local[inner] < window - window // 4)).all()
    assert len(np.unique(start)) <= 2 * 300 // window + 1
    s, n = GV.tile_of(np.array([5]), np.array([0]), np.array([40]), window)    # short contig: one padded tile
    assert s.tolist() == [0] and n.tolist() == [40]


def test_place_checks_reference_and_prefixes_chr():
    bases = np.zeros(1200, dtype=np.uint8)         # all A
    bases[499] = 1                                 # C at chr1:600 (1-based)
    labels = np.zeros(1200, dtype=np.uint8)
    v = pl.DataFrame({"chrom": ["1", "1", "chr2", "5"], "pos": [600, 101, 10, 1], "ref": ["C", "G", "A", "A"],
                      "alt": ["T", "A", "G", "C"], "id": ["a", "b", "c", "d"], "label": [1, 0, 0, 1]})
    t, dropped = GV.place(v, INDEX, bases, labels, {"chr2"})
    assert t["id"].to_list() == ["a", "c"] and t["held_out"].to_list() == [False, True]
    assert dropped == {"outside the contigs": 1, "reference mismatch": 1}
    assert t["ref_code"].to_list() == [1, 0] and t["alt_code"].to_list() == [3, 2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the KronLSTM core needs CUDA")
def test_masked_marginals_match_a_direct_pass():
    torch.manual_seed(0)
    c = G.KronGenomeConfig(d_model=32, n_heads=2, n_layers=1, ff_units=64, stem_channels=8, n_down=3, dropout=0.0)
    model = G.KronGenome(c).cuda().eval()
    window = 256
    rng = np.random.default_rng(0)
    bases = rng.integers(0, 4, 1000).astype(np.uint8)
    g = np.array([300, 520])                       # one contig of 1000 bases at offset 0
    ref = bases[g].astype(np.int64)
    alt = (ref + 1) % 4
    start, length = GV.tile_of(g, np.zeros(2, np.int64), np.full(2, 1000), window)
    got = GV.masked_marginals(model, bases, g, ref, alt, start, length, window, torch.device("cuda"), batch_size=1)
    for i in range(2):
        seq = torch.from_numpy(bases[start[i]:start[i] + window].astype(np.int64)).cuda()
        same = (start == start[i])
        for j in np.nonzero(same)[0]:
            seq[g[j] - start[i]] = G.MASK_CHANNEL
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _, _ = model(seq[None])
        lp = torch.log_softmax(logits.float()[0, g[i] - start[i]], -1)
        assert got[i] == pytest.approx(float(lp[alt[i]] - lp[ref[i]]), abs=3e-3)   # bf16 logits


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the KronLSTM core needs CUDA")
def test_track_deltas_zero_for_a_silent_change_and_shaped():
    torch.manual_seed(0)
    c = G.KronGenomeConfig(d_model=32, n_heads=2, n_layers=1, ff_units=64, stem_channels=8, n_down=3, dropout=0.0,
                           n_tracks=9)
    model = G.KronGenome(c).cuda().eval()
    bases = np.random.default_rng(0).integers(0, 4, 500).astype(np.uint8)
    g = np.array([10, 250, 490])
    ref = bases[g].astype(np.int64)
    off, length = np.zeros(3, np.int64), np.full(3, 500)
    same = GV.track_deltas(model, bases, g, ref, ref, off, length, 128, torch.device("cuda"), batch_size=2)
    moved = GV.track_deltas(model, bases, g, ref, (ref + 1) % 4, off, length, 128, torch.device("cuda"))
    assert same.shape == (3, 9) and np.abs(same).max() == 0 and (moved.max(1) > 0).all()
