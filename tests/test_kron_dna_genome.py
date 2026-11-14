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


def test_track_mask_groups_and_names():
    import pytest
    assert all(G.track_mask("all"))
    reg, gene = G.track_mask("regulatory"), G.track_mask("gene")
    assert not any(r and g for r, g in zip(reg, gene)) and all(r or g for r, g in zip(reg, gene))
    assert G.track_mask("donor,acceptor") == [t in ("donor", "acceptor") for t in G.TRACKS]
    with pytest.raises(ValueError):
        G.track_mask("splice")


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


def test_gene_structure_splits_utrs_by_strand_and_marks_tss():
    import polars as pl
    rows = [  # + strand transcript t1: exon 1-100, CDS 21-80, UTRs 1-20 (5') and 81-100 (3')
        ("chr1", 1, 100, "transcript", "+", "t1", "protein_coding"), ("chr1", 1, 100, "exon", "+", "t1", "protein_coding"),
        ("chr1", 21, 80, "CDS", "+", "t1", "protein_coding"), ("chr1", 1, 20, "UTR", "+", "t1", "protein_coding"),
        ("chr1", 81, 100, "UTR", "+", "t1", "protein_coding"), ("chr1", 21, 23, "start_codon", "+", "t1", "protein_coding"),
        # - strand transcript t2: 201-300, CDS 221-280, so 281-300 is 5' and 201-220 is 3'
        ("chr1", 201, 300, "transcript", "-", "t2", "protein_coding"), ("chr1", 221, 280, "CDS", "-", "t2", "protein_coding"),
        ("chr1", 201, 220, "UTR", "-", "t2", "protein_coding"), ("chr1", 281, 300, "UTR", "-", "t2", "protein_coding"),
        ("chr1", 401, 450, "transcript", "+", "t3", "lncRNA"), ("chr1", 401, 450, "exon", "+", "t3", "lncRNA")]
    gt = pl.DataFrame(rows, schema=["chrom", "start", "end", "type", "strand", "transcript_id", "transcript_type"], orient="row")
    s = G.gene_structure(gt)
    span = lambda k: sorted(zip(s[k]["start"].to_list(), s[k]["end"].to_list()))
    assert span("utr5") == [(0, 20), (280, 300)] and span("utr3") == [(80, 100), (200, 220)]
    assert span("tss") == [(0, 1), (299, 300), (400, 401)]
    assert span("lncRNA_exon") == [(400, 450)] and span("start_codon") == [(20, 23)]
    assert span("gene_plus") == [(0, 100), (400, 450)] and span("gene_minus") == [(200, 300)]


def test_paint_ids_writes_ids_across_contigs():
    import numpy as np
    import polars as pl
    index = pl.DataFrame({"chrom": ["chr1", "chr1"], "start": [0, 100], "end": [50, 150], "offset": [0, 50]})
    out = np.full(100, -1, dtype=np.int32)
    n = G.paint_ids(index, "chr1", np.array([40, 120]), np.array([110, 130]), out, np.array([7, 9]))
    assert n == 3 and (out[40:50] == 7).all() and (out[50:60] == 7).all() and (out[70:80] == 9).all()
    assert (out[:40] == -1).all() and (out[60:70] == -1).all()


def test_channel_rms_norm_is_per_position():
    import torch
    torch.manual_seed(0)
    norm = G.ChannelRMSNorm(8)
    x = torch.randn(2, 8, 300)
    full, part = norm(x), norm(x[:, :, 100:200])
    assert torch.allclose(full[:, :, 100:200], part, atol=1e-6)        # no statistic crosses positions
    assert torch.allclose(full.pow(2).mean(1), torch.ones(2, 300), atol=1e-3)


def test_conv_block2_and_mlp_head_shapes():
    import torch
    x = torch.randn(2, 16, 64)
    assert G.ConvBlock2(16, 24)(x).shape == (2, 24, 64) and G.ConvBlock2(16, 16)(x).shape == (2, 16, 64)
    assert G.MLPHead(16, 32, 4)(x).shape == (2, 4, 64)


def test_bert_masking_proportions():
    import torch
    g = torch.Generator().manual_seed(0)
    seq = torch.randint(0, 4, (1, 400_000))
    seq[0, :1000] = 4                                                    # N is never picked
    inputs, labels = G.mask_bases(seq, 0.15, generator=g, bert=True)
    picked = labels != -100
    assert not picked[0, :1000].any() and abs(picked.float().mean().item() - 0.15) < 0.005
    masked = (inputs == G.MASK_CHANNEL) & picked
    kept = (inputs == seq) & picked
    assert abs(masked.sum().item() / picked.sum().item() - 0.8) < 0.01
    assert abs(kept.sum().item() / picked.sum().item() - (0.1 + 0.1 / 4)) < 0.01   # 10 % kept + random hits own base
    assert ((inputs != seq) == (masked | (picked & (inputs != seq)))).all()
    plain, _ = G.mask_bases(seq, 0.15, generator=torch.Generator().manual_seed(0))
    assert set(plain[labels != -100].unique().tolist()) == {G.MASK_CHANNEL}


def test_reverse_complement():
    import numpy as np
    assert G.reverse_complement(np.array([0, 1, 2, 3, 4, 0], dtype=np.uint8)).tolist() == [3, 4, 0, 1, 2, 3]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the KronLSTM core needs CUDA")
def test_v2_forward_shapes_and_zero_context():
    torch.manual_seed(0)
    c = G.KronGenomeConfig(d_model=32, n_heads=2, n_layers=1, ff_units=64, stem_channels=16, n_down=3, dropout=0.0,
                           n_tracks=5, arch="v2", n_contexts=34)
    model = G.KronGenome(c).cuda().eval()
    seq = torch.randint(0, 4, (2, 256), device="cuda")
    with torch.no_grad():
        logits, core, tracks = model(seq)
        logits_c, _, _ = model(seq, context=torch.tensor([1, 2], device="cuda"))
    assert logits.shape == (2, 256, 4) and tracks.shape == (2, 256, 5) and core.shape == (2, 32, 32)
    assert torch.allclose(logits, logits_c)                             # zero-initialised contexts change nothing
    out = model.forward_contig(seq[0], [0], chunk=128, halo=0, context=2)
    assert out[0][0].shape == (1, 128, 4)


def test_read_genbank_parses_records_and_code():
    text = """LOCUS       NC_000001            20 bp    DNA     circular MAM 01-JAN-2020
VERSION     NC_000001.1
SOURCE      mitochondrion Homo sapiens
  ORGANISM  Homo sapiens
            Eukaryota; Metazoa; Chordata; Mammalia; Primates; Hominidae; Homo.
FEATURES             Location/Qualifiers
     CDS             1..9
                     /transl_table=2
     CDS             10..18
                     /transl_table=2
ORIGIN
        1 acgtacgtac gtacgtacgt
//
LOCUS       NW_1            5999 bp    DNA     linear   CON 17-APR-2019
VERSION     NW_1.1
  ORGANISM  Plasmodium gonderi
            Eukaryota; Sar.
CONTIG      join(x)
//
"""
    recs = list(G.read_genbank(text.splitlines(keepends=True)))
    assert [r["accession"] for r in recs] == ["NC_000001.1", "NW_1.1"]
    a, b = recs
    assert a["circular"] and a["transl_table"] == 2 and a["sequence"] == "ACGTACGTACGTACGTACGT"
    assert "Primates" in a["lineage"] and a["organism"] == "Homo sapiens"
    assert not b["circular"] and b["sequence"] == "" and b["transl_table"] == 1
