"""ClinVar mapping: locations, strand-aware CDS offsets, codon substitutions and AUROC."""
import numpy as np
import polars as pl
import pytest

from kron_dna import variants as V


def test_chromosome_of():
    assert V.chromosome_of("NC_000001.11") == "1"
    assert V.chromosome_of("NC_000023.11") == "X"
    assert V.chromosome_of("NC_000024.10") == "Y"
    assert V.chromosome_of("NC_012920.1") == "MT"
    assert V.chromosome_of("NT_187361.1") is None


def test_parse_location():
    assert V.parse_location("join(65565..65573,69037..70008)") == ("+", [(65565, 65573), (69037, 70008)])
    assert V.parse_location("complement(join(20..25,1..3))") == ("-", [(1, 3), (20, 25)])
    assert V.parse_location("100..199") == ("+", [(100, 199)])
    assert V.parse_location("join(<1..10,20..30)") is None


def test_cds_offset_both_strands():
    exons = [(1, 3), (10, 12)]                       # CDS bases at 1,2,3,10,11,12
    assert [V.cds_offset(p, "+", exons) for p in (1, 3, 10, 12)] == [0, 2, 3, 5]
    assert [V.cds_offset(p, "-", exons) for p in (12, 10, 3, 1)] == [0, 2, 3, 5]
    assert V.cds_offset(5, "+", exons) is None


def _transcript(strand: str) -> dict:
    # ATG GCT AAA TAA on the given strand; exons 1..6 and 11..16 of a toy chromosome
    seq = "ATGGCTAAATAA"
    return {"name": "tx", "gene": "G", "chrom": "1", "strand": strand, "exons": [(1, 6), (11, 16)], "seq": seq}


def _variants(rows):
    return pl.DataFrame(rows, schema=["chrom", "pos", "ref", "alt", "id", "genes", "clnsig", "label"], orient="row")


def test_map_plus_strand():
    tx = {"G": _transcript("+")}
    # pos 6 = CDS offset 5 = third base of GCT (T); T>C keeps Ala (synonymous); pos 11 = offset 6, A>G: AAA->GAA (K->E)
    table, dropped = V.map_variants(_variants([("1", 6, "T", "C", "1", ["G"], "Benign", 0),
                                               ("1", 11, "A", "G", "2", ["G"], "Pathogenic", 1),
                                               ("1", 2, "C", "A", "3", ["G"], "Benign", 0)]), tx)
    assert table["consequence"].to_list() == ["synonymous", "missense"]
    assert table["codon_index"].to_list() == [1, 2]
    assert table["alt_codon"].to_list() == ["GCC", "GAA"]
    assert dropped["reference mismatch"] == 1


def test_map_minus_strand_complements():
    tx = {"G": _transcript("-")}
    # minus strand: CDS offset 0 is genomic 16, offset 5 is genomic 11; offset 5 = T of GCT, genomic base is A
    table, dropped = V.map_variants(_variants([("1", 11, "A", "G", "1", ["G"], "Benign", 0)]), tx)
    assert table["alt_codon"].to_list() == ["GCC"] and table["consequence"].to_list() == ["synonymous"]
    assert not dropped


def test_auroc():
    label = np.array([1, 1, 0, 0])
    assert V.auroc(np.array([-3.0, -2.0, 1.0, 2.0]), label) == 1.0     # pathogenic score lower
    assert V.auroc(np.array([3.0, 2.0, -1.0, -2.0]), label) == 0.0
    assert V.auroc(np.zeros(4), label) == pytest.approx(0.5)
    assert V.auroc(np.zeros(2), np.array([1, 1])) is None


def test_masked_marginals_cpu_plain_graph():
    import torch
    from kron_dna.model import KronDNA, KronDNAConfig
    from kron_dna.onnx_export import KronDNAGraph

    torch.manual_seed(0)
    model = KronDNAGraph(KronDNA(KronDNAConfig(d_model=32, n_heads=2, n_layers=2, ff_units=64))).eval()
    seq = "ATG" + "GCT" * 40 + "TAA"
    tx = {"G": {"name": "tx", "gene": "G", "chrom": "1", "strand": "+", "exons": [(1, len(seq))], "seq": seq}}
    table = pl.DataFrame({"transcript": ["tx", "tx", "tx"], "codon_index": [5, 5, 30],
                          "ref_codon": ["GCT", "GCT", "GCT"], "alt_codon": ["GCC", "GCT", "ACT"]})
    s = V.masked_marginals(model, table, tx, torch.device("cpu"), window=16, batch_size=2)
    assert s.shape == (3,) and np.isfinite(s).all() and s[1] == 0.0


def test_sample_per_class():
    t = pl.DataFrame({"id": [str(i) for i in range(10)], "consequence": ["missense"] * 10,
                      "label": [1] * 3 + [0] * 7})
    out = V.sample_per_class(t, 2)
    assert out.group_by("label").len().sort("label")["len"].to_list() == [2, 2]
    assert V.sample_per_class(t, 0).height == 10


def test_bootstrap_ci_brackets_the_estimate():
    rng = np.random.default_rng(0)
    label = np.repeat([1, 0], 200)
    score = rng.normal(size=400) - label                 # pathogenic lower: AUROC well above 0.5
    est = V.auroc(score, label)
    lo, hi = V.bootstrap_ci(V.auroc, score, label, n=200)
    assert lo < est < hi and 0.5 < lo and hi - lo < 0.2
    assert V.bootstrap_ci(V.auroc, np.zeros(3), np.ones(3, int), n=10) is None   # one class in every resample
