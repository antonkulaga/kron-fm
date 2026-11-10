"""Probe features and the ridge head: right shapes, CAI's definition, no use of the test split for choices."""
from __future__ import annotations

import numpy as np

from kron_dna import probe as P


def test_features_have_the_documented_shapes_and_values():
    seqs = ["ATGGCCGCTTAA", "ATGGCAGCGTGA"]
    gc = P.gc_features(seqs)
    assert gc.shape == (2, 4) and np.isclose(gc[0, 0], 6 / 12)
    usage = P.codon_usage(seqs)
    assert usage.shape == (2, 64) and np.allclose(usage.sum(1), 1)
    assert P.kmer_features(seqs).shape == (2, 4 + 16 + 64 + 256)
    oh = P.onehot_pos(seqs)
    assert oh.shape == (2, 4 * 64) and oh.sum() == 8


def test_cai_is_one_for_the_preferred_codons_and_lower_otherwise():
    w = P.relative_adaptiveness(["ATG" + "GCC" * 50 + "GCA" * 5 + "TAA"])
    assert np.isclose(w["GCC"], 1.0) and w["GCA"] < 0.2
    assert np.isclose(P.cai(["ATGGCCGCCTAA"], w)[0, 0], 1.0)
    assert P.cai(["ATGGCAGCATAA"], w)[0, 0] < 0.2


def test_ridge_probe_recovers_a_linear_signal_and_never_reads_test_for_alpha():
    rng = np.random.default_rng(0)
    x = {k: rng.normal(size=(n, 5)) for k, n in (("train", 200), ("val", 50), ("test", 50))}
    beta = np.array([1.0, -2.0, 0.0, 0.5, 0.0])
    y = {k: v @ beta + 0.1 * rng.normal(size=len(v)) for k, v in x.items()}
    result = P.ridge_probe(x, y)
    assert result["test_spearman"] > 0.95
    y_bad = dict(y, test=rng.normal(size=50))
    assert P.ridge_probe(x, y_bad)["alpha"] == result["alpha"]
