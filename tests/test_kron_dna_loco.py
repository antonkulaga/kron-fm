"""kron_dna.loco: leave-one-chromosome-out logistic combination of a knockout score with distance and expression."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from kron_dna import loco as L


def pairs(n=600, seed=0, informative=True):
    rng = np.random.default_rng(seed)
    distance = 10 ** rng.uniform(2, 6, n)
    effect = rng.normal(0, 1, n)
    logit = -2.0 - 1.0 * (np.log10(distance) - 4) + (-2.0 * effect if informative else 0)
    y = rng.random(n) < 1 / (1 + np.exp(-logit))
    chrom = np.array([f"chr{1 + i % 6}" for i in range(n)])
    return pl.DataFrame(dict(chrom=chrom, distance=distance, effect_mask=effect, rna_ref=rng.gamma(1, 1, n),
                             regulated=y, held_out=np.isin(chrom, ["chr1", "chr2"])))


def test_an_informative_score_adds_to_distance():
    rows = {(r["set"], r["features"].split(" ")[0]): r for r in L.combine(pairs())}
    assert rows[("all", "effect_mask")]["auroc"] > rows[("all", "distance")]["auroc"] + 0.05
    assert rows[("held_out", "effect_mask")]["n"] == 200


def test_an_uninformative_score_adds_little():
    rows = {r["features"].split(" ")[0]: r for r in L.combine(pairs(informative=False), subset=None)}
    assert abs(rows["effect_mask"]["auroc"] - rows["distance"]["auroc"]) < 0.05


def test_predictions_never_see_their_own_chromosome():
    X = np.zeros((4, 1))
    X[:2, 0] = [1, -1]
    y = np.array([1, 0, 1, 1])
    groups = np.array(["a", "a", "b", "b"])
    p = L.loco_predict(X, y, groups)
    assert p[0] == pytest.approx(1.0) and p[1] == pytest.approx(1.0)   # chr b alone: one class, its base rate
    assert np.all((p[2:] > 0) & (p[2:] < 1))


def test_rows_with_missing_values_are_dropped():
    d = pairs(120).with_columns(pl.when(pl.int_range(pl.len()) < 10).then(None).otherwise(pl.col("effect_mask"))
                                .alias("effect_mask"))
    assert L.combine(d, subset=None)[0]["n"] == 110
