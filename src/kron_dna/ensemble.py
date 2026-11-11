"""Supervised ensemble over per-variant scores, cross-validated by gene.

Joins the per-variant parquet files that ``kron-dna clinvar`` writes (one column per score) on the variant id,
fits gradient-boosted trees in folds that keep each gene on one side, and reports out-of-fold AUROC by
consequence, each with a 95% bootstrap interval, plus the paired gain of every score left in versus left out.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from kron_dna.variants import auroc, bootstrap_ci

KEYS = ("id", "gene", "consequence", "label")


def join_scores(paths: list[Path], columns: dict[str, str]) -> pl.DataFrame:
    """One row per variant: the keys from the first file, then each wanted column as its short name.

    ``columns`` maps a column name as the parquet files hold it to the name used here; each is taken from the
    first file that has it. Variants missing from a later file keep a null there."""
    frames = [pl.read_parquet(p) for p in paths]
    out = frames[0].select(KEYS)
    for long, short in columns.items():
        src = next((f for f in frames if long in f.columns), None)
        if src is None:
            raise ValueError(f"no input file holds the column {long!r}")
        out = out.join(src.select("id", pl.col(long).alias(short)), on="id", how="left")
    return out.with_columns((pl.col("consequence") == "missense").cast(pl.Int8).alias("is_missense"))


def out_of_fold(table: pl.DataFrame, features: list[str], folds: int = 5, seed: int = 0,
                max_iter: int = 300) -> np.ndarray:
    """Out-of-fold score (lower = pathogenic, the AUROC convention here) from gradient-boosted trees, with folds
    split by gene so no gene is in both a fit and its test. Missing values stay NaN: the trees route them."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold

    x = table.select(features).to_numpy().astype(float)
    y = table["label"].to_numpy()
    genes = table["gene"].to_numpy()
    pred = np.zeros(len(y))
    for fit, test in GroupKFold(folds).split(x, y, genes):
        model = HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=0.05, random_state=seed)
        pred[test] = model.fit(x[fit], y[fit]).predict_proba(x[test])[:, 1]
    return -pred


def paired_gain(a: np.ndarray, b: np.ndarray, y: np.ndarray, n: int = 1000, seed: int = 0) -> dict:
    """AUROC of ``b`` minus that of ``a`` on the same rows, with a 95% paired bootstrap interval."""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        da, db = auroc(a[i], y[i]), auroc(b[i], y[i])
        if da is not None and db is not None:
            diffs.append(db - da)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"gain": float(auroc(b, y) - auroc(a, y)), "ci": [float(lo), float(hi)]}


def evaluate(table: pl.DataFrame, sets: dict[str, list[str]], n_boot: int = 1000,
             max_iter: int = 300) -> tuple[list[dict], dict]:
    """AUROC with interval for each feature set, by consequence; and each set's out-of-fold predictions."""
    y = table["label"].to_numpy()
    cons = table["consequence"].to_numpy()
    preds = {name: out_of_fold(table, feats, max_iter=max_iter) for name, feats in sets.items()}
    rows = []
    for consequence in ("missense", "synonymous"):
        m = cons == consequence
        for name, p in preds.items():
            rows.append({"consequence": consequence, "inputs": name, "n": int(m.sum()), "auroc": auroc(p[m], y[m]),
                         "ci": bootstrap_ci(auroc, p[m], y[m], n=n_boot)})
    return rows, preds
