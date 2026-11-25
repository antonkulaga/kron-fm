"""Does a model's score add to simple covariates? A logistic regression on (score, covariates), fitted and scored
leave-one-chromosome-out, against the same regression on the covariates alone.

Used for the CRISPRi enhancer knockouts of ``scripts/kron_dna/knockout_bench.py``: the model's knockout effect at the
gene (sign flipped: a larger drop is more likely regulated) with the enhancer-TSS distance and the gene's predicted
expression, against distance alone, the baseline any enhancer-gene benchmark must beat. Each pair's probability comes
from a model fitted without its chromosome, so nothing is scored by a fit that saw it.

    python -m kron_dna.loco PAIRS.parquet [PAIRS ...] --out loco.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import typer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score


def loco_predict(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Out-of-group probabilities: for each group, a logistic regression fitted on the other groups. A group whose
    training rows hold one class only gets that class's base rate."""
    p = np.zeros(len(y))
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        if len(np.unique(y[tr])) < 2:
            p[te] = y[tr].mean()
            continue
        p[te] = LogisticRegression(max_iter=1000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def knockout_features(d: pl.DataFrame, score: str = "effect_mask") -> tuple[np.ndarray, np.ndarray]:
    """(score + distance + expression, distance alone) feature matrices from a knockout_bench pairs table."""
    dist = -np.log10(d["distance"].to_numpy().clip(1))
    full = np.c_[-d[score].to_numpy(), dist, np.log1p(d["rna_ref"].to_numpy())]
    return full, dist[:, None]


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return dict(auroc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)))


def combine(d: pl.DataFrame, score: str = "effect_mask", subset: str | None = "held_out") -> list[dict]:
    """Rows of AUROC and AUPRC for the combination and for distance alone, over all pairs and over ``subset`` (a
    boolean column, e.g. the held-out chromosomes)."""
    d = d.drop_nulls([score, "distance", "rna_ref"])
    y = d["regulated"].to_numpy().astype(int)
    chrom = d["chrom"].to_numpy()
    full, dist = knockout_features(d, score)
    preds = {f"{score} + distance + expression": loco_predict(full, y, chrom), "distance": loco_predict(dist, y, chrom)}
    sets = [("all", np.ones(len(y), dtype=bool))]
    if subset and subset in d.columns:
        sets.append((subset, d[subset].to_numpy().astype(bool)))
    rows = []
    for name, m in sets:
        if len(np.unique(y[m])) < 2:
            continue
        for features, p in preds.items():
            rows.append(dict(set=name, features=features, n=int(m.sum()), positives=int(y[m].sum()), **metrics(y[m], p[m])))
    return rows


app = typer.Typer(add_completion=False)


@app.command()
def main(pairs: list[Path] = typer.Argument(..., help="knockout_bench pairs.parquet files"),
         score: str = typer.Option("effect_mask"), out: Path = typer.Option(Path("loco.csv"))) -> None:
    """One block of rows per pairs file, labelled by the file's parent folder."""
    rows = []
    for f in pairs:
        rows += [dict(label=f.parent.name, **r) for r in combine(pl.read_parquet(f), score)]
    t = pl.DataFrame(rows)
    t.write_csv(out)
    with pl.Config(tbl_rows=-1, tbl_width_chars=200):
        print(t)


if __name__ == "__main__":
    app()
