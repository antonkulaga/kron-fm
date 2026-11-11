"""The gene-split ensemble: joins scores by id, keeps genes apart, and finds a score that carries signal."""
import numpy as np
import polars as pl

from kron_dna import ensemble as E


def _files(tmp_path):
    rng = np.random.default_rng(0)
    n = 600
    label = rng.integers(0, 2, n)
    base = pl.DataFrame({"id": [str(i) for i in range(n)], "gene": [f"g{i % 60}" for i in range(n)],
                         "consequence": ["missense"] * (n // 2) + ["synonymous"] * (n // 2), "label": label,
                         "signal": -label + rng.normal(0, 0.5, n), "noise": rng.normal(size=n)})
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    base.select("id", "gene", "consequence", "label", "signal").write_parquet(a)
    base.select("id", "gene", "consequence", "label", "noise").sample(fraction=1.0, seed=1).write_parquet(b)
    return [a, b]


def test_join_and_gain(tmp_path):
    t = E.join_scores(_files(tmp_path), {"signal": "s", "noise": "z"})
    assert t.columns == ["id", "gene", "consequence", "label", "s", "z", "is_missense"] and t["z"].null_count() == 0
    rows, preds = E.evaluate(t, {"all": ["s", "z", "is_missense"], "all but s": ["z", "is_missense"]},
                             n_boot=50, max_iter=30)
    best = {(r["consequence"], r["inputs"]): r["auroc"] for r in rows}
    assert best[("missense", "all")] > 0.8 and best[("missense", "all but s")] < 0.65
    m = t["consequence"].to_numpy() == "missense"
    g = E.paired_gain(preds["all but s"][m], preds["all"][m], t["label"].to_numpy()[m], n=50)
    assert g["gain"] > 0.15 and g["ci"][0] > 0
