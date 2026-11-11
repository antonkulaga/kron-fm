"""Autonomic Aging HRV head: lead-to-lead gate, then per-person CV (ridge and boosting) against the median and a null."""
import json
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from kron_health.hrv import FEATURES
from kron_ppg.wrist_age import People, evaluate, bootstrap, _r, json_safe

f = pl.read_parquet("data/interim/health/autonomic_aging/hrv.parquet")
two = f.filter(pl.col("lead").is_in(["ECG1", "ECG2"])).pivot(on="lead", index="record", values=["hr", "rmssd", "kept_fraction"])
two = two.drop_nulls()
dhr = (two["hr_ECG1"] - two["hr_ECG2"]).abs().to_numpy()
r_rmssd = np.corrcoef(np.log(two["rmssd_ECG1"]), np.log(two["rmssd_ECG2"]))[0, 1]
print(f"gate, {two.height} two-lead records: |HR lead1 - lead2| median {np.median(dhr):.2f} bpm, p95 {np.percentile(dhr,95):.2f}; "
      f"log RMSSD r {r_rmssd:.3f}; kept fraction median {two['kept_fraction_ECG1'].median():.3f}")
# one row per record: ECG1 where present, else ECG
one = f.filter(pl.col("lead").is_in(["ECG1", "ECG"])).sort("lead").unique("record", keep="first")
one = one.filter(pl.col("age").is_not_null() & pl.col("rmssd").is_not_null())
print("people with age and HRV", one.height, "| age groups", one["Age_group"].n_unique(), "| device", one.group_by("Device").len().to_dicts())
x = one.select(FEATURES + ["Sex"]).to_numpy().astype(float)
x = np.where(np.isfinite(x), x, np.nanmedian(x, axis=0))
y = one["age"].to_numpy()
out = {}
p = People(one["record"].to_numpy(), np.full(one.height, "aa"), y, x[:, :-1], np.ones(one.height, int))
res, _ = evaluate(p, folds=10, repeats=3, null=10); q = res["pooled"]; i = q["interval95"]
print(f"ridge  MAE {q['mae']:.2f} [{i['model'][0]:.2f},{i['model'][1]:.2f}] const {q['mae_constant']:.2f} r {q['r']:.2f} null {res['null']['mean']:.2f}/{res['null']['min']:.2f}")
out["ridge"] = res
rng = np.random.default_rng(0); fold = rng.permutation(len(y)) % 10
def cv(xx, t):
    pr = np.zeros(len(t))
    for k in range(10):
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20, random_state=0)
        m.fit(xx[fold != k], t[fold != k]); pr[fold == k] = m.predict(xx[fold == k])
    return pr
for name, xx in (("gbm", x[:, :-1]), ("gbm+sex", x)):
    pr = cv(xx, y); base = np.array([np.median(y[fold != k]) for k in fold])
    err = {"model": np.abs(pr - y), "constant": np.abs(base - y)}; ii = bootstrap(err)
    null = [np.abs(cv(xx, s) - s).mean() for s in (rng.permutation(y) for _ in range(5))]
    print(f"{name:7s} MAE {err['model'].mean():.2f} [{ii['model'][0]:.2f},{ii['model'][1]:.2f}] const {err['constant'].mean():.2f} r {_r(pr, y):.2f} null {np.mean(null):.2f}/{np.min(null):.2f}")
    out[name] = {"mae": err["model"].mean(), "interval95": ii, "r": _r(pr, y), "null": null}
Path("data/output/runs/kron-health/hrv1-autonomic-aging.json").write_text(json.dumps(json_safe(out), indent=1))
