"""Workout head on the Malaga treadmill tests: per-person grouped CV, median baseline, null; saved model."""
import json
from pathlib import Path
import joblib
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from kron_health.workout import FEATURES, NO_PEAK, treadmill
from kron_ppg.wrist_age import bootstrap, _r, json_safe

f = treadmill(Path("data/input/health/treadmill_malaga"))
f = f.filter(pl.col("age") >= 18)
p = f.group_by("subject").agg(pl.col("age").mean(), pl.col("sex").first(), pl.col("bmi").mean(),
                              *[pl.col(c).median() for c in FEATURES])
print("tests", f.height, "adults", p.height, "age", p["age"].min(), "-", p["age"].max(), "median", p["age"].median())
print("coverage:", {c: round(float(p[c].is_not_nan().mean()), 2) for c in FEATURES})
y = p["age"].to_numpy(); rng = np.random.default_rng(0); fold = rng.permutation(len(y)) % 10
def model(): return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20, random_state=0)
def cv(x, t):
    pr = np.zeros(len(t))
    for k in range(10): pr[fold == k] = model().fit(x[fold != k], t[fold != k]).predict(x[fold == k])
    return pr
res = {}
for name, cols in (("all", FEATURES), ("no_peak", NO_PEAK), ("peak_only", ["hr_peak"]), ("all+sex+bmi", FEATURES + ["sex", "bmi"]), ("sex+bmi_only", ["sex", "bmi"])):
    x = p.select(cols).to_numpy().astype(float); pr = cv(x, y)
    base = np.array([np.median(y[fold != k]) for k in fold])
    err = {"model": np.abs(pr - y), "constant": np.abs(base - y)}; i = bootstrap(err)
    null = [np.abs(cv(x, s) - s).mean() for s in (rng.permutation(y) for _ in range(3))]
    print(f"{name:13s} MAE {err['model'].mean():.2f} [{i['model'][0]:.2f},{i['model'][1]:.2f}] median {err['constant'].mean():.2f} r {_r(pr, y):.2f} slope {np.polyfit(y, pr, 1)[0]:.2f} null {np.mean(null):.2f}")
    res[name] = {"mae": err["model"].mean(), "interval95": i, "r": _r(pr, y), "null": null}
out = Path("data/output/runs/kron-health/workout-gbm"); out.mkdir(parents=True, exist_ok=True)
for name, cols in (("all", FEATURES), ("no_peak", NO_PEAK)):
    x = p.select(cols).to_numpy().astype(float)
    joblib.dump({"model": model().fit(x, y), "features": cols, "people": p.height,
                 "lo": np.nanpercentile(x, 0.5, axis=0), "hi": np.nanpercentile(x, 99.5, axis=0)}, out / f"model_{name}.joblib")
(out / "metrics.json").write_text(json.dumps(json_safe(res), indent=1))
