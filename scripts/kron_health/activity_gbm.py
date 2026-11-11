"""NHANES activity head, nonlinear: gradient boosting, grouped 10-fold CV, bootstrap, permutation null."""
import json
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from kron_health.activity import ALL, UNIT_FREE
from kron_ppg.wrist_age import bootstrap, _r, json_safe

f = pl.read_parquet("data/interim/health/nhanes_pam/activity.parquet")
y = f["age"].to_numpy()
rng = np.random.default_rng(0)
fold = rng.permutation(len(y)) % 10
def cv(x, target):
    pred = np.zeros(len(target))
    for k in range(10):
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
                                          l2_regularization=1.0, random_state=0)
        m.fit(x[fold != k], target[fold != k]); pred[fold == k] = m.predict(x[fold == k])
    return pred
out = {}
for name, cols in (("unit_free", UNIT_FREE), ("all", ALL), ("all+sex", ALL + ["sex"])):
    x = f.select(cols).to_numpy().astype(float)
    pred = cv(x, y)
    base = np.array([np.median(y[fold != k]) for k in fold])
    err = {"model": np.abs(pred - y), "constant": np.abs(base - y)}
    i = bootstrap(err)
    null = [np.abs(cv(x, s) - s).mean() for s in (rng.permutation(y) for _ in range(3))]
    print(f"{name:9s} MAE {err['model'].mean():.2f} [{i['model'][0]:.2f},{i['model'][1]:.2f}] const {err['constant'].mean():.2f} "
          f"d [{i['model_minus_constant'][0]:.2f},{i['model_minus_constant'][1]:.2f}] r {_r(pred, y):.2f} null {np.mean(null):.2f}/{np.min(null):.2f}")
    out[name] = {"mae": err["model"].mean(), "interval95": i, "mae_constant": err["constant"].mean(), "r": _r(pred, y), "null": null}
Path("data/output/runs/kron-health/act2-nhanes-gbm.json").write_text(json.dumps(json_safe(out), indent=1))
