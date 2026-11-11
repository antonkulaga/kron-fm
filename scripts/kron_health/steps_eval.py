"""NHANES 2005-06 step head: features from 10-minute step bins, gates, per-person CV, the saved model."""
import json
from pathlib import Path
import joblib
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from kron_health.steps import FEATURES, nhanes_2005
from kron_ppg.wrist_age import bootstrap, _r, json_safe

cache = Path("data/interim/health/nhanes_pam_2005/steps.parquet"); cache.parent.mkdir(parents=True, exist_ok=True)
f = pl.read_parquet(cache) if cache.is_file() else nhanes_2005(Path("data/input/health/nhanes_pam_2005"))
f.write_parquet(cache)
print("people", f.height, "age", f["age"].min(), "-", f["age"].max(), "median", f["age"].median(), flush=True)
shares = f.select([c for c in FEATURES if c.startswith("share_h")]).to_numpy()
print(f"gate: share of window steps 12-17 h {shares[:, 4:10].sum(1).mean()*100:.1f} %; daily window steps median {np.expm1(f['log_daily_steps']).median():.0f}; empty bins median {f['empty_daytime'].median():.2f}")
y = f["age"].to_numpy(); rng = np.random.default_rng(0); fold = rng.permutation(len(y)) % 10
def model(): return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
                                                  l2_regularization=1.0, random_state=0)
def cv(x, t):
    p = np.zeros(len(t))
    for k in range(10):
        p[fold == k] = model().fit(x[fold != k], t[fold != k]).predict(x[fold == k])
    return p
res = {}
for name, cols in (("steps", FEATURES), ("steps+sex", FEATURES + ["sex"])):
    x = f.select(cols).to_numpy().astype(float); p = cv(x, y)
    base = np.array([np.median(y[fold != k]) for k in fold])
    err = {"model": np.abs(p - y), "constant": np.abs(base - y)}; i = bootstrap(err)
    null = [np.abs(cv(x, s) - s).mean() for s in (rng.permutation(y) for _ in range(3))]
    print(f"{name:10s} MAE {err['model'].mean():.2f} [{i['model'][0]:.2f},{i['model'][1]:.2f}] median {err['constant'].mean():.2f} r {_r(p, y):.2f} null {np.mean(null):.2f}", flush=True)
    res[name] = {"mae": err["model"].mean(), "interval95": i, "mae_constant": err["constant"].mean(), "r": _r(p, y), "null": null}
    if name == "steps":
        sig = p - y
        res["steps"]["signed_by_band"] = {int(b): float(sig[(y // 10 * 10) == b].mean()) for b in np.unique(y // 10 * 10)}
x = f.select(FEATURES).to_numpy().astype(float)
out = Path("data/output/runs/kron-health/steps-gbm"); out.mkdir(parents=True, exist_ok=True)
joblib.dump({"model": model().fit(x, y), "features": FEATURES, "people": f.height,
             "lo": np.nanpercentile(x, 0.5, axis=0), "hi": np.nanpercentile(x, 99.5, axis=0)}, out / "model.joblib")
(out / "metrics.json").write_text(json.dumps(json_safe(res), indent=1))
print("saved", out)
