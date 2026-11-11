"""NHANES activity head: features, the clock-alignment gate, per-person CV against the median and a null."""
import json, sys
from pathlib import Path
import numpy as np
import polars as pl
from kron_health.activity import ALL, UNIT_FREE, nhanes
from kron_ppg.wrist_age import People, evaluate, json_safe

root = Path("data/interim/health/nhanes_pam"); root.mkdir(parents=True, exist_ok=True)
cache = root / "activity.parquet"
if cache.is_file():
    f = pl.read_parquet(cache)
else:
    f = nhanes(Path("data/input/health/nhanes_pam"))
    f.write_parquet(cache)
print("people", f.height, "age", f["age"].min(), "-", f["age"].max(), "median", f["age"].median())
# gate: L5 (least active 5 h) midpoint should fall at night for most people
l5 = (np.degrees(np.arctan2(f["l5_sin"], f["l5_cos"])) % 360) / 15
night = ((l5 >= 0) & (l5 < 7)) | (l5 >= 22)
print(f"gate: L5 midpoint in 22:00-07:00 for {night.mean()*100:.1f} % of people; median clock {np.median(l5):.1f} h")
out = {}
for name, cols in (("unit_free", UNIT_FREE), ("all", ALL)):
    x = f.select(cols).to_numpy().astype(float)
    x = np.where(np.isfinite(x), x, np.nanmean(x, axis=0))
    p = People(f["subject"].to_numpy(), np.full(f.height, "nhanes"), f["age"].to_numpy(), x, np.ones(f.height, int))
    res, _ = evaluate(p, folds=10, repeats=3, null=10)
    q = res["pooled"]; i = q["interval95"]
    print(f"{name:10s} n {q['people']} MAE {q['mae']:.2f} [{i['model'][0]:.2f},{i['model'][1]:.2f}] const {q['mae_constant']:.2f} "
          f"d [{i['model_minus_constant'][0]:.2f},{i['model_minus_constant'][1]:.2f}] r {q['r']:.2f} null {res['null']['mean']:.2f}/{res['null']['min']:.2f}")
    out[name] = res
Path("data/output/runs/kron-health").mkdir(parents=True, exist_ok=True)
Path("data/output/runs/kron-health/act1-nhanes-ridge.json").write_text(json.dumps(json_safe(out), indent=1))
