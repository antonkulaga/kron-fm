"""Per-patient test predictions of an ECG head (CODE-15% test patients, PTB-XL fold 10) and its signed error by age.

Writes predictions.parquet and bias.json beside the run's metrics.json: the mean signed error (predicted - age)
and its SD per 10-year band, which says whether a person of a given age reads old or young on average.
"""
import json, sys, zlib
from pathlib import Path
import numpy as np
import polars as pl
import torch
from kron_health.ecg import ECGAge, band

run = Path(sys.argv[1]); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ck = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
m = ECGAge(mean=ck["mean"], scale=ck["scale"]); m.load_state_dict(ck["state"]); m.eval().to(dev)
filt = bool(ck.get("filtered", False))
def predict(x):
    out = []
    with torch.no_grad():
        for i in range(0, len(x), 2048):
            b = x[i:i + 2048].astype(np.float64)
            b = torch.from_numpy((band(b) if filt else b).astype(np.float32)).to(dev)
            out.append(((m(b) + m(-b)) / 2).float().cpu().numpy())
    return np.concatenate(out)
c = np.load("data/interim/health/code15/lead1.npz")
side = np.array([zlib.crc32(str(p).encode()) % 20 for p in c["patient"]]); te = side >= 18
p = dict(np.load("data/interim/health/ptb_xl/lead1.npz", allow_pickle=True)); pte = p["fold"] == 10
rows = []
for name, x, age, pat in (("code15_test", c["x"][te], c["age"][te], c["patient"][te]),
                          ("ptbxl_fold10", p["x"][pte], p["age"][pte], p["patient"][pte])):
    pr = predict(x)
    f = pl.DataFrame({"set": name, "patient": pat.astype(np.int64), "age": age.astype(float), "predicted": pr.astype(float)})
    rows.append(f.group_by("set", "patient").agg(pl.col("age").mean(), pl.col("predicted").mean(), pl.len().alias("strips")))
pred = pl.concat(rows).with_columns((pl.col("predicted") - pl.col("age")).alias("signed_error"))
pred.write_parquet(run / "predictions.parquet")
bands = (pred.with_columns(((pl.col("age") // 10) * 10).cast(pl.Int64).alias("band"))
         .group_by("set", "band").agg(pl.len().alias("patients"), pl.col("signed_error").mean().alias("mean"),
                                      pl.col("signed_error").std().alias("sd"), pl.col("signed_error").abs().mean().alias("mae"))
         .sort("set", "band"))
near40 = pred.filter(pl.col("age").is_between(35, 45)).group_by("set").agg(
    pl.len().alias("patients"), pl.col("signed_error").mean().alias("mean"), pl.col("signed_error").std().alias("sd"))
out = {"bands": bands.to_dicts(), "ages_35_45": near40.to_dicts()}
(run / "bias.json").write_text(json.dumps(out, indent=1))
print(bands); print(near40)
