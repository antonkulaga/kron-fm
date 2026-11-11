"""Autonomic Aging: RR per lead by XQRS, HRV features per lead, one row per record and lead."""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import polars as pl
from kron_health.hrv import FEATURES, age_mid, autonomic_aging_record, features

folder = Path(sys.argv[1]); out = Path(sys.argv[2])
info = pl.read_csv(folder / "subject-info.csv", schema_overrides={"ID": pl.String}, null_values=["NaN", ""], infer_schema_length=None)
def one(rid):
    rows = []
    try:
        leads = autonomic_aging_record(folder / rid)
    except Exception as e:
        return [{"record": rid, "lead": "error", "error": str(e)[:200]}]
    for lead, d in leads.items():
        f = features(d["rr_ms"], d["t_s"])
        rows.append({"record": rid, "lead": lead, "beats": len(d["rr_ms"]), **(f or {})})
    return rows
ids = info["ID"].to_list()
with ProcessPoolExecutor(int(sys.argv[3])) as ex:
    rows = [r for rs in ex.map(one, ids, chunksize=4) for r in rs]
frame = pl.DataFrame(rows, infer_schema_length=None)
frame = frame.join(info.rename({"ID": "record"}), on="record", how="left")
frame = frame.with_columns(pl.col("Age_group").cast(pl.Int64, strict=False).map_elements(age_mid, return_dtype=pl.Float64, skip_nulls=True).alias("age"))
out.parent.mkdir(parents=True, exist_ok=True)
frame.write_parquet(out)
print(frame.group_by("lead").len())
