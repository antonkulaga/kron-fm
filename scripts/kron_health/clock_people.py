"""Per-person predictions of a kron-ppg clock run on a source (e.g. Aurora alone), saved private, with MAE and slope."""
import json, sys
from pathlib import Path
import numpy as np
import polars as pl
import torch
from kron_ppg.data import Pool, variant_dir
from kron_ppg.train import load_clock, predict
from kron_ppg.wrist_age import json_safe

run, source, seconds, out = Path(sys.argv[1]), sys.argv[2], float(sys.argv[3]), Path(sys.argv[4])
dev = torch.device("cuda")
m = load_clock(run / "checkpoint.pt", dev)
pool = Pool.load([variant_dir(source, m.encoder.config.sample_rate, seconds)])
cfg = run / "config.json"
cutoff = json.loads(cfg.read_text()).get("lowpass_hz") if cfg.is_file() else None
if cutoff:  # the band the clock was trained in
    from kron_ppg.train import lowpass_pool
    lowpass_pool(pool, cutoff)
idx = np.flatnonzero(np.isfinite(pool.age))
subj = predict(m, pool, idx, 256, dev)
f = pl.DataFrame({"subject": list(subj), "predicted": [v[0] for v in subj.values()], "age": [v[1] for v in subj.values()]})
out.parent.mkdir(parents=True, exist_ok=True); f.write_parquet(out)
p, a = f["predicted"].to_numpy(), f["age"].to_numpy()
print(json.dumps(json_safe({"run": str(run), "people": len(a), "mae": float(np.abs(p - a).mean()),
                            "slope_pred_on_age": float(np.polyfit(a, p, 1)[0]), "age_on_pred": float(np.polyfit(p, a, 1)[0]),
                            "r": float(np.corrcoef(p, a)[0, 1])})))
