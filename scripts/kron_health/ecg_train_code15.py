"""The ECG head at scale: CODE-15% lead I, patient-wise split (hash), tested on CODE-15% and on PTB-XL fold 10."""
import json, sys, time, zlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from kron_health.ecg import ECGAge, band, load_code15
from kron_ppg.wrist_age import bootstrap, _r, json_safe

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 40000
filtered = len(sys.argv) > 3 and sys.argv[3] == "filtered"  # 0.5-40 Hz on every strip, as kron_health.ecg.strips(filtered=True)
d = load_code15(Path("data/input/health/code15"), Path("data/interim/health/code15/lead1.npz"))
x, age, pat = d["x"], d["age"], d["patient"]
def filt(a):
    return np.concatenate([band(a[i:i + 8192].astype(np.float64)).astype(np.float16) for i in range(0, len(a), 8192)])
if filtered:
    x = filt(x)
side = np.array([zlib.crc32(str(p).encode()) % 20 for p in pat])  # 0-15 train, 16-17 val, 18-19 test
tr, va, te = side < 16, (side >= 16) & (side < 18), side >= 18
print("strips", len(age), "train/val/test", tr.sum(), va.sum(), te.sum(), "age", age.min(), age.max(), flush=True)
ptb = dict(np.load("data/interim/health/ptb_xl/lead1.npz", allow_pickle=True))
if filtered:
    ptb["x"] = filt(ptb["x"])
pte = ptb["fold"] == 10
dev = torch.device("cuda"); torch.manual_seed(0)
m = ECGAge(width=64, mean=float(age[tr].mean()), scale=float(age[tr].std())).to(dev)
opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps)
X = torch.from_numpy(x).to(dev); Y = torch.from_numpy(age).to(dev)
idx_tr = np.flatnonzero(tr); rng = np.random.default_rng(0); bs = 256
def predict(XX, idx):
    m.eval(); o = []
    with torch.no_grad():
        for i in range(0, len(idx), 2048):
            b = XX[idx[i:i + 2048]].float(); o.append(((m(b) + m(-b)) / 2).float().cpu().numpy())
    m.train(); return np.concatenate(o)
val_idx = np.flatnonzero(va); val_idx = val_idx[: 20000]
best, state, t0 = 1e9, None, time.time()
for step in range(1, steps + 1):
    b = rng.choice(idx_tr, bs)
    xb = X[b].float() * torch.where(torch.rand(bs, 1, device=dev) < 0.5, -1.0, 1.0)
    xb = torch.roll(xb, int(rng.integers(0, 250)), dims=1)
    loss = F.smooth_l1_loss(m(xb), Y[b], beta=2.0)
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if step % 2000 == 0:
        v = np.abs(predict(X, val_idx) - age[val_idx]).mean()
        print(f"step {step} loss {loss.item():.2f} val MAE {v:.2f} ({time.time() - t0:.0f}s)", flush=True)
        if v < best: best, state = v, {k: t.clone() for k, t in m.state_dict().items()}
m.load_state_dict(state)
res = {"val_best_mae": best, "steps": steps, "filtered": filtered}
def per_patient(pred, y, p, base):
    u = np.unique(p); pp = np.array([pred[p == k].mean() for k in u]); yy = np.array([y[p == k].mean() for k in u])
    err = {"model": np.abs(pp - yy), "constant": np.abs(base - yy)}
    return {"patients": len(u), "mae": err["model"].mean(), "mae_constant": err["constant"].mean(), "r": _r(pp, yy),
            "interval95": bootstrap(err)}
base = float(np.median(age[tr]))
res["code15_test"] = per_patient(predict(X, np.flatnonzero(te)), age[te], pat[te], base)
PX = torch.from_numpy(ptb["x"]).to(dev)
pi = np.flatnonzero(pte)
res["ptbxl_fold10"] = per_patient(predict(PX, pi), ptb["age"][pte], ptb["patient"][pte], float(np.median(ptb["age"][ptb["fold"] <= 8])))
print(json.dumps(json_safe(res), indent=1))
torch.save({"state": m.state_dict(), "mean": float(m.mean), "scale": float(m.scale), "fs": 250, "seconds": 10,
            "filtered": filtered}, out / "checkpoint.pt")
(out / "metrics.json").write_text(json.dumps(json_safe(res), indent=1))
