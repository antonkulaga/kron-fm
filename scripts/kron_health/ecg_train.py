"""PTB-XL lead I -> age: train on folds 1-8, choose the step on fold 9, report fold 10 per patient with intervals."""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from kron_health.ecg import ECGAge, load_ptbxl
from kron_ppg.wrist_age import bootstrap, _r, json_safe

folder, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
cache = Path("data/interim/health/ptb_xl/lead1.npz")
if cache.is_file():
    d = dict(np.load(cache, allow_pickle=True))
else:
    d = load_ptbxl(folder); cache.parent.mkdir(parents=True, exist_ok=True); np.savez(cache, **d)
x, age, fold = d["x"], d["age"], d["fold"]
tr, va, te = fold <= 8, fold == 9, fold == 10
print("records", len(age), "train/val/test", tr.sum(), va.sum(), te.sum(), "age", age.min(), age.max(), flush=True)
dev = torch.device("cuda"); torch.manual_seed(0)
m = ECGAge(mean=float(age[tr].mean()), scale=float(age[tr].std())).to(dev)
opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
X = torch.from_numpy(x).to(dev); Y = torch.from_numpy(age).to(dev)
idx_tr = np.flatnonzero(tr); steps, bs = 12000, 128
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps)
def predict(idx):
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(idx), 1024):
            b = X[idx[i:i+1024]]
            out.append(((m(b) + m(-b)) / 2).float().cpu().numpy())   # average both polarities
    m.train(); return np.concatenate(out)
best, best_state, rng, t0 = 1e9, None, np.random.default_rng(0), time.time()
for step in range(1, steps + 1):
    b = rng.choice(idx_tr, bs)
    xb = X[b] * torch.where(torch.rand(bs, 1, device=dev) < 0.5, -1.0, 1.0)  # polarity augmentation: a watch may flip the lead
    shift = int(rng.integers(0, 250)); xb = torch.roll(xb, shift, dims=1)
    loss = F.smooth_l1_loss(m(xb), Y[b], beta=2.0)
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if step % 1000 == 0:
        v = np.abs(predict(np.flatnonzero(va)) - age[va]).mean()
        print(f"step {step} loss {loss.item():.2f} val MAE {v:.2f} ({time.time()-t0:.0f}s)", flush=True)
        if v < best: best, best_state = v, {k: t.clone() for k, t in m.state_dict().items()}
m.load_state_dict(best_state)
p = predict(np.flatnonzero(te)); y = age[te]
# per patient: mean over a patient's records
pat = d["patient"][te]; up = np.unique(pat)
pp = np.array([p[pat == u].mean() for u in up]); yy = np.array([y[pat == u].mean() for u in up])
base = np.median(age[tr])
err = {"model": np.abs(pp - yy), "constant": np.abs(base - yy)}
i = bootstrap(err)
res = {"val_best_mae": best, "test_patients": len(up), "mae": err["model"].mean(), "mae_constant": err["constant"].mean(),
       "r": _r(pp, yy), "interval95": i, "steps": steps}
print(json.dumps(json_safe(res), indent=1))
torch.save({"state": m.state_dict(), "mean": float(m.mean), "scale": float(m.scale), "fs": 250, "seconds": 10}, out / "checkpoint.pt")
(out / "metrics.json").write_text(json.dumps(json_safe(res), indent=1))
