"""Gate for cross-modal distillation: does the ECG age model read Vilnius's Holter ECG?

Per patient, 200 random 10 s strips from the hours their wrist PPG was also recording, each lead of the Holter
read by the filtered ECG head (both polarities), the person's mean against their true age. A teacher worth
distilling onto PulseWatch's unaged people should track age here.
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path
import h5py
import numpy as np
import torch
from kron_health.ecg import ECGAge, strips
from kron_ppg.watch_readers import xlsx_rows, VILNIUS_COLUMNS

folder = Path("data/input/ppg/vilnius_af")
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
m = ECGAge(width=ck.get("width", 64), mean=ck["mean"], scale=ck["scale"]); m.load_state_dict(ck["state"]); m.eval()
filt = bool(ck.get("filtered", False))
s = lambda a: "".join(chr(int(c)) for c in np.asarray(a).ravel())
ages = {}
for row in xlsx_rows(folder / "subject_info.xlsx"):
    if row and row[0].strip().isdigit():
        rec = dict(zip(VILNIUS_COLUMNS, [(row[i] if i < len(row) else "").strip() for i in range(len(VILNIUS_COLUMNS))]))
        try: ages[f"{int(rec['subject']):03d}"] = float(rec["age"])
        except ValueError: pass
rng = np.random.default_rng(0)
rows = []
for ecg_path in sorted(folder.glob("[0-9][0-9][0-9]_ECG.mat")):
    sid = ecg_path.name[:3]
    ppg_path = folder / f"{sid}_PPG.mat"
    if sid not in ages or not ppg_path.is_file():
        continue
    with h5py.File(ecg_path, "r") as g, h5py.File(ppg_path, "r") as f:
        t0 = datetime.strptime(s(g["recording_starttime"][()]), "%H:%M:%S") + timedelta(days=int(s(g["recording_startday"][()])))
        ecg = g["ECG"]; n = ecg.shape[0]; fs = 500
        spans = []
        for k in range(f["PPG_GREEN"].shape[0]):
            d = int(s(f[f["recording_startday"][k, 0]][()])); t = s(f[f["recording_starttime"][k, 0]][()])
            st = datetime.strptime(t.split(".")[0], "%H:%M:%S") + timedelta(days=d)
            a = int((st - t0).total_seconds() * fs); L = f[f["PPG_GREEN"][k, 0]].shape[1] / 100 * fs
            spans.append((max(a, 0), min(int(a + L), n)))
        spans = [(a, b) for a, b in spans if b - a > fs * 60]
        if not spans:
            continue
        preds = []
        for _ in range(200):
            a, b = spans[rng.integers(len(spans))]
            i = int(rng.integers(a, b - fs * 10))
            x = ecg[i:i + fs * 10, 0].astype(float)
            if not np.isfinite(x).all() or np.ptp(x) == 0:
                continue
            st_ = torch.from_numpy(strips(x, fs, filtered=filt))
            if len(st_):
                with torch.no_grad(): preds.append(float(((m(st_) + m(-st_)) / 2).mean()))
    if preds:
        rows.append((sid, ages[sid], float(np.median(preds)), len(preds)))
        print(sid, ages[sid], round(rows[-1][2], 1), len(preds), flush=True)
a = np.array([r[1] for r in rows]); p = np.array([r[2] for r in rows])
print(f"patients {len(rows)}: r {np.corrcoef(a, p)[0,1]:.2f}, MAE {np.abs(p - a).mean():.1f} y, median-age MAE {np.abs(np.median(a) - a).mean():.1f} y, mean signed {np.mean(p - a):+.1f}")
