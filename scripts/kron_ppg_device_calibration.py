"""How many people of known age does a new watch need? Leave one open dataset out, fit the head on the rest, then
correct the held-out dataset by the mean error of k of its people (an offset only) and score the others; Aurora too.
Prints MAE before and after, per k (median over 200 draws of the k people)."""
import numpy as np
from kron_ppg.wrist_age import People, fit_ridge

R = "data/output/runs/kron-ppg/"
rng = np.random.default_rng(0)
for tag, oo, aa in (("c1 full", "wa9-open-c1", "wa9-aurora15-c1"), ("r4 5 Hz", "wa4-open-r4-lp5", "wa4-aurora15-r4-lp5")):
    o, a = People.load(R + oo + "/people.npz"), People.load(R + aa + "/people.npz")
    sets = [(s, fit_ridge(o.x[o.source != s], o.age[o.source != s]).predict(o.x[o.source == s]), o.age[o.source == s])
            for s in np.unique(o.source)]
    sets.append(("aurora (all open)", fit_ridge(o.x, o.age).predict(a.x), a.age))
    print(f"[{tag}]  MAE: none | k=3 | k=5 | k=10 | k=20   (offset from k people of the held-out set)")
    for s, p, y in sets:
        row = [np.abs(p - y).mean()]
        for k in (3, 5, 10, 20):
            if len(y) <= k + 5:
                row.append(np.nan); continue
            m = []
            for _ in range(200):
                i = rng.permutation(len(y)); cal, test = i[:k], i[k:]
                m.append(np.abs(p[test] - (p[cal] - y[cal]).mean() - y[test]).mean())
            row.append(np.median(m))
        print(f"  {s:18s} n {len(y):5d}  " + " | ".join("  -  " if np.isnan(v) else f"{v:5.1f}" for v in row))
