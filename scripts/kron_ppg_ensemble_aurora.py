"""Deployable ensemble on Aurora (never fitted on it): frozen c1 + ridge, and fine-tuned heads, averaged per person."""
import numpy as np, polars as pl
from kron_ppg.wrist_age import People, fit_ridge, bootstrap
R = "data/output/runs/kron-ppg/"
o, a = People.load(R + "wa9-open-c1/people.npz"), People.load(R + "wa9-aurora15-c1/people.npz")
c1 = dict(zip(a.subject, fit_ridge(o.x, o.age).predict(a.x))); age = dict(zip(a.subject, a.age))
ft = {r: dict(pl.read_parquet(R + r + "/aurora15-people.parquet").select("subject", "predicted").iter_rows())
      for r in ("wf5-open-finetune-6k-lr3e-4", "wf6-c1-finetune-6k-lr3e-4", "wf2-open-finetune-lr1e-4", "wf7-r4-finetune-6k-lr3e-4-s1", "wf7-r4-finetune-6k-lr3e-4-s2")}
s = sorted(set(c1).intersection(*ft.values())); y = np.array([age[k] for k in s])
P = {"c1 ridge": np.array([c1[k] for k in s]), **{r: np.array([ft[r][k] for k in s]) for r in ft}}
P["c1 ridge + wf5"] = (P["c1 ridge"] + P["wf5-open-finetune-6k-lr3e-4"]) / 2
P["c1 ridge + wf5 + wf6"] = (P["c1 ridge"] + P["wf5-open-finetune-6k-lr3e-4"] + P["wf6-c1-finetune-6k-lr3e-4"]) / 3
err = {"model": np.abs(P["c1 ridge + wf5"] - y), **{k: np.abs(v - y) for k, v in P.items()}}
ci = bootstrap(err)
print("people", len(s))
for k, v in P.items():
    print(f"{k:28s} MAE {np.abs(v-y).mean():.2f} r {np.corrcoef(v,y)[0,1]:.2f}", "" if k == "c1 ridge + wf5" else f"pair(c1+wf5 minus this) {[round(x,2) for x in ci['model_minus_'+k]]}")
# seed replicates of the fine-tuned member (seeds 1, 2 were not used to choose it)
for r in ("wf7-r4-finetune-6k-lr3e-4-s1", "wf7-r4-finetune-6k-lr3e-4-s2"):
    e = np.abs((P["c1 ridge"] + P[r]) / 2 - y)
    c = bootstrap({"model": e, "c1": np.abs(P["c1 ridge"] - y)})
    print(f"c1 ridge + {r:30s} MAE {e.mean():.2f} minus c1 {[round(x, 2) for x in c['model_minus_c1']]}")
# the fine-tuned member as the mean of its three seeds (no seed chosen)
ft3 = (P["wf5-open-finetune-6k-lr3e-4"] + P["wf7-r4-finetune-6k-lr3e-4-s1"] + P["wf7-r4-finetune-6k-lr3e-4-s2"]) / 3
e = np.abs((P["c1 ridge"] + ft3) / 2 - y); c = bootstrap({"model": e, "c1": np.abs(P["c1 ridge"] - y)})
print(f"c1 ridge + mean of 3 fine-tune seeds  MAE {e.mean():.2f} {[round(x, 2) for x in c['model']]} r {np.corrcoef((P['c1 ridge'] + ft3) / 2, y)[0, 1]:.2f} minus c1 {[round(x, 2) for x in c['model_minus_c1']]}")
