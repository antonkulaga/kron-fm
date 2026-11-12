"""Device confound check: ridge on open wrist embeddings with each source's mean embedding removed (source-centred),
scored on Aurora centred by its own mean (no labels used); and leave-one-source-out mean error per source.
usage: kron_ppg_device_offsets.py [TAG OPEN_RUN AURORA_RUN ...]  (default: c1 full and r4 5 Hz)"""
import sys
import numpy as np
from kron_ppg.wrist_age import People, fit_ridge, bootstrap
R = "data/output/runs/kron-ppg/"
a = sys.argv[1:]
RUNS = [tuple(a[i:i + 3]) for i in range(0, len(a), 3)] or [("c1 full", "wa9-open-c1", "wa9-aurora15-c1"),
                                                             ("r4 5Hz", "wa4-open-r4-lp5", "wa4-aurora15-r4-lp5")]
for tag, oo, aa in RUNS:
    o, a = People.load(R + oo + "/people.npz"), People.load(R + aa + "/people.npz")
    xc = o.x.copy()
    for s in np.unique(o.source): m = o.source == s; xc[m] -= xc[m].mean(0)
    p0 = fit_ridge(o.x, o.age).predict(a.x)
    p1 = fit_ridge(xc, o.age).predict(a.x - a.x.mean(0))  # centred head adds back the training mean age as intercept
    e = {"model": np.abs(p1 - a.age), "plain": np.abs(p0 - a.age)}; ci = bootstrap(e)
    print(f"[{tag}] Aurora plain {e['plain'].mean():.2f} r {np.corrcoef(p0,a.age)[0,1]:.2f} | source-centred {e['model'].mean():.2f} "
          f"r {np.corrcoef(p1,a.age)[0,1]:.2f} slope {np.polyfit(a.age,p1,1)[0]:.2f}; centred minus plain {[round(x,2) for x in ci['model_minus_plain']]}")
    # does the plain head read a held-out source by its device? leave-one-source-out mean signed error
    for s in np.unique(o.source):
        m = o.source == s; pr = fit_ridge(o.x[~m], o.age[~m]).predict(o.x[m])
        print(f"   held out {s:15s} n {m.sum():3d} true median {np.median(o.age[m]):4.0f} read median {np.median(pr):5.1f}")
