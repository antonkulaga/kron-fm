"""MitoAge: phylogenetically independent contrasts (Felsenstein 1985) on the dated MitoAge tree, for log maximum
lifespan, log body mass, GC (whole genome, third codon positions) and ESM-2 features of the 13 mitochondrial proteins:
the 20 leading principal components of the per-protein mean embeddings, and ESM-2's log-probability of the observed
residue at every alignment column. Each feature's contrasts are regressed, through the origin, with lifespan
contrasts as the outcome and mass and GC contrasts as covariates, so shared ancestry cancels by construction.

usage: uv run python scripts/kron_dna/mito/contrasts.py DATA_DIR PHYLOGENY.json OUT_DIR

Also residue identity: each residue seen in 5-95 % of a class's species at a column (gaps <= 10 %) as a 0/1 trait,
with its contrasts tested the same way (OUT_DIR/<class>-residues.parquet).

Writes OUT_DIR/<class>-columns.parquet (gene, column, t, p, q (Benjamini-Hochberg), beta, the residues most common in
the longest- and shortest-lived quarter of species) and OUT_DIR/<class>-summary.json (contrast R^2 of the covariates,
of covariates + embedding components, the components' t values, the top columns).
"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats
from sklearn.decomposition import PCA

data, tree_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
GENES = ["ATP6", "ATP8", "COX1", "COX2", "COX3", "CYTB", "ND1", "ND2", "ND3", "ND4", "ND4L", "ND5", "ND6"]
EPS = 0.01                                           # million years added to every branch: zero lengths occur

sp = (pl.read_parquet(data / "input/mitoage/species.parquet")
      .filter((pl.col("record_status") != "retired") & pl.col("max_longevity_years").is_not_null()
              & pl.col("adult_weight_g").is_not_null())
      .with_columns(y=pl.col("max_longevity_years").log(), lmass=pl.col("adult_weight_g").log()))
aln = pl.read_parquet(data / "input/mitoage/alignments.parquet")
cod = aln.filter(pl.col("kind") == "codon").group_by("slug").agg(pl.col("aligned").str.join(""))
gc3 = {r["slug"]: (lambda t: sum(c in "GC" for c in t) / max(1, len(t)))(r["aligned"].replace("-", "")[2::3])
       for r in cod.iter_rows(named=True)}
sp = sp.with_columns(gc3=pl.col("slug").replace_strict(gc3, default=None, return_dtype=pl.Float64))
prot = {g: dict(zip(d["slug"].to_list(), d["aligned"].to_list()))
        for (g,), d in aln.filter(pl.col("kind") == "protein").group_by("gene")}
esm = {g: np.load(data / f"interim/mito/esm2_650m/{g}.npz") for g in GENES}
idx = {g: {s: i for i, s in enumerate(esm[g]["slug"])} for g in GENES}
tree = json.load(open(tree_path))["root"]


def prune(node, keep):
    """The tree induced on leaf ids in keep: (children, length, leaf); unary nodes merged into their child."""
    if not node.get("children"):
        return {"leaf": node["leaf"], "length": node["length"] + EPS, "children": []} if node["leaf"] in keep else None
    kids = [k for k in (prune(c, keep) for c in node["children"]) if k is not None]
    if not kids:
        return None
    if len(kids) == 1:
        kids[0]["length"] += node["length"]
        return kids[0]
    return {"leaf": None, "length": node["length"] + EPS, "children": kids}


def contrasts(node, X: dict[int, np.ndarray], out_rows: list):
    """Felsenstein's contrasts, post-order; returns (node value, extra length). Polytomies are resolved left to right."""
    if not node["children"]:
        return X[node["leaf"]], node["length"]
    vals = [contrasts(c, X, out_rows) for c in node["children"]]
    x, v = vals[0]
    for x2, v2 in vals[1:]:
        out_rows.append((x - x2) / np.sqrt(v + v2))
        x = (x / v + x2 / v2) / (1 / v + 1 / v2)
        v = v * v2 / (v + v2)
    return x, v + node["length"]


def ols0(y, X):
    """Regression through the origin: beta, t of the last column, R^2 (uncentred)."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    dof = len(y) - X.shape[1]
    s2 = res @ res / dof
    cov = s2 * np.linalg.pinv(X.T @ X)
    return beta, beta[-1] / np.sqrt(cov[-1, -1]), 1 - (res @ res) / (y @ y), dof


for cls in ["Mammalia", "Aves"]:
    d = sp.filter((pl.col("class") == cls) & pl.col("gc3").is_not_null())
    d = d.filter(pl.col("slug").map_elements(lambda s: all(s in idx[g] for g in GENES), return_dtype=pl.Boolean))
    ids = d["id"].to_list()
    t = prune(tree, set(ids))
    d = d.filter(pl.col("id").is_in(_leaves := [n for n in ids]))
    # features per species: covariates, 20 embedding PCs, per-column log p (columns with <= 10 % gaps in this class)
    emb = np.stack([np.concatenate([esm[g]["emb"][idx[g][s]].astype(np.float32) for g in GENES]) for s in d["slug"]])
    pcs = PCA(20, random_state=0).fit_transform((emb - emb.mean(0)) / (emb.std(0) + 1e-6))
    col_meta, col_vals = [], []
    for g in GENES:
        lp = esm[g]["logp"].astype(np.float32)[[idx[g][s] for s in d["slug"]]]
        ok = np.isnan(lp).mean(0) <= 0.10
        for c in np.nonzero(ok)[0]:
            v = lp[:, c]
            col_meta.append((g, int(c)))
            col_vals.append(np.where(np.isnan(v), np.nanmean(v), v))
    C = np.stack(col_vals, 1)
    base = d.select("y", "lmass", "gc_percent", "gc3").to_numpy().astype(float)
    F = np.c_[base, pcs, C]
    X = dict(zip(d["id"].to_list(), F))
    rows: list = []
    contrasts(t, X, rows)
    R = np.stack(rows)
    y, cov = R[:, 0], R[:, 1:4]
    n = len(y)
    _, _, r2_cov, _ = ols0(y, cov)
    _, _, r2_pcs, _ = ols0(y, np.c_[cov, R[:, 4:24]])
    pc_t = [float(ols0(y, np.c_[cov, R[:, 4 + k]])[1]) for k in range(20)]
    # per column
    tvals, betas, trob = np.zeros(C.shape[1]), np.zeros(C.shape[1]), np.zeros(C.shape[1])
    for j in range(C.shape[1]):
        x = R[:, 24 + j]
        b, tv, _, dof = ols0(y, np.c_[cov, x])
        tvals[j], betas[j] = tv, b[-1]
        mad = np.median(np.abs(x)) * 1.4826 + 1e-12          # clip contrasts beyond 4 robust sd: a few large ones
        trob[j] = ols0(y, np.c_[cov, np.clip(x, -4 * mad, 4 * mad)])[1]   # must not carry the column alone
    p = 2 * stats.t.sf(np.abs(tvals), n - 4)
    q = stats.false_discovery_control(p)
    yq = d["y"].to_numpy()
    hi, lo = yq >= np.quantile(yq, 0.75), yq <= np.quantile(yq, 0.25)
    def top_res(g, c, mask):
        res = [prot[g][s][c] for s, m in zip(d["slug"].to_list(), mask) if m and s in prot[g]]
        vals, counts = np.unique(res, return_counts=True)
        o = np.argsort(-counts)[:2]
        return " ".join(f"{vals[i]} {counts[i] / len(res):.0%}" for i in o)
    cols = pl.DataFrame({"gene": [m[0] for m in col_meta], "column": [m[1] for m in col_meta], "t": tvals, "p": p,
                         "q": q, "beta": betas, "t_clipped": trob,
                         "share_nonzero": (np.abs(R[:, 24:]) > 1e-4).mean(0)}).sort("p")
    cols = cols.with_columns(
        long_lived=pl.struct("gene", "column").map_elements(lambda r: top_res(r["gene"], r["column"], hi), return_dtype=pl.Utf8),
        short_lived=pl.struct("gene", "column").map_elements(lambda r: top_res(r["gene"], r["column"], lo), return_dtype=pl.Utf8))
    cols.write_parquet(out / f"{cls}-columns.parquet")
    summ = {"species": d.height, "contrasts": n, "columns_tested": C.shape[1],
            "r2_covariates": float(r2_cov), "r2_covariates_plus_20_pcs": float(r2_pcs),
            "covariate_t": [float(ols0(y, cov[:, [k]])[1]) for k in range(3)],
            "pc_t": pc_t, "columns_q_below_0.05": int((q < 0.05).sum()),
            "columns_q_below_0.05_and_clipped_t_above_3.5": int(((q < 0.05) & (np.abs(trob) > 3.5)).sum()), "columns_q_below_0.2": int((q < 0.2).sum()),
            "top": cols.head(15).to_dicts()}

    # residue identity: for each column and residue seen in 5-95 % of the class's species, a 0/1 trait
    id_meta, id_vals = [], []
    slugs = d["slug"].to_list()
    for g in GENES:
        seqs = [prot[g].get(s) for s in slugs]
        width = len(next(x for x in seqs if x))
        for c in range(width):
            col = np.array([x[c] if x else "-" for x in seqs])
            if (col == "-").mean() > 0.10:
                continue
            for a in np.unique(col):
                f = (col == a).mean()
                if a in "-X" or not 0.05 <= f <= 0.95:
                    continue
                id_meta.append((g, c, a, f))
                id_vals.append((col == a).astype(float))
    if id_vals:
        Xi = dict(zip(d["id"].to_list(), np.c_[base[:, :1], np.stack(id_vals, 1)]))
        rows_i: list = []
        contrasts(t, Xi, rows_i)
        Ri = np.stack(rows_i)
        ti, tci = np.zeros(len(id_meta)), np.zeros(len(id_meta))
        for j in range(len(id_meta)):
            x = Ri[:, 1 + j]
            ti[j] = ols0(y, np.c_[cov, x])[1]
            mad = np.median(np.abs(x[np.abs(x) > 1e-9])) * 1.4826 if (np.abs(x) > 1e-9).any() else 1.0
            tci[j] = ols0(y, np.c_[cov, np.clip(x, -4 * mad, 4 * mad)])[1]
        pi = 2 * stats.t.sf(np.abs(ti), n - 4)
        qi = stats.false_discovery_control(pi)
        ident = pl.DataFrame({"gene": [m[0] for m in id_meta], "column": [m[1] for m in id_meta],
                              "residue": [m[2] for m in id_meta], "share": [m[3] for m in id_meta],
                              "t": ti, "t_clipped": tci, "p": pi, "q": qi,
                              "changes": (np.abs(Ri[:, 1:]) > 1e-9).sum(0)}).sort("p")
        ident.write_parquet(out / f"{cls}-residues.parquet")
        summ["residue_tests"] = len(id_meta)
        summ["residues_q_below_0.05"] = int((qi < 0.05).sum())
        summ["residues_q_below_0.05_and_clipped_t_above_3.5"] = int(((qi < 0.05) & (np.abs(tci) > 3.5)).sum())
        summ["top_residues"] = ident.head(20).to_dicts()
        print(f"{cls} residues: {len(id_meta)} tests, q<0.05 {int((qi < 0.05).sum())}, robust "
              f"{summ['residues_q_below_0.05_and_clipped_t_above_3.5']}", flush=True)
        print(ident.head(15), flush=True)
    (out / f"{cls}-summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print(f"{cls}: {d.height} species, {n} contrasts, {C.shape[1]} columns; contrast R^2 covariates {r2_cov:.3f}, "
          f"+20 PCs {r2_pcs:.3f}; columns q<0.05 {int((q < 0.05).sum())}, q<0.2 {int((q < 0.2).sum())}", flush=True)
    print(cols.head(12), flush=True)
