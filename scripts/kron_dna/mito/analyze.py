"""MitoAge pilot: do Evo 2 and ESM-2 features of a species' mitochondrial genome predict its maximum lifespan beyond
body mass, GC content and taxonomy? Ridge regression of log maximum lifespan, folds held out by taxonomic group (whole
families, or whole orders), one class at a time; R^2 of each feature set added to the baseline.

usage: uv run python scripts/kron_dna/mito/analyze.py DATA_DIR OUT.json

DATA_DIR holds input/mitoage/{species,alignments}.parquet and interim/mito/{evo2_7b,esm2_650m}/.
Feature sets: base = log mass + whole-genome GC + GC3 (third codon positions, from the codon alignments);
taxon = mean log lifespan of the nearest taxon level present in the training folds (genus, family, order, class);
evo2 = per-gene mean Evo 2 log-likelihood (15 genes + whole genome); esm = per-protein mean ESM-2 log p (13);
evo2-emb / esm-emb = the first 20 principal components of the per-gene mean embeddings, fit inside each fold.
"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

data, out = Path(sys.argv[1]), Path(sys.argv[2])
M = data / "interim" / "mito"
sp = (pl.read_parquet(data / "input/mitoage/species.parquet")
      .filter((pl.col("record_status") != "retired") & pl.col("max_longevity_years").is_not_null() & pl.col("refseq").is_not_null())
      .with_columns(y=pl.col("max_longevity_years").log(), lmass=pl.col("adult_weight_g").log()))

# GC3 from the codon alignments
cod = pl.read_parquet(data / "input/mitoage/alignments.parquet").filter(pl.col("kind") == "codon")
def gc3(s: str) -> float:
    s = s.replace("-", "")
    t = s[2::3]
    return sum(c in "GC" for c in t) / max(1, len(t))
g3 = cod.group_by("slug").agg(pl.col("aligned").str.join("")).with_columns(
    gc3=pl.col("aligned").map_elements(gc3, return_dtype=pl.Float64)).select("slug", "gc3")
sp = sp.join(g3, on="slug", how="left")

# Evo 2: per-gene mean log-likelihood and per-gene embeddings
feat = pl.read_parquet(M / "features.parquet")
GENES = ["ATP6", "ATP8", "COX1", "COX2", "COX3", "CYTB", "ND1", "ND2", "ND3", "ND4", "ND4L", "ND5", "ND6", "rRNA-12S", "rRNA-16S"]
ll_rows, emb = [], {}
for f in sorted((M / "evo2_7b").glob("ll-*.parquet")):
    for r in pl.read_parquet(f).iter_rows(named=True):
        ll = np.asarray(r["ll"], dtype=np.float32)
        row = {"refseq": r["accession"], "evo2:genome": float(np.nanmean(ll))}
        for fr in feat.filter((pl.col("accession") == r["accession"]) & pl.col("gene").is_in(GENES)).iter_rows(named=True):
            row[f"evo2:{fr['gene']}"] = float(np.nanmean(ll[fr["start"]:fr["end"]]))
        ll_rows.append(row)
for f in sorted((M / "evo2_7b").glob("emb-*.npz")):
    z = np.load(f)
    for a, g, v in zip(z["accession"], z["gene"], z["emb"]):
        if g in GENES:
            emb.setdefault(str(a), {})[str(g)] = v.astype(np.float32)
evo = pl.DataFrame(ll_rows)
sp = sp.join(evo, on="refseq", how="left")

# ESM-2: per-protein mean log p and embeddings
esm_rows, esm_emb = {}, {}
for f in sorted((M / "esm2_650m").glob("*.npz")):
    z = np.load(f)
    gene = f.stem
    for s_, lp, e in zip(z["slug"], z["logp"].astype(np.float32), z["emb"].astype(np.float32)):
        esm_rows.setdefault(str(s_), {"slug": str(s_)})[f"esm:{gene}"] = float(np.nanmean(lp))
        esm_emb.setdefault(str(s_), {})[gene] = e
sp = sp.join(pl.DataFrame(list(esm_rows.values())), on="slug", how="left")

TAX = ["genus", "family", "order", "class"]


def taxon_prior(train: pl.DataFrame, test: pl.DataFrame) -> np.ndarray:
    pred = np.full(test.height, np.nan)
    for lvl in TAX:
        m = dict(train.group_by(lvl).agg(pl.col("y").mean()).iter_rows())
        v = np.array([m.get(k, np.nan) for k in test[lvl].to_list()])
        pred = np.where(np.isnan(pred), v, pred)
    return np.where(np.isnan(pred), train["y"].mean(), pred)


def emb_matrix(d: pl.DataFrame, source: dict, key: str, genes: list[str]) -> np.ndarray:
    rows = []
    for k in d[key].to_list():
        e = source.get(k, {})
        rows.append(np.concatenate([e[g] if g in e else np.full(next(iter(source.values()))[genes[0]].shape, np.nan) for g in genes]))
    return np.stack(rows)


def evaluate(d: pl.DataFrame, group: str, sets: dict, k: int = 5, seed: int = 0) -> dict:
    groups = d[group].to_numpy()
    uniq = np.array(sorted(set(groups)))
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(uniq, rng.permutation(len(uniq)) % k))
    folds = np.array([fold_of[g] for g in groups])
    y = d["y"].to_numpy()
    preds = {name: np.full(len(y), np.nan) for name in sets}
    for f in range(k):
        tr, te = folds != f, folds == f
        prior_te = taxon_prior(d.filter(pl.Series(tr)), d.filter(pl.Series(te)))
        prior_tr = np.zeros(tr.sum())          # inner prior for training rows: leave-one-group-out by the same level
        dtr = d.filter(pl.Series(tr))
        g_tr = dtr[group].to_numpy()
        for gname in np.unique(g_tr):
            m = g_tr == gname
            prior_tr[m] = taxon_prior(dtr.filter(pl.Series(~m)), dtr.filter(pl.Series(m)))
        for name, build in sets.items():
            Xtr, Xte = build(d, tr, te, prior_tr, prior_te)
            model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
            model.fit(Xtr, y[tr])
            preds[name][te] = model.predict(Xte)
    r2 = lambda p: float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))
    return {name: r2(p) for name, p in preds.items()}


def cols(d, names):
    X = d.select(names).to_numpy().astype(float)
    return np.where(np.isnan(X), np.nanmean(X, 0), X)


def make_sets(d: pl.DataFrame) -> dict:
    base = ["lmass", "gc_percent", "gc3"]
    evo_cols = [c for c in d.columns if c.startswith("evo2:")]
    esm_cols = [c for c in d.columns if c.startswith("esm:")]
    Eevo = emb_matrix(d, emb, "refseq", GENES[:13])
    Eesm = emb_matrix(d, esm_emb, "slug", GENES[:13])

    def tab(names, prior=False):
        def build(d, tr, te, ptr, pte):
            X = cols(d, names) if names else np.zeros((d.height, 0))
            Xtr, Xte = X[tr], X[te]
            if prior:
                Xtr, Xte = np.c_[Xtr, ptr], np.c_[Xte, pte]
            return Xtr, Xte
        return build

    def with_emb(E, names, prior=True):
        def build(d, tr, te, ptr, pte):
            Xb_tr, Xb_te = tab(names, prior)(d, tr, te, ptr, pte)
            Etr, Ete = E[tr], E[te]
            mu = np.nanmean(Etr, 0)
            Etr, Ete = np.where(np.isnan(Etr), mu, Etr), np.where(np.isnan(Ete), mu, Ete)
            sc = StandardScaler().fit(Etr)
            pca = PCA(20, random_state=0).fit(sc.transform(Etr))
            return np.c_[Xb_tr, pca.transform(sc.transform(Etr))], np.c_[Xb_te, pca.transform(sc.transform(Ete))]
        return build

    return {
        "mass": tab(["lmass"]), "GC": tab(["gc_percent", "gc3"]), "base": tab(base),
        "base+taxon": tab(base, True),
        "base+taxon+evo2-ll": tab(base + evo_cols, True), "base+taxon+esm-ll": tab(base + esm_cols, True),
        "base+evo2-ll": tab(base + evo_cols), "base+esm-ll": tab(base + esm_cols),
        "base+evo2-emb": with_emb(Eevo, base, False), "base+esm-emb": with_emb(Eesm, base, False),
        "base+taxon+evo2-emb": with_emb(Eevo, base), "base+taxon+esm-emb": with_emb(Eesm, base),
        "evo2-emb alone": with_emb(Eevo, [], False), "esm-emb alone": with_emb(Eesm, [], False),
    }


res = {}
for cls in ["Mammalia", "Aves", "Actinopterygii", "Reptilia"]:
    d = sp.filter((pl.col("class") == cls) & pl.col("lmass").is_not_null())
    if d.height < 80:
        continue
    sets = make_sets(d)
    for group in ["family", "order"]:
        if d[group].n_unique() < 10:
            continue
        r = evaluate(d, group, sets)
        res[f"{cls} | held out by {group} | n={d.height}"] = r
        print(f"{cls}, whole {group} held out, {d.height} species: " + ", ".join(f"{k} {v:.3f}" for k, v in r.items()), flush=True)
out.write_text(json.dumps(res, indent=1))
