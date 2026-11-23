"""Biology checks of a binned ChIP-seq store (``prep_bigwig_binned.py``), run before a store is cleared for training.

    uv run python scripts/kron_dna/check_chip_store.py histone STORE GENOME_DIR
    uv run python scripts/kron_dna/check_chip_store.py tf STORE GENOME_DIR

``histone``: each core mark's enrichment at loci where it is known to sit (H3K4me3 and H3K27ac at housekeeping
promoters, H3K36me3 over a housekeeping gene body, H3K27me3 at HOXD, H3K9me3 at the chr19 ZNF cluster), as the median
over biosamples of locus mean / genome mean of log1p, and the correlation structure between marks.
``tf``: signal over noise per track; POLR2A, general factors and EZH2 at housekeeping promoters; CTCF at shared CTCF
sites found from the other CTCF tracks (left out), against RAD21 (co-bound) and POLR2A at the same cells.
Coordinates are hg38. The numbers posted for chip492b and chiptf on 22 Nov come from these checks.
"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import typer

app = typer.Typer(add_completion=False)

HG38 = {"GAPDH promoter": ("chr12", 6_533_500, 6_535_500), "ACTB promoter": ("chr7", 5_529_600, 5_531_600),
        "EEF1A1 promoter": ("chr6", 73_520_000, 73_522_000), "SF3B1 gene body": ("chr2", 197_390_000, 197_434_000),
        "HOXD cluster": ("chr2", 176_080_000, 176_190_000), "ZNF cluster": ("chr19", 56_400_000, 57_900_000),
        "OR cluster": ("chr11", 4_400_000, 5_900_000)}


def genes_of(gtf: Path) -> pl.DataFrame:
    """Gene rows of a GTF: chrom, start0, end, strand, name (upper case)."""
    import gzip
    import re
    rows = []
    with (gzip.open if str(gtf).endswith(".gz") else open)(gtf, "rt") as f:
        for line in f:
            c = line.split("\t")
            if len(c) > 8 and c[2] == "gene":
                m = re.search(r'gene_name "([^"]+)"', c[8])
                if m:
                    rows.append((c[0], int(c[3]) - 1, int(c[4]), c[6], m.group(1).upper()))
    return pl.DataFrame(rows, schema=["chrom", "start0", "end", "strand", "name"], orient="row")


def loci_of(gtf: Path | None) -> dict:
    """The checked loci: hg38 coordinates without a GTF; with one (any species), from its genes by name: TSS +-1 kb of
    GAPDH, ACTB, EEF1A1, SF3B1's body, the HOXD1-HOXD13 span, and the 1.5 Mb window richest in ZNF/ZFP genes."""
    if gtf is None:
        return dict(HG38)
    g = genes_of(gtf)
    out = {}
    for name in ("GAPDH", "ACTB", "EEF1A1"):
        r = g.filter(pl.col("name") == name).row(0, named=True)
        tss = r["start0"] if r["strand"] == "+" else r["end"] - 1
        out[f"{name} promoter"] = (r["chrom"], tss - 1000, tss + 1000)
    r = g.filter(pl.col("name") == "SF3B1").row(0, named=True)
    out["SF3B1 gene body"] = (r["chrom"], r["start0"], r["end"])
    hox = g.filter(pl.col("name").str.contains(r"^HOXD\d+$"))
    out["HOXD cluster"] = (hox["chrom"][0], int(hox["start0"].min()), int(hox["end"].max()))
    def densest(pattern: str) -> tuple:
        z = g.filter(pl.col("name").str.contains(pattern)).sort("chrom", "start0")
        best = (0, None)
        for c, d in z.group_by("chrom"):
            st = np.sort(d["start0"].to_numpy())
            k = np.searchsorted(st, st + 1_500_000) - np.arange(len(st))
            i = int(k.argmax())
            if k[i] > best[0]:
                best = (int(k[i]), (c[0], int(st[i]), int(st[i]) + 1_500_000))
        return best[1]
    out["ZNF cluster"] = densest(r"^(ZNF|ZFP)\d")
    out["OR cluster"] = densest(r"^(OR\d|OLFR\d)")                 # olfactory receptors: silenced outside the nose
    return out


@app.command()
def histone(store: Path, genome: Path, gtf: Path = typer.Option(None, help="the genome's annotation (loci by gene "
                                                                       "name); without it hg38 coordinates")) -> None:


    j = json.loads((store / "binned.json").read_text())
    B, T, C = j["bin"], j["n_tracks"], j["cells"]
    lg = np.memmap(store / "log1p.f16", dtype=np.float16, mode="r", shape=(C, T))
    idx = pl.read_parquet(genome / "contigs.parquet")
    marks = np.array([i.split("|")[0] for i in j["ids"]])
    bios = np.array([i.split("|", 1)[1] for i in j["ids"]])


    def cells(chrom, a, b):
        r = idx.filter((pl.col("chrom") == chrom) & (pl.col("start") <= a) & (pl.col("end") >= b))
        assert r.height == 1, (chrom, a, b, r)
        r = r.row(0, named=True)
        g0 = r["offset"] + a - r["start"]
        return g0 // B, (g0 + b - a) // B + 1


    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(C, 50_000, replace=False))
    gmean = np.asarray(lg[pick], dtype=np.float32).mean(0)
    print(f"{T} tracks, {len(set(bios))} biosamples, marks {sorted(set(marks))}")
    print(f"genome-wide mean log1p per mark: " + ", ".join(f"{m} {gmean[marks == m].mean():.3f}" for m in sorted(set(marks))))

    loci = loci_of(gtf)
    expect = {"GAPDH promoter": "H3K4me3, H3K27ac high; H3K9me3, H3K27me3 low", "ACTB promoter": "H3K4me3, H3K27ac high",
              "EEF1A1 promoter": "H3K4me3, H3K27ac high", "SF3B1 gene body": "H3K36me3 high",
              "HOXD cluster": "H3K27me3 high", "ZNF cluster": "H3K9me3 high (with H3K36me3 at ZNF 3' ends)",
              "OR cluster": "H3K9me3 high, active marks low"}
    print("loci:", {k: f"{c}:{a:,}-{b:,}" for k, (c, a, b) in loci.items()})
    ms = ["H3K4me3", "H3K27ac", "H3K4me1", "H3K36me3", "H3K27me3", "H3K9me3"]
    print("\nmedian over biosamples of locus mean / genome mean (share of biosamples > 1):")
    print(f"{'locus':32s} " + " ".join(f"{m:>15s}" for m in ms) + "   expected")
    for name, (c, a, b) in loci.items():
        c0, c1 = cells(c, a, b)
        v = np.asarray(lg[c0:c1], dtype=np.float32).mean(0) / np.maximum(gmean, 1e-6)
        row = []
        for m in ms:
            x = v[marks == m]
            row.append(f"{np.median(x):7.2f} ({(x > 1).mean():4.0%})")
        print(f"{name:32s} " + " ".join(f"{r:>15s}" for r in row) + f"   {expect[name]}")

    # correlation structure on 20k random cells
    x = np.log1p(np.asarray(lg[pick[:20000]], dtype=np.float32))
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    r = (x.T @ x) / len(x)
    same_mark = np.median([r[i, k] for i in range(T) for k in range(i + 1, T) if marks[i] == marks[k]])
    print("\ncorrelation over 20k random 128-bp cells:")
    for m in ms:
        ii = np.where(marks == m)[0]
        rr = r[np.ix_(ii, ii)][np.triu_indices(len(ii), 1)]
        print(f"  {m:9s} same mark, across biosamples: median r {np.median(rr):.2f}")
    for m1, m2 in [("H3K4me3", "H3K27ac"), ("H3K27ac", "H3K4me1"), ("H3K36me3", "H3K9me3"), ("H3K4me3", "H3K9me3"),
                   ("H3K4me3", "H3K27me3")]:
        vals = []
        for bsm in set(bios):
            i1 = np.where((marks == m1) & (bios == bsm))[0]
            i2 = np.where((marks == m2) & (bios == bsm))[0]
            if len(i1) and len(i2):
                vals.append(r[i1[0], i2[0]])
        print(f"  {m1} vs {m2}, same biosample: median r {np.median(vals):.2f} over {len(vals)}")
    print(f"\nall pairs of the same mark: median r {same_mark:.2f}")


@app.command()
def tf(store: Path, genome: Path, gtf: Path = typer.Option(None, help="as for histone")) -> None:


    j = json.loads((store / "binned.json").read_text())
    B, T, C = j["bin"], j["n_tracks"], j["cells"]
    lg = np.memmap(store / "log1p.f16", dtype=np.float16, mode="r", shape=(C, T))
    idx = pl.read_parquet(genome / "contigs.parquet")
    tgt = np.array([t["target"] for t in j["tracks"]])
    bios = np.array([t["biosample"] for t in j["tracks"]])


    def cells(chrom, a, b):
        r = idx.filter((pl.col("chrom") == chrom) & (pl.col("start") <= a) & (pl.col("end") >= b)).row(0, named=True)
        g0 = r["offset"] + a - r["start"]
        return g0 // B, (g0 + b - a) // B + 1


    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(C, 50_000, replace=False))
    x = np.asarray(lg[pick], dtype=np.float32)
    gmean = x.mean(0)
    # signal to noise: mean of the top 0.5 % of sampled cells over the median cell, per track
    top = np.sort(x, 0)[-250:].mean(0)
    snr = top / np.maximum(np.median(x, 0), 1e-3)
    print(f"{T} tracks, {len(set(tgt))} targets, {len(set(bios))} biosamples")
    print(f"signal/noise (top 0.5 % / median): median {np.median(snr):.1f}, 5th pct {np.percentile(snr, 5):.1f}; "
          f"tracks below 2: {int((snr < 2).sum())}; all-zero tracks: {int((x.max(0) == 0).sum())}")
    low = np.argsort(snr)[:8]
    print("lowest:", ", ".join(f"{j['ids'][k]} {snr[k]:.2f}" for k in low))

    proms = {k: v for k, v in loci_of(gtf).items() if k.endswith("promoter")}
    pv = []
    for name, (c, a, b) in proms.items():
        c0, c1 = cells(c, a, b)
        pv.append(np.asarray(lg[c0:c1], dtype=np.float32).mean(0) / np.maximum(gmean, 1e-6))
    pv = np.mean(pv, 0)
    for t in ["POLR2A", "POLR2AphosphoS5", "TAF1", "TBP", "YY1", "MAX", "CTCF", "EZH2"]:
        k = tgt == t
        if k.any():
            print(f"{t:16s} n={k.sum():3d}  housekeeping promoters / genome mean: median {np.median(pv[k]):6.2f}, "
                  f"share > 1: {(pv[k] > 1).mean():.0%}")
    print(f"all tracks: median {np.median(pv):.2f}, share > 1: {(pv > 1).mean():.0%}")

    z = np.log1p(x[:20000])
    z = (z - z.mean(0)) / (z.std(0) + 1e-6)
    r = z.T @ z / len(z)
    for t in ["CTCF", "RAD21", "POLR2A", "EP300", "REST"]:
        ii = np.where(tgt == t)[0]
        if len(ii) > 1:
            print(f"{t:8s} same factor across biosamples: median r {np.median(r[np.ix_(ii, ii)][np.triu_indices(len(ii), 1)]):.2f}")
    ii, jj = np.where(tgt == "CTCF")[0], np.where(tgt == "POLR2A")[0]
    print(f"CTCF vs POLR2A: median r {np.median(r[np.ix_(ii, jj)]):.2f}")
    ii, jj = np.where(tgt == "CTCF")[0], np.where(tgt == "RAD21")[0]
    if len(jj):
        print(f"CTCF vs RAD21 (cohesin, co-bound): median r {np.median(r[np.ix_(ii, jj)]):.2f}")
    s = store
    C, T = j["cells"], j["n_tracks"]
    lg = np.memmap(s / "log1p.f16", dtype=np.float16, mode="r", shape=(C, T))
    tgt = np.array([t["target"] for t in j["tracks"]])
    pick = np.sort(np.random.default_rng(1).choice(C, 200_000, replace=False))
    x = np.asarray(lg[pick], dtype=np.float32)
    ct = np.where(tgt == "CTCF")[0]; pol = np.where(tgt == "POLR2A")[0]; rad = np.where(tgt == "RAD21")[0]
    g = x.mean(0)
    loo = []
    for k in ct:                                   # sites from the other CTCF tracks only (leave this track out)
        others = ct[ct != k]
        cons = x[:, others].mean(1)
        top = cons >= np.quantile(cons, 0.995)     # the top 0.5 % of cells by the other CTCF tracks: shared sites
        loo.append(x[top, k].mean() / max(g[k], 1e-6))
    cons = x[:, ct].mean(1); top = cons >= np.quantile(cons, 0.995)
    print(f"CTCF tracks at shared CTCF sites (left-out consensus), signal / genome mean: median {np.median(loo):.2f}, "
          f"5th pct {np.percentile(loo, 5):.2f}, share > 2: {(np.array(loo) > 2).mean():.0%}")
    for name, ii in (("RAD21", rad), ("POLR2A", pol)):
        v = x[top][:, ii].mean(0) / np.maximum(g[ii], 1e-6)
        print(f"{name} at the same sites: median {np.median(v):.2f}")
    low = np.argsort(loo)[:5]
    print("weakest CTCF tracks:", ", ".join(f"{j['ids'][ct[k]]} {loo[k]:.2f}" for k in low))
    pr = [t.get("check_pearson_cells_vs_bigwig", 1) for t in j["tracks"]]
    print("lowest cell-vs-bigWig r:", sorted(zip(pr, j["ids"]))[:3])


def _open(store: Path, genome: Path):
    j = json.loads((store / "binned.json").read_text())
    C, T, B = j["cells"], j["n_tracks"], j["bin"]
    lg = np.memmap(store / "log1p.f16", dtype=np.float16, mode="r", shape=(C, T))
    idx = pl.read_parquet(genome / "contigs.parquet")

    def mean_at(c, a, b):
        r = idx.filter((pl.col("chrom") == c) & (pl.col("start") <= a) & (pl.col("end") >= b)).row(0, named=True)
        g0 = r["offset"] + a - r["start"]
        return np.asarray(lg[g0 // B:(g0 + b - a) // B + 1], dtype=np.float32).mean(0)

    pick = np.sort(np.random.default_rng(0).choice(C, 50_000, replace=False))
    return j, lg, mean_at, np.asarray(lg[pick], dtype=np.float32).mean(0)


@app.command()
def accessibility(store: Path, genome: Path, gtf: Path = typer.Option(None, help="as for histone")) -> None:
    """DNase-seq and ATAC-seq: open at housekeeping promoters, closed over the ZNF/HOXD clusters' repressed chromatin."""
    j, lg, mean_at, gm = _open(store, genome)
    assay = np.array([t.get("assay", "") for t in j["tracks"]])
    print(f"{j['n_tracks']} tracks: " + ", ".join(f"{a} {int((assay == a).sum())}" for a in sorted(set(assay))))
    for name, (c, a, b) in loci_of(gtf).items():
        v = mean_at(c, a, b) / np.maximum(gm, 1e-6)
        print(f"{name:18s} " + "  ".join(f"{a_}: median {np.median(v[assay == a_]):5.2f}, share > 1 {(v[assay == a_] > 1).mean():4.0%}"
                                          for a_ in sorted(set(assay))))


@app.command()
def rna(store: Path, genome: Path, gtf: Path = typer.Option(..., help="the genome's annotation")) -> None:
    """Stranded RNA-seq: over housekeeping gene bodies, the gene's own strand against the other strand's channel."""
    j, lg, mean_at, gm = _open(store, genome)
    ids = j["ids"]
    g = genes_of(gtf)
    for name in ("GAPDH", "ACTB", "EEF1A1"):
        r = g.filter(pl.col("name") == name).row(0, named=True)
        v = mean_at(r["chrom"], r["start0"], r["end"])
        own = "_P" if r["strand"] == "+" else "_M"
        k_own = [k for k, i in enumerate(ids) if i.endswith(own)]
        k_oth = [k for k, i in enumerate(ids) if i.endswith(("_P", "_M")) and not i.endswith(own)]
        print(f"{name:7s} ({r['strand']}): own strand median {np.median(v[k_own]):.2f}, other strand "
              f"{np.median(v[k_oth]):.2f}; own > other in {(v[k_own] > np.asarray(v[k_oth])).mean():.0%} of pairs; "
              f"own / genome mean {np.median(v[k_own] / np.maximum(gm[k_own], 1e-6)):.1f}")


if __name__ == "__main__":
    app()
