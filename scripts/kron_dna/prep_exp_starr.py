"""Intrinsic enhancer and promoter strengths on hg38 from enhancer x promoter STARR-seq (ExP STARR-seq, Bergman et
al. 2022, Nature 607:176; GEO GSE184426, ``GSE184426_ExP_1Kx1K_counts.txt.gz``): 1,000 enhancer and 1,000 promoter
sequences of 264 bp, every pair in one plasmid, expression read in K562. The paper's finding is that a pair's
expression is close to the product of an enhancer's and a promoter's own strength; those two strengths per element
are what a model reading the genome can be asked for. Not a counterfactual: elements read once, in the table shape
of ``prep_lentimpra.py`` (``variant_bench.py score --ref-only``, ``report-lentimpra``).

    uv run --with pyliftover python scripts/kron_dna/prep_exp_starr.py OUT.parquet COUNTS.txt.gz
        --chain hg19ToHg38.over.chain.gz [--min-dna 10] [--min-barcodes 2] [--min-pairs 100]

The file is per barcode. A pair's activity is log2 of its RNA reads (four replicates summed) over its plasmid
reads, over its barcodes with >= ``--min-dna`` plasmid reads, kept with >= ``--min-barcodes`` of them. The
strengths are the additive fit ``activity = mean + promoter + enhancer`` over the measured pairs (alternating
means); the summary gives the share of variance it explains. One row per genomic element with >= ``--min-pairs``
measured pairs: ``cell`` (K562), ``group`` (enhancer, promoter), ``name``, ``chrom``, ``start``, ``end`` (hg38,
lifted from hg19 in one piece), ``pos`` (the middle), ``ref`` = ``alt`` = "", ``activity`` (the strength, log2),
``pairs``, ``fold`` (0-9, by chromosome, for the report's probe) and ``held_out`` (hg38 chr8/chr9). The scrambled
control sequences have no place in the genome and are left out of the table (they stay in the fit).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import polars as pl
import typer

HELD = ["chr8", "chr9"]


def additive(p: np.ndarray, e: np.ndarray, y: np.ndarray, n_p: int, n_e: int, rounds: int = 50):
    """``y ~ mu + a[p] + b[e]`` by alternating means; returns mu, a, b and the share of variance explained."""
    mu, a, b = y.mean(), np.zeros(n_p), np.zeros(n_e)
    cp, ce = np.bincount(p, minlength=n_p).clip(1), np.bincount(e, minlength=n_e).clip(1)
    for _ in range(rounds):
        a = np.bincount(p, y - mu - b[e], n_p) / cp
        b = np.bincount(e, y - mu - a[p], n_e) / ce
        shift = a.mean() + b.mean()
        mu, a, b = mu + shift, a - a.mean(), b - b.mean()
    return mu, a, b, float(1 - (y - mu - a[p] - b[e]).var() / y.var())


def main(out: Path, counts: Path, chain: Path = typer.Option(...), min_dna: int = 10, min_barcodes: int = 2,
         min_pairs: int = 100) -> None:
    from pyliftover import LiftOver
    t = pl.read_csv(counts, separator="\t", infer_schema_length=10000)
    t = t.filter(pl.col("DNA_input") >= min_dna).with_columns(rna=pl.sum_horizontal(r"^reads_rep\d$"))
    scale = float(np.log2(t["rna"].sum() / t["DNA_input"].sum()))
    g = t.group_by("promoter", "enhancer").agg(barcodes=pl.len(), rna=pl.col("rna").sum(), dna=pl.col("DNA_input").sum())
    g = g.filter(pl.col("barcodes") >= min_barcodes).with_columns(
        activity=((pl.col("rna") + 1) / (pl.col("dna") + 1)).log(2) - scale)
    proms, enhs = g["promoter"].unique().sort().to_list(), g["enhancer"].unique().sort().to_list()
    pi, ei = {x: k for k, x in enumerate(proms)}, {x: k for k, x in enumerate(enhs)}
    p = np.array([pi[x] for x in g["promoter"].to_list()])
    e = np.array([ei[x] for x in g["enhancer"].to_list()])
    mu, a, b, r2 = additive(p, e, g["activity"].to_numpy(), len(proms), len(enhs))
    lift = LiftOver(str(chain))
    rows, dropped = [], {"control": 0, "few pairs": 0, "not lifted in one piece": 0}
    for group, names, strength, counts_ in (("promoter", proms, a, np.bincount(p, minlength=len(proms))),
                                            ("enhancer", enhs, b, np.bincount(e, minlength=len(enhs)))):
        for name, s, n in zip(names, strength, counts_):
            m = re.fullmatch(r"(chr[^:]+):(\d+)-(\d+)-" + group, name)
            if not m:
                dropped["control"] += 1
                continue
            if n < min_pairs:
                dropped["few pairs"] += 1
                continue
            chrom, lo, hi = m[1], int(m[2]), int(m[3])
            x, y = lift.convert_coordinate(chrom, lo), lift.convert_coordinate(chrom, hi - 1)
            if not x or not y or x[0][0] != y[0][0] or x[0][2] != "+" or y[0][1] - x[0][1] != hi - 1 - lo:
                dropped["not lifted in one piece"] += 1
                continue
            rows.append(dict(cell="K562", group=group, name=name, chrom=x[0][0], start=x[0][1], end=x[0][1] + hi - lo,
                             pos=x[0][1] + (hi - lo) // 2, ref="", alt="", activity=float(s), pairs=int(n)))
    v = pl.DataFrame(rows)
    order = {c: k for k, c in enumerate(sorted(v["chrom"].unique().to_list()))}
    v = v.with_columns(fold=pl.col("chrom").replace_strict(order, return_dtype=pl.Int64) % 10, held_out=pl.col("chrom").is_in(HELD))
    out.parent.mkdir(parents=True, exist_ok=True)
    v.write_parquet(out)
    meta = dict(rows=v.height, source="Bergman et al. 2022, GEO GSE184426 ExP_1Kx1K_counts, hg19 lifted to hg38",
                held_out_chroms=HELD, pairs_measured=g.height, pairs_possible=len(proms) * len(enhs),
                additive_r2=round(r2, 4), dropped=dropped,
                by_group=v.group_by("group").agg(elements=pl.len(), held_out=pl.col("held_out").sum(),
                                                 sd=pl.col("activity").std().round(3), pairs=pl.col("pairs").median()).sort("group").to_dicts())
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    typer.run(main)
