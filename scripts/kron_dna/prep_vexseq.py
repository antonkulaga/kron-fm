"""Exon-inclusion variants on hg38 for the splice head: Vex-seq (Adamson et al. 2018, Genome Biol 19:71; the CAGI5
exon-skipping challenge data; scottiadamson/Vex-seq, ``processed_files``): 2,055 ExAC variants (SNVs and short
indels) in or next to 110 human exons, each exon in a minigene, its inclusion (PSI) measured by RNA-seq in K562
and HepG2. The same columns as ``prep_mfass.py``'s table, so ``variant_bench.py score --head splice --read-col
exon_mid --flank-col half`` and ``report-mfass`` read it unchanged.

    uv run --with pyliftover python scripts/kron_dna/prep_vexseq.py OUT.parquet DIR --chain hg19ToHg38.over.chain.gz

``DIR`` holds ``delta_PSI_values.tsv``, ``Vex-seq_positions.tsv`` and ``control_variant_pairs.csv``. The files are
on hg19; positions are lifted with the UCSC chain. One row per variant: ``chrom``, ``pos`` (0-based, of the first
``ref`` base), ``ref``, ``alt`` (as in the file: an indel keeps its anchor base), ``dpsi`` (the file's
``delta_PSI`` / 100: change of inclusion against the reference exon, mean of the two cell lines), ``dpsi_k562``,
``dpsi_hepg2``, ``psi`` (the variant's own inclusion), ``sdv`` (dPSI <= -0.5, MFASS's definition; only a few
here, the correlation with ``dpsi`` is this set's measure), and

- ``exon_id`` (the file's exon: its control name), ``strand``, ``exon_start``/``exon_end`` (hg38, 0-based
  half-open), ``exon_mid``, ``half`` (the exon and 3 bases beyond each end). The file gives each variant's distance
  to the 3' and the 5' splice site in the transcript's direction, without a zero (1 = first base after the site,
  -1 = last base before it) and without a strand; the exon's ends are the most common ones among its SNVs, and
  the strand is the one under which those ends look more like splice sites (annotated in ``splice-sites-v1``,
  AG before the exon, GT or GC after it). ``sites_annotated``: both ends are annotated sites of that strand;
  ``sites_canonical``: AG and GT/GC flank it;
- ``region`` (exon, upstr_intron, downstr_intron), ``ref_ok`` and ``held_out`` (hg38 chr8/chr9).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
import typer

from kron_core.paths import interim_dir

HELD = ["chr8", "chr9"]
BITS = {"+": (2, 1), "-": (8, 4)}      # (acceptor, donor) bits of splice-sites-v1


def ends(g: int, d3: int, d5: int, strand: str) -> tuple[int, int]:
    """The exon (0-based half-open) of a variant at ``g`` with the file's distances to the 3' and 5' splice site."""
    if strand == "+":
        return g - d3 + (1 if d3 > 0 else 0), g - d5 + (1 if d5 > 0 else 0)
    return g + d5 + (0 if d5 > 0 else 1), g + d3 + (0 if d3 > 0 else 1)


def main(out: Path, data: Path, chain: Path = typer.Option(...), name: str = "hg38") -> None:
    from pyliftover import LiftOver
    lift = LiftOver(str(chain))
    src = interim_dir("genome", name)
    bases = np.memmap(src / "bases.u8", dtype=np.uint8, mode="r")
    sites = np.memmap(src.parent.parent / "tracks" / name / "splice-sites-v1" / "splice.u8", dtype=np.uint8, mode="r")
    segs = pl.read_parquet(src / "contigs.parquet").to_dicts()

    def at(chrom: str, x: int) -> int | None:
        return next((s["offset"] + x - s["start"] for s in segs if s["chrom"] == chrom and s["start"] <= x < s["end"]), None)

    def annotated(chrom: str, lo: int, hi: int, strand: str) -> tuple[bool, bool]:
        acc, don = (lo - 1, hi) if strand == "+" else (hi, lo - 1)
        a, d = at(chrom, acc), at(chrom, don)
        return (a is not None and bool(sites[a] & BITS[strand][0]), d is not None and bool(sites[d] & BITS[strand][1]))

    exon_of = {}
    for line in (data / "control_variant_pairs.csv").read_text().splitlines():
        names = [x for x in line.strip().split(",") if x]
        for x in names:
            exon_of[x] = names[0].removesuffix("_Control_Original")
    dist = {r["variant"]: (r["3SS"], r["5SS"]) for r in pl.read_csv(data / "Vex-seq_positions.tsv", separator="\t").to_dicts()}
    rows, dropped = [], Counter()
    for r in pl.read_csv(data / "delta_PSI_values.tsv", separator="\t").to_dicts():
        v = r["variant"]
        chrom19, pos19, ref, alt = v.split("_", 3)
        to = lift.convert_coordinate(chrom19, int(pos19) - 1)
        if not to or to[0][2] != "+" or v not in dist or v not in exon_of:
            dropped["unlifted" if not to or to[0][2] != "+" else "no distance or exon"] += 1
            continue
        chrom, pos = to[0][0], to[0][1]
        g = at(chrom, pos)
        here = "".join("ACGTN"[b] for b in bases[g:g + len(ref)]) if g is not None else ""
        rows.append(dict(id=v, exon_id=exon_of[v], chrom=chrom, pos=pos, ref=ref, alt=alt, d3=dist[v][0], d5=dist[v][1],
                         dpsi=r["delta_PSI"] / 100, dpsi_k562=r["delta_psi_K"] / 100, dpsi_hepg2=r["delta_psi_H"] / 100,
                         psi=r["mean_PSI"] / 100, ref_ok=here == ref))
    by_exon: dict = {}
    for r in rows:
        by_exon.setdefault(r["exon_id"], []).append(r)
    def canonical(chrom: str, lo: int, hi: int, strand: str) -> tuple[bool, bool]:
        a, d = at(chrom, lo - 2), at(chrom, hi)
        if a is None or d is None:
            return False, False
        left, right = ("".join("ACGTN"[b] for b in bases[x:x + 2]) for x in (a, d))
        return (left == "AG", right in ("GT", "GC")) if strand == "+" else (right == "CT", left in ("AC", "GC"))

    for rs in by_exon.values():
        snv = [r for r in rs if len(r["ref"]) == 1 and len(r["alt"]) == 1] or rs      # an indel's distance may be off
        best = None
        for s in "+-":                                           # the strand whose exon ends look like splice sites
            (lo, hi), _ = Counter(ends(r["pos"], r["d3"], r["d5"], s) for r in snv).most_common(1)[0]
            score = sum(annotated(rs[0]["chrom"], lo, hi, s)) + sum(canonical(rs[0]["chrom"], lo, hi, s))
            if best is None or score > best[0]:
                best = (score, s, lo, hi)
        _, strand, lo, hi = best
        ok = all(annotated(rs[0]["chrom"], lo, hi, strand))
        motif = all(canonical(rs[0]["chrom"], lo, hi, strand))
        for r in rs:
            r.update(strand=strand, exon_start=lo, exon_end=hi, sites_annotated=ok, sites_canonical=motif,
                     region="exon" if r["d3"] > 0 > r["d5"] else "upstr_intron" if r["d3"] < 0 else "downstr_intron")
    t = pl.DataFrame(rows).with_columns(
        sdv=pl.col("dpsi") <= -0.5, exon_mid=(pl.col("exon_start") + pl.col("exon_end")) // 2,
        half=(pl.col("exon_end") - pl.col("exon_start") + 1) // 2 + 3, held_out=pl.col("chrom").is_in(HELD),
        kind=pl.when((pl.col("ref").str.len_chars() == 1) & (pl.col("alt").str.len_chars() == 1)).then(pl.lit("snv")).otherwise(pl.lit("indel")))
    out.parent.mkdir(parents=True, exist_ok=True)
    t.write_parquet(out)
    summ = t.group_by("region").agg(variants=pl.len(), sdv=pl.col("sdv").sum(), exons=pl.col("exon_id").n_unique(),
                                    indels=(pl.col("kind") == "indel").sum(), held_out=pl.col("held_out").sum(),
                                    ref_ok=pl.col("ref_ok").mean().round(4)).sort("region")
    ex = t.unique("exon_id")
    meta = dict(rows=t.height, source="Vex-seq, Adamson et al. 2018 (scottiadamson/Vex-seq processed_files), hg19 lifted to hg38",
                held_out_chroms=HELD, dropped=dict(dropped), exons=ex.height, exons_both_sites_annotated=int(ex["sites_annotated"].sum()),
                exons_ag_gt=int(ex["sites_canonical"].sum()),
                strands=dict(ex["strand"].value_counts().iter_rows()), sdv=int(t["sdv"].sum()), by_region=summ.to_dicts())
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(summ)
    print({k: meta[k] for k in ("rows", "dropped", "exons", "exons_both_sites_annotated", "exons_ag_gt", "strands", "sdv")})


if __name__ == "__main__":
    typer.run(main)
