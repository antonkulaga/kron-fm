"""Exon-skipping variants on hg38 for the splice head: MFASS (Cheung et al. 2019, Mol Cell 73:183; KosuriLab/MFASS,
``processed_data/snv/snv_data_clean.txt``): 27,733 rare ExAC SNVs in or next to 2,198 short human exons, each
exon with its flanking introns (170 nt) placed in a three-exon reporter and its inclusion measured by sorting.

    uv run python scripts/kron_dna/prep_mfass.py OUT.parquet snv_data_clean.txt [--name hg38]

One row per variant with a measurement: ``chrom``, ``pos`` (0-based), ``ref``, ``alt`` (forward strand of the
assembly, whichever strand the file gives them on), ``strand`` of the gene, ``exon_start``/``exon_end`` (hg38,
0-based half-open: the construct's start plus ``intron1_len``, the low-coordinate flank on either strand), ``exon_mid`` and ``half`` (where and how wide
a per-base head is read: the exon and 3 bases beyond each end, so both splice sites), ``region`` (exon,
upstr_intron, downstr_intron), ``dpsi`` (the file's ``v2_dpsi``: change of the inclusion index against the natural
exon), ``sdv`` (its ``strong_lof``: a splice-disrupting variant, dPSI <= -0.5), ``nat_index`` (the natural exon's
inclusion), ``ref_ok`` and ``held_out`` (hg38 chr8/chr9).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import typer

from kron_core.paths import interim_dir

HELD = ["chr8", "chr9"]
COMP = {"A": "T", "C": "G", "G": "C", "T": "A"}


def main(out: Path, table: Path, name: str = "hg38") -> None:
    t = pl.read_csv(table, separator="\t", null_values=["NA"], infer_schema_length=40000)
    t = t.filter((pl.col("category") == "mutant") & pl.col("v2_dpsi").is_not_null() & pl.col("strong_lof").is_not_null()
                 & pl.col("snp_position_hg38_0based_start").is_not_null())
    s, e = pl.col("start_hg38_0based"), pl.col("end_hg38_0based")
    ex_lo = s + pl.col("intron1_len")          # intron1 is the low-coordinate flank on both strands (checked against
                                               # the splice labels: 99.5 % and 99.9 % of exon ends are annotated sites)
    t = t.select(
        id="id", exon_id="ensembl_id", chrom="chr", pos=pl.col("snp_position_hg38_0based_start").cast(pl.Int64),
        ref_given="ref_allele", alt_given="alt_allele", strand="strand", exon_start=ex_lo.cast(pl.Int64),
        exon_end=(ex_lo + pl.col("exon_len")).cast(pl.Int64), region="label", dpsi="v2_dpsi", sdv="strong_lof",
        nat_index="nat_v2_index", construct_start=s.cast(pl.Int64), construct_end=e.cast(pl.Int64))
    src = interim_dir("genome", name)
    bases = np.memmap(src / "bases.u8", dtype=np.uint8, mode="r")
    contigs = pl.read_parquet(src / "contigs.parquet").select("chrom", c_start="start", c_end="end", offset="offset")
    j = t.with_row_index("row").join(contigs, on="chrom").filter(
        (pl.col("pos") >= pl.col("c_start")) & (pl.col("pos") < pl.col("c_end")))
    here = np.full(t.height, 4, dtype=np.uint8)
    here[j["row"].to_numpy()] = bases[(j["offset"] + j["pos"] - j["c_start"]).to_numpy()]
    letter = np.array(list("ACGTN"))[here]
    ref_g, alt_g = t["ref_given"].to_list(), t["alt_given"].to_list()
    ref, alt, how = [], [], []
    for g, r, a in zip(letter, ref_g, alt_g):                   # the file's alleles: forward, or the gene's strand
        if r == g:
            ref.append(r), alt.append(a), how.append("forward")
        elif a == g:                                            # the assembly carries the file's alternative
            ref.append(r), alt.append(a), how.append("swapped")
        elif COMP.get(r) == g:
            ref.append(g), alt.append(COMP.get(a, "N")), how.append("complement")
        else:
            ref.append(r), alt.append(a), how.append("mismatch")
    t = t.with_columns(ref=pl.Series(ref), alt=pl.Series(alt), allele_strand=pl.Series(how)).with_columns(
        ref_ok=pl.col("allele_strand").is_in(["forward", "complement"]), exon_mid=(pl.col("exon_start") + pl.col("exon_end")) // 2,
        half=(pl.col("exon_end") - pl.col("exon_start") + 1) // 2 + 3, held_out=pl.col("chrom").is_in(HELD))
    out.parent.mkdir(parents=True, exist_ok=True)
    t.write_parquet(out)
    summ = t.group_by("region").agg(variants=pl.len(), sdv=pl.col("sdv").sum(), exons=pl.col("exon_id").n_unique(),
                                    held_out=pl.col("held_out").sum(), ref_ok=pl.col("ref_ok").mean().round(4)).sort("region")
    meta = dict(rows=t.height, source="MFASS, Cheung et al. 2019 (KosuriLab/MFASS snv_data_clean.txt)", held_out_chroms=HELD,
                exons=t["exon_id"].n_unique(), sdv=int(t["sdv"].sum()),
                allele_strand=dict(t["allele_strand"].value_counts().iter_rows()), by_region=summ.to_dicts())
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(summ)
    print(meta["allele_strand"], "exons", meta["exons"], "sdv", meta["sdv"], "held-out sdv", int(t.filter("held_out")["sdv"].sum()))


if __name__ == "__main__":
    typer.run(main)
