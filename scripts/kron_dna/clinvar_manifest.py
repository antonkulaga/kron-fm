"""Freeze the ClinVar-by-region cohort of ``kron-dna genome-clinvar`` as one manifest, before any scoring.

    uv run python scripts/kron_dna/clinvar_manifest.py OUT_DIR [--name hg38] [--per-class 2000] [--seed 0]

Writes ``OUT_DIR/manifest.parquet`` (one row per reviewed pathogenic or benign ClinVar SNV, placed or not) and
``OUT_DIR/manifest.json`` (checksums, versions, settings, counts). Placement, regions and the per-class sample are
``genome-clinvar``'s own (``genome_variants.place`` and the CLI's sampling expression), so ``in_sample`` marks the
rows that command scores with the same settings. Rows the placement drops stay in the manifest with ``excluded``
set, so every region keeps its full denominator.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import polars as pl
import typer

from kron_core.paths import input_dir, interim_dir
from kron_dna import genome_variants as GV
from kron_dna import variants as V


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def header_value(path: Path, key: str) -> str | None:
    with gzip.open(path, "rt") as f:
        for line in f:
            if not line.startswith("#"):
                return None
            if key in line:
                return line.strip().lstrip("#")


def main(out: Path, vcf: Path = typer.Option(None), name: str = "hg38", val_chroms: str = "chr8,chr9",
         near: int = 10, per_class: int = 2000, seed: int = 0, gtf: Path = typer.Option(None)) -> None:
    vcf = vcf or input_dir("variants", "clinvar") / "clinvar.vcf.gz"
    gtf = gtf or input_dir("regulation", "gencode") / "gencode.basic.gtf.gz"
    src = interim_dir("genome", name)
    bases = np.memmap(src / "bases.u8", dtype=np.uint8, mode="r")
    labels = np.memmap(src / "labels.u8", dtype=np.uint8, mode="r")
    index = pl.read_parquet(src / "contigs.parquet")
    held = set(val_chroms.split(","))

    variants = V.read_clinvar(vcf)
    assert variants["id"].n_unique() == variants.height, "ClinVar ids repeat"
    table, dropped = GV.place(variants, index, bases, labels, held, near)
    # genome-clinvar's sample, the same expression on the same row order
    sample = table.filter(pl.int_range(pl.len()).shuffle(seed).over("region", "label", "held_out") < per_class) \
        if per_class else table
    placed = table.with_columns(
        splice_distance=pl.Series(GV.splice_distance(labels, table["g"].to_numpy())),
        in_sample=pl.col("id").is_in(sample["id"].to_list()), excluded=pl.lit(None, dtype=pl.Utf8))

    # the rows placement dropped, with the reason
    rest = variants.join(table.select("id"), on="id", how="anti")
    chrom = rest["chrom"].cast(pl.Utf8).to_numpy().astype(str)
    chrom = np.where(np.char.startswith(chrom, "chr"), chrom, np.char.add("chr", chrom))
    g, _, _ = GV.locate(index, chrom, rest["pos"].to_numpy().astype(np.int64) - 1)
    rest = rest.with_columns(chrom=pl.Series(chrom), held_out=pl.Series(np.isin(chrom, list(held))),
                             excluded=pl.Series(np.where(g < 0, "outside the contigs", "reference mismatch")),
                             region=pl.lit(None, dtype=pl.Utf8), splice_distance=pl.lit(None, dtype=pl.Int64),
                             in_sample=pl.lit(False))
    cols = ["assembly", "chrom", "pos", "ref", "alt", "id", "genes", "clnsig", "label", "region", "held_out",
            "split", "splice_distance", "in_sample", "excluded"]
    both = pl.concat([placed.select(rest.columns), rest.select(rest.columns)], how="vertical_relaxed")
    both = both.with_columns(assembly=pl.lit("GRCh38"),
                             split=pl.when("held_out").then(pl.lit("held-out")).otherwise(pl.lit("training")))
    both = both.select(cols).sort("chrom", "pos", "ref", "alt", "id")
    assert both.height == variants.height and dropped == {
        k: int((both["excluded"] == k).sum()) for k in dropped}, "exclusion counts disagree with place()"

    out.mkdir(parents=True, exist_ok=True)
    both.write_parquet(out / "manifest.parquet")
    counts = (both.group_by("split", "region", "label")
              .agg(pl.len().alias("n"), pl.col("in_sample").sum().alias("in_sample"))
              .sort("split", "region", "label"))
    meta = {
        "assembly": "GRCh38", "vcf_sha256": sha256(vcf), "vcf_fileDate": header_value(vcf, "fileDate"),
        "annotation": header_value(gtf, "description"), "gtf_sha256": sha256(gtf),
        "labels_sha256": sha256(src / "labels.u8"), "bases_sha256": sha256(src / "bases.u8"),
        "prepared_genome": name, "val_chroms": sorted(held), "near": near, "per_class": per_class, "seed": seed,
        "filter": "SNV; review status criteria_provided, reviewed_by_expert_panel or practice_guideline; "
                  "pathogenic/likely pathogenic = 1, benign/likely benign = 0; GENEINFO present",
        "code": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "rows": both.height, "placed": table.height, "dropped": dropped, "sampled": sample.height,
        "manifest_sha256": sha256(out / "manifest.parquet"), "counts": counts.to_dicts(),
    }
    (out / "manifest.json").write_text(json.dumps(meta, indent=1))
    with pl.Config(tbl_rows=-1):
        typer.echo(str(counts))
    typer.echo(f"{both.height} rows, {table.height} placed, dropped {dropped}, {sample.height} in the sample")


if __name__ == "__main__":
    typer.run(main)
