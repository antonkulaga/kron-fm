"""AlphaGenome's variant scores for the frozen ClinVar-by-region cohort (clinvar_manifest.py), by its API.

    uv run --group baselines python scripts/kron_dna/clinvar_alphagenome.py MANIFEST.parquet OUT_DIR --key KEYFILE \
        [--version 1 | --version 2 --condition UBERON:... --condition ...] [--split held-out] [--limit N]

Scores the manifest's ``in_sample`` rows of ``--split`` with the audited scorer (``variants.alphagenome_scores``):
one row per allele (id, chrom, pos, ref, alt), model version and condition in a cache under ``OUT_DIR``, appended as
results arrive, so a rerun or a killed job continues where it stopped and never reuses a score of another
condition set. Writes ``OUT_DIR/alphagenome<version>.parquet``: the allele key and one column per scorer, lower =
a larger predicted effect; a row the API never answered stays NaN and is counted, not dropped.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import polars as pl
import typer

from kron_dna import variants as V


def main(manifest: Path, out: Path, key: Path = typer.Option(..., help="a file holding the API key"),
         version: int = 1, condition: list[str] = typer.Option([]), split: str = "held-out", limit: int = 0,
         workers: int = 4) -> None:
    rows = pl.read_parquet(manifest).filter(pl.col("in_sample") & (pl.col("split") == split))
    if limit:                                         # a timing pilot: the first rows by position, every region mixed
        rows = rows.sample(n=min(limit, rows.height), seed=0)
    # the scorer adds the "chr" prefix itself
    positions = rows.select("id", pl.col("chrom").str.strip_prefix("chr"), "pos", "ref", "alt")
    out.mkdir(parents=True, exist_ok=True)
    cache = out / f"cache_alphagenome{version}.v2.parquet"
    t0 = time.time()
    scores = V.alphagenome_scores(rows, positions, key.read_text().strip(), cache, workers=workers, version=version,
                                  conditions=tuple(condition))
    table = rows.select("assembly", "chrom", "pos", "ref", "alt", "id", "region", "label", "split").with_columns(
        [pl.Series(name, s) for name, s in scores.items()])
    table.write_parquet(out / f"alphagenome{version}.parquet")
    missing = {name: int(np.isnan(s).sum()) for name, s in scores.items()}
    typer.echo(f"AlphaGenome {version}: {rows.height} alleles, conditions {list(condition) or '-'}, "
               f"{time.time() - t0:.0f} s this run, unanswered per score {missing}")


if __name__ == "__main__":
    typer.run(main)
