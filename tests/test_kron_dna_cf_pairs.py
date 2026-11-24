"""The exclusion of the counterfactual training pairs: nothing on the held-out chromosomes, splits by chromosome."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import polars as pl
import pytest

spec = importlib.util.spec_from_file_location("prep_cf_pairs", Path(__file__).parents[1] / "scripts/kron_dna/prep_cf_pairs.py")
CF = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CF)


def rows(chroms: list[str]) -> pl.DataFrame:
    return pl.DataFrame(dict(chrom=chroms, pos=list(range(len(chroms))))).with_columns(
        ref=pl.lit("A"), alt=pl.lit("C"), read_pos=pl.col("pos"), head=pl.lit("chiptf"),
        channel_ids=pl.lit([], dtype=pl.List(pl.String)), effect=pl.lit(1.0), effect_kind=pl.lit("log2"), weight=pl.lit(1.0),
        source=pl.lit("s"))


def test_split_and_check():
    """The exclusion list: chr8 and chr9 are never written; chr3, chr10 and chr16 are pair_test; the rest train."""
    assert CF.HELD == ["chr8", "chr9"] and CF.PAIR_TEST == ["chr3", "chr10", "chr16"]
    t = CF.split_of(rows(["chr1", "chr3", "chr8", "chr9", "chr10", "chr16", "chrX", "chr18", "chr19"]))
    assert dict(zip(t["chrom"], t["split"])) == {"chr1": "train", "chr3": "pair_test", "chr10": "pair_test",
                                                 "chr16": "pair_test", "chrX": "train", "chr18": "train", "chr19": "train"}
    CF.check(t)
    with pytest.raises(AssertionError):                  # a held-out row
        CF.check(pl.concat([t, rows(["chr8"]).with_columns(split=pl.lit("train"))]))
    with pytest.raises(AssertionError):                  # a train row on a pair-test chromosome
        CF.check(t.with_columns(split=pl.lit("train")))


def test_built_tables_keep_the_exclusion():
    """The tables on this machine, where they were built."""
    from kron_core.paths import interim_dir
    d = interim_dir("genome", "hg38").parent.parent / "train" / "cf-pairs-v1"
    files = sorted(d.glob("*.parquet")) if d.exists() else []
    if not files:
        pytest.skip("no built tables here")
    for f in files:
        CF.check(pl.read_parquet(f))
