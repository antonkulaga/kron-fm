"""Saturation mutagenesis MPRAs of disease-associated promoters and enhancers on hg38 (Kircher et al. 2019; the
CAGI5 regulation challenge data): the ``GRCh38_<experiment>.tsv`` files of the paper's OSF project (75b2m,
``Variants/GRCh38``), every single-base change and 1-bp deletion of 20 elements, each with its measured effect on
reporter expression.

    uv run python scripts/kron_dna/prep_mpra_satmut.py OUT.parquet DIR_WITH_TSVS [--name hg38]

One row per (experiment, variant): ``chrom``, ``pos`` (0-based), ``ref``, ``alt`` (empty for a deletion),
``effect`` (the file's ``Value``: log2 change of expression, alt over ref), ``p_value``, ``tags`` (barcodes behind
it; the paper's analyses keep >= 10: ``confident``), and

- ``experiment`` (the file), ``element`` (the locus), ``kind`` (promoter, enhancer), ``cell`` (the cell line the
  library was assayed in; ``human_cell`` false for the mouse lines), ``primary`` (one experiment per element and
  cell line: the first replicate, PKLR at 48 h, no flipped insert and no siRNA arm; this file's choice);
- ``centre`` and ``half``: the element's middle and half-length, where a track head is read
  (``variant_bench.py score --read-col centre``);
- ``ref_ok`` (the assembly's base is ``ref``) and ``held_out`` (hg38 chr8/chr9: the MYC and FOXE1 elements).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import typer

from kron_core.paths import interim_dir

HELD = ["chr8", "chr9"]
# experiment: (element, kind, cell line, primary)       (Kircher et al. 2019, Table 1 and Methods)
EXPERIMENTS = {
    "F9": ("F9", "promoter", "HepG2", True), "GP1BA": ("GP1BA", "promoter", "HEL 92.1.7", True),
    "HBB": ("HBB", "promoter", "HEL 92.1.7", True), "HBG1": ("HBG1", "promoter", "HEL 92.1.7", True),
    "HNF4A": ("HNF4A", "promoter", "HEK293T", True), "LDLR": ("LDLR", "promoter", "HepG2", True),
    "LDLR.2": ("LDLR", "promoter", "HepG2", False), "MSMB": ("MSMB", "promoter", "HEK293T", True),
    "PKLR-24h": ("PKLR", "promoter", "K562", False), "PKLR-48h": ("PKLR", "promoter", "K562", True),
    "TERT-HEK": ("TERT", "promoter", "HEK293T", True), "TERT-GBM": ("TERT", "promoter", "SF7996", True),
    "TERT-GAa": ("TERT", "promoter", "SF7996", False), "TERT-GSc": ("TERT", "promoter", "SF7996", False),
    "FOXE1": ("FOXE1", "promoter", "HeLa", True),
    "BCL11A": ("BCL11A", "enhancer", "HEL 92.1.7", True), "IRF4": ("IRF4", "enhancer", "SK-MEL-28", True),
    "IRF6": ("IRF6", "enhancer", "HaCaT", True), "MYCrs6983267": ("MYC rs6983267", "enhancer", "HEK293T", True),
    "MYCrs11986220": ("MYC rs11986220", "enhancer", "LNCaP", True), "RET": ("RET", "enhancer", "Neuro-2a", True),
    "SORT1": ("SORT1", "enhancer", "HepG2", True), "SORT1.2": ("SORT1", "enhancer", "HepG2", False),
    "SORT1-flip": ("SORT1", "enhancer", "HepG2", False), "TCF7L2": ("TCF7L2", "enhancer", "MIN6", True),
    "UC88": ("UC88", "enhancer", "Neuro-2a", True), "ZFAND3": ("ZFAND3", "enhancer", "MIN6", True),
    "ZRSh-13": ("ZRS", "enhancer", "NIH/3T3", True), "ZRSh-13h2": ("ZRS", "enhancer", "NIH/3T3", False),
}
MOUSE = {"Neuro-2a", "MIN6", "NIH/3T3"}


def main(out: Path, data: Path, name: str = "hg38") -> None:
    parts = []
    for f in sorted(data.glob("GRCh38_*.tsv")):
        exp = f.stem.removeprefix("GRCh38_")
        if exp not in EXPERIMENTS:
            raise typer.BadParameter(f"{f.name}: an experiment this script does not know")
        element, kind, cell, primary = EXPERIMENTS[exp]
        t = pl.read_csv(f, separator="\t", comment_prefix="#", infer_schema_length=0, null_values=["NA"])
        t = t.filter(pl.col("Value").is_not_null())             # a few variants have no estimate
        parts.append(t.select(
            chrom="chr" + pl.col("Chromosome"), pos=pl.col("Position").cast(pl.Int64) - 1, ref="Ref",
            alt=pl.col("Alt").replace("-", ""), effect=pl.col("Value").cast(pl.Float64),
            p_value=pl.col("P-Value").cast(pl.Float64), tags=pl.col("Tags").cast(pl.Int64),
            experiment=pl.lit(exp), element=pl.lit(element), kind=pl.lit(kind), cell=pl.lit(cell),
            human_cell=pl.lit(cell not in MOUSE), primary=pl.lit(primary)))
    t = pl.concat(parts)
    span = t.group_by("element").agg(lo=pl.col("pos").min(), hi=pl.col("pos").max() + 1)   # one read site per locus
    t = t.join(span, on="element").with_columns(
        centre=(pl.col("lo") + pl.col("hi")) // 2, half=(pl.col("hi") - pl.col("lo") + 1) // 2,
        confident=pl.col("tags") >= 10, held_out=pl.col("chrom").is_in(HELD)).drop("lo", "hi")
    src = interim_dir("genome", name)
    bases = np.memmap(src / "bases.u8", dtype=np.uint8, mode="r")
    contigs = pl.read_parquet(src / "contigs.parquet").select("chrom", c_start="start", c_end="end", offset="offset")
    j = t.with_row_index("row").join(contigs, on="chrom").filter(
        (pl.col("pos") >= pl.col("c_start")) & (pl.col("pos") < pl.col("c_end")))
    here = np.full(t.height, 4, dtype=np.uint8)
    here[j["row"].to_numpy()] = bases[(j["offset"] + j["pos"] - j["c_start"]).to_numpy()]
    code = t["ref"].replace_strict({"A": 0, "C": 1, "G": 2, "T": 3}, default=9, return_dtype=pl.Int64).to_numpy()
    t = t.with_columns(ref_ok=pl.Series(here == code))
    out.parent.mkdir(parents=True, exist_ok=True)
    t.write_parquet(out)
    summ = t.group_by("experiment", "element", "kind", "cell", "primary", "chrom", "centre", "half").agg(
        variants=pl.len(), confident=pl.col("confident").sum(), deletions=(pl.col("alt") == "").sum(),
        significant=((pl.col("p_value") < 1e-5) & pl.col("confident")).sum(), ref_ok=pl.col("ref_ok").mean().round(4),
        held_out=pl.col("held_out").first()).sort("kind", "experiment")
    out.with_suffix(".json").write_text(json.dumps(dict(
        rows=t.height, source="Kircher et al. 2019, OSF 75b2m Variants/GRCh38", held_out_chroms=HELD,
        experiments=summ.to_dicts()), indent=1))
    with pl.Config(tbl_rows=40, tbl_cols=20, tbl_width_chars=220):
        print(summ)


if __name__ == "__main__":
    typer.run(main)
