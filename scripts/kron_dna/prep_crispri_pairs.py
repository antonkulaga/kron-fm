"""CRISPRi enhancer-gene pairs on hg38 for the knockout benchmark: the ENCODE-rE2G CRISPR benchmark
(Gschwind et al. 2025; EngreitzLab/CRISPR_comparison ``resources/crispr_data``), which pools Fulco 2019,
Gasperini 2019, Schraivogel 2020 and the rest in K562 (``training_K562``) and five further cell types
(``heldout_5_cell_types``).

    uv run python scripts/kron_dna/prep_crispri_pairs.py OUT.parquet TSV.gz [TSV.gz ...] [--half 524288]

One row per element-gene pair: element (chrom, start, end), target gene (symbol, Ensembl id, TSS), effect size,
adjusted p, ``regulated`` (significant and negative, the benchmark's positive), power at a 25 % effect, cell type,
``study`` (the original screen: Fulco2019, Gasperini2019, ...; ``Dataset`` names the pooled release) and the set.
Flags: ``valid`` (the benchmark's ValidConnection is TRUE: not a promoter or the gene's own exon), ``held_out``
(element or TSS on hg38 chr8/chr9: score these apart, they are the trained models' held-out split),
``in_context`` (element and TSS both within ``--half`` of their midpoint, so one window centred there holds
both; 524,288 for the 1,048,576 context).
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import polars as pl
import typer

HELD = ["chr8", "chr9"]


def study_of(ref: str, dataset: str) -> str:
    m = re.search(r"from (\w+) et al\., (\d{4})", ref or "")
    if m:
        return f"{m.group(1)}{m.group(2)}"
    m = re.match(r"(\w+) et al\., (\d{4})", ref or "")
    return f"{m.group(1)}{m.group(2)}" if m else dataset


def main(out: Path, tsv: list[Path], half: int = 524_288) -> None:
    parts = []
    for p in tsv:
        s = re.search(r"combined_data\.(.+)\.GRCh38", p.name)
        parts.append(pl.read_csv(p, separator="\t", infer_schema_length=0).with_columns(
            set=pl.lit(s.group(1) if s else p.name)))
    t = pl.concat(parts)
    t = t.select(
        chrom="chrom", start=pl.col("chromStart").cast(pl.Int64), end=pl.col("chromEnd").cast(pl.Int64),
        gene="measuredGeneSymbol", gene_id="measuredGeneEnsemblId", chrom_tss="chrTSS",
        tss=pl.col("startTSS").cast(pl.Int64), effect=pl.col("EffectSize").cast(pl.Float64),
        p_adj=pl.col("pValueAdjusted").cast(pl.Float64), regulated=pl.col("Regulated") == "TRUE",
        significant=pl.col("Significant") == "TRUE", power25=pl.col("PowerAtEffectSize25").cast(pl.Float64),
        valid=pl.col("ValidConnection") == "TRUE", valid_reason="ValidConnection", cell_type="CellType",
        study=pl.struct("Reference", "Dataset").map_elements(lambda r: study_of(r["Reference"], r["Dataset"]),
                                                             return_dtype=pl.String),
        dataset="Dataset", reference="Reference", set="set",
        distance=pl.col("distanceToTSS").cast(pl.Int64), chromatin="elementChromatinCategory")
    lo = pl.min_horizontal("start", "tss")
    hi = pl.max_horizontal("end", pl.col("tss") + 1)
    t = t.with_columns(held_out=pl.col("chrom").is_in(HELD) | pl.col("chrom_tss").is_in(HELD),
                       in_context=(pl.col("chrom") == pl.col("chrom_tss")) & (hi - lo <= 2 * half),
                       centre=((lo + hi) // 2))
    out.parent.mkdir(parents=True, exist_ok=True)
    t.write_parquet(out)
    summ = (t.group_by("set", "study", "cell_type").agg(
        pairs=pl.len(), regulated=pl.col("regulated").sum(), in_context=pl.col("in_context").sum(),
        held_out=pl.col("held_out").sum(), held_out_regulated=(pl.col("held_out") & pl.col("regulated")).sum())
        .sort("set", "study", "cell_type"))
    meta = dict(rows=t.height, half=half, held_out_chroms=HELD,
                inputs={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in tsv},
                by_study=summ.to_dicts())
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    with pl.Config(tbl_rows=40):
        print(summ)


if __name__ == "__main__":
    typer.run(main)
