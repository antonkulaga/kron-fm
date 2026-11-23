"""lentiMPRA activity of native regulatory elements on hg38 (Agarwal et al. 2025, Nature 639:411; Zenodo 10558183):
~680,000 200-bp candidate enhancers and promoters tested in HepG2, K562 and WTC11, with the paper's ten folds.

    uv run --with openpyxl python scripts/kron_dna/prep_lentimpra.py OUT.parquet DESIGN.xlsx ACTIVITY.xlsx

``DESIGN`` is the record's "Table S3 - large-scale lib design" (per cell line: name, category, hg38 coordinates,
strand, the 230-nt oligo) and ``ACTIVITY`` its "Table S6 - folds and performance" (per cell line: fold, sequence id,
observed log2(RNA/DNA), on the forward orientation). One row per (cell line, measured element): ``cell``, ``name``,
``category``, ``chrom``, ``start``, ``end`` (hg38, 0-based half-open as given), ``strand``, ``activity``, ``fold``,
``pos`` (the element's middle: where a track head is read; ``ref`` and ``alt`` are empty, for ``variant_bench.py
score --ref-only``), ``held_out`` (hg38 chr8/chr9). Elements without hg38 coordinates (synthetic controls, hg19
elements with no perfect hg38 match) are left out and counted. This is not a counterfactual set: the element is
scored in its genomic place, the assay measures it in a reporter.
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import typer

HELD = ["chr8", "chr9"]


def main(out: Path, design: Path, activity: Path) -> None:
    import openpyxl
    d = openpyxl.load_workbook(design, read_only=True)
    a = openpyxl.load_workbook(activity, read_only=True)
    parts, counts = [], {}
    for cell in ("HepG2", "K562", "WTC11"):
        rows = list(d[f"{cell} large-scale"].iter_rows(values_only=True))
        h = next(i for i, r in enumerate(rows) if r[0] == "name")
        col = {str(c): k for k, c in enumerate(rows[h]) if c is not None}
        des = {}
        for r in rows[h + 1:]:
            if r[0] is not None:
                des[str(r[0]).removesuffix("_F")] = r
        act = list(a[f"Fold assignment, {cell}"].iter_rows(values_only=True))[1:]
        recs, lost = [], 0
        for fold, sid, y in act:
            r = des.get(str(sid))
            c = r[col["chr.hg38"]] if r is not None else None
            if r is None or c in (None, "NA") or r[col["start.hg38"]] in (None, "NA") or y is None:
                lost += 1
                continue
            c = str(c)
            recs.append(dict(cell=cell, name=str(sid), category=str(r[col["category"]]),
                             chrom=c if c.startswith("chr") else "chr" + c, start=int(r[col["start.hg38"]]),
                             end=int(r[col["stop.hg38"]]), strand=str(r[col["str.hg38"]]), activity=float(y), fold=int(fold)))
        counts[cell] = dict(measured=len(act), kept=len(recs), without_hg38=lost)
        parts.append(pl.DataFrame(recs))
    t = pl.concat(parts).with_columns(pos=(pl.col("start") + pl.col("end")) // 2, ref=pl.lit(""), alt=pl.lit(""),
                                      held_out=pl.col("chrom").is_in(HELD))
    out.parent.mkdir(parents=True, exist_ok=True)
    t.write_parquet(out)
    summ = t.group_by("cell", "category").agg(elements=pl.len(), held_out=pl.col("held_out").sum(),
                                              mean_activity=pl.col("activity").mean().round(3)).sort("cell", "category")
    out.with_suffix(".json").write_text(json.dumps(dict(
        rows=t.height, source="Agarwal et al. 2025, Zenodo 10558183, Tables S3 and S6", held_out_chroms=HELD,
        cells=counts, sites=t.select(pl.struct("chrom", "pos").n_unique()).item(), by_category=summ.to_dicts()), indent=1))
    with pl.Config(tbl_rows=40):
        print(summ)
    print(counts)


if __name__ == "__main__":
    typer.run(main)
