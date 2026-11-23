"""Allele-specific TF binding on hg38 for the variant benchmark: ADASTRA (Abramov et al. 2021), release BillCipher,
its cell-type x TF tables (``adastra.cltf.bill_cipher.zip``, unpacked: one ``<UNIPROT_NAME>@<cell type>.tsv`` per
pair, every row an SNV whose ChIP-seq reads favour one allele at 5 % FDR).

    uv run python scripts/kron_dna/prep_asb.py OUT.parquet RELEASE_DATA_DIR --uniprot UNIPROT.tsv
        [--store interim tracks/hg38/chip-tf-v1-bin128] [--name hg38]

One row per (SNV, TF, cell type): ``chrom``, ``pos`` (0-based), ``ref``, ``alt``, ``rsid``, ``tf`` (gene symbol, by
the UniProt entry name; ``--uniprot`` is UniProt's table of entry name and primary gene name), ``cell`` (the
TF-ChIP store's biosample when ADASTRA's cell type names one, else null), the allele-wise effect sizes and FDRs,
``side`` (the favoured allele: the smaller FDR) and ``effect`` (log2, positive when the alternative allele binds
more: ``es_alt`` on the alt side, ``-es_ref`` on the ref side), the HOCOMOCO motif columns (``motif_fc``: the
sequence-only baseline), coverage, and

- ``channels``: the store's tracks to read, ``TF|cell[|experiment]``; ``match`` says how they were found:
  ``tf_cell`` (the pair itself is a track), ``tf`` (the factor is in the store in other cell types only: all its
  tracks), ``none``;
- ``ref_ok``: the assembly's base at ``pos`` is ``ref``;
- ``held_out``: on hg38 chr8/chr9, the trained models' held-out split (score these apart).
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import typer

from kron_core.paths import interim_dir

HELD = ["chr8", "chr9"]
NUM = ["mean_BAD", "total_cover", "n_aggregated", "es_mean_ref", "es_mean_alt", "fdrp_bh_ref", "fdrp_bh_alt",
       "motif_log_pref", "motif_log_palt", "motif_fc"]


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def cell_of(adastra: str, ours: dict) -> str | None:
    """ADASTRA's cell type (``WA01__H1__human_embryonic_stem_cells_``) -> the store's biosample (``H1``): the name
    before the first ``__``, or the first word of a later part, compared without case and punctuation."""
    parts = adastra.split("__")
    for cand in [parts[0], adastra] + [p.split("_")[0] for p in parts[1:]]:
        if norm(cand) in ours:
            return ours[norm(cand)]
    return None


def main(out: Path, data: Path, uniprot: Path = typer.Option(...), store: Path = None, name: str = "hg38") -> None:
    src = interim_dir("genome", name)
    store = store or src.parent.parent / "tracks" / name / "chip-tf-v1-bin128"
    ids = json.loads((store / "binned.json").read_text())["ids"]
    by_pair, by_tf = defaultdict(list), defaultdict(list)
    for i in ids:
        tf, cell = i.split("|")[:2]
        by_pair[(tf, cell)].append(i)
        by_tf[tf].append(i)
    cells = {norm(c): c for _, c in by_pair}
    u = pl.read_csv(uniprot, separator="\t")
    sym = dict(zip(u["Entry Name"], u["Gene Names (primary)"]))
    parts, unnamed = [], []
    for f in sorted(data.glob("*.tsv")):
        entry, adastra = f.stem.split("@", 1)
        tf = sym.get(entry)
        if tf is None:
            unnamed.append(entry)
            continue
        t = pl.read_csv(f, separator="\t", infer_schema_length=0)
        if not t.height:
            continue
        cell = cell_of(adastra, cells)
        ch = by_pair.get((tf, cell)) or by_tf.get(tf) or []
        match = "tf_cell" if (tf, cell) in by_pair else "tf" if tf in by_tf else "none"
        parts.append(t.with_columns(tf=pl.lit(tf), cell=pl.lit(cell, dtype=pl.String), adastra_cell=pl.lit(adastra),
                                    match=pl.lit(match), channels=pl.lit(ch, dtype=pl.List(pl.String))))
    t = pl.concat(parts).with_columns(pl.col(NUM).cast(pl.Float64, strict=False))
    alt_side = pl.col("fdrp_bh_alt") < pl.col("fdrp_bh_ref")
    t = t.select(
        chrom="#chr", pos=pl.col("pos").cast(pl.Float64).cast(pl.Int64) - 1, ref="ref", alt="alt", rsid="ID",
        tf="tf", cell="cell", adastra_cell="adastra_cell", match="match", channels="channels",
        es_ref="es_mean_ref", es_alt="es_mean_alt", fdr_ref="fdrp_bh_ref", fdr_alt="fdrp_bh_alt",
        side=pl.when(alt_side).then(pl.lit("alt")).otherwise(pl.lit("ref")),
        effect=pl.when(alt_side).then(pl.col("es_mean_alt")).otherwise(-pl.col("es_mean_ref")),
        motif_fc="motif_fc", motif_conc="motif_conc", motif_log_pref="motif_log_pref", motif_log_palt="motif_log_palt",
        n_aggregated=pl.col("n_aggregated").cast(pl.Int64), total_cover="total_cover", mean_bad="mean_BAD",
        repeat="repeat_type", novel=pl.col("novel") == "True",
        held_out=pl.col("#chr").is_in(HELD))
    bases = np.memmap(src / "bases.u8", dtype=np.uint8, mode="r")
    contigs = pl.read_parquet(src / "contigs.parquet").select("chrom", c_start="start", c_end="end", offset="offset")
    j = t.with_row_index("row").join(contigs, on="chrom").filter(
        (pl.col("pos") >= pl.col("c_start")) & (pl.col("pos") < pl.col("c_end")))
    at = (j["offset"] + j["pos"] - j["c_start"]).to_numpy()
    here = np.full(t.height, 4, dtype=np.uint8)
    here[j["row"].to_numpy()] = bases[at]
    code = t["ref"].replace_strict({"A": 0, "C": 1, "G": 2, "T": 3}, default=9, return_dtype=pl.Int64).to_numpy()
    t = t.with_columns(ref_ok=pl.Series(here == code))
    out.parent.mkdir(parents=True, exist_ok=True)
    t.write_parquet(out)
    summ = t.group_by("match").agg(
        rows=pl.len(), variants=pl.struct("chrom", "pos", "alt").n_unique(), pairs=pl.struct("tf", "adastra_cell").n_unique(),
        tfs=pl.col("tf").n_unique(), alt_side=(pl.col("side") == "alt").mean().round(3),
        with_motif=pl.col("motif_fc").is_not_null().sum(), held_out=pl.col("held_out").sum(),
        ref_ok=pl.col("ref_ok").mean().round(4)).sort("match")
    meta = dict(rows=t.height, source="ADASTRA BillCipher, cell-type x TF tables (5 % FDR)", store=str(store.name),
                held_out_chroms=HELD, entries_without_gene=sorted(set(unnamed)), by_match=summ.to_dicts())
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(summ)


if __name__ == "__main__":
    typer.run(main)
