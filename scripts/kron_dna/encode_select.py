"""One signal bigWig per ENCODE experiment for an assembly, as the selection ``encode_fetch.py`` downloads.

    uv run python scripts/kron_dna/encode_select.py OUT.tsv --assembly mm10

Per assay, one output type (the human selections' choice): ChIP-seq and ATAC-seq ``fold change over control``,
DNase-seq ``read-depth normalized signal``, RNA-seq (total and polyA plus) the two strands of ``signal of unique
reads``. Per experiment, the file (for RNA-seq, the plus and minus pair) over the most biological replicates (the
pooled one when there is one), ties to the lowest accession. Released files only; single-cell, single-nucleus and
microRNA assays are not taken. Columns as the human selections: i, experiment, file, assay, biosample, target,
output_type, size, bio_reps, url.
"""
from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from pathlib import Path

import polars as pl
import typer

API = "https://www.encodeproject.org"
WANT = {
    "ChIP-seq": ["fold change over control"],
    "ATAC-seq": ["fold change over control"],
    "DNase-seq": ["read-depth normalized signal"],
    "RNA-seq": ["plus strand signal of unique reads", "minus strand signal of unique reads"],
    "polyA plus RNA-seq": ["plus strand signal of unique reads", "minus strand signal of unique reads"],
    "total RNA-seq": ["plus strand signal of unique reads", "minus strand signal of unique reads"],
}
FIELDS = ["accession", "dataset", "assay_term_name", "output_type", "file_size", "biological_replicates",
          "target.label", "biosample_ontology.term_name", "href", "status"]


def search(assembly: str) -> list[dict]:
    q = (f"{API}/search/?type=File&status=released&file_format=bigWig&assembly={assembly}&limit=all&format=json"
         + "".join(f"&field={f}" for f in FIELDS))
    out = subprocess.run(["curl", "-sS", "--fail", "--retry", "5", "--retry-all-errors", "--max-time", "900",
                          "-H", "Accept: application/json", q], capture_output=True, text=True, check=True).stdout
    return json.loads(out)["@graph"]


def main(out: Path, assembly: str = typer.Option(...)) -> None:
    files = search(assembly)
    by_exp: dict = defaultdict(list)
    for f in files:
        a = f.get("assay_term_name")
        a = "/".join(a) if isinstance(a, list) else a
        if a in WANT and f.get("output_type") in WANT[a]:
            by_exp[(a, f["dataset"])].append(f)
    rows = []
    for (assay, exp), fs in sorted(by_exp.items()):
        reps = defaultdict(dict)                 # replicate set -> {output_type: file}
        for f in fs:
            reps[tuple(sorted(f.get("biological_replicates") or []))][f["output_type"]] = f
        whole = [(r, d) for r, d in reps.items() if len(d) == len(WANT[assay])]
        if not whole:
            continue
        r, d = sorted(whole, key=lambda x: (-len(x[0]), min(f["accession"] for f in x[1].values())))[0]
        for ot in WANT[assay]:
            f = d[ot]
            rows.append(dict(experiment=exp.strip("/").split("/")[-1], file=f["accession"], assay=assay,
                             biosample=(f.get("biosample_ontology") or {}).get("term_name", ""),
                             target=(f.get("target") or {}).get("label", ""), output_type=ot,
                             size=int(f.get("file_size") or 0), bio_reps=",".join(map(str, r)),
                             url=f"{API}{f['href']}"))
    df = pl.DataFrame(rows).with_row_index("i")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(out, separator="\t")
    summ = df.group_by("assay").agg(pl.len().alias("files"), pl.col("experiment").n_unique().alias("experiments"),
                                    (pl.col("size").sum() / 1e9).round(1).alias("GB"))
    typer.echo(summ)
    typer.echo(f"{df.height} files, {df['size'].sum() / 1e12:.2f} TB -> {out}")


if __name__ == "__main__":
    typer.run(main)
