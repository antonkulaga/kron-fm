"""One contact map per biosample from ENCODE's intact Hi-C, as the selection ``encode_fetch.py`` downloads.

    uv run python scripts/kron_dna/hic_select.py OUT.tsv [--assembly GRCh38] [--biosamples K562,HepG2] [--max-gb 0]

Intact Hi-C is ENCODE's uniformly processed set: every experiment went through the same pipeline, so the maps are
comparable. Taken: released experiments whose biosamples carry no treatment and no genetic modification; per
experiment the ``mapping quality thresholded contact matrix`` (``.hic``) over the most biological replicates; per
biosample the experiment whose file is largest (the deepest map). ``--biosamples`` keeps the named ones,
``--max-gb`` leaves out files above a size. Columns as ``encode_select.py`` writes them: i, experiment, file,
assay, biosample, target, output_type, size, bio_reps, url, channel (the biosample), plus classification.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import polars as pl
import typer

API = "https://www.encodeproject.org"
OUTPUT = "mapping quality thresholded contact matrix"


def get(path: str) -> list[dict]:
    out = subprocess.run(["curl", "-sS", "--fail", "--retry", "5", "--retry-all-errors", "--max-time", "300",
                          "-H", "Accept: application/json", API + path], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)["@graph"]


def plain(exp: dict) -> bool:
    """No treatment and no genetic modification on any replicate's biosample."""
    for r in exp.get("replicates", []):
        b = r.get("library", {}).get("biosample", {})
        if b.get("treatments") or b.get("genetic_modifications") or b.get("applied_modifications"):
            return False
    return True


def choose(exps: list[dict], files: list[dict]) -> list[dict]:
    """Per biosample of the untreated experiments, the largest of their best files (most replicates, then size)."""
    ok = {e["@id"]: e for e in exps if plain(e)}
    best: dict[str, dict] = {}
    for f in files:
        if f["dataset"] not in ok:
            continue
        key = (len(f.get("biological_replicates", [])), f["file_size"])
        if f["dataset"] not in best or key > best[f["dataset"]][0]:
            best[f["dataset"]] = (key, f)
    per: dict[str, dict] = {}
    for d, (_, f) in best.items():
        name = ok[d]["biosample_ontology"]["term_name"]
        if name not in per or f["file_size"] > per[name]["size"]:
            per[name] = dict(experiment=ok[d]["accession"], file=f["accession"], assay="intact Hi-C", biosample=name,
                             target="", output_type=OUTPUT, size=f["file_size"],
                             bio_reps=",".join(str(x) for x in sorted(f.get("biological_replicates", []))),
                             url=API + f["href"], channel=name,
                             classification=ok[d]["biosample_ontology"].get("classification", ""))
    return [dict(i=i, **r) for i, r in enumerate(sorted(per.values(), key=lambda r: r["biosample"]))]


def main(out: Path, assembly: str = "GRCh38", biosamples: str = "", max_gb: float = 0.0) -> None:
    q = "&assay_title=intact+Hi-C&status=released&limit=all&format=json"
    exps = get(f"/search/?type=Experiment{q}&field=accession&field=biosample_ontology.term_name"
               "&field=biosample_ontology.classification&field=replicates.library.biosample.treatments"
               "&field=replicates.library.biosample.genetic_modifications"
               "&field=replicates.library.biosample.applied_modifications")
    files = get(f"/search/?type=File{q}&assembly={assembly}&file_format=hic&output_type={OUTPUT.replace(' ', '+')}"
                "&field=accession&field=dataset&field=file_size&field=biological_replicates&field=href")
    rows = choose(exps, files)
    keep = {b.strip() for b in biosamples.split(",") if b.strip()}
    rows = [r for r in rows if (not keep or r["biosample"] in keep) and (not max_gb or r["size"] <= max_gb * 1e9)]
    for i, r in enumerate(rows):
        r["i"] = i
    out.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(out, separator="\t")
    typer.echo(f"{len(rows)} maps, {sum(r['size'] for r in rows) / 1e12:.2f} TB -> {out}")


if __name__ == "__main__":
    typer.run(main)
