"""``prep_hic.py`` over every downloaded file of a selection, one contact store per biosample.

    uv run python scripts/kron_dna/prep_hic_batch.py SELECT.tsv FILES_DIR OUT_ROOT [--workers 4] [--prefix encode]

``SELECT.tsv`` has the columns ``file``, ``assembly`` and ``biosample`` (the Hi-C selections' layout). A file not yet
in ``FILES_DIR`` (still downloading) is skipped, as is a biosample whose store already has its ``hic.json``, so the
command is run again as files arrive. Stores go to ``OUT_ROOT/<genome>/<prefix>-<biosample>-5kb`` (genome: hg38 for
GRCh38, else the assembly's name; the biosample in lower case, other characters than letters and digits as ``-``).
A mouse file is read for chr1..chr19 and chrX. A file ``prep_hic.py`` refuses (no 5 kb level, no SCALE vector) is
listed in ``OUT_ROOT/<genome>/<prefix>-failed.tsv`` with its last line of output, and the run goes on.
"""
from __future__ import annotations

import csv
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import typer

GENOME = {"GRCh38": "hg38"}
CHROMS = {"mm10": ",".join([f"chr{k}" for k in range(1, 20)] + ["chrX"])}


def slug(biosample: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", biosample.lower()).strip("-")


def plan(rows: list[dict], files: Path, out: Path, prefix: str) -> list[tuple[dict, Path, Path]]:
    """(row, the .hic file, the store's directory) for every row whose file is there and whose store is not."""
    todo, seen = [], set()
    for r in rows:
        store = out / GENOME.get(r["assembly"], r["assembly"]) / f"{prefix}-{slug(r['biosample'])}-5kb"
        if store in seen:
            raise ValueError(f"two rows give {store}")
        seen.add(store)
        hic = files / f"{r['file']}.hic"
        if hic.exists() and not (store / "hic.json").exists():
            todo.append((r, hic, store))
    return todo


def main(select: Path, files: Path, out: Path, workers: int = 4, prefix: str = "encode") -> None:
    rows = list(csv.DictReader(select.open(), delimiter="\t"))
    todo = plan(rows, files, out, prefix)
    typer.echo(f"{len(rows)} rows, {len(todo)} to convert")
    script = Path(__file__).with_name("prep_hic.py")

    def one(item: tuple[dict, Path, Path]) -> tuple[dict, Path, int, str]:
        r, hic, store = item
        cmd = [sys.executable, str(script), str(hic), str(store), "--check"]
        if r["assembly"] in CHROMS:
            cmd += ["--chroms", CHROMS[r["assembly"]]]
        p = subprocess.run(cmd, capture_output=True, text=True)
        last = (p.stdout + p.stderr).strip().splitlines()[-1:] or [""]
        return r, store, p.returncode, last[0]

    failed: dict[Path, list[str]] = {}
    with ThreadPoolExecutor(workers) as pool:
        for r, store, code, last in pool.map(one, todo):
            typer.echo(f"{r['file']} {r['biosample']}: {'ok' if code == 0 else 'FAILED'} {last[:300]}")
            if code:
                failed.setdefault(store.parent, []).append(f"{r['file']}\t{r['biosample']}\t{last[:500]}")
    for folder, lines in failed.items():
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / f"{prefix}-failed.tsv").open("a") as f:
            f.write("\n".join(lines) + "\n")
    typer.echo(f"done: {len(todo) - sum(map(len, failed.values()))} stores, {sum(map(len, failed.values()))} failed")


if __name__ == "__main__":
    typer.run(main)
