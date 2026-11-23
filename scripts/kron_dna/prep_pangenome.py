"""HPRC release 2 haplotype assemblies as ONE prepared genome (masked bases only, no tracks), for ``--genome``.

    uv run python scripts/kron_dna/prep_pangenome.py part INDEX.csv WORK [--part K --parts N]
    uv run python scripts/kron_dna/prep_pangenome.py assemble INDEX.csv WORK --name hprc_r2

``INDEX.csv`` is the release's ``data_tables/assemblies_release2_v1.0.index.csv`` (one row per haplotype; the two
haplotype-0 reference rows are skipped). ``part`` takes every Nth haplotype: downloads its ``.fa.gz`` (checked against
the release's md5) and its ``chromAlias.txt`` (assembly sequence -> chr1..chr22, X, Y, M), keeps the sequences
assigned to chr1-22, X and Y, names each ``<sample>#<haplotype>#<chr>``, splits them at runs of N as genome-prepare
does (``genome.contigs_of``: gaps of 50 N or more, contigs under 10 kb dropped) and writes ``WORK/parts/<assembly>/``
(``bases.u8``, ``contigs.parquet``); the FASTA is then deleted. A finished part is skipped on a rerun.
``assemble`` concatenates every part, in index order, into ``data/interim/genome/<name>/`` (``bases.u8`` and
``contigs.parquet`` with a boolean ``held_out`` column: chr8 and chr9 of every haplotype, as in hg38), built in a
staging folder and published only when complete. ``WORK/provenance.jsonl`` records each haplotype's accession,
md5, size and contig counts.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import polars as pl
import typer

from kron_core.paths import interim_dir
from kron_dna import genome as G

app = typer.Typer(add_completion=False)
S3 = "https://s3-us-west-2.amazonaws.com/human-pangenomics/"
KEEP = {f"chr{c}" for c in [*map(str, range(1, 23)), "X", "Y"]}
HELD = {"chr8", "chr9"}


def https(s3: str) -> str:
    return S3 + s3.removeprefix("s3://human-pangenomics/")


def fetch(url: str, dest: Path) -> None:
    subprocess.run(["curl", "-sSL", "--fail", "--retry", "5", "--retry-all-errors", "-C", "-", "-o", str(dest), url],
                   check=True)


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def rows_of(index: Path) -> pl.DataFrame:
    return pl.read_csv(index, infer_schema_length=0).filter(pl.col("haplotype").is_in(["1", "2"]))


def alias_url(r: dict) -> str:
    base = r["assembly"].rsplit("/", 1)[0]
    return https(f"{base}/annotation/chrom_assignment/{r['assembly_name']}.chromAlias.txt")


@app.command()
def part(index: Path, work: Path, part: int = 0, parts: int = 1) -> None:
    rows = rows_of(index)
    for k, r in enumerate(rows.iter_rows(named=True)):
        if k % parts != part:
            continue
        name = r["assembly_name"]
        d = work / "parts" / name
        if (d / "contigs.parquet").exists():
            continue
        tmp = work / "tmp" / name
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        t0 = time.time()
        fa, al, sm = tmp / "a.fa.gz", tmp / "alias.txt", tmp / "a.md5"
        fetch(https(r["assembly"]), fa)
        fetch(https(r["assembly_md5"]), sm)
        fetch(alias_url(r), al)
        want = sm.read_text().split()[0]
        got = md5(fa)
        if got != want:
            raise SystemExit(f"{name}: md5 {got} != the release's {want}")
        alias = {}
        for line in al.read_text().splitlines():
            if line and not line.startswith("#"):
                f = line.split("\t")
                alias[f[0]] = f[1]
        tag = f"{r['sample_id']}#{r['haplotype']}#"
        out_rows, offset, kept = [], 0, {}
        (tmp / "out").mkdir()
        name_ = seq_ = None
        with gzip.open(fa, "rt") as f, open(tmp / "out" / "bases.u8", "wb") as fh:
            def flush(nm, chunks):
                nonlocal offset
                chrom = alias.get(nm)
                if chrom not in KEEP:
                    return
                codes = G.encode_bases("".join(chunks))
                kept[chrom] = kept.get(chrom, 0) + 1
                for a, b in G.contigs_of(codes, 50, 10_000):
                    fh.write(codes[a:b].tobytes())
                    out_rows.append((tag + chrom, a, b, offset, chrom in HELD))
                    offset += b - a
            for line in f:
                if line.startswith(">"):
                    if name_ is not None:
                        flush(name_, seq_)
                    name_, seq_ = line[1:].split()[0], []
                else:
                    seq_.append(line.strip())
            if name_ is not None:
                flush(name_, seq_)
        if not out_rows:
            raise SystemExit(f"{name}: no sequence assigned to chr1-22, X or Y (alias names {list(alias)[:3]})")
        pl.DataFrame(out_rows, schema=["chrom", "start", "end", "offset", "held_out"], orient="row").write_parquet(
            tmp / "out" / "contigs.parquet")
        assert (tmp / "out" / "bases.u8").stat().st_size == offset
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(d, ignore_errors=True)
        os.replace(tmp / "out", d)                    # published whole: contigs.parquet marks it done
        rec = dict(assembly=name, sample=r["sample_id"], haplotype=r["haplotype"], accession=r["genbank_accession"],
                   method=r["assembly_method"], md5=got, fasta_bytes=fa.stat().st_size, bases=offset,
                   contigs=len(out_rows), chromosomes=len(kept), multi=[c for c, n in kept.items() if n > 1],
                   seconds=round(time.time() - t0, 1))
        with open(work / "provenance.jsonl", "a") as fp:
            fp.write(json.dumps(rec) + "\n")
        shutil.rmtree(tmp, ignore_errors=True)
        typer.echo(f"{name}: {offset / 1e9:.2f} Gb, {len(out_rows)} contigs on {len(kept)} chromosomes, "
                   f"{rec['seconds']:.0f} s")


@app.command()
def assemble(index: Path, work: Path, name: str = "hprc_r2") -> None:
    rows = rows_of(index)
    names = rows["assembly_name"].to_list()
    missing = [n for n in names if not (work / "parts" / n / "contigs.parquet").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(names)} haplotypes not prepared yet: {missing[:5]}")
    out = interim_dir("genome", name)
    out.mkdir(parents=True, exist_ok=True)
    stage = out / f".assemble-{os.getpid()}"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir()
    try:
        tables, offset = [], 0
        with open(stage / "bases.u8", "wb") as fh:
            for n in names:
                d = work / "parts" / n
                t = pl.read_parquet(d / "contigs.parquet").with_columns(pl.col("offset") + offset)
                with open(d / "bases.u8", "rb") as src:
                    shutil.copyfileobj(src, fh, 1 << 24)
                offset += (d / "bases.u8").stat().st_size
                tables.append(t)
        index_ = pl.concat(tables)
        index_.write_parquet(stage / "contigs.parquet")
        assert (stage / "bases.u8").stat().st_size == offset == int((index_["offset"] + index_["end"]
                                                                       - index_["start"]).max())
        for f in ("bases.u8", "contigs.parquet"):
            os.replace(stage / f, out / f)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    held = index_.filter(pl.col("held_out"))
    typer.echo(f"{len(names)} haplotypes, {index_.height:,} contigs, {offset / 1e9:.1f} Gb; held out (chr8, chr9): "
               f"{int((held['end'] - held['start']).sum()) / 1e9:.1f} Gb -> {out}")


if __name__ == "__main__":
    app()
