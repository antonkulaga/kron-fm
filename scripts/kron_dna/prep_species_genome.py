"""Any species' genome as a prepared genome (``bases.u8`` + ``contigs.parquet``), genome-prepare's layout and rules,
for evaluation genomes whose chromosome names are not human's (genome-prepare keeps chr1-22, X, Y only).

    uv run python scripts/kron_dna/prep_species_genome.py FASTA --name dmel_r6 [--keep 2L+2R+3L+3R+4+X+Y]
        [--min-seq 1000000] [--prefix chr]

Sequences listed by ``--keep`` (all, without it) of at least ``--min-seq`` bases are kept, named ``PREFIX + name``
(so a GTF with ``--chrom-prefix`` of the label scripts matches), split at runs of 50 N or more with contigs under 10 kb
dropped (``genome.contigs_of``, as genome-prepare does). Published only when complete, never over an existing genome.
"""
from __future__ import annotations

import gzip
import os
import shutil
from pathlib import Path

import polars as pl
import typer

from kron_core.paths import interim_dir
from kron_dna import genome as G


def held_pick(lengths: dict, target: float = 0.09) -> list[str]:
    """One or two sequences whose combined share of the genome is closest to ``target`` (deterministic; the largest and the
    smallest sequence are never picked, so neither the main chromosome nor a fragment stands for the held-out set)."""
    names = sorted(lengths, key=lambda k: -lengths[k])
    total = sum(lengths.values())
    chroms = [n for n in names if n.startswith("NC_")] or names   # RefSeq: assembled chromosomes, not scaffolds
    pool = chroms[1:-1] if len(chroms) > 3 else chroms
    sets = [(a,) for a in pool] + [(a, b) for i, a in enumerate(pool) for b in pool[i + 1:]]   # one or two: a genome
    best = min(sets, key=lambda ab: (abs(sum(lengths[x] for x in ab) / total - target), len(ab), ab))  # of few large
    return sorted(best, key=names.index)                                                           # chromosomes needs one


def main(fasta: Path, name: str = typer.Option(...), keep: str = "", min_seq: int = 1_000_000, prefix: str = "",
         held: str = typer.Option("", help="'+'-joined sequence names to hold out, or 'auto' (two whose share is "
                                           "closest to 9 %); written as contigs.parquet's boolean held_out column")) -> None:
    out = interim_dir("genome", name)
    if (out / "bases.u8").exists():
        raise typer.BadParameter(f"{out} already holds a genome; choose another --name")
    want = set(keep.split("+")) if keep else None
    stage = out / f".prepare-{os.getpid()}"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    rows, offset, kept = [], 0, []
    try:
        with (gzip.open if str(fasta).endswith(".gz") else open)(fasta, "rt") as f, open(stage / "bases.u8", "wb") as fh:
            def flush(nm, chunks):
                nonlocal offset
                if nm is None or (want is not None and nm not in want):
                    return
                seq = "".join(chunks)
                if len(seq) < min_seq:
                    return
                codes = G.encode_bases(seq)
                kept.append(nm)
                for a, b in G.contigs_of(codes, 50, 10_000):
                    fh.write(codes[a:b].tobytes())
                    rows.append((prefix + nm, int(a), int(b), offset))
                    offset += int(b - a)
            nm, chunks = None, []
            for line in f:
                if line.startswith(">"):
                    flush(nm, chunks)
                    nm, chunks = line[1:].split()[0], []
                else:
                    chunks.append(line.strip())
            flush(nm, chunks)
        if want is not None and set(kept) != want:
            raise typer.BadParameter(f"{fasta}: --keep names not found: {sorted(want - set(kept))}")
        table = pl.DataFrame(rows, schema=["chrom", "start", "end", "offset"], orient="row")
        if held:
            lens = {c: int(n) for c, n in table.group_by("chrom").agg((pl.col("end") - pl.col("start")).sum()).iter_rows()}
            pick = held_pick(lens) if held == "auto" else [prefix + h for h in held.split("+")]
            missing = sorted(set(pick) - set(lens))
            if missing:
                raise typer.BadParameter(f"--held names not in the genome: {missing}")
            table = table.with_columns(held_out=pl.col("chrom").is_in(pick))
            typer.echo(f"held out: {pick} = {sum(lens[p] for p in pick) / sum(lens.values()):.1%} of the bases")
        table.write_parquet(stage / "contigs.parquet")
        assert (stage / "bases.u8").stat().st_size == offset
        for fn in ("bases.u8", "contigs.parquet"):
            os.replace(stage / fn, out / fn)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    typer.echo(f"{name}: {len(kept)} sequences, {len(rows)} contigs, {offset / 1e6:.1f} Mb -> {out}")


if __name__ == "__main__":
    typer.run(main)
