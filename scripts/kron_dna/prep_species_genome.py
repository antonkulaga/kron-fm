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


def main(fasta: Path, name: str = typer.Option(...), keep: str = "", min_seq: int = 1_000_000, prefix: str = "") -> None:
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
        pl.DataFrame(rows, schema=["chrom", "start", "end", "offset"], orient="row").write_parquet(stage / "contigs.parquet")
        assert (stage / "bases.u8").stat().st_size == offset
        for fn in ("bases.u8", "contigs.parquet"):
            os.replace(stage / fn, out / fn)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    typer.echo(f"{name}: {len(kept)} sequences, {len(rows)} contigs, {offset / 1e6:.1f} Mb -> {out}")


if __name__ == "__main__":
    typer.run(main)
