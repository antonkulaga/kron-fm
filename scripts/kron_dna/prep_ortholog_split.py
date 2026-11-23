"""A further genome's training split that holds out its orthologs of the primary's held-out genes, where no synteny
chain exists (birds, reptiles, fish, insects): conserved exon-intron structure there would otherwise supervise the
genes head on what the primary's held-out score reads (hg38 chr8, chr9).

    uv run python scripts/kron_dna/prep_ortholog_split.py GTF OUT_DIR --name tgut_btaegut7 \
        --gene-info Homo_sapiens.gene_info.gz --orthologs gene_orthologs.gz [--held 8+9] [--flank 0]

A gene of ``GTF`` (a RefSeq annotation of the genome, as the label scripts read it) is held out when any of:

- NCBI calls it the ortholog of a human gene on a ``--held`` chromosome: its own ``db_xref "NCBIOrtholog:<GeneID>"``,
  or a pair in ``--orthologs`` (NCBI Gene's ``gene_orthologs.gz``, both directions);
- its symbol, upper-cased, is the symbol of such a human gene (``--gene-info``: NCBI Gene's ``Homo_sapiens.gene_info``,
  every type but ``biological-region``), or is one after dropping a teleost paralog's ``a``/``b`` (``sox9a``) or
  Xenopus's ``.L``/``.S`` when the full symbol names no human gene at all.

Its span (gene start to end, both strands alike, widened by ``--flank``) is split code 4, as are the contigs the
genome's ``contigs.parquet`` marks ``held_out`` (its own held-out chromosomes); every other base is 1. Writes
``OUT_DIR/split.u8`` (uint8 per base of the layout, for ``--xhead-genome-split``) and ``split.json`` (counts by
reason, the inputs' checksums).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
from bisect import bisect_right
from pathlib import Path

import numpy as np
import polars as pl
import typer

from kron_core.paths import interim_dir

ATTR = re.compile(r'(\w+) "([^"]*)"')


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def human_genes(gene_info: Path, held: set[str]) -> tuple[set[int], set[str], set[str]]:
    """(GeneIDs on the held chromosomes, their symbols upper-cased, every human symbol upper-cased)."""
    ids, sym, every = set(), set(), set()
    with gzip.open(gene_info, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if p[0] != "9606" or p[9] == "biological-region":
                continue
            every.add(p[2].upper())
            if p[6] in held:
                ids.add(int(p[1]))
                sym.add(p[2].upper())
    return ids, sym, every


def ortholog_pairs(path: Path, human_ids: set[int]) -> set[int]:
    """Other species' GeneIDs NCBI pairs with any of ``human_ids`` (every species; the GTF's ids pick them)."""
    out = set()
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            t1, g1, _, t2, g2 = line.rstrip("\n").split("\t")
            if t1 == "9606" and int(g1) in human_ids:
                out.add(int(g2))
            elif t2 == "9606" and int(g2) in human_ids:
                out.add(int(g1))
    return out


def symbol_hit(s: str, held_sym: set[str], every: set[str]) -> bool:
    u = s.upper()
    if u in held_sym:
        return True
    if u in every:
        return False
    for suf in (".L", ".S"):
        if u.endswith(suf) and u[:-2] in held_sym:
            return True
    return len(u) > 2 and u[-1] in "AB" and u[:-1] in held_sym


def genes_of(gtf: Path) -> list[dict]:
    out = []
    with (gzip.open if str(gtf).endswith(".gz") else open)(gtf, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 9 or p[2] != "gene":
                continue
            kv = ATTR.findall(p[8])
            a = {k: v for k, v in kv}
            xref = [v for k, v in kv if k == "db_xref"]
            gid = next((int(x.split(":")[1]) for x in xref if x.startswith("GeneID:")), None)
            orth = {int(x.split(":")[1]) for x in xref if x.startswith("NCBIOrtholog:")}
            out.append(dict(chrom=p[0], start=int(p[3]) - 1, end=int(p[4]), symbol=a.get("gene", a.get("gene_id", "")),
                            gene_id=gid, orth=orth))
    return out


def main(gtf: Path, out: Path, name: str = typer.Option(...), gene_info: Path = typer.Option(...),
         orthologs: Path = typer.Option(None), held: str = "8+9", flank: int = 0, chrom_prefix: str = "") -> None:
    src = interim_dir("genome", name)
    idx = pl.read_parquet(src / "contigs.parquet")
    total = int((idx["end"] - idx["start"]).sum())
    split = np.ones(total, dtype=np.uint8)
    own = 0
    if "held_out" in idx.columns:
        for r in idx.filter(pl.col("held_out")).iter_rows(named=True):
            split[r["offset"]:r["offset"] + r["end"] - r["start"]] = 4
            own += r["end"] - r["start"]
    h_ids, h_sym, every = human_genes(gene_info, set(held.split("+")))
    paired = ortholog_pairs(orthologs, h_ids) if orthologs else set()
    genes = genes_of(gtf)
    if not genes:
        raise typer.BadParameter(f"{gtf}: no gene rows")
    by = {}
    for r in idx.sort("chrom", "start").iter_rows(named=True):
        by.setdefault(r["chrom"], []).append((r["start"], r["end"], r["offset"]))
    starts = {c: [x[0] for x in v] for c, v in by.items()}
    why = dict(ncbi_ortholog_tag=0, ncbi_ortholog_pair=0, symbol=0)
    held_genes, placed, unplaced, masked = [], 0, 0, np.zeros(total, dtype=bool)
    for g in genes:
        r = ("ncbi_ortholog_tag" if g["orth"] & h_ids else
             "ncbi_ortholog_pair" if g["gene_id"] in paired else
             "symbol" if symbol_hit(g["symbol"], h_sym, every) else None)
        if r is None:
            continue
        why[r] += 1
        held_genes.append(g["symbol"])
        c = chrom_prefix + g["chrom"]
        if c not in by:
            unplaced += 1
            continue
        placed += 1
        a, b = g["start"] - flank, g["end"] + flank
        for k in range(max(bisect_right(starts[c], a) - 1, 0), len(by[c])):
            cs, ce, o = by[c][k]
            if cs >= b:
                break
            x, y = max(a, cs), min(b, ce)
            if x < y:
                masked[o + x - cs:o + y - cs] = True
    split[masked] = 4
    out.mkdir(parents=True, exist_ok=True)
    split.tofile(out / "split.u8")
    meta = dict(genome=name, total_bases=total, codes={"1": "training", "4": "held out"},
                held_out_bases=int((split == 4).sum()), held_out_share=float((split == 4).mean()),
                own_held_out_bases=int(own), ortholog_span_bases=int(masked.sum()),
                ortholog_span_share=float(masked.mean()), genes=len(genes), held_genes=len(held_genes),
                held_genes_placed=placed, held_genes_unplaced=unplaced, by_reason=why,
                human_held_chromosomes=held, human_held_gene_ids=len(h_ids), flank=flank,
                inputs={str(p): sha256(p) for p in (gtf, gene_info, orthologs) if p})
    (out / "split.json").write_text(json.dumps(meta, indent=1))
    typer.echo(f"{name}: {meta['held_out_share']:.1%} held out ({own / total:.1%} own chromosomes, "
               f"{meta['ortholog_span_share']:.1%} in {placed:,} ortholog spans: {why})")


if __name__ == "__main__":
    typer.run(main)
