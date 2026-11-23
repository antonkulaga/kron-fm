"""A further genome's training split that holds out what is orthologous to the primary genome's held-out
chromosomes, so conserved labels there cannot leak into the primary's held-out scores.

    uv run python scripts/kron_dna/prep_syntenic_split.py CHAIN OUT_DIR --name mm10 [--held chr8+chr9]
        [--own chr8+chr9] [--merge 20000]

``CHAIN`` is a UCSC chain from the primary assembly (target) to the further one (query), reciprocal best
(``hg38.mm10.rbest.chain.gz``): one-to-one orthology. Every aligned block whose target is a ``--held`` chromosome
(hg38 chr8, chr9) marks its query interval; intervals closer than ``--merge`` bases are joined (the unaligned bases
between syntenic blocks are held out with them). Those intervals, and the further genome's own ``--own``
chromosomes (a held-out set native to it), are split code 4; every other base of the prepared genome's layout is
1 (training). Writes ``OUT_DIR/split.u8`` (uint8 per base of the layout) and ``split.json`` (counts per query
chromosome, the chain's checksum).
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
import typer

from kron_core.paths import interim_dir


def blocks(chain: Path, held: set[str]):
    """(query chrom, start, end) of every aligned block whose target chrom is in ``held``, forward coordinates."""
    out = []
    with gzip.open(chain, "rt") as f:
        cur = None
        for line in f:
            p = line.split()
            if not p:
                cur = None
                continue
            if p[0] == "chain":
                t_name, q_name, q_size, q_strand, q_start = p[2], p[7], int(p[8]), p[9], int(p[10])
                cur = dict(keep=t_name in held, q=q_name, size=q_size, strand=q_strand, qpos=q_start)
                continue
            if cur is None:
                continue
            size = int(p[0])
            if cur["keep"]:
                a, b = cur["qpos"], cur["qpos"] + size
                if cur["strand"] == "-":
                    a, b = cur["size"] - b, cur["size"] - a
                out.append((cur["q"], a, b))
            cur["qpos"] += size + (int(p[2]) if len(p) == 3 else 0)
    return out


def main(chain: Path, out: Path, name: str = typer.Option(...), held: str = "chr8+chr9", own: str = "chr8+chr9",
         merge: int = 20_000) -> None:
    src = interim_dir("genome", name)
    idx = pl.read_parquet(src / "contigs.parquet")
    total = int((idx["end"] - idx["start"]).sum())
    split = np.ones(total, dtype=np.uint8)
    bl = pl.DataFrame(blocks(chain, set(held.split("+"))), schema=["chrom", "a", "b"], orient="row").sort("chrom", "a")
    merged = []
    for c, d in bl.group_by("chrom", maintain_order=True):
        a_, b_ = d["a"].to_numpy(), d["b"].to_numpy()
        s, e = int(a_[0]), int(b_[0])
        for x, y in zip(a_[1:], b_[1:]):
            if x - e <= merge:
                e = max(e, int(y))
            else:
                merged.append((c[0], s, e))
                s, e = int(x), int(y)
        merged.append((c[0], s, e))
    own_set = set(own.split("+")) if own else set()
    per = {}
    for r in idx.iter_rows(named=True):
        o, cs, ce = r["offset"], r["start"], r["end"]
        if r["chrom"] in own_set:
            split[o:o + ce - cs] = 4
            continue
        for c, s, e in merged:
            if c != r["chrom"] or e <= cs or s >= ce:
                continue
            a, b = max(s, cs), min(e, ce)
            split[o + a - cs:o + b - cs] = 4
            per[c] = per.get(c, 0) + b - a
    out.mkdir(parents=True, exist_ok=True)
    split.tofile(out / "split.u8")
    meta = dict(genome=name, total_bases=total, codes={"1": "training", "4": "held out"},
                held_out_bases=int((split == 4).sum()), held_out_share=float((split == 4).mean()),
                syntenic_to=held, own_held_out=own, merge=merge, intervals=len(merged),
                syntenic_bases_per_chrom=dict(sorted(per.items(), key=lambda kv: -kv[1])),
                chain=str(chain), chain_sha256=hashlib.sha256(chain.read_bytes()).hexdigest())
    (out / "split.json").write_text(json.dumps(meta, indent=1))
    typer.echo(f"{meta['held_out_share']:.1%} of {total:,} bases held out ({len(merged):,} syntenic intervals + "
               f"{own or 'no'} own); syntenic bases by chromosome: "
               + ", ".join(f"{c} {v / 1e6:.1f} Mb" for c, v in list(meta["syntenic_bases_per_chrom"].items())[:8]))


if __name__ == "__main__":
    typer.run(main)
