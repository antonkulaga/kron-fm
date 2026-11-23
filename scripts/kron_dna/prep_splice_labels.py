"""Stranded one-base splice-site labels on a prepared genome's contig layout, as a new label store.

    uv run --group genomics python scripts/kron_dna/prep_splice_labels.py OUT_DIR [--name hg38] [--gtf GTF]

The definition, stated once: for every intron of every GENCODE basic transcript, in the transcript's own direction,
the **donor** is the intron's first base and the **acceptor** its last base. ``OUT_DIR/splice.u8`` has one byte per
contig base, laid out like the genome's ``bases.u8``, with four bits:

    bit 0  donor,    + strand        bit 2  donor,    - strand
    bit 1  acceptor, + strand        bit 3  acceptor, - strand

so a window read on the other strand is the reversed array with bits 0 and 2, and 1 and 3, exchanged. This is not the
prepared genome's own ``labels.u8`` definition (unstranded, the intron's first and last TWO bases), which stays as it
is and is the baseline the earlier splice-delta score used.

Checked before the file is put in place, on both strands: the share of donors whose intron starts with GT (GC is the
known minor class) and of acceptors whose intron ends with AG, read from ``bases.u8`` in the transcript's direction.
A wrong offset or a strand mix-up drops these to about 1 in 16. ``OUT_DIR/splice.json`` records the definition, the
counts, the motif shares per strand, and checksums.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
from pathlib import Path

import numpy as np
import polars as pl
import typer

from kron_core.paths import input_dir, interim_dir
from kron_dna import genome_variants as GV

BITS = {"donor_plus": 0, "acceptor_plus": 1, "donor_minus": 2, "acceptor_minus": 3}
COMPLEMENT = np.array([3, 2, 1, 0, 4], dtype=np.uint8)           # A C G T N -> T G C A N


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def main(out: Path, name: str = "hg38", gtf: Path = typer.Option(None),
         split: Path = typer.Option(None, help="uint8 per base, 1 = training: positives are also counted there for "
                                               "the head's class weights (default: the genome's ntv3/split.u8)")) -> None:
    from kron_dna.prep.features import introns as introns_rule, read_features

    gtf = gtf or input_dir("regulation", "gencode") / "gencode.basic.gtf.gz"
    src = interim_dir("genome", name)
    if src.resolve() in [out.resolve(), *out.resolve().parents]:
        raise typer.BadParameter("OUT_DIR lies inside the prepared genome; choose a new path")
    index = pl.read_parquet(src / "contigs.parquet")
    bases = np.memmap(src / "bases.u8", dtype=np.uint8, mode="r")
    # the shared reader and intron rule (B23); 4 bases at least, as these labels always took
    introns = introns_rule(read_features(gtf), 4).rename({"s": "i0", "e": "i1"})
    chrom = introns["chrom"].to_numpy().astype(str)
    plus = introns["strand"].to_numpy() == "+"
    first, last = introns["i0"].to_numpy(), introns["i1"].to_numpy() - 1      # the intron's two end bases
    g_first, _, _ = GV.locate(index, chrom, first)
    g_last, _, _ = GV.locate(index, chrom, last)
    # an intron counts when both its ends lie in one contig of the layout
    ok = (g_first >= 0) & (g_last >= 0) & (g_last - g_first == last - first)
    labels = np.zeros(len(bases), dtype=np.uint8)
    sites = {"donor_plus": g_first[ok & plus], "acceptor_plus": g_last[ok & plus],
             "donor_minus": g_last[ok & ~plus], "acceptor_minus": g_first[ok & ~plus]}
    for key, at in sites.items():
        np.bitwise_or.at(labels, np.unique(at), np.uint8(1 << BITS[key]))

    # the check, in each transcript's direction: donor + next base = GT (or GC), previous base + acceptor = AG
    def pair(at: np.ndarray, step: int, minus: bool) -> np.ndarray:
        """Two bases as codes read 5' to 3' on the transcript's strand: the site and its neighbour at ``step``."""
        a, b = np.asarray(bases[at]), np.asarray(bases[at + step])
        if minus:
            a, b = COMPLEMENT[a], COMPLEMENT[b]
        return np.stack([a, b] if (step > 0) != minus else [b, a], 1)

    code = {b: i for i, b in enumerate("ACGT")}
    check = {}
    for key, at in sites.items():
        at = np.unique(at)
        minus, donor = key.endswith("minus"), key.startswith("donor")
        # donor: the site and the next intron base; acceptor: the previous intron base and the site
        step = (1 if donor else -1) * (-1 if minus else 1)
        two = pair(at, step, minus)
        if donor:
            share = {"GT": float(((two[:, 0] == code["G"]) & (two[:, 1] == code["T"])).mean()),
                     "GC": float(((two[:, 0] == code["G"]) & (two[:, 1] == code["C"])).mean())}
        else:
            share = {"AG": float(((two[:, 0] == code["A"]) & (two[:, 1] == code["G"])).mean())}
        check[key] = dict(sites=int(len(at)), **share)
        typer.echo(f"{key}: {len(at):,} sites, " + ", ".join(f"{k} {v:.4f}" for k, v in share.items()))
    placed = float(ok.sum()) / max(len(ok), 1)
    if placed < 0.5:        # chromosome names that do not match the layout place nothing: refuse, never pass empty
        raise typer.Exit(f"only {placed:.1%} of {len(ok):,} introns lie on the layout: chromosome names or genome wrong")
    worst = min(min(check[k].get("GT", 1.0) + check[k].get("GC", 0.0), check[k].get("AG", 1.0)) for k in check)
    if not worst >= 0.9:    # NaN (no site of a class) fails too
        raise typer.Exit(f"the canonical motif share is {worst:.3f} for one site class: offsets or strands are wrong")

    out.mkdir(parents=True, exist_ok=True)
    part = out / f"splice.u8.part.{socket.gethostname()}.{os.getpid()}"
    labels.tofile(part)
    os.replace(part, out / "splice.u8")
    sp_ = split or (src / "ntv3" / "split.u8")
    train = None
    if Path(sp_).exists():                            # the class weights count the training split only
        from kron_dna.xheads import train_counts
        tsites, tbases = train_counts(np.memmap(out / "splice.u8", dtype=np.uint8, mode="r"),
                                      np.memmap(sp_, dtype=np.uint8, mode="r"), BITS)
        for k in check:
            check[k]["train_sites"] = tsites[k]
        train = (tbases, str(sp_))
    meta = dict(
        genome=name, total_bases=int(len(bases)), bits=BITS,
        definition="per intron of each GENCODE basic transcript, in the transcript's direction: donor = the intron's "
                   "first base, acceptor = the intron's last base; one base each, per strand",
        other_strand="reverse the window and exchange bits (0, 2) and (1, 3)",
        introns=int(introns.height), introns_placed=int(ok.sum()), check=check,
        **({"train_bases": train[0], "train_split": train[1]} if train else {}),
        annotation_sha256=sha256(gtf), contigs_sha256=sha256(src / "contigs.parquet"),
        splice_sha256=sha256(out / "splice.u8"))
    (out / "splice.json").write_text(json.dumps(meta, indent=1))
    typer.echo(f"{int(ok.sum()):,} of {introns.height:,} introns placed; wrote {out / 'splice.u8'}")


if __name__ == "__main__":
    typer.run(main)
