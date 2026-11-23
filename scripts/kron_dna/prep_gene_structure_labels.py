"""Stranded per-base gene-structure labels on a prepared genome's contig layout, as a label store for a binary
(multi-label) further head: ``genome-train --xhead genes:OUT_DIR``.

    uv run python scripts/kron_dna/prep_gene_structure_labels.py OUT_DIR --name hg38 --gtf GTF [--basic-only]

Definition, per transcript of the annotation (GENCODE: the ``basic`` tag with ``--basic-only``), in its own strand's
channels (``_plus`` or ``_minus``); a base takes every label any transcript gives it (alternative isoforms overlap, so
the channels are not exclusive):

    tss      the transcript's first base (5' end)             polya    its last base (3' end)
    utr5     an exon base 5' of the CDS                       cds      a CDS base, the stop codon included
    utr3     an exon base 3' of the CDS                       ncexon   an exon base of a transcript with no CDS
    intron   a base between two consecutive exons of the transcript

so a window read on the other strand is the reversed array with each ``_plus`` channel exchanged for its ``_minus``
partner. ``OUT_DIR/labels.u16`` has one uint16 per base of the layout, bit k = channel k (``labels.json``: ``bits``).

Checked before the file is put in place, in each transcript's direction: introns start with GT (or GC) and end with
AG, and coding transcripts' CDS start with ATG where the annotation marks a start codon; offsets or strands that
are wrong drop these to about 1 in 16. ``labels.json`` records the definition, per-channel base counts, the checks and
checksums.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import socket
from pathlib import Path

import numpy as np
import polars as pl
import typer

from kron_core.paths import interim_dir

KINDS = ["tss", "polya", "utr5", "cds", "utr3", "ncexon", "intron"]
BITS = {f"{k}_{s}": i + 7 * j for j, s in enumerate(("plus", "minus")) for i, k in enumerate(KINDS)}
COMPLEMENT = np.array([3, 2, 1, 0, 4], dtype=np.uint8)
CODE = {b: i for i, b in enumerate("ACGT")}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def read_gtf(gtf: Path, basic_only: bool, source: str = "") -> pl.DataFrame:
    """exon, CDS, stop_codon and start_codon rows of a GTF or GFF3 (by name or its ``##gff-version 3`` line): chrom,
    type, start0 (0-based), end (exclusive), strand, tid (GTF transcript_id; GFF3 the row's first Parent)."""
    opener = gzip.open if str(gtf).endswith(".gz") else open
    rows = []
    want = {"exon", "CDS", "stop_codon", "start_codon"}
    tid_re = re.compile(r'transcript_id "([^"]+)"')        # GTF
    parent_re = re.compile(r'(?:^|;)Parent=([^;,]+)')      # GFF3: an exon's or CDS's (first) parent transcript
    gff3 = ".gff" in str(gtf).lower()
    with opener(gtf, "rt") as f:
        for line in f:
            if line.startswith("#"):
                if line.startswith("##gff-version 3"):
                    gff3 = True
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 9 or c[2] not in want or c[6] not in "+-" or (source and c[1] != source):
                continue
            if basic_only and not ('basic' in (re.search(r'(?:^|;)tag=([^;]*)', c[8]) or [None, ""])[1].split(",")
                                       if gff3 else 'tag "basic"' in c[8]):
                continue
            m = (parent_re if gff3 else tid_re).search(c[8])
            if m:
                rows.append((c[0], c[2], int(c[3]) - 1, int(c[4]), c[6], m.group(1)))
    return pl.DataFrame(rows, schema=["chrom", "type", "start0", "end", "strand", "tid"], orient="row")


def intervals(g: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Per label kind, [start0, end) intervals with chrom and strand (points as 1-base intervals)."""
    ex = g.filter(pl.col("type") == "exon")
    cds = (g.filter(pl.col("type").is_in(["CDS", "stop_codon"])).group_by("tid")
           .agg(c0=pl.col("start0").min(), c1=pl.col("end").max()))
    tx = ex.group_by("tid", "chrom", "strand").agg(t0=pl.col("start0").min(), t1=pl.col("end").max())
    plus = pl.col("strand") == "+"
    out = {
        "tss": tx.select("chrom", "strand", s=pl.when(plus).then(pl.col("t0")).otherwise(pl.col("t1") - 1)),
        "polya": tx.select("chrom", "strand", s=pl.when(plus).then(pl.col("t1") - 1).otherwise(pl.col("t0"))),
    }
    out = {k: v.with_columns(e=pl.col("s") + 1) for k, v in out.items()}
    exc = ex.join(cds, on="tid", how="left")
    coding = exc.filter(pl.col("c0").is_not_null())
    out["ncexon"] = exc.filter(pl.col("c0").is_null()).select("chrom", "strand", s="start0", e="end")
    out["cds"] = (g.filter(pl.col("type").is_in(["CDS", "stop_codon"]))
                  .select("chrom", "strand", s="start0", e="end"))
    left = coding.select("chrom", "strand", s="start0", e=pl.min_horizontal("end", "c0"))      # exon part below CDS
    right = coding.select("chrom", "strand", s=pl.max_horizontal("start0", "c1"), e="end")    # exon part above
    left, right = left.filter(pl.col("e") > pl.col("s")), right.filter(pl.col("e") > pl.col("s"))
    out["utr5"] = pl.concat([left.filter(plus), right.filter(~plus)])
    out["utr3"] = pl.concat([right.filter(plus), left.filter(~plus)])
    out["intron"] = (ex.sort("tid", "start0").with_columns(nxt=pl.col("start0").shift(-1).over("tid"))
                     .filter(pl.col("nxt").is_not_null() & (pl.col("nxt") > pl.col("end")))
                     .select("chrom", "strand", s="end", e="nxt"))
    return out


def main(out: Path, name: str = typer.Option(...), gtf: Path = typer.Option(...),
         basic_only: bool = typer.Option(False, help="only transcripts tagged basic (GENCODE)"),
         chrom_prefix: str = typer.Option("", help="prepend to the GTF's chromosome names (e.g. 'chr' for Ensembl)"),
         source: str = typer.Option("", help="only rows of this source (column 2), e.g. WormBase's curated models in "
                                             "its full GFF3")) -> None:
    src = interim_dir("genome", name)
    if src.resolve() in [out.resolve(), *out.resolve().parents]:
        raise typer.BadParameter("OUT_DIR lies inside the prepared genome; choose a new path")
    index = pl.read_parquet(src / "contigs.parquet")
    bases = np.memmap(src / "bases.u8", dtype=np.uint8, mode="r")
    g = read_gtf(gtf, basic_only, source).with_columns(chrom=pl.lit(chrom_prefix) + pl.col("chrom"))
    if g.is_empty():
        raise typer.BadParameter(f"{gtf}: no exon rows" + (" tagged basic" if basic_only else ""))
    iv = intervals(g)
    out.mkdir(parents=True, exist_ok=True)
    part = out / f"labels.u16.part.{socket.gethostname()}.{os.getpid()}"
    labels = np.memmap(part, dtype=np.uint16, mode="w+", shape=(len(bases),))
    counts = {k: 0 for k in BITS}
    chroms = sorted(set(index["chrom"].to_list()))
    for chrom in chroms:
        pieces = index.filter(pl.col("chrom") == chrom).sort("start")
        n = int(pieces["end"].max())
        lab = np.zeros(n, dtype=np.uint16)
        for kind, df in iv.items():
            for strand, sname in (("+", "plus"), ("-", "minus")):
                d = df.filter((pl.col("chrom") == chrom) & (pl.col("strand") == strand))
                if d.is_empty():
                    continue
                s = np.clip(d["s"].to_numpy(), 0, n)
                e = np.clip(d["e"].to_numpy(), 0, n)
                diff = np.zeros(n + 1, dtype=np.int32)
                np.add.at(diff, s, 1)
                np.add.at(diff, e, -1)
                lab[np.cumsum(diff[:-1]) > 0] |= np.uint16(1 << BITS[f"{kind}_{sname}"])
        for r in pieces.iter_rows(named=True):
            seg = lab[r["start"]:r["end"]]
            labels[r["offset"]:r["offset"] + r["end"] - r["start"]] = seg
            for k, b in BITS.items():
                counts[k] += int(np.count_nonzero(seg & np.uint16(1 << b)))
        typer.echo(f"{chrom}: done")
    labels.flush()

    # checks in the transcript's direction, on the layout
    def at_layout(chrom: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """Global positions of (chrom, pos); -1 outside the gap-free contigs (or on a chromosome the layout lacks)."""
        res = np.full(len(pos), -1, dtype=np.int64)
        for c in np.unique(chrom):
            pc = index.filter(pl.col("chrom") == c).sort("start")
            if pc.is_empty():
                continue
            st, en, off = pc["start"].to_numpy(), pc["end"].to_numpy(), pc["offset"].to_numpy()
            sel = np.where(chrom == c)[0]
            k = np.searchsorted(st, pos[sel], side="right") - 1
            ok = (k >= 0) & (pos[sel] < en[np.maximum(k, 0)])
            res[sel[ok]] = off[k[ok]] + pos[sel][ok] - st[k[ok]]
        return res

    def along(chrom, first, strand, steps):
        """Codes of the bases ``first + step`` along the transcript (5'->3', complemented on -) for each step, as
        ``[n, len(steps)]``, and the rows whose bases all lie in the layout."""
        sign = np.where(strand == "+", 1, -1)
        at = np.stack([at_layout(chrom, first + sign * k) for k in steps], 1)
        ok = (at >= 0).all(1)
        x = np.asarray(bases[at[ok].reshape(-1)]).reshape(-1, len(steps))
        x = np.where((strand[ok] == "+")[:, None], x, COMPLEMENT[x])
        return x, ok

    intr = iv["intron"].unique()
    ch, st = intr["chrom"].to_numpy().astype(str), intr["strand"].to_numpy()
    s0, e0 = intr["s"].to_numpy(), intr["e"].to_numpy()
    don, _ = along(ch, np.where(st == "+", s0, e0 - 1), st, (0, 1))          # donor: the first two intron bases
    acc, _ = along(ch, np.where(st == "+", e0 - 1, s0), st, (-1, 0))        # acceptor: the last two
    check = dict(introns=int(len(intr)), introns_checked=int(len(don)),
                 donor_GT=float(((don[:, 0] == CODE["G"]) & (don[:, 1] == CODE["T"])).mean()),
                 donor_GC=float(((don[:, 0] == CODE["G"]) & (don[:, 1] == CODE["C"])).mean()),
                 acceptor_AG=float(((acc[:, 0] == CODE["A"]) & (acc[:, 1] == CODE["G"])).mean()))
    sc = g.filter(pl.col("type") == "start_codon").unique(["chrom", "start0", "strand"])
    if not sc.is_empty():
        cst = sc["strand"].to_numpy()
        first = np.where(cst == "+", sc["start0"].to_numpy(), sc["end"].to_numpy() - 1)
        codon, _ = along(sc["chrom"].to_numpy().astype(str), first, cst, (0, 1, 2))
        check["start_codons"] = int(len(codon))
        check["start_ATG"] = float(((codon[:, 0] == CODE["A"]) & (codon[:, 1] == CODE["T"])
                                    & (codon[:, 2] == CODE["G"])).mean())
    typer.echo(f"checks: {check}")
    placed = check["introns_checked"] / max(check["introns"], 1)
    if placed < 0.5 or not sum(counts.values()):  # names that do not match the layout place nothing: never pass empty
        part.unlink()
        raise typer.Exit(f"only {placed:.1%} of {check['introns']:,} introns lie on the layout: chromosome names "
                         "(--chrom-prefix?) or genome wrong")
    worst = min(check["donor_GT"] + check["donor_GC"], check["acceptor_AG"], check.get("start_ATG", 1.0))
    if not worst >= 0.85:                         # NaN fails too
        part.unlink()
        raise typer.Exit(f"a canonical motif share is {worst:.3f}: offsets or strands are wrong")
    os.replace(part, out / "labels.u16")
    meta = dict(
        genome=name, total_bases=int(len(bases)), dtype="uint16", bits=BITS,
        definition="per transcript (" + ("GENCODE basic" if basic_only else "every transcript") + "), its own strand: "
                   "tss/polya = first/last base; utr5/cds/utr3 = exon bases 5' of, in, 3' of the CDS (stop codon in "
                   "cds); ncexon = exon bases of a transcript without CDS; intron = bases between consecutive exons; "
                   "union over transcripts",
        other_strand="reverse the window and exchange each _plus channel with its _minus partner",
        annotation=str(gtf), annotation_sha256=sha256(gtf), transcripts=int(g["tid"].n_unique()),
        check={**{k: dict(sites=v) for k, v in counts.items()}, "motifs": check},
        contigs_sha256=sha256(src / "contigs.parquet"), labels_sha256=sha256(out / "labels.u16"))
    (out / "labels.json").write_text(json.dumps(meta, indent=1))
    typer.echo(f"{meta['transcripts']:,} transcripts; bases per channel: "
               + ", ".join(f"{k} {v:,}" for k, v in counts.items()))


if __name__ == "__main__":
    typer.run(main)
