"""Gene annotations (GTF or GFF3) as one normalised table, and the intron rule, for every label store."""
from __future__ import annotations

import gzip
import re
from pathlib import Path

import polars as pl


def read_features(gtf: Path, basic_only: bool = False, source: str = "") -> pl.DataFrame:
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



def introns(features: pl.DataFrame, min_len: int = 1) -> pl.DataFrame:
    """Distinct introns, 0-based half-open ``[s, e)`` with chrom and strand: the gap between consecutive exons of a
    transcript, at least ``min_len`` bases (splice sites take 4, as their labels always did; gene structure 1)."""
    ex = features.filter(pl.col("type") == "exon")
    return (ex.sort("tid", "start0").with_columns(nxt=pl.col("start0").shift(-1).over("tid"))
            .filter(pl.col("nxt").is_not_null() & (pl.col("nxt") - pl.col("end") >= min_len))
            .select("chrom", "strand", s="end", e="nxt").unique())
