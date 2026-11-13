"""NCBI feature tables (efetch rettype=ft) of mitochondrial genomes -> one row per CDS, rRNA, tRNA and D-loop:
accession, kind, gene, start (0-based), end (exclusive), strand. Joined pieces (a feature over the circle's origin) keep
their first and last coordinate.

usage: uv run python scripts/kron_dna/mito/features.py FEATURES.ft OUT.parquet
"""
import re
import sys

import polars as pl

GENES = {"ATP6", "ATP8", "COX1", "COX2", "COX3", "CYTB", "ND1", "ND2", "ND3", "ND4", "ND4L", "ND5", "ND6"}
ALIAS = {"COI": "COX1", "COII": "COX2", "COIII": "COX3", "CO1": "COX1", "CO2": "COX2", "CO3": "COX3", "COB": "CYTB",
         "NAD1": "ND1", "NAD2": "ND2", "NAD3": "ND3", "NAD4": "ND4", "NAD4L": "ND4L", "NAD5": "ND5", "NAD6": "ND6",
         "ATPASE6": "ATP6", "ATPASE8": "ATP8", "CYT B": "CYTB", "NADH4L": "ND4L"}


def gene_name(kind: str, quals: dict[str, str]) -> str | None:
    if kind == "D-loop":
        return "D-loop"
    if kind == "rRNA":
        p = quals.get("product", "").lower()
        small = "12s" in p or "small" in p or "rrns" in p or p.startswith("s-r")
        large = "16s" in p or "large" in p or "rrnl" in p or p.startswith("l-r")
        return "rRNA-12S" if small else "rRNA-16S" if large else None
    if kind == "tRNA":
        return quals.get("product")
    g = quals.get("gene", "").upper().replace(" ", "")
    g = ALIAS.get(g, g)
    if g in GENES:
        return g
    p = quals.get("product", "").upper()
    for pat, name in ((r"CYTOCHROME C OXIDASE SUBUNIT (III|3)\b", "COX3"), (r"CYTOCHROME C OXIDASE SUBUNIT (II|2)\b", "COX2"),
                      (r"CYTOCHROME C OXIDASE SUBUNIT (I|1)\b", "COX1"), (r"CYTOCHROME B", "CYTB"),
                      (r"ATP SYNTHASE (F0 )?SUBUNIT (6|A)\b", "ATP6"), (r"ATP SYNTHASE (F0 )?SUBUNIT (8)\b", "ATP8"),
                      (r"NADH DEHYDROGENASE SUBUNIT (4L)\b", "ND4L"), (r"NADH DEHYDROGENASE SUBUNIT ([1-6])\b", None)):
        m = re.search(pat, p)
        if m:
            return name or f"ND{m.group(1)}"
    return None


def parse(path: str) -> pl.DataFrame:
    rows, acc, cur = [], None, None

    def flush():
        if cur and cur["kind"] in ("CDS", "rRNA", "tRNA", "D-loop"):
            name = gene_name(cur["kind"], cur["quals"])
            if name:
                a, b = cur["spans"][0][0], cur["spans"][-1][1]
                strand = "-" if a > b else "+"
                lo, hi = min(a, b), max(a, b)
                rows.append({"accession": acc, "kind": cur["kind"], "gene": name, "start": lo - 1, "end": hi,
                             "strand": strand})

    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith(">Feature"):
            flush()
            cur = None
            acc = line.split("|")[1]
            continue
        parts = line.split("\t")
        if parts[0] and len(parts) >= 3 and parts[2]:          # a new feature's first span
            flush()
            cur = {"kind": parts[2], "spans": [], "quals": {}}
        if parts[0] and len(parts) >= 2 and cur is not None:
            cur["spans"].append((int(parts[0].strip("<>^")), int(parts[1].strip("<>^"))))
        elif len(parts) >= 5 and cur is not None:
            cur["quals"].setdefault(parts[3], parts[4])
    flush()
    return pl.DataFrame(rows)


if __name__ == "__main__":
    t = parse(sys.argv[1])
    t.write_parquet(sys.argv[2])
    per = t.filter(pl.col("kind") != "tRNA").group_by("accession").agg(pl.col("gene").n_unique().alias("n"))
    print(f"{t.height:,} features, {t['accession'].n_unique():,} genomes; genomes with all 13 CDS + 2 rRNA:",
          per.filter(pl.col("n") >= 15).height)
    print(t.filter(pl.col("kind") != "tRNA")["gene"].value_counts().sort("gene"))
