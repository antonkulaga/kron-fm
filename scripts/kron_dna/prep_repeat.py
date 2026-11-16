"""Build ``repeat.u8`` for a prepared KronGenome genome from its soft-masked source FASTA (for ``genome-train
--repeat-channel``).

``repeat.u8`` is laid out exactly like the genome's ``bases.u8``: one byte per base, at the same offset, over the
contigs of ``contigs.parquet`` (chrom, start, end, offset), 1 where the source FASTA has a lower-case letter (soft-masked
by RepeatMasker / the assembly team) and 0 elsewhere. Every contig's bases are re-encoded from the FASTA with the
repository's own ``kron_dna.genome.encode_bases`` and compared with ``bases.u8`` before anything is written, so a
FASTA or name mapping that does not match the prepared layout stops the script.

Stack: polars-bio ``scan_fasta`` (lazy) for the FASTA, polars LazyFrames for the contig table and the NCBI assembly
report, duckdb for the contig-to-record join, pathlib. polars-bio keeps the letters' case; that is checked against a
direct read of the first record's first megabase, and the script stops if it does not hold.

The array is built in ``--stage-dir`` (on the same file system) and then moved to ``<genome-dir>/repeat.u8``, which
must not exist yet: nothing else in the genome directory is touched. A JSON summary (lower-case fraction overall, per
chromosome, training vs held-out) goes to the stage directory.

``--check-only`` compares every contig with ``bases.u8`` and reports each mismatch (count of differing bases, first
position) and the lower-case fractions, writing nothing but the summary JSON in the stage directory; exit 0 when every
contig matches, 2 otherwise.

Run:  uv run --no-sync --with duckdb python prep_repeat.py --fasta F --genome-dir D [--report R] --stage-dir S
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
import polars_bio as pb

from kron_dna import genome as G

REPORT_COLS = ["sequence_name", "role", "molecule", "location", "genbank", "relationship", "refseq", "unit",
               "length", "ucsc"]
LOWER = np.zeros(256, dtype=np.uint8)
LOWER[ord("a"):ord("z") + 1] = 1


def record_map(fasta: Path, report: Path | None) -> pl.LazyFrame:
    """FASTA record name -> chromosome name of contigs.parquet. With an NCBI assembly report: GenBank accession ->
    'chr' + molecule for assembled molecules (as prep_assembly.py named them); without: the record name itself."""
    names = pb.scan_fasta(str(fasta)).select("name")
    if report is None:
        return names.with_columns(pl.col("name").alias("chrom"))
    rep = (pl.scan_csv(report, separator="\t", comment_prefix="#", has_header=False, new_columns=REPORT_COLS,
                       infer_schema=False)
           .filter(pl.col("role") == "assembled-molecule")
           .select(pl.col("genbank").alias("name"), (pl.lit("chr") + pl.col("molecule")).alias("chrom")))
    return names.join(rep, on="name", how="inner")


def raw_prefix(fasta: Path, n: int) -> tuple[str, str]:
    """The first record's name and its first ``n`` letters, read directly (case kept) - the polars-bio case check."""
    name, parts, have = None, [], 0
    with fasta.open("rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    break
                name = line[1:].split()[0]
                continue
            s = line.strip()
            parts.append(s)
            have += len(s)
            if have >= n:
                break
    return name, "".join(parts)[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", type=Path, required=True, help="soft-masked FASTA (uncompressed)")
    ap.add_argument("--genome-dir", type=Path, required=True, help="prepared genome: bases.u8 + contigs.parquet")
    ap.add_argument("--report", type=Path, default=None, help="NCBI assembly report (accession -> chromosome)")
    ap.add_argument("--stage-dir", type=Path, required=True, help="where repeat.u8 is built before the move")
    ap.add_argument("--check-only", action="store_true", help="compare every contig, report, write no repeat.u8")
    ap.add_argument("--held", default="", help="held-out chromosomes for the summary (comma-separated; default: "
                                               "assembly.json's held_out)")
    ap.add_argument("--min-lower", type=float, default=0.01,
                    help="stop without writing when less than this fraction of contig bases is lower-case")
    a = ap.parse_args()
    t0 = time.time()
    gdir, stage = a.genome_dir, a.stage_dir
    target = gdir / "repeat.u8"
    if target.exists() and not a.check_only:
        sys.exit(f"{target} exists; refusing to overwrite it")
    stage.mkdir(parents=True, exist_ok=True)
    name = gdir.name

    # contigs, each with the FASTA record it comes from (duckdb join of two lazy tables, collected once)
    contigs = pl.scan_parquet(gdir / "contigs.parquet").select("chrom", "start", "end", "offset")
    rmap = record_map(a.fasta, a.report)
    con = duckdb.connect()
    con.register("contigs", contigs.collect())
    con.register("rmap", rmap.collect())
    plan = con.sql("""SELECT c.chrom, r.name, c.start, c."end", c."offset"
                      FROM contigs c LEFT JOIN rmap r ON c.chrom = r.chrom ORDER BY c."offset" """).pl()
    unmatched = plan.filter(pl.col("name").is_null())["chrom"].unique().to_list()
    if unmatched:
        sys.exit(f"contigs whose chromosome has no FASTA record: {unmatched}")
    if plan.select(pl.col("name").n_unique()).item() != plan.select(pl.col("chrom").n_unique()).item():
        sys.exit("the chromosome -> record mapping is not one-to-one")
    bases = np.memmap(gdir / "bases.u8", dtype=np.uint8, mode="r")
    total = int((plan["offset"] + plan["end"] - plan["start"]).max())
    if total != len(bases):
        sys.exit(f"contigs reach {total:,} but bases.u8 holds {len(bases):,}")
    print(f"{name}: {plan.height} contigs on {plan['name'].n_unique()} records, {total:,} bases", flush=True)

    # polars-bio must keep the case: compare its first record's first megabase with a direct read
    first_name, first_raw = raw_prefix(a.fasta, 1 << 20)
    pb_first = (pb.scan_fasta(str(a.fasta)).filter(pl.col("name") == first_name)
                .select(pl.col("sequence").str.slice(0, 1 << 20)).collect().item())
    if pb_first != first_raw:
        sys.exit(f"polars-bio's sequence of {first_name} differs from the file's letters (case not kept?)")
    print(f"polars-bio {pb.__version__} keeps the letters' case ({first_name}, first {len(first_raw):,} bases, "
          f"{sum(c.islower() for c in first_raw):,} lower-case)", flush=True)

    wanted = plan["name"].unique().to_list()
    seqs = (pb.scan_fasta(str(a.fasta)).filter(pl.col("name").is_in(wanted)).select("name", "sequence").collect())
    held = set(x for x in a.held.split(",") if x)
    meta = gdir / "assembly.json"
    if not held and meta.is_file():
        held = set(json.loads(meta.read_text()).get("held_out", []))
    part = stage / f"{name}.repeat.u8.part"
    out = None if a.check_only else np.memmap(part, dtype=np.uint8, mode="w+", shape=(total,))
    mismatches = []
    per_chrom, digest = {}, hashlib.md5()
    lower_all = bases_all = 0
    by_record = {r: plan.filter(pl.col("name") == r) for r in wanted}
    for rec, seq in seqs.iter_rows():
        raw = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
        codes = G.encode_bases(seq)
        rec_bases = rec_lower = 0
        for chrom, _, s, e, off in by_record[rec].iter_rows():
            n = e - s
            have = np.asarray(bases[off:off + n])
            if len(codes) < e or not np.array_equal(codes[s:e], have):
                diff = (np.flatnonzero(codes[s:e] != have) if len(codes) >= e
                        else np.arange(max(0, len(codes) - s), n))
                mismatches.append(dict(chrom=chrom, start=s, end=e, offset=off, n_diff=int(len(diff)),
                                       first=int(s + diff[0])))
                if not a.check_only:
                    sys.exit(f"{chrom}:{s + int(diff[0])}: the FASTA's base differs from bases.u8 at offset "
                             f"{off + int(diff[0])}; not the FASTA this genome was prepared from")
            rep = LOWER[raw[s:e]]
            if out is not None:
                out[off:off + n] = rep
            k = int(rep.sum())
            d = per_chrom.setdefault(chrom, dict(bases=0, lower=0, held_out=chrom in held))
            d["bases"] += n
            d["lower"] += k
            lower_all += k
            bases_all += n
            rec_lower += k
            rec_bases += n
        print(f"{rec}: {len(raw):,} bp, {by_record[rec].height} contigs, {rec_bases:,} contig bases, "
              f"lower-case {100 * rec_lower / max(1, rec_bases):.2f}%", flush=True)
        del raw, codes, seq
    if set(seqs["name"].to_list()) != set(wanted) or bases_all != total:
        sys.exit(f"covered {bases_all:,} of {total:,} bases; records missing: "
                 f"{sorted(set(wanted) - set(seqs['name'].to_list()))}")
    frac = lower_all / bases_all
    if a.check_only:
        res = dict(genome=name, fasta=str(a.fasta), check_only=True, polars_bio=pb.__version__, contigs=plan.height,
                   contigs_matching=plan.height - len(mismatches), mismatches=mismatches, total_bases=total,
                   lower_fraction=frac,
                   per_chrom={c: dict(d, lower_fraction=d["lower"] / max(1, d["bases"])) for c, d in per_chrom.items()},
                   held_out=sorted(held), seconds=round(time.time() - t0, 1))
        (stage / f"{name}.check.json").write_text(json.dumps(res, indent=1) + "\n")
        print(json.dumps({k: res[k] for k in ("genome", "contigs", "contigs_matching", "total_bases", "lower_fraction",
                                                "seconds")} | {"mismatches": mismatches[:20]}), flush=True)
        sys.exit(0 if not mismatches else 2)
    out.flush()
    del out
    with part.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            digest.update(block)
    split = {k: dict(bases=sum(d["bases"] for d in per_chrom.values() if d["held_out"] == v),
                     lower=sum(d["lower"] for d in per_chrom.values() if d["held_out"] == v))
             for k, v in (("train", False), ("held_out", True))}
    summary = dict(genome=name, fasta=str(a.fasta), report=None if a.report is None else str(a.report),
                   polars_bio=pb.__version__, duckdb=duckdb.__version__, total_bases=total, lower_bases=lower_all,
                   lower_fraction=frac, split={k: dict(v, lower_fraction=v["lower"] / max(1, v["bases"]))
                                               for k, v in split.items()},
                   per_chrom={c: dict(d, lower_fraction=d["lower"] / max(1, d["bases"])) for c, d in per_chrom.items()},
                   repeat_md5=digest.hexdigest(), seconds=round(time.time() - t0, 1))
    if frac < a.min_lower:
        summary["written"] = False
        (stage / f"{name}.repeat.json").write_text(json.dumps(summary, indent=1) + "\n")
        part.unlink()
        sys.exit(f"only {100 * frac:.3f}% lower-case: {a.fasta} is not soft-masked; nothing written")
    if target.exists():
        sys.exit(f"{target} appeared while building; leaving {part}")
    os.replace(part, target)                                  # same file system: an atomic move, a new file only
    summary["written"] = str(target)
    (stage / f"{name}.repeat.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: summary[k] for k in ("genome", "total_bases", "lower_fraction", "split", "repeat_md5",
                                                "seconds", "written")}), flush=True)


if __name__ == "__main__":
    main()
