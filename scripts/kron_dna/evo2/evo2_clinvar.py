"""Evo 2 7B zero-shot on ClinVar SNVs: mean log-likelihood of a genomic window with the alternate base minus the
same window with the reference base (Evo 2 paper's variant-effect recipe), for a fixed subsample of the variants
kron-dna clinvar scores. Writes a parquet of id, evo2 (lower = larger predicted damage), resumable.

usage: python evo2_clinvar.py POSITIONS.parquet hg38.fa.gz OUT.parquet [PER_CLASS] [WINDOW]

Runs in its own environment (this folder's pyproject.toml: torch 2.7.1 cu128, flash-attn, evo2), not the
project's: `cd scripts/kron_dna/evo2 && uv sync && uv run python evo2_clinvar.py ...` on an A100 (7B, bf16).
POSITIONS comes from `scripts/kron_dna/export_positions.py` (the project's environment); hg38 from UCSC.
"""
import sys
from pathlib import Path

import polars as pl
import polars_bio as pb
import torch

positions, genome, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
per_class = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
window = int(sys.argv[5]) if len(sys.argv) > 5 else 4096

# the subsample: up to PER_CLASS per (consequence, label), sorted by id then drawn by seed, as sample_per_class does
pos = pl.read_parquet(positions).sort("id")
pos = pl.concat([g.sample(min(per_class, g.height), seed=1) for _, g in pos.group_by("consequence", "label", maintain_order=True)])
done = pl.read_parquet(out) if out.exists() else pl.DataFrame(schema={"id": pl.Utf8, "evo2": pl.Float64})
todo = pos.join(done, on="id", how="anti")
print(f"{pos.height:,} variants in the subsample, {todo.height:,} to score", flush=True)

chroms = {f"chr{c}" for c in [*map(str, range(1, 23)), "X", "Y"]} | {"chrM"}
names = {"MT": "chrM"}
seqs = {r["name"]: r["sequence"] for r in pb.scan_fasta(genome).select("name", "sequence")
        .filter(pl.col("name").is_in(list(chroms))).collect().iter_rows(named=True)}
print(f"genome: {len(seqs)} chromosomes", flush=True)

from evo2 import Evo2  # noqa: E402  (after the genome, so a missing file fails before the 15 GB download)

model = Evo2("evo2_7b")
half = window // 2
rows = []
mismatch = 0
for i, r in enumerate(todo.iter_rows(named=True)):
    chrom = names.get(r["chrom"], f"chr{r['chrom']}")
    seq = seqs[chrom]
    p = r["pos"] - 1                                     # 0-based
    lo = max(0, p - half)
    ref_win = seq[lo:lo + window].upper()
    k = p - lo
    if ref_win[k] != r["ref"]:
        mismatch += 1
        continue
    alt_win = ref_win[:k] + r["alt"] + ref_win[k + 1:]
    with torch.no_grad():
        ref_ll, alt_ll = model.score_sequences([ref_win, alt_win], batch_size=2)
    rows.append({"id": r["id"], "evo2": float(alt_ll - ref_ll)})
    if len(rows) % 250 == 0:
        done = pl.concat([done, pl.DataFrame(rows, schema=done.schema)])
        done.write_parquet(out)
        rows = []
        print(f"{done.height:,} scored ({i + 1:,} of {todo.height:,} this run; {mismatch} reference mismatches)", flush=True)
done = pl.concat([done, pl.DataFrame(rows, schema=done.schema)])
done.write_parquet(out)
print(f"done: {done.height:,} scored, {mismatch} reference mismatches", flush=True)
