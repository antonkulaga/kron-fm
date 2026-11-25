"""Per-residue ESM C features (esm_atlas_fetch.py) laid on the genome: one row per codon of one transcript per gene.

    uv run python scripts/kron_dna/prep_esm_atlas_codons.py GTF.gz FEATURES.parquet OUT_DIR [--name hg38]

A protein's features belong to a transcript when the transcript's CDS, read from the prepared genome and
translated, is exactly the protein (same MD5 as the Atlas key), so residue i is codon i with no alignment. Per gene
one such transcript is kept: the one tagged ``Ensembl_canonical`` when it matches, else the longest. Transcripts
with an incomplete CDS start, a gap in the prepared genome or a length that is not whole codons are left out.

OUT_DIR, rows sorted by the codon's first base in the genome's global layout (``bases.u8``):

- ``pos.i64``   ``[codons, 3]``: the layout offsets of the codon's three bases in reading order (a codon split by
  an intron has distant offsets; on the minus strand they descend);
- ``feat.u16``  ``[codons, 64]`` and ``val.u8`` ``[codons, 64]``: the residue's 64 active features of 16,384 and
  their stored values, strongest first as the Atlas has them;
- ``scale.f32`` ``[codons]``: the protein's scale for ``val``;
- ``codons.parquet``: per kept transcript its gene, transcript, protein name, strand, residues;
- ``codons.json``: counts and what was left out.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import typer

from kron_core.paths import interim_dir

K = 64
BASES = "ACGT"
AA = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODON = {a + b + c: AA[16 * i + 4 * j + k] for i, a in enumerate("TCAG") for j, b in enumerate("TCAG")
         for k, c in enumerate("TCAG")}
COMP = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}


def transcripts(gtf: Path) -> dict[str, dict]:
    """``{transcript: {gene, chrom, strand, canonical, ok, cds: [(start, end)]}}`` (0-based half-open), CDS rows only."""
    tx: dict[str, dict] = {}
    gid, tid = re.compile(r'gene_id "([^"]+)"'), re.compile(r'transcript_id "([^"]+)"')
    with gzip.open(gtf, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            c = line.split("\t", 9)
            if c[2] != "CDS":
                continue
            t = tx.setdefault(tid.search(c[8]).group(1), dict(
                gene=gid.search(c[8]).group(1), chrom=c[0], strand=c[6], cds=[],
                canonical="Ensembl_canonical" in c[8], ok="cds_start_NF" not in c[8] and "cds_end_NF" not in c[8]))
            t["cds"].append((int(c[3]) - 1, int(c[4])))
    return tx


def coding_positions(cds: list[tuple[int, int]], strand: str) -> np.ndarray:
    """Chromosome positions of the coding bases in reading order."""
    parts = [np.arange(a, b) for a, b in sorted(cds)]
    pos = np.concatenate(parts)
    return pos if strand == "+" else pos[::-1]


def decode(blob: bytes) -> tuple[np.ndarray, np.ndarray, float]:
    """One protein's Atlas record as ``(feat [L, 64] uint16, val [L, 64] uint8, scale)``."""
    z = np.load(io.BytesIO(blob), allow_pickle=False)
    n = int(z["shape"][0])
    row, col, data = z["row"].astype(np.int64), z["col"], z["data"]
    if len(row) != n * K or not np.array_equal(np.bincount(row, minlength=n), np.full(n, K)):
        raise ValueError("not 64 features per residue")
    order = np.argsort(row, kind="stable")
    return col[order].reshape(n, K).astype(np.uint16), data[order].reshape(n, K).astype(np.uint8), float(z["scale"][0])


def main(gtf: Path, features: Path, out: Path, name: str = "hg38") -> None:
    genome = interim_dir("genome", name)
    contigs = pl.read_parquet(genome / "contigs.parquet").sort("chrom", "start")
    by_chrom = {c[0]: g for c, g in contigs.group_by("chrom", maintain_order=True)}
    bases = np.memmap(genome / "bases.u8", dtype=np.uint8, mode="r")
    feats = {r["md5"]: r for r in pq.read_table(features, columns=["name", "md5", "length"]).to_pylist()}
    left = dict(incomplete=0, gap=0, not_codons=0, no_protein=0)
    best: dict[str, tuple] = {}
    for tid, t in transcripts(gtf).items():
        g = by_chrom.get(t["chrom"])
        if not t["ok"] or g is None:
            left["incomplete"] += 1
            continue
        pos = coding_positions(t["cds"], t["strand"])
        if len(pos) % 3:
            left["not_codons"] += 1
            continue
        starts, ends, offs = g["start"].to_numpy(), g["end"].to_numpy(), g["offset"].to_numpy()
        j = np.searchsorted(starts, pos, side="right") - 1
        if (j < 0).any() or (pos >= ends[np.maximum(j, 0)]).any():
            left["gap"] += 1
            continue
        glob = offs[j] + pos - starts[j]
        seq = "".join(BASES[b] if b < 4 else "N" for b in np.asarray(bases[np.sort(glob)])[np.argsort(np.argsort(glob))])
        if t["strand"] == "-":
            seq = "".join(COMP[x] for x in seq)
        prot = "".join(CODON.get(seq[i:i + 3], "X") for i in range(0, len(seq), 3))
        md5 = hashlib.md5(prot.encode()).hexdigest()
        if md5 not in feats:
            left["no_protein"] += 1
            continue
        rank = (t["canonical"], len(prot))
        if t["gene"] not in best or rank > best[t["gene"]][0]:
            best[t["gene"]] = (rank, tid, t, glob.reshape(-1, 3), md5)
    blobs = {r["md5"]: r["per_position"] for r in pq.read_table(features, columns=["md5", "per_position"]).to_pylist()}
    P, F, V, S, rows = [], [], [], [], []
    for gene, (_, tid, t, glob, md5) in best.items():
        f, v, scale = decode(blobs[md5])
        if len(f) != len(glob):
            raise ValueError(f"{tid}: {len(glob)} codons, {len(f)} residues with features")
        P.append(glob), F.append(f), V.append(v), S.append(np.full(len(f), scale, dtype=np.float32))
        rows.append(dict(gene=gene, transcript=tid, protein=feats[md5]["name"], strand=t["strand"], residues=len(f)))
    out.mkdir(parents=True, exist_ok=True)
    P, F, V, S = np.concatenate(P), np.concatenate(F), np.concatenate(V), np.concatenate(S)
    order = np.argsort(P.min(1), kind="stable")
    P[order].astype(np.int64).tofile(out / "pos.i64")
    F[order].tofile(out / "feat.u16")
    V[order].tofile(out / "val.u8")
    S[order].tofile(out / "scale.f32")
    pl.DataFrame(rows).write_parquet(out / "codons.parquet")
    info = dict(genome=name, gtf=gtf.name, codons=int(len(P)), transcripts=len(rows), features_per_codon=K,
                feature_space=16384, left_out=left, proteins_offered=len(feats),
                source="ESM Atlas v1 (Chan Zuckerberg Biohub, CC BY 4.0): sparse-autoencoder features of ESM C 6B",
                layout="rows sorted by the codon's first base in the genome's global layout")
    (out / "codons.json").write_text(json.dumps(info, indent=1))
    typer.echo(json.dumps(info))


if __name__ == "__main__":
    typer.run(main)
