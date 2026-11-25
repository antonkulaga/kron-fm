"""prep_esm_atlas_codons.py: a protein's per-residue features land on the codons of the transcript that encodes it,
on both strands and across an intron."""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

spec = importlib.util.spec_from_file_location(
    "prep_esm_atlas_codons", Path(__file__).resolve().parents[1] / "scripts" / "kron_dna" / "prep_esm_atlas_codons.py")
PC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PC)


def record(n: int, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    row = np.repeat(np.arange(n), 64).astype(np.uint16)
    col = np.concatenate([rng.choice(16384, 64, replace=False) for _ in range(n)]).astype(np.uint16)
    data = rng.integers(1, 256, n * 64).astype(np.uint8)
    perm = rng.permutation(n * 64)                                    # the Atlas does not promise row order
    buf = io.BytesIO()
    np.savez(buf, row=row[perm], col=col[perm], data=data[perm], shape=np.array([n, 16384], dtype=np.uint32),
             scale=np.array([9.5], dtype=np.float32), is_1d=np.array([False]))
    return buf.getvalue()


def test_features_follow_the_codons(tmp_path, monkeypatch):
    comp = str.maketrans("ACGT", "TGCA")
    plus = "ATGAAAGGG"                                                # M K G
    minus = "ATGCCCTTTTGG"                                            # M P F W, on the minus strand, split 5 + 7
    mrc = minus.translate(comp)[::-1]                                 # as the forward strand shows it
    chrom = "TT" + plus[:4] + "GTAAG" + plus[4:] + "CC" + mrc[:7] + "ACGTAC" + mrc[7:] + "AA"
    #        0    2..6         intron   11..16      16   18..25      intron    31..36
    genome = tmp_path / "genome"
    genome.mkdir()
    np.array(["ACGT".index(x) for x in chrom], dtype=np.uint8).tofile(genome / "bases.u8")
    pl.DataFrame({"chrom": ["chr1"], "start": [0], "end": [len(chrom)], "offset": [0]}).write_parquet(genome / "contigs.parquet")
    lines = []
    for gene, tx, strand, cds, tag in (("g1", "t1", "+", [(2, 6), (11, 16)], 'tag "Ensembl_canonical";'),
                                       ("g2", "t2", "-", [(18, 25), (31, 36)], ""),
                                       ("g3", "t3", "+", [(2, 6)], "")):                       # not whole codons
        for a, b in cds:
            lines.append(f'chr1\tx\tCDS\t{a + 1}\t{b}\t.\t{strand}\t0\tgene_id "{gene}"; transcript_id "{tx}"; {tag}\n')
    gtf = tmp_path / "a.gtf.gz"
    with gzip.open(gtf, "wt") as f:
        f.writelines(lines)
    prots = {"P1": "MKG", "P2": "MPFW", "P3": "MAAA"}                 # P3: no transcript encodes it
    rows = [{"name": n, "md5": hashlib.md5(s.encode()).hexdigest(), "length": len(s), "per_position": record(len(s), i)}
            for i, (n, s) in enumerate(prots.items())]
    feats = tmp_path / "features.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=pa.schema([("name", pa.string()), ("md5", pa.string()),
                                                                 ("length", pa.int32()), ("per_position", pa.large_binary())])), feats)
    monkeypatch.setattr(PC, "interim_dir", lambda *a: genome)
    out = tmp_path / "out"
    PC.main(gtf, feats, out, name="x")
    info = json.loads((out / "codons.json").read_text())
    assert info["codons"] == 7 and info["transcripts"] == 2 and info["left_out"]["not_codons"] == 1
    from kron_dna.genome import layout_fingerprint
    assert info["layout_sha256"] == layout_fingerprint(["chr1"], [0], [len(chrom)], [0])
    assert info["layout_sha256"] != layout_fingerprint(["chr1"], [0], [len(chrom)], [8])     # another layout
    pos = np.fromfile(out / "pos.i64", dtype=np.int64).reshape(-1, 3)
    feat = np.fromfile(out / "feat.u16", dtype=np.uint16).reshape(-1, 64)
    assert pos[:3].tolist() == [[2, 3, 4], [5, 11, 12], [13, 14, 15]]                 # the K codon spans the intron
    assert pos[3:].tolist() == [[20, 19, 18], [23, 22, 21], [32, 31, 24], [35, 34, 33]]   # minus: descending; W first
    f1 = PC.decode(rows[0]["per_position"])[0]
    f2 = PC.decode(rows[1]["per_position"])[0]
    assert np.array_equal(feat[:3], f1)                                                # residue i on codon i
    assert np.array_equal(feat[3:], f2[::-1])                                          # W, F, P, M along the genome
    for p, aa in zip(pos, "MKG" + "WFPM"):                                             # each codon reads its residue
        s = "".join(chrom[i] for i in p)
        s = s if p[0] < p[2] else s.translate(comp)
        assert PC.CODON[s] == aa
    assert np.fromfile(out / "scale.f32", dtype=np.float32).tolist() == [9.5] * 7
