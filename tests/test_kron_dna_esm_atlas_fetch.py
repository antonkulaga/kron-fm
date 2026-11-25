"""esm_atlas_fetch.py: which shards can hold a protein, which row groups to read, and one shard fetched from a
local stand-in for the bucket."""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

spec = importlib.util.spec_from_file_location(
    "esm_atlas_fetch", Path(__file__).resolve().parents[1] / "scripts" / "kron_dna" / "esm_atlas_fetch.py")
EF = importlib.util.module_from_spec(spec)
spec.loader.exec_module(EF)


def test_sixteen_shards_per_hash_prefix():
    s = EF.shards_of("c133dfce69f606f20865e9008199f852")              # found in the Atlas in shard c_0304
    assert len(s) == 16 and s[0] == "deduplicated_c_0304.parquet" and s[-1] == "deduplicated_c_0319.parquet"
    assert EF.shards_of("9f97af79d53dc90b58cbd279fd5946d3")[14] == "deduplicated_9_3998.parquet"


def test_fasta_reading_and_row_group_choice(tmp_path, monkeypatch):
    fa = tmp_path / "p.fasta.gz"
    with gzip.open(fa, "wt") as f:
        f.write(">sp|A|A_X first protein\nMKT\nAA*\n>sp|B|B_X\nMSS\n>empty\n")
    seqs = EF.read_fasta(fa)
    assert seqs == {"sp|A|A_X": "MKTAA", "sp|B|B_X": "MSS"}            # joined, the stop dropped, an empty entry left out
    h = {n: hashlib.md5(s.encode()).hexdigest() for n, s in seqs.items()}
    # a stand-in shard: sorted hashes in row groups of two, one wanted protein in it
    others = sorted(f"{i:032x}" for i in (1, 2, 3, 2 ** 127, 2 ** 127 + 5))
    rows = sorted(others + [h["sp|A|A_X"]])
    shard = tmp_path / "shard.parquet"
    pq.write_table(pa.table({"header": rows, "max_pooled": [b""] * len(rows),
                             "per_position": pa.array([x.encode() for x in rows], pa.large_binary()),
                             "protein_hash": rows}), shard, row_group_size=2)
    pf = pq.ParquetFile(shard)
    groups = EF.pick(pf.metadata, [h["sp|A|A_X"], h["sp|B|B_X"]])
    assert sum(h["sp|A|A_X"] in v for v in groups.values()) == 1       # exactly the group that holds it
    monkeypatch.setattr(EF, "Ranged", lambda url: open(shard, "rb"))   # the bucket: this file
    parts = tmp_path / "parts"
    parts.mkdir()
    assert EF.fetch_shard("deduplicated_x_0000.parquet", [h["sp|A|A_X"], h["sp|B|B_X"]], parts) == 1
    got = pq.read_table(parts / "deduplicated_x_0000.parquet").to_pylist()
    assert got == [{"md5": h["sp|A|A_X"], "per_position": h["sp|A|A_X"].encode()}]
    assert EF.fetch_shard("deduplicated_x_0000.parquet", [], parts) == 1          # a finished shard is kept
