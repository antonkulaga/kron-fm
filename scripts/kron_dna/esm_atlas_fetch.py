"""Per-residue ESM C features for a set of proteins, fetched from the public ESM Atlas with no inference.

    uv run python scripts/kron_dna/esm_atlas_fetch.py PROTEINS.fasta[.gz] OUT_DIR [--workers 32] [--limit N]

The Atlas (Chan Zuckerberg Biohub, CC BY 4.0; bucket ``esm-protein-atlas``, release ``v1``) holds, for about 6.8
billion deduplicated public protein sequences, the sparse-autoencoder features of ESM C 6B: per residue 64 active
features out of 16,384. A protein is keyed by the MD5 of its sequence. Its features sit in one of the 16 shards
``sae/data_shards/deduplicated_<h0>_<NNNN>.parquet`` with ``NNNN = int(h[1:3], 16) * 16 + j``, ``j`` in 0..15 (found
by reading; the Atlas does not document it), each shard sorted by hash in row groups of 100 proteins. The groups
of 16 are equal-count pieces of the sorted hashes, so their edges drift a little from the three-character prefix:
a hash near the start of its prefix can sit in the group before. A second pass looks there for
what the first did not find (``parts-neighbours/``).

So for the proteins of one species: hash every sequence; per shard that could hold any of them read the footer,
then of each row group whose hash range covers a wanted hash only the hash column, and the feature column only
where the hash is there. Nothing else of the Atlas is downloaded.

OUT_DIR gets ``features.parquet`` (name, md5, length, ``per_position``: the Atlas's own bytes, an ``.npz`` with
``row``, ``col`` (feature id), ``data`` (uint8), ``shape`` [residues, 16384] and ``scale``; value = data / 255 *
scale is an assumption to check against the Atlas's documentation), ``missing.tsv`` (proteins whose exact
sequence the Atlas does not hold) and ``fetch.json`` (counts, release, time). A rerun keeps the shards already
done (``parts/``). Sequences with characters outside the 20 amino acids are looked up as they are.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import typer

BASE = "https://esm-protein-atlas.s3.us-west-2.amazonaws.com/v1/sae/data_shards/"


class Ranged(io.RawIOBase):
    """A remote file read by HTTP range requests (through the environment's proxy), for pyarrow."""

    def __init__(self, url: str, tries: int = 6) -> None:
        self.url, self.pos, self.tries = url, 0, tries
        self.size = int(self._get(urllib.request.Request(url, method="HEAD"), body=False)["Content-Length"])

    def _get(self, req, body: bool = True):
        """The response's body (or its headers), the whole exchange tried again on any failure: a timeout can
        come while the body is read, not only when the connection opens."""
        for k in range(self.tries):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    return r.read() if body else r.headers
            except Exception:
                if k == self.tries - 1:
                    raise
                time.sleep(min(60, 2 ** k))

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{min(self.size, self.pos + n) - 1}"})
        data = self._get(req)
        self.pos += len(data)
        return data

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def read_fasta(path: Path) -> dict[str, str]:
    seqs: dict[str, list[str]] = {}
    name = None
    with (gzip.open(path, "rt") if path.suffix == ".gz" else open(path)) as f:
        for line in f:
            if line.startswith(">"):
                name = line[1:].split()[0]
                seqs[name] = []
            elif name is not None:
                seqs[name].append(line.strip().rstrip("*"))
    return {k: "".join(v) for k, v in seqs.items() if v}


def shards_of(md5: str) -> list[str]:
    """The 16 shard names that can hold a protein with this hash."""
    first = int(md5[1:3], 16) * 16
    return [f"deduplicated_{md5[0]}_{first + j:04d}.parquet" for j in range(16)]


def chunk_shards(n: int) -> list[str]:
    """The 16 shards of chunk ``n`` (0..4095, the hashes' first three characters as a number)."""
    return [f"deduplicated_{n >> 8:x}_{(n & 255) * 16 + j:04d}.parquet" for j in range(16)]


def pick(meta, wanted: list[str]) -> dict[int, list[str]]:
    """Row groups of a shard whose hash range covers a wanted hash: ``{row group: hashes}``."""
    col = next(i for i in range(meta.num_columns) if meta.schema.column(i).path == "protein_hash")
    out: dict[int, list[str]] = {}
    for g in range(meta.num_row_groups):
        st = meta.row_group(g).column(col).statistics
        hit = [h for h in wanted if st.min <= h <= st.max]
        if hit:
            out[g] = hit
    return out


def fetch_shard(shard: str, wanted: list[str], parts: Path) -> int:
    """The wanted proteins this shard holds, written to ``parts/<shard>``; returns how many."""
    dst = parts / shard
    if dst.exists():
        return pq.read_metadata(dst).num_rows
    pf = pq.ParquetFile(Ranged(BASE + shard))
    rows: list[dict] = []
    for g, hashes in pick(pf.metadata, wanted).items():
        have = pf.read_row_group(g, columns=["protein_hash"]).column(0).to_pylist()
        if not set(hashes) & set(have):
            continue
        for r in pf.read_row_group(g, columns=["protein_hash", "per_position"]).to_pylist():
            if r["protein_hash"] in hashes:
                rows.append({"md5": r["protein_hash"], "per_position": r["per_position"]})
    tmp = dst.with_suffix(".part")
    pq.write_table(pa.Table.from_pylist(rows, schema=pa.schema([("md5", pa.string()), ("per_position", pa.large_binary())])), tmp)
    tmp.replace(dst)
    return len(rows)


def main(fasta: Path, out: Path, workers: int = 32, limit: int = 0) -> None:
    t0 = time.time()
    seqs = read_fasta(fasta)
    if limit:
        seqs = dict(list(seqs.items())[:limit])
    md5 = {n: hashlib.md5(s.encode()).hexdigest() for n, s in seqs.items()}
    by_shard: dict[str, list[str]] = {}
    for h in set(md5.values()):
        for s in shards_of(h):
            by_shard.setdefault(s, []).append(h)
    parts = out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    typer.echo(f"{len(seqs):,} proteins, {len(set(md5.values())):,} distinct sequences, {len(by_shard):,} shards to look in")
    done = found = 0
    failed: list[str] = []

    def one(kv) -> int:
        try:
            return fetch_shard(kv[0], kv[1], parts)
        except Exception as e:                                           # one shard never ends the run
            failed.append(f"{kv[0]}: {e!r}"[:200])
            return 0

    with ThreadPoolExecutor(workers) as pool:
        for n in pool.map(one, by_shard.items()):
            done, found = done + 1, found + n
            if done % 500 == 0:
                typer.echo(f"shards {done:,} of {len(by_shard):,}; proteins found {found:,}; failed {len(failed)}; "
                           f"{time.time() - t0:.0f} s")
    if failed:
        typer.echo(f"{len(failed)} shards could not be read; run again to fetch them (finished shards are kept):")
        for line in failed[:10]:
            typer.echo("  " + line)
        raise typer.Exit(1)
    have: set[str] = set()                                              # hashes only: the records are read at the end
    for p in sorted(parts.glob("*.parquet")):
        have.update(pq.read_table(p, columns=["md5"]).column(0).to_pylist())
    # second pass: a hash the first pass did not find may sit in the chunk before its own (seen: the pieces' edges
    # run up to a quarter of a prefix late, never early; the human proteome's misses pile up at the prefix's start)
    rest = sorted(set(md5.values()) - have)
    near: dict[int, list[str]] = {}
    for h in rest:
        if int(h[:3], 16) > 0:
            near.setdefault(int(h[:3], 16) - 1, []).append(h)
    parts2 = out / "parts-neighbours"
    parts2.mkdir(exist_ok=True)

    def neighbour(item) -> int:
        try:
            return fetch_shard(item[0], item[1], parts2)                # the footer's ranges decide what is read
        except Exception as e:
            failed.append(f"{item[0]}: {e!r}"[:200])
            return 0

    with ThreadPoolExecutor(workers) as pool:
        extra = sum(pool.map(neighbour, [(sh, hs) for m, hs in near.items() for sh in chunk_shards(m)]))
    typer.echo(f"second pass: {len(rest):,} not found at first, {len(near):,} neighbouring chunks looked at, {extra:,} found there")
    if failed:
        typer.echo(f"{len(failed)} neighbouring chunks could not be read; run again: {failed[:5]}")
        raise typer.Exit(1)
    got: dict[str, bytes] = {}
    for p in sorted(parts.glob("*.parquet")) + sorted(parts2.glob("*.parquet")):
        for r in pq.read_table(p).to_pylist():
            got[r["md5"]] = r["per_position"]
    rows = [{"name": n, "md5": md5[n], "length": len(seqs[n]), "per_position": got[md5[n]]} for n in seqs if md5[n] in got]
    pq.write_table(pa.Table.from_pylist(rows, schema=pa.schema([("name", pa.string()), ("md5", pa.string()),
                                                                 ("length", pa.int32()), ("per_position", pa.large_binary())])),
                   out / "features.parquet", row_group_size=200)
    missing = [n for n in seqs if md5[n] not in got]
    (out / "missing.tsv").write_text("name\tmd5\tlength\n" + "".join(f"{n}\t{md5[n]}\t{len(seqs[n])}\n" for n in missing))
    (out / "fetch.json").write_text(json.dumps({
        "source": "ESM Atlas v1 (Chan Zuckerberg Biohub), sparse-autoencoder features of ESM C 6B, per residue",
        "license": "CC BY 4.0", "bucket_prefix": BASE, "proteins": len(seqs), "found": len(rows), "missing": len(missing),
        "shards_read": len(by_shard), "seconds": round(time.time() - t0), "fasta": fasta.name}, indent=1))
    typer.echo(f"wrote {out / 'features.parquet'}: {len(rows):,} of {len(seqs):,} proteins; {len(missing):,} missing")


if __name__ == "__main__":
    typer.run(main)
