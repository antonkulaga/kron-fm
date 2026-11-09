"""Streaming, sharded ``prepare`` for corpora too large to hold in memory (the SynCodonLM corpus: 43.5M
sequences, ~18B codons in one 17 GB gzipped CSV).

The CSV is read line by line; worker processes check and encode chunks of sequences with numpy lookup
tables (:func:`encode_checked`, the same verdicts and tokens as :func:`kron_dna.codons.check_cds` and
:func:`kron_dna.codons.encode`); the main process packs documents into fixed-length rows in input order
and writes them as ``uint8`` shards. Each value of the split column (the corpus's own ``Set``: train or
test) becomes its own prepared directory::

    <out>/train/tokens_00000.npy ...  meta.json
    <out>/test/tokens_00000.npy  ...  meta.json
    <out>/meta.json                    (both splits' summaries)

A split's ``meta.json`` lists its shards with their row counts, which :class:`kron_dna.data.PackedRows`
reads by memory map, and tallies sequences, codons, filter rejects, species and groups.
"""
from __future__ import annotations

import csv
import gzip
import json
import sys
import time
from collections import Counter, deque
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from kron_dna.codons import CLS_ID, EOS_ID, GENETIC_CODE, PAD_ID, STOP_CODONS, TOKEN_ID, UNK_ID, VOCAB_SIZE

_BASE = np.full(256, 4, dtype=np.uint8)
for _i, _b in enumerate("ACGT"):
    _BASE[ord(_b)] = _BASE[ord(_b.lower())] = _i
_BASE[ord("U")] = _BASE[ord("u")] = 3
#: base indices (a, b, c) in ACGT order -> token id, with ``<unk>`` for any base outside ACGT (index 4).
_CODON_TOKEN = np.full(125, UNK_ID, dtype=np.uint8)
_IS_STOP = np.zeros(125, dtype=bool)
for _a in range(4):
    for _b in range(4):
        for _c in range(4):
            _codon = "ACGT"[_a] + "ACGT"[_b] + "ACGT"[_c]
            _CODON_TOKEN[25 * _a + 5 * _b + _c] = TOKEN_ID[_codon]
            _IS_STOP[25 * _a + 5 * _b + _c] = _codon in STOP_CODONS
assert len(GENETIC_CODE) == 64
_WHITESPACE = b" \t\r\n\x0b\x0c"


def encode_checked(seq: str, starts: tuple[str, ...] = ("ATG",), min_codons: int = 30
                   ) -> tuple[str | None, np.ndarray | None]:
    """``(reason, None)`` for a sequence :func:`~kron_dna.codons.check_cds` refuses, else ``(None, tokens)``
    with the ``uint8`` tokens of :func:`~kron_dna.codons.encode` (``<cls>`` codons ``<eos>``).

    Same verdicts in the same order as ``check_cds`` with its defaults (a stop required, no unknown bases).
    """
    raw = seq.encode("ascii", "replace")
    if any(w in raw for w in _WHITESPACE):
        raw = b"".join(raw.split())
    n = len(raw)
    if n % 3:
        return "frame", None
    n_codons = n // 3
    if n_codons < min_codons:
        return "short", None
    if raw[:3].upper().replace(b"U", b"T").decode("ascii", "replace") not in starts:
        return "start", None
    bases = _BASE[np.frombuffer(raw, dtype=np.uint8)].reshape(-1, 3)
    index = bases[:, 0].astype(np.int16) * 25 + bases[:, 1] * 5 + bases[:, 2]
    if (bases == 4).any():
        return "alphabet", None
    stops = _IS_STOP[index]
    if not stops[-1]:
        return "stop", None
    if stops[:-1].any():
        return "internal_stop", None
    tokens = np.empty(n_codons + 2, dtype=np.uint8)
    tokens[0], tokens[-1] = CLS_ID, EOS_ID
    tokens[1:-1] = _CODON_TOKEN[index]
    return None, tokens


def documents_np(tokens: np.ndarray, seq_len: int) -> Iterator[np.ndarray]:
    """:func:`kron_dna.data.documents` for a ``uint8`` array (``<cls>`` codons ``<eos>``)."""
    body = tokens[1:-1]
    step = seq_len - 1
    for start in range(0, len(body), step):
        chunk = body[start:start + step]
        last = start + step >= len(body)
        if last and len(chunk) + 1 < seq_len:
            piece = np.empty(len(chunk) + 2, dtype=np.uint8)
            piece[0], piece[1:-1], piece[-1] = CLS_ID, chunk, EOS_ID
            yield piece
            return
        piece = np.empty(len(chunk) + 1, dtype=np.uint8)
        piece[0], piece[1:] = CLS_ID, chunk
        yield piece
        if last:
            yield np.array([CLS_ID, EOS_ID], dtype=np.uint8)


class ShardWriter:
    """Greedy packing of documents into rows (as :func:`kron_dna.data.pack`), written ``shard_rows`` rows at a
    time as ``tokens_NNNNN.npy``."""

    def __init__(self, out: Path, seq_len: int, shard_rows: int) -> None:
        self.out, self.seq_len, self.shard_rows = out, seq_len, shard_rows
        out.mkdir(parents=True, exist_ok=True)
        self.buffer = np.full((shard_rows, seq_len), PAD_ID, dtype=np.uint8)
        self.n_in_buffer = 0     # complete rows in the buffer
        self.fill = 0            # tokens in the row being filled (row index n_in_buffer)
        self.shards: list[dict] = []
        self.non_pad = 0

    def add(self, doc: np.ndarray) -> None:
        if self.fill + len(doc) > self.seq_len:
            self._close_row()
        self.buffer[self.n_in_buffer, self.fill:self.fill + len(doc)] = doc
        self.fill += len(doc)
        self.non_pad += len(doc)

    def _close_row(self) -> None:
        self.n_in_buffer += 1
        self.fill = 0
        if self.n_in_buffer == self.shard_rows:
            self._flush()

    def _flush(self) -> None:
        if self.n_in_buffer == 0:
            return
        name = f"tokens_{len(self.shards):05d}.npy"
        tmp = self.out / (name + ".tmp")
        with tmp.open("wb") as fh:
            np.save(fh, self.buffer[:self.n_in_buffer])
        tmp.replace(self.out / name)
        self.shards.append({"file": name, "rows": int(self.n_in_buffer)})
        self.buffer[:] = PAD_ID
        self.n_in_buffer = 0

    def close(self) -> list[dict]:
        if self.fill:
            self._close_row()
        self._flush()
        return self.shards

    @property
    def rows(self) -> int:
        return sum(s["rows"] for s in self.shards) + self.n_in_buffer


def _read_rows(path: Path, sequence_column: str, columns: tuple[str, ...]) -> Iterator[tuple[str, ...]]:
    """``(sequence, *columns)`` tuples from a CSV, plain or gzipped; a cut last line is skipped."""
    opener = gzip.open if path.name.endswith(".gz") else open
    csv.field_size_limit(sys.maxsize)
    with opener(path, "rt", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        missing = [c for c in (sequence_column, *columns) if c not in header]
        if missing:
            raise ValueError(f"{path} has no column(s) {missing} (columns: {header})")
        idx = [header.index(sequence_column), *(header.index(c) for c in columns)]
        width = len(header)
        try:
            for row in reader:
                if len(row) == width:
                    yield tuple(row[i] for i in idx)
        except (EOFError, csv.Error):
            return


def _encode_chunk(args: tuple[list[tuple[str, ...]], tuple[str, ...]]) -> list[tuple]:
    """Worker: ``(reason, tokens, *columns)`` per row of a chunk."""
    rows, starts = args
    out = []
    for row in rows:
        reason, tokens = encode_checked(row[0], starts)
        out.append((reason, tokens, *row[1:]))
    return out


def _chunks(rows: Iterator[tuple[str, ...]], size: int, limit: int | None) -> Iterator[list[tuple[str, ...]]]:
    chunk, n = [], 0
    for row in rows:
        if limit is not None and n >= limit:
            break
        chunk.append(row)
        n += 1
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _bounded_map(pool: ProcessPoolExecutor, items: Iterator, window: int) -> Iterator:
    """``pool.map`` in order, with at most ``window`` chunks in flight (``map`` itself would read the whole input)."""
    pending: deque = deque()
    for item in items:
        pending.append(pool.submit(_encode_chunk, item))
        if len(pending) >= window:
            yield pending.popleft().result()
    while pending:
        yield pending.popleft().result()


def prepare_stream(path: str | Path, out_dir: str | Path, seq_len: int = 1024, starts: tuple[str, ...] = ("ATG",),
                   split_column: str = "Set", species_column: str = "Species", group_column: str = "Group",
                   sequence_column: str = "CDS", shard_rows: int = 262_144, workers: int = 1,
                   chunk_size: int = 20_000, max_rows: int | None = None, log_every: int = 1_000_000) -> dict:
    """Stream one CSV into sharded packed rows, one prepared directory per value of ``split_column``."""
    path, out = Path(path), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    columns = (split_column, species_column, group_column)
    splits: dict[str, dict] = {}

    def split(name: str) -> dict:
        key = name.strip().lower() or "unsplit"
        if key not in splits:
            splits[key] = {"writer": ShardWriter(out / key, seq_len, shard_rows), "sequences": 0, "codons": 0,
                           "rejected": Counter(), "species": Counter(), "groups": Counter()}
        return splits[key]

    t0, n_read = time.time(), 0
    chunks = ((c, starts) for c in _chunks(_read_rows(path, sequence_column, columns), chunk_size, max_rows))
    pool = ProcessPoolExecutor(workers) if workers > 1 else None
    results = _bounded_map(pool, chunks, 2 * workers) if pool else map(_encode_chunk, chunks)
    try:
        for result in results:
            for reason, tokens, set_name, species, group in result:
                s = split(set_name)
                if reason is not None:
                    s["rejected"][reason] += 1
                    continue
                s["sequences"] += 1
                s["codons"] += len(tokens) - 2
                s["species"][species] += 1
                s["groups"][group] += 1
                for doc in documents_np(tokens, seq_len):
                    s["writer"].add(doc)
            before, n_read = n_read, n_read + len(result)
            if log_every and n_read // log_every > before // log_every:
                rate = n_read / (time.time() - t0)
                print(f"{n_read:,} rows read, {rate:,.0f} rows/s; "
                      + ", ".join(f"{k}: {v['sequences']:,} kept, {v['writer'].rows:,} packed rows"
                                  for k, v in splits.items()), flush=True)
    finally:
        if pool:
            pool.shutdown()

    summary = {}
    for key, s in splits.items():
        shards = s["writer"].close()
        rows = sum(sh["rows"] for sh in shards)
        meta = {"seq_len": seq_len, "rows": rows, "sequences": s["sequences"], "codons": s["codons"],
                "fill": s["writer"].non_pad / max(1, rows * seq_len), "rejected": dict(s["rejected"]),
                "starts": list(starts), "vocab_size": VOCAB_SIZE, "inputs": [str(path)],
                "keep": {split_column: key}, "shards": shards, "n_species": len(s["species"]),
                "groups": dict(s["groups"].most_common()), "species": dict(s["species"].most_common())}
        (out / key / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
        summary[key] = {k: meta[k] for k in ("rows", "sequences", "codons", "fill", "rejected", "n_species", "groups")}
        summary[key]["shards"] = len(shards)
    summary = {"seq_len": seq_len, "rows_read": n_read, "seconds": round(time.time() - t0, 1), "splits": summary}
    (out / "meta.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
