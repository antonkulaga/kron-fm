"""Coding sequences to packed training rows, and the masking that turns rows into a masked-LM batch.

Pipeline: FASTA or CSV (plain or gzipped) -> :func:`kron_dna.codons.check_cds` -> codon tokens -> packed rows
of ``seq_len`` tokens, saved as one ``uint8`` array (``tokens.npy``, shape ``[rows, seq_len]``) with a
``meta.json`` of counts. Every document in a row starts with ``<cls>``, so document boundaries are
recovered from the tokens alone (:func:`doc_ids_from_tokens`) and need not be stored. A sequence
longer than a row is cut into pieces of at most ``seq_len - 1`` codons, each its own document; the
last piece ends with ``<eos>``. A row's unused tail is ``<pad>``.

No published codon model packs sequences (the report's sizing section), which at a mean of ~407 codons
against a 1,024-token context leaves most positions empty; the KronLSTM layer restarts its state at
every document (``cu_seqlens``), so packing costs nothing in correctness.
"""
from __future__ import annotations

import csv
import gzip
import json
import re
import sys
import zlib
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import torch

from kron_dna.codons import CLS_ID, EOS_ID, FIRST_CODON_ID, MASK_ID, PAD_ID, VOCAB_SIZE, check_cds, encode
from kron_dna.losses import IGNORE_INDEX


def read_fasta(path: str | Path) -> Iterator[tuple[str, str]]:
    """``(header, sequence)`` pairs from a FASTA file (gzipped or not), read with polars-bio; the header is the name
    and, when present, the description after a space, as on the ``>`` line."""
    import polars_bio as pb

    frame = pb.scan_fasta(str(path)).collect()
    for name, description, sequence in frame.select("name", "description", "sequence").iter_rows():
        yield (f"{name} {description}" if description else name), sequence


def read_csv(path: str | Path, sequence_column: str = "CDS", keep: dict[str, str] | None = None
             ) -> Iterator[tuple[str, str]]:
    """``(header, sequence)`` pairs from a CSV with a sequence column (the SynCodonLM corpus: CDS, Species,
    Group, Set). The header joins the other columns with ``|``; ``keep`` keeps only rows whose columns hold
    the given values, e.g. ``{"Set": "train"}``. A cut last line, as in a truncated download, is skipped."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    csv.field_size_limit(sys.maxsize)
    with opener(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or sequence_column not in reader.fieldnames:
            raise ValueError(f"{path} has no {sequence_column!r} column (columns: {reader.fieldnames})")
        others = [c for c in reader.fieldnames if c != sequence_column]
        try:
            for row in reader:
                if keep and any(row.get(k) != v for k, v in keep.items()):
                    continue
                yield "|".join(row.get(c) or "" for c in others), row[sequence_column] or ""
        except (EOFError, csv.Error):
            return


def is_csv(path: str | Path) -> bool:
    name = Path(path).name.lower()
    return name.endswith(".csv") or name.endswith(".csv.gz")


def read_sequences(path: str | Path, keep: dict[str, str] | None = None) -> Iterator[tuple[str, str]]:
    """FASTA or CSV, chosen by the file name."""
    return read_csv(path, keep=keep) if is_csv(path) else read_fasta(path)


def sequence_files(folder: str | Path) -> list[Path]:
    """The FASTA and CSV files of a downloaded source folder, in name order."""
    patterns = ("*.fna.gz", "*.fa.gz", "*.fasta.gz", "*.fna", "*.fa", "*.fasta", "*.csv.gz", "*.csv")
    return sorted({p for pattern in patterns for p in Path(folder).glob(pattern)})


def documents(tokens: list[int], seq_len: int) -> Iterator[list[int]]:
    """One encoded sequence (``<cls>`` codons ``<eos>``) -> documents of at most ``seq_len`` tokens.

    Each piece starts with ``<cls>``; only the last keeps the ``<eos>``.
    """
    if seq_len < 3:
        raise ValueError("seq_len must leave room for <cls>, a codon and <eos>")
    body = tokens[1:-1] if tokens and tokens[0] == CLS_ID and tokens[-1] == EOS_ID else tokens
    step = seq_len - 1
    for start in range(0, len(body), step):
        piece = [CLS_ID, *body[start:start + step]]
        if start + step >= len(body):
            if len(piece) < seq_len:
                piece.append(EOS_ID)
            else:
                # the last piece filled the row: its <eos> starts one more, tiny document
                yield piece
                piece = [CLS_ID, EOS_ID]
        yield piece


def pack(docs: Iterable[list[int]], seq_len: int) -> Iterator[np.ndarray]:
    """Greedy first-fit packing of documents (each at most ``seq_len`` tokens) into ``uint8`` rows."""
    row: list[int] = []
    for doc in docs:
        if len(doc) > seq_len:
            raise ValueError(f"a document of {len(doc)} tokens does not fit a row of {seq_len}")
        if len(row) + len(doc) > seq_len:
            yield np.array(row + [PAD_ID] * (seq_len - len(row)), dtype=np.uint8)
            row = []
        row.extend(doc)
    if row:
        yield np.array(row + [PAD_ID] * (seq_len - len(row)), dtype=np.uint8)


_GENE = re.compile(r"\[gene=([^\]]+)\]")


def gene_of(header: str) -> str:
    """The gene a FASTA record belongs to: NCBI's ``[gene=...]`` tag, else the whole header."""
    m = _GENE.search(header)
    return m.group(1) if m else header


def held_out(gene: str, fraction: float, seed: int = 0) -> bool:
    """Whether a gene falls in the held-out set: a fixed hash of its name, so every isoform goes the same way."""
    return fraction > 0 and zlib.crc32(f"{seed}:{gene}".encode()) / 2**32 < fraction


def prepare(paths: Iterable[str | Path], out_dir: str | Path, seq_len: int = 1024,
            starts: tuple[str, ...] = ("ATG",), max_sequences: int | None = None,
            keep: dict[str, str] | None = None, holdout_genes: float = 0.0, seed: int = 0) -> dict:
    """Filter, tokenize and pack coding sequences (FASTA or CSV) into ``out_dir/tokens.npy`` and ``meta.json``.

    With ``holdout_genes`` > 0, that fraction of genes (by :func:`gene_of`, all isoforms together) goes to a
    sibling directory ``<out_dir>_holdout`` instead, for a gene-held-out validation. Paralogs with other gene
    names still cross the split; only clustering removes those.

    Holds the packed rows in memory, which is fine for tens of millions of codons; for a corpus of
    SynCodonLM's size use :func:`kron_dna.stream.prepare_stream`.
    """
    out = Path(out_dir)
    targets = {"train": out} | ({"holdout": out.parent / f"{out.name}_holdout"} if holdout_genes > 0 else {})
    counts = {k: {"kept": 0, "codons": 0, "genes": set()} for k in targets}
    rejected: Counter[str] = Counter()
    docs: dict[str, list[list[int]]] = {k: [] for k in targets}
    kept = 0
    for path in paths:
        for header, seq in read_sequences(path, keep=keep):
            if max_sequences is not None and kept >= max_sequences:
                break
            reason = check_cds(seq, starts=starts)
            if reason is not None:
                rejected[reason] += 1
                continue
            gene = gene_of(header)
            split = "holdout" if held_out(gene, holdout_genes, seed) else "train"
            tokens = encode(seq)
            kept += 1
            c = counts[split]
            c["kept"] += 1
            c["codons"] += len(tokens) - 2
            c["genes"].add(gene)
            docs[split].extend(documents(tokens, seq_len))
    metas = {}
    for split, target in targets.items():
        rows = list(pack(docs[split], seq_len))
        if not rows:
            raise ValueError(f"no sequence passed the checks for the {split} split")
        tokens = np.stack(rows)
        target.mkdir(parents=True, exist_ok=True)
        np.save(target / "tokens.npy", tokens)
        c = counts[split]
        meta = {"seq_len": seq_len, "rows": int(tokens.shape[0]), "sequences": c["kept"], "codons": c["codons"],
                "fill": float((tokens != PAD_ID).mean()), "rejected": dict(rejected), "starts": list(starts),
                "vocab_size": VOCAB_SIZE, "inputs": [str(p) for p in paths], "keep": keep or {}}
        if holdout_genes > 0:
            meta |= {"split": split, "genes": len(c["genes"]), "holdout_genes": holdout_genes, "seed": seed}
        (target / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        metas[split] = meta
    return metas["train"] if holdout_genes <= 0 else metas


def doc_ids_from_tokens(tokens: torch.Tensor) -> torch.Tensor:
    """``[B, T]`` packed tokens -> ``[B, T]`` document ids: one per ``<cls>``, -1 on padding.

    A row that does not start with ``<cls>`` has its leading tokens in document 0.
    """
    ids = (tokens == CLS_ID).long().cumsum(dim=1)
    return torch.where(tokens == PAD_ID, torch.full_like(ids, -1), ids)


def mask_codons(tokens: torch.Tensor, mask_prob: float = 0.15, generator: torch.Generator | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
    """BERT masking over codon tokens only: ``(inputs, labels)``, labels ``-100`` where no loss is taken.

    Of the chosen codons, 80% become ``<mask>``, 10% a random codon and 10% stay; special tokens and
    ``<unk>`` codons are never chosen.
    """
    tokens = tokens.long()
    is_codon = tokens >= FIRST_CODON_ID
    draw = torch.rand(tokens.shape, generator=generator, device=tokens.device)
    chosen = is_codon & (draw < mask_prob)
    labels = torch.where(chosen, tokens, torch.full_like(tokens, IGNORE_INDEX))
    how = torch.rand(tokens.shape, generator=generator, device=tokens.device)
    random_codons = torch.randint(FIRST_CODON_ID, VOCAB_SIZE, tokens.shape, generator=generator,
                                  device=tokens.device)
    inputs = torch.where(chosen & (how < 0.8), torch.full_like(tokens, MASK_ID), tokens)
    inputs = torch.where(chosen & (how >= 0.8) & (how < 0.9), random_codons, inputs)
    return inputs, labels


def shard_files(directory: str | Path) -> list[Path]:
    """The token arrays of a prepared directory: ``tokens.npy``, or the shards ``meta.json`` lists."""
    directory = Path(directory)
    if (directory / "tokens.npy").is_file():
        return [directory / "tokens.npy"]
    meta_path = directory / "meta.json"
    if meta_path.is_file():
        shards = json.loads(meta_path.read_text()).get("shards")
        if shards:
            return [directory / s["file"] for s in shards]
    raise FileNotFoundError(f"{directory} holds neither tokens.npy nor a meta.json listing shards")


def is_packed(directory: str | Path) -> bool:
    try:
        return bool(shard_files(directory))
    except FileNotFoundError:
        return False


class PackedRows(torch.utils.data.Dataset):
    """The rows of a prepared directory, memory-mapped, as ``long`` tensors.

    The directory holds one ``tokens.npy`` or the shards its ``meta.json`` lists (``prepare-stream``);
    shards are opened lazily, per process, so DataLoader workers never copy them.
    """

    def __init__(self, directory: str | Path, rows: np.ndarray | None = None) -> None:
        self.files = shard_files(directory)
        counts = [int(np.load(f, mmap_mode="r").shape[0]) for f in self.files]
        self.offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self.n_total = int(self.offsets[-1])
        self.rows = np.arange(self.n_total) if rows is None else np.asarray(rows)
        self._maps: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _shard(self, k: int) -> np.ndarray:
        if k not in self._maps:
            self._maps[k] = np.load(self.files[k], mmap_mode="r")
        return self._maps[k]

    def __getitem__(self, i: int) -> torch.Tensor:
        row = int(self.rows[i])
        k = int(np.searchsorted(self.offsets, row, side="right") - 1)
        return torch.from_numpy(np.asarray(self._shard(k)[row - self.offsets[k]], dtype=np.int64))

    def __getstate__(self) -> dict:
        return {**self.__dict__, "_maps": {}}
