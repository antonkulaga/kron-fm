"""Coding sequences to packed training rows, and the masking that turns rows into a masked-LM batch.

Pipeline: FASTA (plain or gzipped) -> :func:`kron_dna.codons.check_cds` -> codon tokens -> packed rows
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

import gzip
import json
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import torch

from kron_dna.codons import CLS_ID, EOS_ID, FIRST_CODON_ID, MASK_ID, PAD_ID, VOCAB_SIZE, check_cds, encode
from kron_dna.losses import IGNORE_INDEX


def read_fasta(path: str | Path) -> Iterator[tuple[str, str]]:
    """``(header, sequence)`` pairs from a FASTA file, gzipped when its name ends in ``.gz``."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    header, parts = None, []
    with opener(path, "rt") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(parts)
                header, parts = line[1:], []
            else:
                parts.append(line)
    if header is not None:
        yield header, "".join(parts)


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


def prepare(fasta_paths: Iterable[str | Path], out_dir: str | Path, seq_len: int = 1024,
            starts: tuple[str, ...] = ("ATG",), max_sequences: int | None = None) -> dict:
    """Filter, tokenize and pack FASTA coding sequences into ``out_dir/tokens.npy`` and ``meta.json``.

    Holds the packed rows in memory, which is fine for tens of millions of codons; a corpus of
    SynCodonLM's size (43.5M sequences, ~18B codons) needs sharded output, not written yet.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rejected: Counter[str] = Counter()
    kept = codons = 0

    def docs() -> Iterator[list[int]]:
        nonlocal kept, codons
        for path in fasta_paths:
            for _, seq in read_fasta(path):
                if max_sequences is not None and kept >= max_sequences:
                    return
                reason = check_cds(seq, starts=starts)
                if reason is not None:
                    rejected[reason] += 1
                    continue
                tokens = encode(seq)
                kept += 1
                codons += len(tokens) - 2
                yield from documents(tokens, seq_len)

    rows = list(pack(docs(), seq_len))
    if not rows:
        raise ValueError("no sequence passed the checks")
    tokens = np.stack(rows)
    np.save(out / "tokens.npy", tokens)
    meta = {"seq_len": seq_len, "rows": int(tokens.shape[0]), "sequences": kept, "codons": codons,
            "fill": float((tokens != PAD_ID).mean()), "rejected": dict(rejected), "starts": list(starts),
            "vocab_size": VOCAB_SIZE}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


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


class PackedRows(torch.utils.data.Dataset):
    """The rows of a prepared ``tokens.npy``, memory-mapped, as ``long`` tensors."""

    def __init__(self, directory: str | Path, rows: np.ndarray | None = None) -> None:
        self.tokens = np.load(Path(directory) / "tokens.npy", mmap_mode="r")
        self.rows = np.arange(len(self.tokens)) if rows is None else np.asarray(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> torch.Tensor:
        return torch.from_numpy(np.asarray(self.tokens[self.rows[i]], dtype=np.int64))
