"""Codon vocabulary, the standard genetic code, and coding-sequence checks.

The vocabulary is 5 special tokens followed by the 64 codons in the standard table's TCAG order, 69
tokens in all, so a token fits in a ``uint8``. Sequences are read as DNA (``U`` is read as ``T``).
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import product

import torch

BASES = "TCAG"
#: The standard genetic code (NCBI table 1), in TCAG order; ``*`` is a stop.
_AMINO_ACIDS_TCAG = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODONS: tuple[str, ...] = tuple("".join(c) for c in product(BASES, repeat=3))
GENETIC_CODE: dict[str, str] = dict(zip(CODONS, _AMINO_ACIDS_TCAG, strict=True))
STOP_CODONS: tuple[str, ...] = tuple(c for c in CODONS if GENETIC_CODE[c] == "*")
#: The 20 amino acids, then the stop.
AMINO_ACIDS: tuple[str, ...] = (*sorted(set(_AMINO_ACIDS_TCAG) - {"*"}), "*")

SPECIAL_TOKENS: tuple[str, ...] = ("<pad>", "<mask>", "<unk>", "<cls>", "<eos>")
PAD_ID, MASK_ID, UNK_ID, CLS_ID, EOS_ID = range(len(SPECIAL_TOKENS))
N_SPECIAL = len(SPECIAL_TOKENS)
VOCAB: tuple[str, ...] = (*SPECIAL_TOKENS, *CODONS)
VOCAB_SIZE = len(VOCAB)
TOKEN_ID: dict[str, int] = {tok: i for i, tok in enumerate(VOCAB)}
#: The id of the first codon; codon ``i`` of :data:`CODONS` has id ``FIRST_CODON_ID + i``.
FIRST_CODON_ID = N_SPECIAL


def codon_amino_acid_index() -> torch.Tensor:
    """``[VOCAB_SIZE]`` long: each codon token's index into :data:`AMINO_ACIDS`, -1 for special tokens."""
    index = torch.full((VOCAB_SIZE,), -1, dtype=torch.long)
    for i, codon in enumerate(CODONS):
        index[FIRST_CODON_ID + i] = AMINO_ACIDS.index(GENETIC_CODE[codon])
    return index


def synonym_matrix() -> torch.Tensor:
    """``[VOCAB_SIZE, VOCAB_SIZE]`` bool: true where two tokens are codons of the same amino acid (or stop).

    A special token is synonymous with itself only, so a row is never empty.
    """
    aa = codon_amino_acid_index()
    same = (aa.unsqueeze(0) == aa.unsqueeze(1)) & (aa >= 0).unsqueeze(0)
    return same | torch.eye(VOCAB_SIZE, dtype=torch.bool)


def clean(seq: str) -> str:
    """Upper case, ``U`` read as ``T``, whitespace removed."""
    return "".join(seq.split()).upper().replace("U", "T")


def encode(seq: str, add_cls: bool = True, add_eos: bool = True) -> list[int]:
    """A coding sequence -> token ids, one per codon; a codon with any base outside ACGT is ``<unk>``.

    A trailing partial codon is dropped; :func:`check_cds` refuses such sequences before they get here.
    """
    seq = clean(seq)
    ids = [TOKEN_ID.get(seq[i:i + 3], UNK_ID) for i in range(0, len(seq) - len(seq) % 3, 3)]
    return [CLS_ID] * add_cls + ids + [EOS_ID] * add_eos


def decode(ids: Iterable[int]) -> str:
    """Token ids -> the codons they stand for; special tokens are skipped, ``<unk>`` becomes ``NNN``."""
    out = []
    for i in ids:
        i = int(i)
        if i >= FIRST_CODON_ID:
            out.append(VOCAB[i])
        elif i == UNK_ID:
            out.append("NNN")
    return "".join(out)


def translate(seq: str) -> str:
    """A coding sequence -> its protein in one-letter codes, ``*`` for a stop and ``X`` for an unreadable codon."""
    seq = clean(seq)
    return "".join(GENETIC_CODE.get(seq[i:i + 3], "X") for i in range(0, len(seq) - len(seq) % 3, 3))


def check_cds(seq: str, starts: Sequence[str] = ("ATG",), require_stop: bool = True,
              max_unknown_fraction: float = 0.0, min_codons: int = 30) -> str | None:
    """Why a sequence is not a usable coding sequence, or ``None`` if it is.

    The checks are the ones every published codon model applies (CodonBERT, SynCodonLM): in frame
    (length a multiple of 3), a start codon from ``starts`` (``("ATG", "GTG", "TTG")`` keeps bacterial
    alternative starts), a stop at the end and none inside, and at most ``max_unknown_fraction`` of
    codons with a base outside ACGT.
    """
    seq = clean(seq)
    if len(seq) % 3:
        return "frame"
    n_codons = len(seq) // 3
    if n_codons < min_codons:
        return "short"
    if seq[:3] not in starts:
        return "start"
    codons = [seq[i:i + 3] for i in range(0, len(seq), 3)]
    unknown = sum(c not in GENETIC_CODE for c in codons)
    if unknown > max_unknown_fraction * n_codons:
        return "alphabet"
    if require_stop and codons[-1] not in STOP_CODONS:
        return "stop"
    if any(c in STOP_CODONS for c in codons[:-1]):
        return "internal_stop"
    return None
