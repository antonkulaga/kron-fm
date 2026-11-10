"""Downstream probes: a frozen representation, a ridge head, Spearman on the task's held-out split.

The research report's evaluation advice, in its cheapest form: every model-based number is reported next
to model-free baselines on the same split and head, so a gain that a GC count or a (position, codon)
lookup also gets is visible as such.

Features (:data:`FEATURES`):

* ``gc``: GC content overall and at each codon position (4 numbers).
* ``codon_usage``: the 64 codon frequencies (codon-usage bias).
* ``cai``: the codon adaptation index against a reference codon usage (a genome's CDS), 1 number.
* ``kmer``: nucleotide 1- to 4-mer frequencies (340 numbers).
* ``onehot_pos``: one-hot (position, codon), CodonBench's memorisation canary.
* ``model``: a kron-dna run's mean final hidden state over the sequence's tokens; with ``random_init`` the
  same configuration untrained (the random-init twin).

The head is ridge regression with its penalty chosen on the task's validation split and the score is
Spearman's rho on its test split; no choice ever sees the test split.
"""
from __future__ import annotations

import csv
import math
from collections import Counter
from collections.abc import Iterable, Sequence
from itertools import product
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from kron_dna.codons import CODONS, FIRST_CODON_ID, GENETIC_CODE, PAD_ID, clean, encode

FEATURES = ("gc", "codon_usage", "cai", "kmer", "onehot_pos", "model")
_CODON_INDEX = {c: i for i, c in enumerate(CODONS)}


def read_task(path: str | Path, sequence_column: str = "Sequence", target_column: str = "Value",
              split_column: str = "Split") -> dict[str, tuple[list[str], np.ndarray]]:
    """A CodonBERT-style CSV -> ``{split: (sequences as DNA, targets)}``."""
    out: dict[str, tuple[list[str], list[float]]] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            seqs, ys = out.setdefault(row[split_column].strip().lower(), ([], []))
            seqs.append(clean(row[sequence_column]))
            ys.append(float(row[target_column]))
    return {k: (s, np.asarray(y, dtype=np.float64)) for k, (s, y) in out.items()}


def _codons(seq: str) -> list[str]:
    return [seq[i:i + 3] for i in range(0, len(seq) - len(seq) % 3, 3)]


def gc_features(seqs: Sequence[str]) -> np.ndarray:
    out = np.zeros((len(seqs), 4))
    for n, seq in enumerate(seqs):
        gc = np.array([b in "GC" for b in seq], dtype=float)
        out[n, 0] = gc.mean() if len(gc) else 0.0
        for k in range(3):
            out[n, 1 + k] = gc[k::3].mean() if len(gc[k::3]) else 0.0
    return out


def codon_usage(seqs: Sequence[str]) -> np.ndarray:
    out = np.zeros((len(seqs), 64))
    for n, seq in enumerate(seqs):
        for c in _codons(seq):
            if c in _CODON_INDEX:
                out[n, _CODON_INDEX[c]] += 1
        out[n] /= max(1.0, out[n].sum())
    return out


def relative_adaptiveness(reference: Iterable[str]) -> dict[str, float]:
    """Sharp and Li's w: each codon's count over its most used synonym's, from reference CDS (pseudocount 0.5)."""
    counts: Counter[str] = Counter()
    for seq in reference:
        counts.update(c for c in _codons(clean(seq)) if c in GENETIC_CODE)
    best: dict[str, float] = {}
    for c in CODONS:
        aa = GENETIC_CODE[c]
        best[aa] = max(best.get(aa, 0.0), counts[c] + 0.5)
    return {c: (counts[c] + 0.5) / best[GENETIC_CODE[c]] for c in CODONS}


def cai(seqs: Sequence[str], w: dict[str, float]) -> np.ndarray:
    """Geometric mean of w over the codons that have a synonym (Met, Trp and stops left out)."""
    skip = {"ATG", "TGG"} | {c for c in CODONS if GENETIC_CODE[c] == "*"}
    out = np.zeros((len(seqs), 1))
    for n, seq in enumerate(seqs):
        logs = [math.log(w[c]) for c in _codons(seq) if c in w and c not in skip]
        out[n, 0] = math.exp(sum(logs) / len(logs)) if logs else 0.0
    return out


def kmer_features(seqs: Sequence[str], ks: Sequence[int] = (1, 2, 3, 4)) -> np.ndarray:
    blocks = []
    for k in ks:
        index = {"".join(p): i for i, p in enumerate(product("ACGT", repeat=k))}
        block = np.zeros((len(seqs), len(index)))
        for n, seq in enumerate(seqs):
            for i in range(len(seq) - k + 1):
                j = index.get(seq[i:i + k])
                if j is not None:
                    block[n, j] += 1
            block[n] /= max(1.0, block[n].sum())
        blocks.append(block)
    return np.concatenate(blocks, axis=1)


def onehot_pos(seqs: Sequence[str], max_codons: int | None = None) -> np.ndarray:
    max_codons = max_codons or max(len(s) // 3 for s in seqs)
    out = np.zeros((len(seqs), max_codons * 64), dtype=np.float32)
    for n, seq in enumerate(seqs):
        for p, c in enumerate(_codons(seq)[:max_codons]):
            if c in _CODON_INDEX:
                out[n, p * 64 + _CODON_INDEX[c]] = 1.0
    return out


@torch.no_grad()
def model_embeddings(model: torch.nn.Module, seqs: Sequence[str], device: torch.device, batch_size: int = 32,
                     max_tokens: int = 1024, bf16: bool = True) -> np.ndarray:
    """Mean final hidden state over each sequence's own tokens (``<cls>`` to ``<eos>``), one row per sequence."""
    model = model.to(device).eval()
    out = []
    autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16 and device.type == "cuda")
    for start in range(0, len(seqs), batch_size):
        batch = [encode(s)[:max_tokens] for s in seqs[start:start + batch_size]]
        width = max(len(t) for t in batch)
        tokens = torch.full((len(batch), width), PAD_ID, dtype=torch.long)
        for i, t in enumerate(batch):
            tokens[i, :len(t)] = torch.tensor(t)
        tokens = tokens.to(device)
        with autocast:
            _, hidden = model(tokens)
        valid = (tokens != PAD_ID).unsqueeze(-1).float()
        out.append(((hidden.float() * valid).sum(1) / valid.sum(1)).cpu().numpy())
    return np.concatenate(out)


def _standardize(train: np.ndarray, *others: np.ndarray) -> list[np.ndarray]:
    mean, std = train.mean(0), train.std(0)
    std[std == 0] = 1.0
    return [(x - mean) / std for x in (train, *others)]


def ridge_probe(x: dict[str, np.ndarray], y: dict[str, np.ndarray],
                alphas: Sequence[float] = tuple(10.0 ** np.arange(-3, 6))) -> dict:
    """Ridge on ``train``, penalty by Spearman on ``val``, then refit on train+val and score ``test``."""
    xtr, xva, xte = _standardize(x["train"], x["val"], x["test"])
    ymean = y["train"].mean()

    def fit(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
        u, s, vt = np.linalg.svd(a, full_matrices=False)
        return vt.T @ ((s / (s ** 2 + alpha)) * (u.T @ b))

    def rho(pred: np.ndarray, truth: np.ndarray) -> float:
        r = spearmanr(pred, truth).statistic
        return float(r) if np.isfinite(r) else 0.0

    scores = {alpha: rho(xva @ fit(xtr, y["train"] - ymean, alpha), y["val"]) for alpha in alphas}
    best = max(scores, key=scores.get)
    xall, xte2 = _standardize(np.concatenate([x["train"], x["val"]]), x["test"])
    yall = np.concatenate([y["train"], y["val"]])
    w = fit(xall, yall - yall.mean(), best)
    return {"alpha": best, "val_spearman": scores[best], "test_spearman": rho(xte2 @ w, y["test"]),
            "dims": int(x["train"].shape[1])}
