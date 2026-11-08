"""Masked-codon objectives: plain cross-entropy and the two synonymy-aware losses the research favours.

Both synonymy-aware losses use the many-to-one map from codons to amino acids, cost nothing extra to
compute, and have never been compared on the same corpus and backbone (see the report in ``reports/``):

* :func:`synonymous_cross_entropy` (SynCodonLM, NAR 2026): logits of codons that are not synonymous
  with the true codon are set to minus infinity, so the loss asks only "which synonymous codon";
* :func:`hierarchical_cross_entropy` (HELM, arXiv:2410.12459): a cross-entropy over a codon tree,
  root -> {sense, stop} -> amino acid -> codon, with each edge's term weighted down the deeper it sits,
  so confusing two synonymous codons costs less than confusing two amino acids.

Every loss takes ``logits [..., VOCAB_SIZE]`` and ``labels [...]`` holding a codon token id where the
loss is taken and ``-100`` elsewhere (the masking in :mod:`kron_dna.data` never asks for a special token).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from kron_dna.codons import AMINO_ACIDS, FIRST_CODON_ID, VOCAB_SIZE, codon_amino_acid_index, synonym_matrix

IGNORE_INDEX = -100
OBJECTIVES = ("mlm", "synonymous", "hierarchical")


def _selected(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    keep = labels != IGNORE_INDEX
    return logits[keep].float(), labels[keep]


def mlm_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Plain cross-entropy over the whole vocabulary at the labelled positions, in nats per codon."""
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), labels.reshape(-1),
                           ignore_index=IGNORE_INDEX)


def synonymous_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy with every non-synonymous codon's logit at minus infinity (SynCodonLM).

    Methionine and tryptophan have one codon each, so their terms are zero; they still count in the
    mean, as in a plain masked-LM loss.
    """
    sel, lab = _selected(logits, labels)
    allowed = synonym_matrix().to(sel.device)[lab]
    return F.cross_entropy(sel.masked_fill(~allowed, float("-inf")), lab)


class CodonTree:
    """The index maps of the tree root -> {sense, stop} -> amino acid (and ``*``) -> codon."""

    def __init__(self, device: torch.device | None = None) -> None:
        aa = codon_amino_acid_index()[FIRST_CODON_ID:]
        n_aa = len(AMINO_ACIDS)
        #: ``[64, n_aa]`` and ``[n_aa, 2]`` membership matrices (class 0: sense, 1: stop).
        self.codon_to_aa = F.one_hot(aa, n_aa).float().to(device)
        aa_class = torch.tensor([1 if a == "*" else 0 for a in AMINO_ACIDS])
        self.aa_to_class = F.one_hot(aa_class, 2).float().to(device)
        self.aa_of_codon = aa.to(device)
        self.class_of_aa = aa_class.to(device)


def hierarchical_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, alpha: float = 0.2) -> torch.Tensor:
    """HELM's hierarchical cross-entropy over the codon tree, in nats per codon.

    The prediction is the softmax over the 64 codon logits (the special tokens are never targets).
    The loss telescopes the codon's log-probability along its path,

        - [ w_1 log p(class) + w_2 log p(aa | class) + w_3 log p(codon | aa) ],

    with ``w_d = exp(-alpha * (d - 1))`` for an edge ending at depth ``d``: the top edge counts fully and
    the synonymous choice least, so a synonymous confusion is penalised less than a cross-amino-acid
    one. At ``alpha = 0`` it is the plain cross-entropy over codons. HELM reports ``alpha = 0.2`` best
    among 0.2, 0.4 and 0.6. (Bertinetto et al.'s HXE writes the weight as ``exp(-alpha * h(C))`` with
    the node's height; this is the convention that matches HELM's stated intent.)
    """
    sel, lab = _selected(logits, labels)
    tree = CodonTree(sel.device)
    log_codon = F.log_softmax(sel[:, FIRST_CODON_ID:VOCAB_SIZE], dim=-1)          # [N, 64]
    # log p(aa) and log p(class): log-sum-exp of the children, through the membership matrices
    log_aa = torch.logsumexp(log_codon.unsqueeze(-1) + tree.codon_to_aa.log().unsqueeze(0), dim=1)  # [N, 21]
    log_class = torch.logsumexp(log_aa.unsqueeze(-1) + tree.aa_to_class.log().unsqueeze(0), dim=1)  # [N, 2]
    codon = (lab - FIRST_CODON_ID).long()
    aa = tree.aa_of_codon[codon]
    cls = tree.class_of_aa[aa]
    lp_codon = log_codon.gather(1, codon.unsqueeze(1)).squeeze(1)
    lp_aa = log_aa.gather(1, aa.unsqueeze(1)).squeeze(1)
    lp_class = log_class.gather(1, cls.unsqueeze(1)).squeeze(1)
    w1, w2, w3 = (math.exp(-alpha * d) for d in range(3))
    return -(w1 * lp_class + w2 * (lp_aa - lp_class) + w3 * (lp_codon - lp_aa)).mean()


def objective_loss(name: str, logits: torch.Tensor, labels: torch.Tensor, alpha: float = 0.2) -> torch.Tensor:
    """The training loss called ``name`` (one of :data:`OBJECTIVES`)."""
    if name == "mlm":
        return mlm_cross_entropy(logits, labels)
    if name == "synonymous":
        return synonymous_cross_entropy(logits, labels)
    if name == "hierarchical":
        return hierarchical_cross_entropy(logits, labels, alpha=alpha)
    raise ValueError(f"objective must be one of {OBJECTIVES}, got {name!r}")
