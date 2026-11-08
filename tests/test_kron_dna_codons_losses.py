"""kron-dna's vocabulary, genetic code, coding-sequence checks and losses (CPU only)."""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from kron_dna import codons as C
from kron_dna.losses import (
    IGNORE_INDEX,
    hierarchical_cross_entropy,
    mlm_cross_entropy,
    objective_loss,
    synonymous_cross_entropy,
)


def test_vocabulary_is_five_specials_and_the_64_codons():
    assert C.VOCAB_SIZE == 69 and C.VOCAB_SIZE < 256
    assert len(set(C.CODONS)) == 64 and C.VOCAB[C.FIRST_CODON_ID] == "TTT"
    assert sorted(C.STOP_CODONS) == ["TAA", "TAG", "TGA"]
    assert sum(aa != "*" for aa in C.GENETIC_CODE.values()) == 61
    assert len(C.AMINO_ACIDS) == 21 and C.AMINO_ACIDS[-1] == "*"


def test_genetic_code_matches_known_codons():
    assert C.translate("ATGTGGTTATAA") == "MWL*"
    assert C.translate("AUGGCC") == "MA", "RNA is read as DNA"
    assert C.translate("ATGNNN") == "MX"


def test_synonym_matrix_groups_codons_by_amino_acid():
    syn = C.synonym_matrix()
    tok = C.TOKEN_ID
    assert syn[tok["TTA"]].sum() == 6, "leucine has six codons"
    assert syn[tok["ATG"]].sum() == 1 and syn[tok["TGG"]].sum() == 1
    assert syn[tok["TAA"], tok["TGA"]] and not syn[tok["TAA"], tok["TGG"]]
    assert syn[C.MASK_ID].sum() == 1 and syn[C.MASK_ID, C.MASK_ID], "a special token is its own only synonym"
    assert torch.equal(syn, syn.T)


def test_encode_decode_round_trip_and_unknown_codons():
    seq = "ATGGCCAAATAA"
    ids = C.encode(seq)
    assert ids[0] == C.CLS_ID and ids[-1] == C.EOS_ID and len(ids) == 6
    assert C.decode(ids) == seq
    assert C.encode("ATGNCA", add_cls=False, add_eos=False) == [C.TOKEN_ID["ATG"], C.UNK_ID]
    assert C.decode([C.TOKEN_ID["ATG"], C.UNK_ID, C.PAD_ID]) == "ATGNNN"


@pytest.mark.parametrize("seq, reason", [
    ("ATG" + "GCC" * 40 + "TA", "frame"),
    ("ATG" + "GCC" * 5 + "TAA", "short"),
    ("GTG" + "GCC" * 40 + "TAA", "start"),
    ("ATG" + "GCC" * 40 + "GCC", "stop"),
    ("ATG" + "GCC" * 20 + "TAG" + "GCC" * 20 + "TAA", "internal_stop"),
    ("ATG" + "GNC" * 40 + "TAA", "alphabet"),
    ("ATG" + "GCC" * 40 + "TAA", None),
])
def test_check_cds(seq, reason):
    assert C.check_cds(seq) == reason


def test_check_cds_accepts_alternative_starts_when_asked():
    assert C.check_cds("GTG" + "GCC" * 40 + "TAA", starts=("ATG", "GTG", "TTG")) is None


def _logits_and_labels(n=64, seed=0):
    torch.manual_seed(seed)
    logits = torch.randn(n, C.VOCAB_SIZE)
    labels = torch.randint(C.FIRST_CODON_ID, C.VOCAB_SIZE, (n,))
    labels[::7] = IGNORE_INDEX
    return logits, labels


def test_hierarchical_loss_at_alpha_zero_is_cross_entropy_over_codons():
    logits, labels = _logits_and_labels()
    keep = labels != IGNORE_INDEX
    codon_ce = F.cross_entropy(logits[keep, C.FIRST_CODON_ID:], labels[keep] - C.FIRST_CODON_ID)
    assert torch.allclose(hierarchical_cross_entropy(logits, labels, alpha=0.0), codon_ce, atol=1e-5)


def test_hierarchical_loss_penalises_a_synonymous_confusion_less():
    tok = C.TOKEN_ID
    label = torch.tensor([tok["TTA"]])                     # leucine
    for wrong, synonymous in ((tok["CTG"], True), (tok["ATG"], False)):
        logits = torch.zeros(1, C.VOCAB_SIZE)
        logits[0, wrong] = 8.0                             # confident in the wrong codon
        loss = float(hierarchical_cross_entropy(logits, label, alpha=0.2))
        if synonymous:
            syn_loss = loss
        else:
            assert syn_loss < loss
    # at alpha 0 the two confusions cost the same: only the tree weighting separates them
    a = hierarchical_cross_entropy(torch.zeros(1, C.VOCAB_SIZE).index_fill(1, torch.tensor([tok["CTG"]]), 8.0),
                                   label, alpha=0.0)
    b = hierarchical_cross_entropy(torch.zeros(1, C.VOCAB_SIZE).index_fill(1, torch.tensor([tok["ATG"]]), 8.0),
                                   label, alpha=0.0)
    assert torch.allclose(a, b)


def test_synonymous_loss_only_sees_synonyms():
    logits, labels = _logits_and_labels()
    assert float(synonymous_cross_entropy(logits, labels)) <= float(mlm_cross_entropy(logits, labels))
    met = torch.tensor([C.TOKEN_ID["ATG"]])
    assert float(synonymous_cross_entropy(torch.randn(1, C.VOCAB_SIZE), met)) == 0.0, "one codon, nothing to choose"
    # raising a non-synonymous logit changes nothing
    leu = torch.tensor([C.TOKEN_ID["TTA"]])
    base = torch.zeros(1, C.VOCAB_SIZE)
    bumped = base.clone()
    bumped[0, C.TOKEN_ID["GCC"]] = 50.0
    assert torch.allclose(synonymous_cross_entropy(base, leu), synonymous_cross_entropy(bumped, leu))
    assert math.isclose(float(synonymous_cross_entropy(base, leu)), math.log(6), rel_tol=1e-6)


@pytest.mark.parametrize("name", ["mlm", "synonymous", "hierarchical"])
def test_objectives_are_differentiable_and_finite(name):
    logits, labels = _logits_and_labels()
    logits.requires_grad_(True)
    loss = objective_loss(name, logits, labels)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    with pytest.raises(ValueError):
        objective_loss("nope", logits, labels)
