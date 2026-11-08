"""kron-dna's packing, masking and model: documents never read each other, and both directions are read."""
from __future__ import annotations

import gzip
import json

import numpy as np
import pytest
import torch

from kron_core.segments import cu_seqlens_from_doc_ids, reverse_cu_seqlens
from kron_dna import codons as C
from kron_dna.data import PackedRows, doc_ids_from_tokens, documents, mask_codons, pack, prepare, read_fasta
from kron_dna.losses import IGNORE_INDEX
from kron_dna.model import KronDNA, KronDNAConfig


def _cds(n_codons: int, seed: int) -> str:
    rng = np.random.default_rng(seed)
    sense = [c for c in C.CODONS if C.GENETIC_CODE[c] != "*"]
    return "ATG" + "".join(rng.choice(sense, n_codons)) + "TAA"


# ---- segments and packing ------------------------------------------------------------------------

def test_cu_seqlens_break_at_rows_and_documents():
    doc_ids = torch.tensor([[1, 1, 2, 2, -1], [1, 2, 2, 2, 2]])
    cu = cu_seqlens_from_doc_ids(doc_ids)
    assert cu.dtype == torch.int32 and cu.tolist() == [0, 2, 4, 5, 6, 10]
    assert reverse_cu_seqlens(cu).tolist() == [0, 4, 5, 6, 8, 10]


def test_documents_split_long_sequences_and_keep_one_eos():
    tokens = C.encode(_cds(10, 0))                          # cls + 12 codons + eos
    docs = list(documents(tokens, seq_len=5))
    assert all(d[0] == C.CLS_ID and len(d) <= 5 for d in docs)
    assert sum(d.count(C.EOS_ID) for d in docs) == 1 and docs[-1][-1] == C.EOS_ID
    body = [t for d in docs for t in d if t not in (C.CLS_ID, C.EOS_ID)]
    assert body == tokens[1:-1]


def test_pack_fills_rows_without_splitting_documents():
    docs = [[C.CLS_ID] + [C.FIRST_CODON_ID] * n + [C.EOS_ID] for n in (3, 4, 2, 6)]
    rows = list(pack(docs, seq_len=10))
    assert all(r.dtype == np.uint8 and len(r) == 10 for r in rows)
    flat = [list(r[r != C.PAD_ID]) for r in rows]
    assert sum(len(f) for f in flat) == sum(len(d) for d in docs)
    for r in rows:
        ids = doc_ids_from_tokens(torch.from_numpy(r.astype(np.int64)).unsqueeze(0))[0]
        assert ((ids == -1) == torch.from_numpy(r == C.PAD_ID)).all()


def test_prepare_filters_packs_and_reports(tmp_path):
    fasta = tmp_path / "cds.fa.gz"
    good = [_cds(80, i) for i in range(5)]
    with gzip.open(fasta, "wt") as fh:
        for i, seq in enumerate(good):
            fh.write(f">good{i}\n{seq[:60]}\n{seq[60:]}\n")
        fh.write(">frameshift\n" + good[0][:-1] + "\n")
        fh.write(">nostart\nGGG" + good[1][3:] + "\n")
    assert [h for h, _ in read_fasta(fasta)][:2] == ["good0", "good1"]
    meta = prepare([fasta], tmp_path / "packed", seq_len=128)
    assert meta["sequences"] == 5 and meta["codons"] == 5 * 82
    assert meta["rejected"] == {"frame": 1, "start": 1}
    assert json.loads((tmp_path / "packed" / "meta.json").read_text())["rows"] == meta["rows"]
    rows = PackedRows(tmp_path / "packed")
    assert len(rows) == meta["rows"] and rows[0].dtype == torch.int64 and rows[0].shape == (128,)


def test_masking_touches_codons_only_at_the_requested_rate():
    tokens = torch.full((64, 256), C.TOKEN_ID["GCC"])
    tokens[:, 0], tokens[:, -8:] = C.CLS_ID, C.PAD_ID
    tokens[:, 5] = C.UNK_ID
    inputs, labels = mask_codons(tokens, 0.15, generator=torch.Generator().manual_seed(0))
    chosen = labels != IGNORE_INDEX
    assert not chosen[:, 0].any() and not chosen[:, -8:].any() and not chosen[:, 5].any()
    assert abs(chosen.float().mean().item() / (247 / 256) - 0.15) < 0.01
    assert ((inputs == C.MASK_ID) <= chosen).all()
    assert abs((inputs[chosen] == C.MASK_ID).float().mean().item() - 0.8) < 0.02
    assert (inputs[~chosen] == tokens[~chosen]).all()


# ---- the model, on the CPU stand-in and on the GPU ------------------------------------------------

def _model(device, **overrides):
    kw = dict(d_model=64, n_heads=2, n_layers=4, ff_units=128, dropout=0.0, attention_every=2)
    kw.update(overrides)
    torch.manual_seed(0)
    return KronDNA(KronDNAConfig(**kw)).to(device).eval()


def _row(device):
    """Two packed rows: row 0 holds documents A and B and padding, row 1 holds document C."""
    a, b, c = (C.encode(_cds(n, s)) for n, s in ((20, 1), (12, 2), (50, 3)))
    row0 = a + b + [C.PAD_ID] * (48 - len(a) - len(b))
    row1 = c[:48]
    return torch.tensor([row0, row1], device=device), len(a)


def _tolerance(device):
    return 2e-2 if device.type == "cuda" else 1e-5


@pytest.mark.parametrize("tie", [False, True])
def test_logits_have_the_vocabulary_and_are_finite(device, tie):
    model = _model(device, tie_directions=tie)
    tokens, _ = _row(device)
    logits, hidden = model(tokens)
    assert logits.shape == (2, 48, C.VOCAB_SIZE) and hidden.shape == (2, 48, 64)
    assert torch.isfinite(logits).all()


def test_a_packed_document_never_reads_its_neighbours(device):
    model = _model(device)
    tokens, len_a = _row(device)
    logits, _ = model(tokens)
    changed = tokens.clone()
    changed[0, len_a + 1:len_a + 6] = C.TOKEN_ID["GGG"]        # inside document B
    changed[1, 10:15] = C.TOKEN_ID["GGG"]                       # inside row 1's document
    logits2, _ = model(changed)
    tol = _tolerance(device)
    assert torch.allclose(logits[0, :len_a], logits2[0, :len_a], atol=tol), "A read B"
    assert not torch.allclose(logits[0, len_a:len_a + 13], logits2[0, len_a:len_a + 13], atol=tol)


def test_every_position_reads_both_directions(device):
    model = _model(device, attention_every=0)                   # recurrence only: no attention to help
    tokens, len_a = _row(device)
    logits, _ = model(tokens)
    changed = tokens.clone()
    changed[0, len_a - 3] = C.TOKEN_ID["GGG"]                   # near the end of document A
    logits2, _ = model(changed)
    tol = _tolerance(device)
    assert not torch.allclose(logits[0, 1:4], logits2[0, 1:4], atol=tol), "the start of A did not read its end"
    assert torch.allclose(logits[0, len_a:], logits2[0, len_a:], atol=tol)


def test_a_document_alone_and_packed_give_the_same_logits(device):
    model = _model(device)
    tokens, len_a = _row(device)
    packed, _ = model(tokens)
    alone, _ = model(tokens[:1, :len_a])
    assert torch.allclose(packed[0, :len_a], alone[0], atol=_tolerance(device))


def test_training_step_reduces_the_loss(device):
    from kron_dna.losses import objective_loss

    model = _model(device).train()
    tokens, _ = _row(device)
    inputs, labels = mask_codons(tokens, 0.3, generator=torch.Generator(device=device).manual_seed(0))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(8):
        logits, _ = model(inputs)
        loss = objective_loss("hierarchical", logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_species_embedding_starts_silent(device):
    model = _model(device, n_species=10)
    tokens, _ = _row(device)
    without, _ = model(tokens)
    with_species, _ = model(tokens, species=torch.tensor([3, 7], device=device))
    assert torch.allclose(without, with_species)
