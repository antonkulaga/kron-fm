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


# ---- sharded rows and resumable training ----------------------------------------------------------

def test_packed_rows_read_shards_listed_in_meta(tmp_path):
    a = np.arange(3 * 8, dtype=np.uint8).reshape(3, 8)
    b = np.arange(2 * 8, dtype=np.uint8).reshape(2, 8) + 100
    np.save(tmp_path / "tokens_00000.npy", a)
    np.save(tmp_path / "tokens_00001.npy", b)
    (tmp_path / "meta.json").write_text(json.dumps({"shards": [{"file": "tokens_00000.npy", "rows": 3},
                                                               {"file": "tokens_00001.npy", "rows": 2}]}))
    rows = PackedRows(tmp_path)
    assert len(rows) == 5 and rows.n_total == 5
    assert rows[2].tolist() == a[2].tolist() and rows[3].tolist() == b[0].tolist()
    sub = PackedRows(tmp_path, np.array([4, 0]))
    assert sub[0].tolist() == b[1].tolist() and sub[1].tolist() == a[0].tolist()


def test_train_stops_on_time_and_resumes_from_its_checkpoint(tmp_path, monkeypatch):
    import kron_core.blocks
    from kron_core.testing import CausalMixer
    from typer.testing import CliRunner

    from kron_dna.cli import app

    monkeypatch.setattr(kron_core.blocks, "kronlstm_mixer", CausalMixer)
    fasta = tmp_path / "cds.fa"
    fasta.write_text("".join(f">g{i}\n{_cds(40, i)}\n" for i in range(40)))
    prepare([fasta], tmp_path / "packed", seq_len=64)
    run = tmp_path / "run"
    args = ["train", str(tmp_path / "packed"), "--out", str(run), "--device", "cpu", "--no-bf16",
            "--d-model", "32", "--n-heads", "2", "--n-layers", "2", "--ff-units", "64", "--attention-every", "1",
            "--batch-size", "4", "--steps", "6", "--eval-every", "2", "--eval-batches", "1", "--num-workers", "0",
            "--val-fraction", "0.2"]
    result = CliRunner().invoke(app, args + ["--max-hours", "1e-9"])
    assert result.exit_code == 0, result.output
    first = torch.load(run / "checkpoint.pt", weights_only=False)
    assert first["step"] == 1 and "optimizer" in first
    result = CliRunner().invoke(app, args + ["--snapshot-every", "3"])
    assert result.exit_code == 0, result.output
    assert "resuming at step 1" in result.output
    assert sorted(p.name for p in (run / "snapshots").iterdir()) == ["step_3.pt", "step_6.pt"]
    assert torch.load(run / "checkpoint.pt", weights_only=False)["step"] == 6
    steps = [int(line.split(",")[0]) for line in (run / "metrics.csv").read_text().splitlines()[1:]]
    assert steps == [1, 2, 4, 6]
    result = CliRunner().invoke(app, ["evaluate", str(run), str(tmp_path / "packed"), "--device", "cpu", "--no-bf16",
                                      "--rows", "8", "--batch-size", "4"])
    assert result.exit_code == 0, result.output
    scores = json.loads(next((run / "eval").glob("*.json")).read_text())
    assert scores["step"] == 6 and scores["rows"] == 8
    assert abs(scores["amino_acid_bits"] + scores["synonymous_bits"] - scores["bits_per_codon"]) < 1e-6


def test_masked_scores_split_codon_bits_into_amino_acid_and_synonymous_parts():
    from kron_dna.cli import masked_scores

    gen = torch.Generator().manual_seed(0)
    logits = torch.randn(2, 50, C.VOCAB_SIZE, generator=gen)
    labels = torch.randint(C.FIRST_CODON_ID, C.VOCAB_SIZE, (2, 50), generator=gen)
    labels[:, ::3] = IGNORE_INDEX
    s = masked_scores(logits, labels)
    keep = labels != IGNORE_INDEX
    ce = torch.nn.functional.cross_entropy(logits[keep], labels[keep], reduction="sum")
    assert s["n"] == keep.sum() and abs(s["codon_nats"] - float(ce)) < 1e-3
    assert abs(s["aa_nats"] + s["syn_nats"] - s["codon_nats"]) < 1e-3 and s["syn_nats"] >= 0
    assert 0 <= s["codon_correct"] <= s["aa_correct"] <= s["n"]


def test_prepare_holds_out_whole_genes(tmp_path):
    from kron_dna.data import gene_of, held_out

    fasta = tmp_path / "cds.fa"
    records = [(f"g{g}", _cds(40 + i, 10 * g + i)) for g in range(40) for i in range(3)]
    fasta.write_text("".join(f">lcl|x_{n} [gene={gene}] [protein=p]\n{seq}\n" for n, (gene, seq) in enumerate(records)))
    assert gene_of(">lcl|x [gene=OR4F5] [db_xref=a]") == "OR4F5" and gene_of("plain") == "plain"
    metas = prepare([fasta], tmp_path / "packed", seq_len=128, holdout_genes=0.25)
    out_genes = {g for g, _ in records if held_out(g, 0.25)}
    assert 0 < len(out_genes) < 40
    assert metas["holdout"]["genes"] == len(out_genes) and metas["train"]["genes"] == 40 - len(out_genes)
    assert metas["holdout"]["sequences"] == 3 * len(out_genes)
    assert metas["train"]["sequences"] + metas["holdout"]["sequences"] == 120
    assert (tmp_path / "packed_holdout" / "tokens.npy").is_file() and (tmp_path / "packed" / "tokens.npy").is_file()


def test_read_fasta_keeps_headers_and_joins_lines(tmp_path):
    import gzip
    text = ">a desc one\nACG\nTT\n\n>b\nGG\n"
    (tmp_path / "x.fa").write_text(text)
    with gzip.open(tmp_path / "x.fa.gz", "wt") as fh:
        fh.write(text)
    for name in ("x.fa", "x.fa.gz"):
        assert list(read_fasta(tmp_path / name)) == [("a desc one", "ACGTT"), ("b", "GG")]
