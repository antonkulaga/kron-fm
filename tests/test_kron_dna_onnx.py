"""kron-dna's ONNX graph: the plain-PyTorch path matches the Triton kernels, and ONNX Runtime matches the path.

The export and ONNX Runtime run on the CPU with a tiny model in seconds: the plain layer needs no Triton
kernel, so the model is built with the real KronLSTM layer (its constructor runs anywhere). The comparison
with the kernels needs CUDA.
"""
from __future__ import annotations

import json

import pytest
import torch

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from kron_dna import codons as C  # noqa: E402
from kron_dna.model import KronDNA, KronDNAConfig  # noqa: E402
from kron_dna.onnx_export import KronDNAGraph, example_tokens, export_onnx, load_run, reverse_index  # noqa: E402


def _model(**overrides) -> KronDNA:
    kw = dict(d_model=32, n_heads=2, n_layers=2, ff_units=64, dropout=0.0, attention_every=1)
    kw.update(overrides)
    torch.manual_seed(0)
    model = KronDNA(KronDNAConfig(**kw)).eval()
    with torch.no_grad():  # the zero-initialised output-gate weights would hide part of the path
        for name, p in model.named_parameters():
            if "g_proj.weight" in name:
                p.normal_(0, 0.05)
    return model


def test_reverse_index_reverses_each_rows_prefix_in_place():
    valid = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    assert reverse_index(valid).tolist() == [[2, 1, 0, 3, 4], [4, 3, 2, 1, 0]]


def test_example_rows_are_one_sequence_each():
    rows = example_tokens(4, 40)
    assert (rows[:, 0] == C.CLS_ID).all()
    assert ((rows == C.EOS_ID).sum(1) == 1).all()


def test_padding_never_changes_the_real_tokens_outputs():
    graph = KronDNAGraph(_model()).eval()
    rows = example_tokens(2, 30)
    longer = torch.cat([rows, torch.full((2, 10), C.PAD_ID)], dim=1)
    with torch.no_grad():
        a, _, emb_a = graph(rows)
        b, _, emb_b = graph(longer)
    valid = rows != C.PAD_ID
    assert torch.allclose(a[valid], b[:, :30][valid], atol=1e-5)
    assert torch.allclose(emb_a, emb_b, atol=1e-5)


@pytest.mark.parametrize("tie", [False, True])
def test_export_matches_pytorch_and_records_its_checks(tmp_path, tie):
    meta = export_onnx(_model(tie_directions=tie), tmp_path, seq_len=24, check_kernels=False)
    assert (tmp_path / "kron_dna.onnx").is_file()
    assert json.loads((tmp_path / "onnx_meta.json").read_text())["seq_len"] == 24
    assert all(v["max_abs_diff"] < 1e-4 for v in meta["verification"])
    assert meta["outputs"][0]["shape"] == ["batch", 24, C.VOCAB_SIZE]


def test_load_run_round_trips_a_checkpoint(tmp_path):
    model = _model()
    torch.save({"model": model.state_dict(), "config": model.config.to_dict(), "step": 7, "objective": "mlm"},
               tmp_path / "checkpoint.pt")
    loaded, record = load_run(tmp_path)
    assert record["step"] == 7 and record["checkpoint"].endswith("checkpoint.pt")
    rows = example_tokens(2, 20)
    with torch.no_grad():
        assert torch.equal(KronDNAGraph(model)(rows)[0], KronDNAGraph(loaded)(rows)[0])


def test_plain_graph_matches_the_triton_kernels():
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("the KronLSTM kernels need a CUDA device")
    from kron_dna.onnx_export import kernel_parity

    result = kernel_parity(_model(), example_tokens(3, 48))
    assert result["max_abs_diff"] < 2e-3, result
