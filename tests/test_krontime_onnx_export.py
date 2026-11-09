"""The streamed KronLSTM's ONNX graphs (krontime.onnx_export) agree with their PyTorch modules.

A tiny model on the CPU: the plain layer needs no Triton kernel, and the export and ONNX Runtime run in
seconds. The comparison against the Triton kernels needs CUDA and runs inside the publishing script.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from krontime.glucose import KronLSTMForecaster  # noqa: E402
from krontime.onnx_export import (  # noqa: E402
    consumer_doc,
    export_stream_bundle,
    plain_stream_forecast,
    synthetic_future,
    synthetic_stretches,
)

AFFINES = [(1 / 360, -40 / 360), (1 / 10, 0.0), (1 / 25, 0.0), (1 / 200, 0.0)]


class _Affine:
    def __init__(self, a: float, b: float) -> None:
        self.a, self.b = a, b

    def transform(self, x):
        return np.asarray(x) * self.a + self.b


def _model() -> KronLSTMForecaster:
    torch.manual_seed(0)
    model = KronLSTMForecaster(n_time_steps=288, prediction_horizon=24, d_model=64, n_heads=4, n_layers=2,
                               ff_units=64, dropout=0.0, decoder_layers=1, announced_channels=(3,),
                               plan_channels=(2,), aux_weight=0.0, input_norm="physical", channel_affine=AFFINES,
                               glucose_mean=AFFINES[0][1], glucose_scale=AFFINES[0][0])
    with torch.no_grad():  # the zero-initialised gate and head would hide most of the path
        for name, p in model.named_parameters():
            if "g_proj.weight" in name:
                p.normal_(0, 0.05)
        model.head.weight.normal_(0, 1e-4)
    model.serving = "stream"
    return model.eval()


def test_plain_path_answers_an_announced_meal_and_a_planned_dose():
    model = _model()
    history = synthetic_stretches(3, 100, 0)
    with torch.no_grad():
        forecast = plain_stream_forecast(model, AFFINES, history, synthetic_future(model, 3), 12, 12)
    assert forecast.shape == (3, 24)
    assert torch.isfinite(forecast).all()


def test_bundle_exports_and_matches_pytorch(tmp_path):
    scalers = {name: _Affine(a, b) for name, (a, b) in zip(("glucose", "basal", "bolus", "carbs"), AFFINES)}
    info = export_stream_bundle(_model(), {"history_bridge_steps": 288}, scalers, tmp_path)
    assert (tmp_path / "history.onnx").is_file() and (tmp_path / "decode.onnx").is_file()
    assert info["plan_channels"] == ["bolus"] and info["announced_channels"] == ["carbs"]
    assert all(c["max_abs_diff_mgdl"] <= c["tolerance_mgdl"] for c in info["verification"])
    assert info["onnx_vs_torch_max_abs_diff"] < 0.1
    decode_inputs = [i["name"] for i in info["graphs"]["decode"]["inputs"]]
    assert decode_inputs[:2] == ["glucose_tail", "future"]
    doc = consumer_doc({**info, "checkpoint": "best_model.pt"})
    assert "planned extra dose" in doc and "288" in doc
