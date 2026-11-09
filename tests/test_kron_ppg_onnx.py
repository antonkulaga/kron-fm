"""The PPG age clock's ONNX streaming graph: built with the real KronLSTM layer, read through the plain path.

These tests do not use the ``device`` fixture: the graph reads the vendored layer's own weights, which
build on the CPU, and the CPU stand-in has none of them. The kernel comparison runs where CUDA is.
"""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from kron_core.blocks import LayerStates  # noqa: E402
from kron_ppg.model import AgeHead, KronPPG, KronPPGConfig, PPGAgeClock, patchify  # noqa: E402
from kron_ppg.onnx_export import (PPGStreamGraph, empty_state, export_stream, plain_stream_age,  # noqa: E402
                                  state_from_layers, synthetic_streams)

SMALL = dict(d_model=32, n_heads=2, n_layers=2, ff_units=64, dropout=0.0, patch=5, horizon=3, proj_dim=16)


def real_clock() -> PPGAgeClock:
    torch.manual_seed(0)
    clock = PPGAgeClock(KronPPG(KronPPGConfig(**SMALL)), AgeHead(32)).eval()
    torch.nn.init.normal_(clock.head.net[-1].weight, std=0.5)
    clock.head.set_scale(torch.tensor([25.0, 75.0]))
    return clock


def test_the_exported_graph_matches_pytorch_and_the_kernels(tmp_path):
    clock = real_clock()
    record = export_stream(clock, tmp_path, chunk=10, check_seconds=12.0)
    assert (tmp_path / "stream.onnx").is_file() and (tmp_path / "stream.json").is_file()
    assert record["inputs"][:4] == ["samples", "valid", "pool_sum", "pool_count"]
    assert record["chunk_samples"] == 50 and record["state_shapes"]["recurrent_state_0"] == [2, 16, 16]
    for batch in (1, 3):
        assert record["checks"][f"batch_{batch}_onnx_vs_torch_max_abs_diff_years"] < 1e-3
    if torch.cuda.is_available():
        checks = record["checks"]
        # in float32 the plain path is the kernels' recurrence; under bfloat16 autocast it rounds
        assert checks["kernel_stream_fp32_max_abs_diff_years"] < 1e-2
        assert checks["kernel_whole_fp32_max_abs_diff_years"] < 1e-2
        assert checks["kernel_stream_as_served_max_abs_diff_years"] < 1.0


def test_an_invalid_patch_leaves_the_stream_as_if_it_were_absent():
    clock = real_clock()
    graph = PPGStreamGraph(clock).eval()
    x = synthetic_streams(1, 4.0, 25.0)  # 100 samples, 20 patches
    valid = torch.ones(1, 20, dtype=torch.bool)
    valid[0, 7] = False
    noisy = x.clone()
    noisy[0, 35:40] = 40.0
    with torch.no_grad():
        a, *sa = graph(x, valid, *empty_state(clock, 1))
        b, *sb = graph(noisy, valid, *empty_state(clock, 1))
        drop = torch.cat([x[:, :35], x[:, 40:]], 1)
        c, *sc = graph(drop, torch.ones(1, 19, dtype=torch.bool), *empty_state(clock, 1))
    assert torch.allclose(a, b) and all(torch.allclose(u, v) for u, v in zip(sa, sb))
    assert torch.allclose(a, c, atol=1e-5) and torch.allclose(sa[1], sc[1])


def test_the_plain_stream_gives_the_same_age_in_any_chunking():
    clock = real_clock()
    x = synthetic_streams(2, 8.0, 25.0, seed=3)
    assert torch.allclose(plain_stream_age(clock, x, 8), plain_stream_age(clock, x, 20), atol=1e-4)


def test_the_kernel_memory_continues_in_the_graph():
    """A stream started on the GPU and finished in the graph reads the age the GPU reads."""
    if not torch.cuda.is_available():
        pytest.skip("needs the Triton kernels")
    pytest.importorskip("triton")
    clock = real_clock()
    x = synthetic_streams(2, 8.0, 25.0, seed=5)
    patches, ok = patchify(x, 5)
    gpu = clock.cuda()
    with torch.no_grad():
        states = LayerStates.empty(2)
        pool_sum, pool_count = torch.zeros(2, 32, device="cuda"), torch.zeros(2, 1, device="cuda")
        _, pool_sum, pool_count = gpu.stream(patches[:, :20].cuda(), ok[:, :20].cuda(), states, pool_sum, pool_count)
        age_gpu, _, _ = gpu.stream(patches[:, 20:].cuda(), ok[:, 20:].cuda(), states, pool_sum, pool_count)
        clock.cpu()
        first = [pool_sum.float().cpu(), pool_count.float().cpu()]
        # the GPU's first half as the graph's state, then the second half through the graph
        states_half = LayerStates.empty(2)
        ps, pc = torch.zeros(2, 32, device="cuda"), torch.zeros(2, 1, device="cuda")
        _, ps, pc = clock.cuda().stream(patches[:, :20].cuda(), ok[:, :20].cuda(), states_half, ps, pc)
        clock.cpu()
        state = [ps.float().cpu(), pc.float().cpu(), *state_from_layers(clock, states_half, 2)]
        age_graph, *_ = PPGStreamGraph(clock).eval()(x[:, 100:], ok[:, 20:], *state)
    assert first[1].tolist() == [[20.0], [20.0]]
    assert torch.allclose(age_graph, age_gpu.float().cpu(), atol=0.5)
