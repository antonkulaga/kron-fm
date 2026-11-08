"""Every model test runs twice: on the CPU with :class:`kron_core.testing.CausalMixer` standing in for the
KronLSTM layer, and on a CUDA device with the vendored layer and its Triton kernels, which skips without one.

The stand-in keeps the layer's call, its padding rule and its per-document restart, so the CPU run checks a
model's own logic and the GPU run checks the same claims through the real kernels.
"""
from __future__ import annotations

import pytest
import torch


def needs_gpu() -> None:
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("the KronLSTM layer runs Triton kernels and needs a CUDA device")


@pytest.fixture(params=["cpu", "cuda"])
def device(request, monkeypatch):
    """The model's device: the CPU with the stand-in layer, or CUDA with the vendored one."""
    if request.param == "cuda":
        needs_gpu()
    else:
        import kron_core.blocks
        from kron_core.testing import CausalMixer

        monkeypatch.setattr(kron_core.blocks, "kronlstm_mixer", CausalMixer)
    return torch.device(request.param)
