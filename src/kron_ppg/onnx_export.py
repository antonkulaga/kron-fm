"""The PPG age clock as one ONNX streaming graph for a phone: a chunk of PPG in, the memory and the age out.

The trained layer runs Triton kernels, which neither run on a CPU nor export, so the graph reads the
same weights through :class:`kron_core.onnx.PlainKronMixer` (the layer's token-by-token recurrence in
plain PyTorch), as krontime's export does. A watch or phone app keeps the graph's state between calls:

* every layer's memory, ``recurrent_state_<i>`` ``(B, H, K, V)`` and ``conv_state_<i>`` ``(B, D, W)``,
  exactly the :class:`kron_core.blocks.LayerStates` the GPU streaming path carries;
* the running pool, ``pool_sum`` ``(B, d_model)`` and ``pool_count`` ``(B, 1)``, whose ratio the age
  head reads.

``stream.onnx`` inputs: ``samples`` ``(B, chunk * patch)`` float32, the scaled PPG at the model's rate
(band-passed 0.5-8 Hz and robust-scaled per window, :mod:`kron_ppg.signal`); ``valid`` ``(B, chunk)``
bool, false for a patch that is missing or padding (the memory and the pool skip it); then the pool
and the memory. Outputs: ``age`` ``(B,)`` in years, the pool and the memory after the chunk. A
stream starts from zeros everywhere. The recurrence is unrolled, so the graph reads a fixed number of
patches: a short chunk is padded with ``valid`` false.

The age is a within-person trend, never a biological age: see ``reports/PPG foundation models and
aging clocks 2026.md`` and the clock run's ``metrics.json`` for the error band to show with it.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from kron_core.blocks import LayerStates
from kron_core.onnx import PlainKronMixer, rms
from kron_ppg.model import PPGAgeClock, patchify


def memory_shapes(clock: PPGAgeClock) -> list[tuple[str, tuple[int, ...]]]:
    """Each memory tensor's name and its shape after the batch axis, in the graph's order."""
    shapes = []
    for i, blk in enumerate(clock.encoder.blocks):
        layer = blk.mixer
        shapes.append((f"recurrent_state_{i}", (layer.num_heads, layer.k_head_dim, layer.v_head_dim)))
        shapes.append((f"conv_state_{i}", (2 * layer.key_dim + layer.value_dim,
                                           int(layer.qkv_conv1d.kernel_size[0]))))
    return shapes


def empty_state(clock: PPGAgeClock, batch: int) -> list[torch.Tensor]:
    """``[pool_sum, pool_count, *memory]`` at a stream's start: all zeros."""
    d = clock.encoder.config.d_model
    return [torch.zeros(batch, d), torch.zeros(batch, 1)] + [torch.zeros(batch, *s) for _, s in memory_shapes(clock)]


def state_from_layers(clock: PPGAgeClock, states: LayerStates, batch: int) -> list[torch.Tensor]:
    """The GPU path's :class:`LayerStates` as the graph's flat memory (zeros for a layer that read nothing)."""
    out = []
    shapes = memory_shapes(clock)
    for i in range(len(clock.encoder.blocks)):
        entry = states[i]
        for key, (_, shape) in zip(("recurrent_state", "conv_state"), shapes[2 * i:2 * i + 2], strict=True):
            value = None if entry is None else entry.get(key)
            out.append(torch.zeros(batch, *shape) if value is None else value.float().cpu())
    return out


class PPGStreamGraph(nn.Module):
    """``(samples, valid, pool_sum, pool_count, *memory)`` -> ``(age, pool_sum, pool_count, *memory)``.

    :meth:`kron_ppg.model.PPGAgeClock.stream` through :class:`PlainKronMixer`.
    """

    def __init__(self, clock: PPGAgeClock) -> None:
        super().__init__()
        self.clock = clock
        self.mixers = nn.ModuleList(PlainKronMixer(blk.mixer) for blk in clock.encoder.blocks)

    def forward(self, samples: torch.Tensor, valid: torch.Tensor, pool_sum: torch.Tensor, pool_count: torch.Tensor,
                *memory: torch.Tensor) -> tuple[torch.Tensor, ...]:
        enc = self.clock.encoder
        batch = samples.size(0)
        patches = samples.reshape(batch, -1, enc.config.patch).masked_fill(~valid.unsqueeze(-1), 0.0)
        h = rms(enc.embed_norm, enc.embed(patches)).masked_fill(~valid.unsqueeze(-1), 0.0)
        out: list[torch.Tensor] = []
        for i, (blk, mixer) in enumerate(zip(enc.blocks, self.mixers, strict=True)):
            mixed, rec, conv = mixer(rms(blk.mixer_norm, h), valid, memory[2 * i], memory[2 * i + 1])
            out += [rec, conv]
            h = h + mixed
            h = h + blk.mlp(rms(blk.mlp_norm, h))
        h = rms(enc.final_norm, h).masked_fill(~valid.unsqueeze(-1), 0.0)
        pool_sum = pool_sum + h.sum(1)
        pool_count = pool_count + valid.to(h.dtype).sum(1, keepdim=True)
        age = self.clock.head(pool_sum / pool_count.clamp_min(1.0))
        return (age, pool_sum, pool_count, *out)


def _chunks(samples: torch.Tensor, chunk_samples: int, patch: int):
    """``samples`` ``(B, L)`` in chunks of ``chunk_samples``, the last padded, with per-patch validity."""
    batch, length = samples.shape
    for start in range(0, length, chunk_samples):
        piece = samples[:, start:start + chunk_samples]
        n = piece.size(1) // patch
        valid = torch.zeros(batch, chunk_samples // patch, dtype=torch.bool)
        valid[:, :n] = True
        if piece.size(1) < chunk_samples:
            piece = torch.cat([piece, piece.new_zeros(batch, chunk_samples - piece.size(1))], 1)
        yield piece, valid


def plain_stream_age(clock: PPGAgeClock, samples: torch.Tensor, chunk: int) -> torch.Tensor:
    """The graph's PyTorch module run as a phone runs the ONNX one: chunk after chunk from zero state."""
    graph = PPGStreamGraph(clock).eval()
    state = empty_state(clock, samples.size(0))
    age = None
    with torch.no_grad():
        for piece, valid in _chunks(samples, chunk * clock.encoder.config.patch, clock.encoder.config.patch):
            age, *state = graph(piece, valid, *state)
    assert age is not None
    return age


def synthetic_streams(batch: int, seconds: float, fs: float, seed: int = 0) -> torch.Tensor:
    """``(batch, seconds * fs)`` scaled pulse-like streams with a dicrotic wave, 55-95 bpm."""
    gen = torch.Generator().manual_seed(seed)
    t = torch.arange(int(seconds * fs)) / fs
    hr = 55 + 40 * torch.rand(batch, 1, generator=gen)
    phase = (t * hr / 60 + torch.rand(batch, 1, generator=gen)) % 1.0
    x = torch.exp(-((phase - 0.2) / 0.07) ** 2) + 0.35 * torch.exp(-((phase - 0.5) / 0.08) ** 2)
    x = x + 0.03 * torch.randn(batch, len(t), generator=gen)
    return (x - x.median(1, keepdim=True).values) / x.std(1, keepdim=True)


def kernel_parity(clock: PPGAgeClock, samples: torch.Tensor, reference: torch.Tensor, chunk: int) -> dict[str, Any]:
    """The plain path's ages against the Triton path on the GPU, in years; empty without CUDA.

    ``as_served`` runs the kernels under bfloat16 autocast as training does; ``fp32`` turns autocast off,
    so its difference is the plain path's own and should be near float32 rounding.
    """
    if not torch.cuda.is_available():
        return {}
    import contextlib

    gpu = clock.cuda().eval()
    patch, d = clock.encoder.config.patch, clock.encoder.config.d_model
    patches, ok = patchify(samples.cuda(), patch)
    out: dict[str, Any] = {"device": torch.cuda.get_device_name(0)}
    real_autocast = torch.autocast
    try:
        for label, plain in (("as_served", False), ("fp32", True)):
            if plain:
                torch.autocast = lambda *a, **k: contextlib.nullcontext()  # noqa: E731
            with torch.no_grad():
                states = LayerStates.empty(len(gpu.encoder.blocks))
                pool_sum = torch.zeros(len(samples), d, device="cuda")
                pool_count = torch.zeros(len(samples), 1, device="cuda")
                for a in range(0, patches.size(1), chunk):
                    age, pool_sum, pool_count = gpu.stream(patches[:, a:a + chunk], ok[:, a:a + chunk], states,
                                                           pool_sum, pool_count)
                whole = gpu(samples.cuda())
            torch.autocast = real_autocast
            out[f"kernel_stream_{label}_max_abs_diff_years"] = float((age.float().cpu() - reference).abs().max())
            out[f"kernel_whole_{label}_max_abs_diff_years"] = float((whole.float().cpu() - reference).abs().max())
    finally:
        torch.autocast = real_autocast
        clock.cpu()
    return out


def export_stream(clock: PPGAgeClock, out: str | Path, chunk: int = 25, opset: int = 17,
                  check_seconds: float = 60.0) -> dict[str, Any]:
    """Write ``stream.onnx`` and ``stream.json`` into ``out``, checked with ONNX Runtime; return the record.

    Checks: the graph against its PyTorch module, chunk by chunk over ``check_seconds`` of synthetic
    streams for one stream and for three, with a stream that drops a patch mid-chunk; on a machine with
    CUDA, the PyTorch module against the model's own Triton streaming path.
    """
    import onnx
    import onnxruntime as ort

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    clock = clock.float().cpu().eval()
    config = clock.encoder.config
    graph = PPGStreamGraph(clock).eval()
    names = [n for n, _ in memory_shapes(clock)]
    state_names = ["pool_sum", "pool_count", *names]
    inputs = ["samples", "valid", *state_names]
    outputs = ["age", *[n + "_next" for n in state_names]]
    axes = {n: {0: "batch"} for n in inputs + outputs}
    example = synthetic_streams(2, chunk * config.patch_seconds, config.sample_rate)
    valid = torch.ones(2, chunk, dtype=torch.bool)
    with torch.no_grad():
        torch.onnx.export(graph, (example, valid, *empty_state(clock, 2)), out / "stream.onnx", input_names=inputs,
                          output_names=outputs, dynamic_axes=axes, opset_version=opset, dynamo=False)
    onnx.checker.check_model(str(out / "stream.onnx"))
    session = ort.InferenceSession(str(out / "stream.onnx"), providers=["CPUExecutionProvider"])

    def run_onnx(samples: torch.Tensor, drop: Sequence[tuple[int, int]] = ()) -> tuple[np.ndarray, np.ndarray]:
        state = [s.numpy() for s in empty_state(clock, samples.size(0))]
        torch_state = empty_state(clock, samples.size(0))
        age_o = age_t = None
        for k, (piece, ok) in enumerate(_chunks(samples, chunk * config.patch, config.patch)):
            for lane, patch_idx in drop:
                if k == patch_idx // chunk:
                    ok[lane, patch_idx % chunk] = False
            age_o, *state = session.run(None, {"samples": piece.numpy(), "valid": ok.numpy(),
                                               **dict(zip(state_names, state, strict=True))})
            with torch.no_grad():
                age_t, *torch_state = graph(piece, ok, *torch_state)
        return np.asarray(age_o), age_t.numpy()

    checks = {}
    for batch in (1, 3):
        samples = synthetic_streams(batch, check_seconds, config.sample_rate, seed=batch)
        drop = [(batch - 1, chunk + 3)]
        age_o, age_t = run_onnx(samples, drop)
        checks[f"batch_{batch}_onnx_vs_torch_max_abs_diff_years"] = float(np.abs(age_o - age_t).max())
    samples = synthetic_streams(3, check_seconds, config.sample_rate, seed=7)
    reference = plain_stream_age(clock, samples, chunk)
    checks.update(kernel_parity(clock, samples, reference, chunk))
    record = {
        "graph": "stream.onnx", "opset": opset, "chunk_patches": chunk, "chunk_samples": chunk * config.patch,
        "chunk_seconds": chunk * config.patch_seconds, "sample_rate": config.sample_rate, "patch": config.patch,
        "inputs": inputs, "outputs": outputs,
        "state_shapes": {"pool_sum": [config.d_model], "pool_count": [1],
                         **{n: list(s) for n, s in memory_shapes(clock)}},
        "preprocessing": "band-pass 0.5-8 Hz, resample to sample_rate, robust scale (median, IQR/1.349) per window",
        "bytes": (out / "stream.onnx").stat().st_size, "checks": checks,
    }
    (out / "stream.json").write_text(json.dumps(record, indent=2) + "\n")
    return record
