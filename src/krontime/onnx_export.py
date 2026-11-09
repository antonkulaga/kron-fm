"""ONNX serving graphs for a streamed KronLSTM: the history read and the decode, with the memory as tensors.

Ported from glucose_forecasting_metabonet's ``kronlstm/onnx_export.py`` (commit 6e9bbbe91); the plain
layer and the RMSNorm helper now live in :mod:`kron_core.onnx`.

The trained layer runs FLA's Triton kernels, which neither run on a CPU nor export to ONNX, so the
graphs here read the same weights through a plain-PyTorch copy of the layer (:class:`PlainKronMixer`):
the token-by-token recurrence of ``kron_core/vendor/fla/ops/kronlstm/naive.py``, the short convolution as the
cache-update kernel computes it, and the sigmoid-gated RMSNorm. Each layer's memory is the pair the
streaming path keeps in :class:`kron_core.blocks.LayerStates`: ``recurrent_state`` ``(B, H, K, V)`` and
``conv_state`` ``(B, D, W)``, the last ``W`` convolution inputs, oldest first.

A streamed model is served as it was trained (C66): a person's stretch of record is read once, front to
back, and every forecast is decoded from the memory at its origin. So there are two graphs:

* :class:`KronHistoryGraph` reads ``chunk`` five-minute steps of a stretch into the memory. A step whose
  ``valid`` is false leaves the memory as it was, so a short chunk is padded at its end.
* :class:`KronDecodeGraph` decodes the forecast at an origin from that memory, the last glucose readings
  and the known-future block: the announced carbs and, in a model with a plan channel (C86), a planned dose.

Both take and return physical units (mg/dL, U/h, U, g). The run's scalers are affine, and the graphs apply
them inside, so a phone needs no copy of ``scalers.json`` to use them.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from kron_core.onnx import PlainKronMixer, rms
from krontime.glucose import N_CHANNELS, KronLSTMForecaster, LayerStates

#: The four channels every history and block step holds, in this order, in physical units.
CHANNEL_UNITS: tuple[tuple[str, str], ...] = (("glucose", "mg/dL"), ("basal", "U/h"), ("bolus", "U"),
                                              ("carbs", "g"))


def scaler_affine(scaler: Any) -> tuple[float, float]:
    """``(a, b)`` with ``scaler.transform(x) == a * x + b``, read off two points of the transform."""
    import numpy as np

    zero, one = scaler.transform(np.array([[0.0], [1.0]], dtype=np.float64)).ravel()
    a, b = float(one - zero), float(zero)
    probe = scaler.transform(np.array([[137.0]], dtype=np.float64)).ravel()[0]
    if abs(probe - (a * 137.0 + b)) > 1e-6 * max(1.0, abs(probe)):
        raise ValueError("the run's scaler is not affine; the ONNX graphs can only apply an affine scaler")
    return a, b


def memory_shapes(model: KronLSTMForecaster) -> list[tuple[str, tuple[int, ...]]]:
    """Each memory tensor's name and its shape after the batch axis, in the order the graphs take them."""
    shapes = []
    for i, blk in enumerate(model.blocks):
        layer = blk.mixer
        shapes.append((f"recurrent_state_{i}", (layer.num_heads, layer.k_head_dim, layer.v_head_dim)))
        shapes.append((f"conv_state_{i}", (2 * layer.key_dim + layer.value_dim, int(layer.qkv_conv1d.kernel_size[0]))))
    return shapes


def empty_memory(model: KronLSTMForecaster, batch: int) -> list[torch.Tensor]:
    """The memory at a stretch's start: every tensor zero, as a layer reads ``None``."""
    return [torch.zeros(batch, *shape) for _, shape in memory_shapes(model)]


def memory_from_states(model: KronLSTMForecaster, states: LayerStates, batch: int) -> list[torch.Tensor]:
    """The streaming path's :class:`LayerStates` as the graphs' flat list of tensors (zeros for an unread layer)."""
    out = []
    for i, ((_, rec_shape), (_, conv_shape)) in enumerate(zip(*[iter(memory_shapes(model))] * 2, strict=True)):
        entry = states[i]
        rec = None if entry is None else entry.get("recurrent_state")
        conv = None if entry is None else entry.get("conv_state")
        out.append(torch.zeros(batch, *rec_shape) if rec is None else rec.float().cpu())
        out.append(torch.zeros(batch, *conv_shape) if conv is None else conv.float().cpu())
    return out


class _Units(nn.Module):
    """The run's affine scalers as buffers: physical units in, the model's scaled units out, and back."""

    def __init__(self, affines: Sequence[tuple[float, float]]) -> None:
        super().__init__()
        self.register_buffer("gain", torch.tensor([a for a, _ in affines], dtype=torch.float32))
        self.register_buffer("offset", torch.tensor([b for _, b in affines], dtype=torch.float32))

    def scale(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.gain[:values.size(-1)] + self.offset[:values.size(-1)]

    def glucose(self, scaled: torch.Tensor) -> torch.Tensor:
        return (scaled - self.offset[0]) / self.gain[0]


def _check_servable(model: KronLSTMForecaster) -> None:
    if model.post_gap_steps:
        raise ValueError("the ONNX graphs serve forecast-only KronLSTM runs; this one has a hidden span (gap mode)")
    if model.clock:
        raise ValueError("the ONNX graphs do not yet carry the time-of-day inputs of a clock run (C74)")
    if model.side_channels:
        raise ValueError("the ONNX graphs do not yet carry the wearable channels of a wearable run (R.89)")


class KronHistoryGraph(nn.Module):
    """``(history (B, T, 4) physical, valid (B, T), memory...)`` -> the memory after the chunk.

    The same projections and layers as :meth:`KronLSTMForecaster.stream_history`, through
    :class:`PlainKronMixer`. Missing values are NaN, exactly as the streaming path hands them over.
    """

    def __init__(self, model: KronLSTMForecaster, affines: Sequence[tuple[float, float]]) -> None:
        super().__init__()
        _check_servable(model)
        self.model = model
        self.units = _Units(affines)
        self.mixers = nn.ModuleList(PlainKronMixer(blk.mixer) for blk in model.blocks)

    def forward(self, history: torch.Tensor, valid: torch.Tensor, *memory: torch.Tensor) -> tuple[torch.Tensor, ...]:
        model = self.model
        pad = ~valid
        x = self.units.scale(history).masked_fill(pad.unsqueeze(-1), float("nan"))
        h = model.history_tokens(x).masked_fill(pad.unsqueeze(-1), 0.0)
        out: list[torch.Tensor] = []
        for i, (blk, mixer) in enumerate(zip(model.blocks, self.mixers, strict=True)):
            mixed, rec, conv = mixer(rms(blk.mixer_norm, h), valid, memory[2 * i], memory[2 * i + 1])
            out += [rec, conv]
            if i + 1 < len(model.blocks):  # the last layer's output feeds nothing the memory needs
                h = h + mixed
                h = h + blk.mlp(rms(blk.mlp_norm, h))
        return tuple(out)


class KronDecodeGraph(nn.Module):
    """``(glucose_tail (B, W) mg/dL, future (B, S, 4) physical, [static (B, F)], memory...)`` -> ``(B, horizon)`` mg/dL.

    :meth:`KronLSTMForecaster.decode_from_states` through :class:`PlainKronMixer`. ``future`` holds what is
    known of each forecast step, NaN where nothing is: the announced carbs (zero where no meal is
    announced, NaN over the whole horizon when the announcement is withheld) and, in a model with a plan
    channel, a planned dose in units at its step and NaN elsewhere.
    """

    def __init__(self, model: KronLSTMForecaster, affines: Sequence[tuple[float, float]]) -> None:
        super().__init__()
        _check_servable(model)
        self.model = model
        self.units = _Units(affines)
        self.mixers = nn.ModuleList(PlainKronMixer(blk.mixer) for blk in model.blocks)

    def forward(self, glucose_tail: torch.Tensor, future: torch.Tensor, *rest: torch.Tensor) -> torch.Tensor:
        model = self.model
        static, memory = (rest[0], rest[1:]) if model.static_project is not None else (None, rest)
        glucose = self.units.scale(glucose_tail.unsqueeze(-1)).squeeze(-1)
        values = model.decode_block(self.units.scale(future), glucose.size(0), glucose)
        block, features, _ = model.block_tokens(values, None, static)
        h = block
        for i, (blk, mixer) in enumerate(zip(model.blocks, self.mixers, strict=True)):
            mixed, _, _ = mixer(rms(blk.mixer_norm, h), None, memory[2 * i], memory[2 * i + 1])
            h = h + mixed
            h = h + blk.mlp(rms(blk.mlp_norm, h))
        h = rms(model.final_norm, h.float())
        if model.skip_project is not None:
            h = h + model.skip_project(features)
        decoded = model.decoder_norm(model.decoder(h))
        scaled = model.starting_line(glucose, values) + model._block_delta(decoded, values)
        return self.units.glucose(scaled)


def channel_affines(scalers: Mapping[str, Any]) -> list[tuple[float, float]]:
    """The four channels' affines from a run's scalers, in the window's channel order."""
    return [scaler_affine(scalers[name]) for name, _ in CHANNEL_UNITS]


def plain_stream_forecast(model: KronLSTMForecaster, affines: Sequence[tuple[float, float]], history: torch.Tensor,
                          future: torch.Tensor, chunk: int, tail_steps: int, static: torch.Tensor | None = None
                          ) -> torch.Tensor:
    """The two graphs' PyTorch modules run as a phone would run the ONNX ones: the history in chunks, then the decode.

    ``history`` ``(B, T, 4)`` physical units, every step valid. Returns ``(B, horizon)`` mg/dL.
    """
    hist = KronHistoryGraph(model, affines).eval()
    dec = KronDecodeGraph(model, affines).eval()
    memory = empty_memory(model, history.size(0))
    for piece, valid in _chunks(history, chunk):
        memory = list(hist(piece, valid, *memory))
    tail = history[:, -tail_steps:, 0]
    return dec(tail, future, *([static] if static is not None else []), *memory)


#: Steps the history graph reads per call: an hour. A phone appends a reading every five minutes, so it
#: calls the graph with one valid step and eleven padded ones, or with whole hours to catch up.
CHUNK_STEPS = 12
#: Glucose steps the decode graph reads for its starting line, as the streaming scorer hands them over.
TAIL_STEPS = 12
#: Names of the three serving cases the checks run, in the order the batch of three holds them.
CASES = ("announced meal", "announcement withheld", "planned dose")


def synthetic_stretches(batch: int, steps: int, seed: int) -> torch.Tensor:
    """``(batch, steps, 4)`` plausible records in physical units with the masked rule's missing values.

    Glucose swings between about 90 and 190 mg/dL with a ten-step sensor gap; basal is carried; meals
    and boluses come every few hours. The first stretch of a batch of two or more logs no carbs, which
    the masked rule leaves missing over the whole stretch, so the checks cover a missing channel too.
    """
    gen = torch.Generator().manual_seed(seed)
    t = torch.arange(steps, dtype=torch.float32)
    phase = torch.rand(batch, 1, generator=gen) * 6.28
    glucose = 140 + 45 * torch.sin(t / 40.0 + phase) + 6 * torch.randn(batch, steps, generator=gen)
    glucose[:, steps // 3:steps // 3 + 10] = float("nan")
    basal = (0.6 + 0.6 * torch.rand(batch, 1, generator=gen)).expand(batch, steps).clone()
    bolus = torch.zeros(batch, steps)
    carbs = torch.zeros(batch, steps)
    for lane in range(batch):
        start = int(torch.randint(0, 40, (1,), generator=gen))
        for step in range(start, steps, 60):
            bolus[lane, step] = float(2 + 4 * torch.rand(1, generator=gen))
            carbs[lane, step] = float(25 + 50 * torch.rand(1, generator=gen))
    if batch >= 2:
        carbs[0] = float("nan")
    return torch.stack([glucose, basal, bolus, carbs], dim=-1)


def synthetic_future(model: KronLSTMForecaster, batch: int) -> torch.Tensor:
    """``(batch, horizon, 4)`` blocks cycling through :data:`CASES`: a 60 g meal announced at step 6, the
    announcement withheld, and a 4 U dose planned at step 3 beside announced carbs (a model without a plan
    channel gets the announced meal again)."""
    horizon = model.prediction_horizon
    future = torch.full((batch, horizon, N_CHANNELS), float("nan"))
    for lane in range(batch):
        case = CASES[lane % len(CASES)]
        if case == "announcement withheld" or not model.announced_channels:
            continue
        for c in model.announced_channels:
            future[lane, :, c] = 0.0
        future[lane, 6, 3] = 60.0
        if case == "planned dose" and model.plan_channels:
            future[lane, 6, 3] = 0.0
            future[lane, 3, model.plan_channels[0]] = 4.0
    return future


def _chunks(history: torch.Tensor, chunk: int):
    batch, steps, width = history.shape
    for start in range(0, steps, chunk):
        piece = history[:, start:start + chunk]
        valid = torch.ones(batch, chunk, dtype=torch.bool)
        if piece.size(1) < chunk:
            valid[:, piece.size(1):] = False
            piece = torch.cat([piece, piece.new_full((batch, chunk - piece.size(1), width), float("nan"))], 1)
        yield piece, valid


def kernel_parity(model: KronLSTMForecaster, affines: Sequence[tuple[float, float]], history: torch.Tensor,
                  future: torch.Tensor, reference: torch.Tensor, static: torch.Tensor | None) -> dict[str, Any]:
    """The plain path against the model's own streaming path on the GPU, in mg/dL.

    ``as_served`` runs the kernels as every scorer does, under bfloat16 autocast; ``fp32`` turns the autocast
    off, so its difference is the plain path's own and should be small. Returns an empty dict without CUDA.
    """
    if not torch.cuda.is_available():
        return {}
    import contextlib

    gain = torch.tensor([a for a, _ in affines]); offset = torch.tensor([b for _, b in affines])
    scaled_history = (history * gain + offset).cuda()
    scaled_future = (future * gain + offset).cuda()
    out: dict[str, Any] = {"device": torch.cuda.get_device_name(0)}
    gpu = model.cuda()
    real_autocast = torch.autocast
    try:
        for label, patch in (("as_served", False), ("fp32", True)):
            if patch:
                torch.autocast = lambda *a, **k: contextlib.nullcontext()  # noqa: E731
            with torch.no_grad():
                states = LayerStates.empty(len(gpu.blocks))
                for start in range(0, scaled_history.size(1), 4096):
                    gpu.stream_history(scaled_history[:, start:start + 4096], states)
                scaled = gpu.decode_from_states(states, scaled_future, scaled_history[:, -TAIL_STEPS:, 0],
                                                static=None if static is None else static.cuda())
            torch.autocast = real_autocast
            mgdl = (scaled.float().cpu() - offset[0]) / gain[0]
            out[f"{label}_max_abs_diff_mgdl"] = float((mgdl - reference).abs().max())
    finally:
        torch.autocast = real_autocast
        model.cpu()
    return out


def export_stream_bundle(model: KronLSTMForecaster, meta: Mapping[str, Any], scalers: Mapping[str, Any],
                         work: Any, opset: int = 17, chunk: int = CHUNK_STEPS) -> dict[str, Any]:
    """Write ``history.onnx`` and ``decode.onnx`` for a streamed run into ``work`` and check them; return the record.

    The checks: each graph against its PyTorch module through ONNX Runtime, for one stretch and for three,
    on synthetic stretches with a sensor gap and a channel the history never logs, decoded in the three
    :data:`CASES`; and, on a machine with CUDA, the PyTorch modules against the model's own Triton path.
    """
    from pathlib import Path

    import numpy as np
    import onnx
    import onnxruntime as ort

    if getattr(model, "serving", "window") != "stream":
        raise ValueError("the two ONNX graphs serve a streamed KronLSTM (kronlstm_stream); this run is window-served")
    work = Path(work)
    model = model.float().cpu().eval()
    affines = channel_affines(scalers)
    history_graph = KronHistoryGraph(model, affines).eval()
    decode_graph = KronDecodeGraph(model, affines).eval()
    names = [n for n, _ in memory_shapes(model)]
    static_width = model.static_project.in_features if model.static_project is not None else 0
    decode_inputs = ["glucose_tail", "future", *(["static"] if static_width else []), *names]
    torch.backends.mha.set_fastpath_enabled(False)  # the encoder's fused kernel has no ONNX form
    with torch.no_grad():
        memory = [torch.randn(2, *shape) * 0.1 for _, shape in memory_shapes(model)]
        piece, valid = next(_chunks(synthetic_stretches(2, chunk, 0), chunk))
        valid[1, chunk // 2:] = False
        axes = {n: {0: "batch"} for n in ["history", "valid", *names, *[n + "_next" for n in names]]}
        torch.onnx.export(history_graph, (piece, valid, *memory), work / "history.onnx",
                          input_names=["history", "valid", *names], output_names=[n + "_next" for n in names],
                          dynamic_axes=axes, opset_version=opset, dynamo=False)
        static = torch.randn(2, static_width) if static_width else None
        future = synthetic_future(model, 2)
        axes = {n: {0: "batch"} for n in [*decode_inputs, "glucose_mgdl"]}
        torch.onnx.export(decode_graph, (synthetic_stretches(2, TAIL_STEPS, 1)[..., 0], future,
                                         *([static] if static is not None else []), *memory),
                          work / "decode.onnx", input_names=decode_inputs, output_names=["glucose_mgdl"],
                          dynamic_axes=axes, opset_version=opset, dynamo=False)
    for name in ("history.onnx", "decode.onnx"):
        onnx.checker.check_model(str(work / name))
    options = ort.SessionOptions()
    options.inter_op_num_threads = 1
    sessions = {name: ort.InferenceSession(str(work / f"{name}.onnx"), options, providers=["CPUExecutionProvider"])
                for name in ("history", "decode")}
    if [i.name for i in sessions["decode"].get_inputs()] != decode_inputs:
        raise RuntimeError("the exporter dropped or reordered a decode input; refusing to publish a different interface")

    checks, parity, example = [], {}, {}
    for batch, steps in ((1, 300), (3, 700)):
        history = synthetic_stretches(batch, steps, 10 + batch)
        future = synthetic_future(model, batch)
        static = torch.randn(batch, static_width) * 0.5 if static_width else None
        with torch.no_grad():
            reference = plain_stream_forecast(model, affines, history, future, chunk, TAIL_STEPS, static)
        state = [np.zeros((batch, *shape), dtype=np.float32) for _, shape in memory_shapes(model)]
        for piece, valid in _chunks(history, chunk):
            feed = {"history": piece.numpy(), "valid": valid.numpy(), **dict(zip(names, state, strict=True))}
            state = sessions["history"].run(None, feed)
        feed = {"glucose_tail": history[:, -TAIL_STEPS:, 0].numpy(), "future": future.numpy(),
                **({"static": static.numpy()} if static is not None else {}), **dict(zip(names, state, strict=True))}
        got = sessions["decode"].run(None, feed)[0]
        ref = reference.numpy()
        if got.shape != ref.shape or not np.isfinite(ref).all() or not np.isfinite(got).all():
            raise RuntimeError("ONNX and PyTorch forecasts differ in shape or hold nonfinite values")
        difference = float(np.abs(got - ref).max())
        tolerance = 1e-3 * max(1.0, float(np.abs(ref).max()))
        if difference > tolerance:
            raise RuntimeError(f"onnxruntime disagrees with PyTorch: max abs diff {difference:.3g} > {tolerance:.3g} mg/dL")
        checks.append({"stretches": batch, "steps": steps, "cases": [CASES[i % 3] for i in range(batch)],
                       "max_abs_diff_mgdl": difference, "tolerance_mgdl": tolerance})
        if batch == 3:
            parity = kernel_parity(model, affines, history, future, reference, static)
            # The Triton kernels multiply in TF32, which carries about three decimal digits, so their
            # difference grows with the forecast's size; a wrong layer would differ by far more.
            limit = max(1.0, 5e-3 * float(reference.abs().max()))
            if parity:
                parity["fp32_limit_mgdl"] = limit
            if parity and parity["fp32_max_abs_diff_mgdl"] > limit:
                raise RuntimeError(f"the plain layer disagrees with the Triton kernels by "
                                   f"{parity['fp32_max_abs_diff_mgdl']:.3g} mg/dL in fp32 (limit {limit:.3g}); "
                                   "refusing to publish")
            example = {"decode": {k: v[:1] for k, v in feed.items()}, "forecast": got[:1]}
    first_piece, first_valid = next(_chunks(synthetic_stretches(1, chunk, 3), chunk))
    np.savez(work / "example_history.npz", history=first_piece.numpy(), valid=first_valid.numpy(),
             **{n: np.zeros((1, *s), dtype=np.float32) for n, s in memory_shapes(model)})
    np.savez(work / "example_decode.npz", **example["decode"])

    rule = {k: meta.get(k) for k in ("history_gaps", "history_bridge_steps", "history_basal_carry_steps",
                                     "history_unlogged_events", "mdi_basal_missing", "unsized_meal_grams")}
    channel_names = [name for name, _ in CHANNEL_UNITS]
    return {
        "version": 3, "family_kind": "kronlstm", "serving": "stream", "step_minutes": 5,
        "channels": [{"name": n, "unit": u} for n, u in CHANNEL_UNITS],
        "horizon_steps": model.prediction_horizon, "chunk_steps": chunk, "tail_steps": TAIL_STEPS,
        "announced_channels": [channel_names[c] for c in model.announced_channels],
        "plan_channels": [channel_names[c] for c in model.plan_channels],
        "static_features": static_width, "static_version": meta.get("kronlstm_static_version") if static_width else None,
        "memory": [{"name": n, "shape_after_batch": list(s)} for n, s in memory_shapes(model)],
        "graphs": {name: {"inputs": [{"name": i.name, "shape": i.shape, "dtype": i.type} for i in s.get_inputs()],
                          "outputs": [o.name for o in s.get_outputs()]} for name, s in sessions.items()},
        "history_rule": rule, "opset": opset, "export_route": "torchscript",
        "onnx_vs_torch_max_abs_diff": max(c["max_abs_diff_mgdl"] for c in checks), "verification": checks,
        "kernel_parity": parity or "not measured: no CUDA device on the exporting machine",
    }


def consumer_doc(info: Mapping[str, Any]) -> str:
    """``ONNX.md`` for a streamed KronLSTM bundle: what the two graphs take and how a phone runs them."""
    plan = info["plan_channels"]
    announced = info["announced_channels"]
    rule = info.get("history_rule") or {}
    bridge = rule.get("history_bridge_steps")
    carry = rule.get("history_basal_carry_steps")
    unlogged = rule.get("history_unlogged_events")
    plan_text = (f"`future[..., 2]` is the planned extra dose in units at the step it is taken and NaN at every "
                 f"other step. Real data never fills it: it means one dose beyond what the person usually does."
                 if plan == ["bolus"] else "This model has no planned-dose channel; keep bolus NaN.")
    static_text = ("" if not info.get("static_features") else
                   f"\n`static` `(B, {info['static_features']})` is the person's own vector, version "
                   f"{info.get('static_version')} of `krontime.statics.person_statics`, the same for every origin.\n")
    return f"""# ONNX inference: streamed KronLSTM

This bundle holds the selected `{info['checkpoint']}` checkpoint as two self-contained graphs at opset
{info['opset']}. The model reads a person's record once, front to back, and keeps a memory; every forecast is
decoded from the memory at its origin. A phone keeps that memory between readings.

* `history.onnx` reads `{info['chunk_steps']}` five-minute steps into the memory.
* `decode.onnx` forecasts the next `{info['horizon_steps']}` steps ({info['horizon_steps'] * 5 // 60} hours) from
  the memory, the last `{info['tail_steps']}` glucose steps and what is known of the future.

Both graphs take and return physical units, with NaN for a missing value; the run's scalers are inside.
The channel order everywhere is glucose (mg/dL), basal (U/h), bolus (U), carbs (g).

## The memory

There are {len(info['memory'])} memory tensors, `recurrent_state_i` and `conv_state_i` for each layer,
float32 with the batch first (shapes in `onnx_meta.json`). A stretch of record starts from all zeros.
`history.onnx` returns the next memory as `<name>_next`; feed it back as `<name>` on the next call.
`valid` `(B, {info['chunk_steps']})` is true for a real step and false for padding: a padded step leaves
the memory as it was, so a phone can append one reading with one valid step and pad the rest.

## Preparing the history (the masked-history rule the model trained on)

* A reading is a finite glucose above zero; anything else is NaN and stays NaN, never filled.
* A stretch starts at the first reading after more than {bridge} missing five-minute steps. When a gap
  that long occurs, reset the memory to zeros and start again at the next reading.
* Basal is carried forward from its last recorded value for at most {carry} steps, never backwards; after
  that it is NaN.
* Bolus and carbs are zero where nothing was logged{', but only once the stretch has logged that channel: until the first logged bolus (or meal) the channel is NaN' if unlogged == 'missing' else ''}.

## The forecast

`glucose_tail` `(B, {info['tail_steps']})` is glucose in mg/dL ending at the origin, NaN where there was no
reading; the origin should hold a reading. `future` `(B, {info['horizon_steps']}, 4)` says what is known of
each forecast step and is NaN everywhere else. Announced channels: {', '.join(announced) or 'none'}. Announced
carbs are grams at each step, zero where no meal is announced; to withhold the announcement set the whole
channel to NaN. A what-if meal adds its grams at its step. {plan_text}
{static_text}
The output `glucose_mgdl` `(B, {info['horizon_steps']})` is the forecast at 5, 10, ... minutes.

```python
import numpy as np
import onnxruntime as ort
from huggingface_hub import snapshot_download

folder = snapshot_download("OWNER/REPO", allow_patterns=["*.onnx", "*.npz", "onnx_meta.json"])
history = ort.InferenceSession(folder + "/history.onnx", providers=["CPUExecutionProvider"])
decode = ort.InferenceSession(folder + "/decode.onnx", providers=["CPUExecutionProvider"])
memory_names = [i.name for i in history.get_inputs()][2:]

with np.load(folder + "/example_history.npz") as ex:  # one hour of history from an empty memory
    memory = history.run(None, dict(ex))
with np.load(folder + "/example_decode.npz") as ex:  # a decode example, with its own memory
    forecast = decode.run(None, dict(ex))[0]
print(forecast.shape)
```

`onnx_meta.json` records the inputs, the memory shapes, the checkpoint SHA-256 and the checks: ONNX Runtime
against PyTorch on synthetic stretches with a sensor gap, an unlogged channel, an announced meal, a withheld
announcement and a planned dose, and the PyTorch path against the model's own GPU kernels where measured. These
are conversion checks; they do not measure forecast accuracy.
"""


__all__ = [
    "CASES", "CHANNEL_UNITS", "CHUNK_STEPS", "KronDecodeGraph", "KronHistoryGraph", "N_CHANNELS", "PlainKronMixer",
    "TAIL_STEPS", "channel_affines", "consumer_doc", "empty_memory", "export_stream_bundle", "kernel_parity",
    "memory_from_states", "memory_shapes", "plain_stream_forecast", "scaler_affine", "synthetic_future",
    "synthetic_stretches",
]
