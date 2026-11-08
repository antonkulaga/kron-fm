#!/usr/bin/env python3
"""KronLSTM forecaster: a causal KronLSTM stack over weeks of history, read out by a short decoder.

Ported from glucose_forecasting_metabonet's ``kronlstm/model.py`` at commit 6e9bbbe91, where it is trained
and scored on MetaboNet; that repository's ``stream.py``, ``stream_train.py``, ``spec.py`` and ONNX export
stay there, because they are wired into its competition trainer and data pipeline. The residual block,
the SwiGLU and the recurrent-state cache now live in :mod:`kron_core.blocks`; module and parameter names
are unchanged, so the glucose repository's checkpoints load here as they are.

KronLSTM (Hartl et al., NX-AI, arXiv 2610.26834) is a linear-attention recurrence: each head keeps
a ``K x V`` matrix memory written along one rank-one Kronecker key per step, so a step costs the
same at the first reading of a record as at the ten-thousandth. Training runs NX-AI's chunkwise
Triton kernels (``KronLSTM`` from their flash-linear-attention snapshot, vendored at
``kronlstm.vendor.fla`` by ``scripts/vendor_kronlstm.py``). On an RTX 4090
one layer of width 256 runs forward and backward at about two million steps a second whether the
history is ten days or ninety, which is what makes a whole sensor wear period (10 to 14 days, 2,880
to 4,032 steps) affordable as context where an attention encoder pays for the square of it.

The model reads the shared forecast trainer's windows exactly as ``ConvTransformerModel`` does:

* ``x`` is the ``(B, T, 4)`` scaled history (glucose, basal, bolus, carbs). NaN means "no
  reading". Steps before the first finite value of a window are padding in front of a short
  history and are removed from the recurrence through the layer's padding mask, so a left-padded
  short window and the short window alone give the same forecast. A NaN glucose after that point
  is a missing reading inside the history: it stays a step, with its "given" flag at zero, so the
  recurrence knows that time passed.
* ``future`` is the optional known-future block. With announced channels (``--future-channels
  carbs``) it is ``(B, horizon, 4)`` and only those channels are read from it. In gap mode (C55,
  ``post_gap_steps > 0``) it is ``(B, gap + post, 4)``: the glucose observed after the hidden span
  is read, the glucose inside the span is forced to "not given" whatever the caller put there,
  and ``gap_covariates`` are read across the whole block.
* With a masked history (C61, ``history_rule`` set from a ``--history-gaps masked`` run's
  metadata) the windows keep every glucose gap as NaN and start at a reading, so a random training
  crop is moved on to its first glucose reading as well, and a gap window whose post-gap glucose
  is all missing still counts as two-sided when its block carries in-gap covariates. Without the
  rule the model is exactly the filled one.

Every window is laid out as one causal sequence, the history followed by a **decode block**: one
token per future step, carrying what is known about that step (nothing in strict forecasting,
the announced carbohydrates, or the in-gap covariates and the post-gap glucose in gap mode) plus
its distance from the origin. The block's tokens read the history only through the recurrent
memory, at their own queries. A two-layer bidirectional transformer over the block then reads it
whole, and the first ``horizon`` steps of the hidden span give the forecast.

**The known future is read first** (layout ``known_future_first``, C63). In a gap-mode window
the block's steps after the hidden span come before the span's own steps in the sequence, so the
recurrence has read the glucose observed after a gap by the time it reaches the steps it fills.
A window that knows nothing after the span, which is every forecast-mode window, leaves those
steps out of the recurrence and the decoder altogether through the layer's padding mask, so its
forecast positions follow its history directly, exactly as in strict forecasting. The decoder
also reads every block step's raw values and "given" flags through ``skip_project``, beside the
recurrence's reading of them, so post-gap glucose and an announced meal reach the forecast
positions in one attention layer. The forecast is a change from a starting line: flat at the
last glucose reading in forecasting, and in a two-sided window the straight line from the last
reading before the gap to the first reading after it.

C60's runs (R.59, R.60) were trained with the step-order layout (``step_order``): the hidden span
first and the steps after it last, no skip and a flat start. There the post-gap tokens came after
the forecast positions in the causal order, and their raw values reached the decoder only through
six layers dominated by the history, so the trained models ignored the block: two-sided and
announced-carbs readings equalled strict forecasting. A checkpoint records its layout in
``kronlstm_decode_layout``; one without it is rebuilt in ``step_order``, so old checkpoints keep
their own behaviour, and :meth:`KronLSTMForecaster.load_initial_state` warm-starts the current
layout from them.

**Inputs in their own units (C64).** The shared windows hand every channel over as a MinMax
fraction of the training frame's range, and on MetaboNet those ranges are set by outliers: carbs
reach 855 g, basal 384 U/h and bolus 50 U, so a 60 g meal arrives as 0.07 and 1 U/h of basal as
0.003, beside glucose fractions of 0.2 to 0.5. R.62 (C63's validation) trained that way and a 60 g
meal moved its forecast by 0.20 mg/dL; a synthetic world in which a 60 g meal adds 71 mg/dL an hour
later reproduced it exactly on the same ranges (0.3 mg/dL) and was cured by this change (71 mg/dL),
through the real trainer as well. With ``input_norm="physical"`` the model maps each channel back
to its own units through the run's scalers (``channel_affine``, recorded by the trainer as
``kronlstm_channel_affine``) and reads ``(value - centre) / scale`` clipped to ten units: glucose
around 140 mg/dL in 50 mg/dL steps, basal around 1 U/h, bolus in 5 U and carbs in 30 g. Both heads
then speak :data:`OUTPUT_MGDL` per unit rather than the 499 mg/dL a MinMax glucose unit is. The
scaled windows, the scalers and every scorer are unchanged: only what the projections read and how
the heads' outputs are read differ, and a checkpoint of either norm loads into either.
``minmax``, the constructor's default and what metadata without ``kronlstm_input_norm`` rebuilds,
is the model every run before C64 trained.

**Dense past-only auxiliary loss.** A causal model produces a state at every history step, so
every step whose next two hours lie inside the same history is a free training example. The
auxiliary head forecasts the next ``aux_horizon`` steps from each history position and is scored
against glucose that is already in the window, never against the window's own target. The loss
is the shared task loss's MARD (a fraction, with the reference clamped at 40 mg/dL) of those
forecasts when the glucose scaler is known, and is handed to the shared training loop through
:meth:`get_auxiliary_loss`, weighted by ``aux_weight``; it is computed only in training mode.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from kron_core.blocks import KronLSTMBlock, LayerStates, SwiGLU, _read_block, kronlstm_mixer  # noqa: F401
from krontime.statics import STATIC_FEATURES

#: The smallest reference glucose a MARD divides by, in mg/dL (glucose_forecasting_metabonet's
#: ``common.losses.MARD_MIN_REFERENCE_MGDL``).
MARD_MIN_REFERENCE_MGDL: float = 40.0

#: Channel order of the shared window (``ConvTransformerFamilySpec.feature_names``).
N_CHANNELS = 4
#: Steps per chunk when a large evaluation batch is split: 64 windows of 4,032 steps and the block,
#: which fits a 16 GB card beside a training run's optimizer state.
INFERENCE_TOKENS = 1 << 18
#: Value plus a "given" flag per channel.
N_TOKEN_FEATURES = 2 * N_CHANNELS
#: Time of day (C74, the night forecaster): the local hour as a sine and a cosine, and whether it is known.
CLOCK_FEATURES = 3
#: The decode-block layouts a checkpoint can carry, recorded as ``kronlstm_decode_layout``.
LAYOUT_STEP_ORDER = "step_order"
LAYOUT_KNOWN_FUTURE_FIRST = "known_future_first"
DECODE_LAYOUTS = (LAYOUT_STEP_ORDER, LAYOUT_KNOWN_FUTURE_FIRST)
#: How the history and the block reach the projections and how the heads' outputs are read,
#: recorded as ``kronlstm_input_norm`` (C64). ``minmax`` is every run before C64 (21 October 2026): the
#: shared MinMax fractions go in and the heads speak them. On the MetaboNet training frame those
#: span 0 to 855 g of carbs and 0 to 384 U/h of basal, so a 60 g meal arrives as 0.07 and 1 U/h as
#: 0.003, and one unit of a head's output is 499 mg/dL. ``physical`` maps each channel back to its
#: own units through the run's scalers and then to ``(value - centre) / scale``, clipped, so a 60 g
#: meal arrives as 2, and one unit of a head's output is :data:`OUTPUT_MGDL`.
INPUT_MINMAX = "minmax"
INPUT_PHYSICAL = "physical"
INPUT_NORMS = (INPUT_MINMAX, INPUT_PHYSICAL)
#: Glucose mg/dL, basal U/h, bolus U and carbs g: the centre and the size of one physical unit.
PHYSICAL_CENTRE = (140.0, 1.0, 0.0, 0.0)
PHYSICAL_SCALE = (50.0, 1.0, 5.0, 30.0)
#: A physical feature is clipped to this many units: 300 g of carbs, 50 U of bolus, 11 U/h of basal.
PHYSICAL_CLIP = 10.0
#: One unit of a physical head's output, in mg/dL.
OUTPUT_MGDL = 50.0


def _value_and_given(x: torch.Tensor) -> torch.Tensor:
    """``(..., C)`` with NaN for "no reading" -> ``(..., 2C)``: values with NaN as zero, then the flags."""
    given = torch.isfinite(x)
    return torch.cat([torch.where(given, x, torch.zeros_like(x)), given.to(x.dtype)], dim=-1)


def clock_features(hours: torch.Tensor) -> torch.Tensor:
    """``(...)`` hours since local midnight, NaN where the clock is unknown -> ``(..., 3)``: sine, cosine, known."""
    known = torch.isfinite(hours)
    angle = torch.where(known, hours, torch.zeros_like(hours)) * (2 * torch.pi / 24.0)
    flag = known.to(hours.dtype)
    return torch.stack([torch.sin(angle) * flag, torch.cos(angle) * flag, flag], dim=-1)


def block_clock(origin_hour: torch.Tensor, steps: int) -> torch.Tensor:
    """``(N,)`` local hour at each origin -> ``(N, steps)`` the hour at each block step, five minutes apart.

    Block step ``k`` is ``k + 1`` steps after the origin, in the forecast and after the hidden span
    alike, so the clock of a forecast is known before it happens.
    """
    ahead = torch.arange(1, steps + 1, device=origin_hour.device, dtype=origin_hour.dtype) * (5.0 / 60.0)
    return torch.remainder(origin_hour.unsqueeze(1) + ahead.unsqueeze(0), 24.0)


def history_padding(x: torch.Tensor) -> torch.Tensor:
    """``(B, T)`` True on the steps before a window's first finite value: padding, not history."""
    seen = torch.isfinite(x).any(dim=-1).to(torch.int32).cumsum(dim=1)
    return seen == 0


def pad_before_first_glucose(x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
    """Extend ``pad`` ``(B, T)`` to every step before a window's first kept glucose reading.

    A masked-history window always starts at a reading; a random crop can leave covariate-only
    steps in front of the first one, and those become padding too, as a served history would be.
    """
    seen = (torch.isfinite(x[..., 0]) & ~pad).to(torch.int32).cumsum(dim=1)
    return pad | (seen == 0)


def last_glucose(glucose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(B, T)`` -> the last finite value at or before every step, and whether one exists."""
    steps = glucose.size(1)
    finite = torch.isfinite(glucose)
    index = torch.where(finite, torch.arange(steps, device=glucose.device).expand_as(glucose),
                        torch.full_like(glucose, -1, dtype=torch.long))
    index = index.cummax(dim=1).values
    value = torch.gather(torch.nan_to_num(glucose, nan=0.0), 1, index.clamp(min=0))
    return value, index >= 0




class KronLSTMForecaster(nn.Module):
    """``(B, T, 4)`` scaled history [+ known-future block] -> ``(B, horizon)`` scaled glucose."""

    #: Read by the two-sided scorer (``forecast_contract.gap_predict``): steps before a person's
    #: first record are handed over as NaN padding, as in training, rather than back-filled.
    pads_short_history = True

    def __init__(
        self,
        *,
        n_time_steps: int,
        prediction_horizon: int = 24,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 6,
        ff_units: int = 512,
        dropout: float = 0.1,
        decoder_layers: int = 2,
        announced_channels: Sequence[int] = (),
        post_gap_steps: int = 0,
        gap_steps: int = 48,
        gap_covariates: Sequence[int] = (),
        min_input_steps: int = 0,
        future_dropout: float = 0.0,
        aux_weight: float = 0.5,
        aux_horizon: int = 24,
        glucose_mean: float | None = None,
        glucose_scale: float | None = None,
        history_rule: object | None = None,
        decode_layout: str = LAYOUT_KNOWN_FUTURE_FIRST,
        input_norm: str = INPUT_MINMAX,
        channel_affine: Sequence[Sequence[float]] | None = None,
        clock: bool = False,
        static: bool = False,
        static_features: int = STATIC_FEATURES,
        plan_channels: Sequence[int] = (),
        side_channels: int = 0,
    ) -> None:
        super().__init__()
        # R.89: wearable channels after the four, read in the history only (kronlstm.wearables).
        self.side_channels = int(side_channels)
        if self.side_channels < 0:
            raise ValueError("side_channels must be non-negative")
        # C74: every history and block token also reads the local time of day (clock_features).
        self.clock = bool(clock)
        # C79: every block token also reads the person's static vector (krontime.statics).
        self.static = bool(static)
        self.n_time_steps = int(n_time_steps)
        self.prediction_horizon = int(prediction_horizon)
        self.d_model = int(d_model)
        announced = tuple(int(c) for c in announced_channels)
        if 0 in announced or any(c < 0 or c >= N_CHANNELS for c in announced):
            raise ValueError(f"announced_channels {announced} must be covariate channels 1..3, never glucose")
        self.announced_channels = announced
        self.post_gap_steps, self.gap_steps = int(post_gap_steps), int(gap_steps)
        if self.post_gap_steps and self.gap_steps < self.prediction_horizon and (
                decode_layout != LAYOUT_KNOWN_FUTURE_FIRST
                or self.gap_steps + self.post_gap_steps < self.prediction_horizon):
            raise ValueError("in gap mode the hidden span must cover the forecast horizon, or, for a universal model "
                             "(the known_future_first layout), the hidden span and the steps after it must")
        gap_cov = tuple(int(c) for c in gap_covariates)
        if gap_cov and not self.post_gap_steps:
            raise ValueError("gap_covariates need gap mode (post_gap_steps > 0)")
        if 0 in gap_cov:
            raise ValueError("gap_covariates are covariate channels, never glucose")
        self.gap_covariates = gap_cov
        # C86: plan channels, read from the block over the horizon like an announcement but never filled
        # by real data (the windows, the stream plan and every scorer fill only the announced channels and
        # the in-gap covariates). Only the intervention-consistency loss and what-if serving write them, so
        # a planned dose is read as one extra dose beyond what the person usually does
        # (metabonet.physio_consistency). A forecast-only model only: in gap mode the block's insulin is real.
        plan = tuple(int(c) for c in plan_channels)
        if plan and (0 in plan or any(c < 0 or c >= N_CHANNELS for c in plan)):
            raise ValueError(f"plan_channels {plan} must be covariate channels 1..3, never glucose")
        if plan and (set(plan) & set(announced) or self.post_gap_steps):
            raise ValueError("plan_channels must differ from the announced channels, in a forecast-only model")
        self.plan_channels = plan
        self.min_input_steps = int(min_input_steps)
        if not 0 <= self.min_input_steps <= self.n_time_steps:
            raise ValueError(f"min_input_steps must lie in [0, {self.n_time_steps}], got {min_input_steps}")
        if not 0.0 <= float(future_dropout) <= 1.0:
            raise ValueError(f"future_dropout must be in [0, 1], got {future_dropout}")
        if future_dropout and not announced:
            raise ValueError("future_dropout withholds announced channels; this model has none")
        self.future_dropout = float(future_dropout)
        self.aux_weight, self.aux_horizon = float(aux_weight), int(aux_horizon)
        if (glucose_mean is None) != (glucose_scale is None):
            raise ValueError("glucose_mean and glucose_scale come together or not at all")
        self.glucose_mean = None if glucose_mean is None else float(glucose_mean)
        self.glucose_scale = None if glucose_scale is None else float(glucose_scale)
        # The run's masked-history rule (C61), or None for a filled run. A plain attribute, not a
        # buffer: it is not in the state dict, and the two-sided scorer reads it to serve the
        # model its gaps as gaps.
        self.history_rule = history_rule
        # How this model is served (C66): "window", a fixed-length history per origin as every
        # family is, or "stream", each person's record read once with the memory carried to every
        # origin. Set by the spec from the run's metadata; a plain attribute, not in the state dict.
        self.serving = "window"
        if decode_layout not in DECODE_LAYOUTS:
            raise ValueError(f"decode_layout must be one of {DECODE_LAYOUTS}, got {decode_layout!r}")
        self.decode_layout = decode_layout
        if input_norm not in INPUT_NORMS:
            raise ValueError(f"input_norm must be one of {INPUT_NORMS}, got {input_norm!r}")
        self.input_norm = input_norm
        self.channel_affine = None if channel_affine is None else tuple(
            (float(a), float(b)) for a, b in channel_affine)
        if input_norm == INPUT_PHYSICAL:
            if self.channel_affine is None or len(self.channel_affine) != N_CHANNELS:
                raise ValueError(f"physical inputs need the {N_CHANNELS} channels' scaler affines (channel_affine)")
            if self.glucose_scale is None:
                raise ValueError("physical inputs need the glucose affine (glucose_mean and glucose_scale)")
            gain = torch.tensor([a for a, _ in self.channel_affine]) / torch.tensor(PHYSICAL_SCALE)
            offset = (torch.tensor([b for _, b in self.channel_affine]) - torch.tensor(PHYSICAL_CENTRE)
                      ) / torch.tensor(PHYSICAL_SCALE)
            # Rebuilt from the run's metadata, never saved: a checkpoint of either norm loads into either.
            self.register_buffer("feature_gain", gain, persistent=False)
            self.register_buffer("feature_offset", offset, persistent=False)
            self.output_gain = OUTPUT_MGDL / self.glucose_scale
        else:
            self.feature_gain = self.feature_offset = None
            self.output_gain = 1.0

        max_block = max(self.prediction_horizon, self.gap_steps + self.post_gap_steps)
        self.max_block = max_block
        n_in = N_TOKEN_FEATURES + (CLOCK_FEATURES if self.clock else 0)
        self.history_project = nn.Linear(n_in, self.d_model)
        self.block_project = nn.Linear(n_in, self.d_model)
        # 0: a step inside the horizon or the hidden span, 1: a step after the hidden span.
        self.block_kind = nn.Embedding(2, self.d_model)
        self.block_lead = nn.Embedding(max_block, self.d_model)
        # The block's raw values and given flags, added to the decoder's input (known_future_first).
        self.skip_project = (nn.Linear(n_in, self.d_model)
                             if decode_layout == LAYOUT_KNOWN_FUTURE_FIRST else None)
        # The person's static vector, added to every block token; zero at the start (see _init_weights).
        # Its width is the static version's (krontime.statics.static_features): 11 for C79's, longer for version 2.
        self.static_project = nn.Linear(int(static_features), self.d_model, bias=False) if self.static else None
        # The wearable channels' values and given flags, added to every history token; zero at the start, with
        # no bias, so a step whose wearables are unknown adds nothing (see _init_weights).
        self.side_project = (nn.Linear(2 * self.side_channels, self.d_model, bias=False)
                             if self.side_channels else None)
        self.blocks = nn.ModuleList(
            KronLSTMBlock(self.d_model, int(n_heads), int(ff_units), float(dropout), i) for i in range(int(n_layers))
        )
        self.final_norm = nn.RMSNorm(self.d_model, eps=1e-5)
        layer = nn.TransformerEncoderLayer(self.d_model, int(n_heads), 2 * self.d_model, float(dropout),
                                           activation="gelu", batch_first=True, norm_first=True)
        self.decoder = nn.TransformerEncoder(layer, int(decoder_layers), enable_nested_tensor=False)
        self.decoder_norm = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, 1)
        self.aux_head = nn.Linear(self.d_model, self.aux_horizon) if self.aux_weight > 0 else None
        self._aux_loss: torch.Tensor | None = None
        self._init_weights()

    @torch.no_grad()
    def _init_weights(self) -> None:
        # The KronLSTM layer initialises its own decay and output gate (an output-gate bias of 3, so
        # the layer starts nearly ungated) and keeps nn.Linear's default for its projections; the
        # decoder keeps PyTorch's transformer init. Everything else is small and centred.
        for name, p in self.named_parameters():
            if ".mixer." in name or name.startswith("decoder."):
                continue
            if p.dim() == 2:
                nn.init.normal_(p, std=0.02)
            elif name in ("history_project.bias", "block_project.bias"):
                # A step with nothing recorded reads as all-zero features (C67's missing events beside a
                # glucose gap and no basal). With a zero bias its token is exactly zero, stays zero through
                # every layer, and each RMSNorm passes it a gradient 1/sqrt(eps) times larger: the
                # kernel's backward then overflowed into NaN within four batches of a from-scratch run.
                nn.init.normal_(p, std=0.02)
            elif name.endswith(".bias"):
                nn.init.zeros_(p)
        # The forecast starts at "no change from the last reading" and the auxiliary head likewise.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        if self.aux_head is not None:
            nn.init.zeros_(self.aux_head.weight)
            nn.init.zeros_(self.aux_head.bias)
        # A static model starts as the model without statics, from scratch or warm-started from one.
        if self.static_project is not None:
            nn.init.zeros_(self.static_project.weight)
        # Likewise a wearable model starts as the model without wearables.
        if self.side_project is not None:
            nn.init.zeros_(self.side_project.weight)

    def history_tokens(self, x: torch.Tensor, hours: torch.Tensor | None = None) -> torch.Tensor:
        """``(B, T, 4 [+ K])`` scaled history -> ``(B, T, d_model)`` tokens, before the padding is zeroed.

        Columns after the four are the wearable channels (R.89): a wearable model adds their projection, a
        history without them reads as unknown wearables, and a model without wearables ignores them.
        """
        tokens = self.history_project(self.featurise(x[..., :N_CHANNELS], hours))
        if self.side_project is None or x.size(-1) <= N_CHANNELS:
            return tokens
        side = x[..., N_CHANNELS:N_CHANNELS + self.side_channels]
        if side.size(-1) != self.side_channels:
            raise ValueError(f"this model reads {self.side_channels} wearable channels; the history holds "
                             f"{side.size(-1)}")
        return tokens + self.side_project(_value_and_given(side))

    def featurise(self, values: torch.Tensor, hours: torch.Tensor | None = None) -> torch.Tensor:
        """``(..., 4)`` scaled values, NaN for "no reading" -> ``(..., 8)``: what the projections read.

        A clock model (C74) also reads ``hours`` ``(...)``, each step's local hour, as three more
        features; without them its clock reads as unknown, the same features a step with no clock has.
        """
        if self.feature_gain is not None:
            # NaN stays NaN through the affine and the clip, so the given flags are unchanged.
            values = (values * self.feature_gain + self.feature_offset).clamp(-PHYSICAL_CLIP, PHYSICAL_CLIP)
        features = _value_and_given(values)
        if not self.clock:
            return features
        if hours is None:
            hours = torch.full(values.shape[:-1], float("nan"), device=values.device, dtype=values.dtype)
        return torch.cat([features, clock_features(hours.to(device=values.device, dtype=values.dtype))], dim=-1)

    # ---- the decode block -----------------------------------------------------------------
    @property
    def block_steps(self) -> int:
        """Steps of the known-future block a gap-mode model reads: the gap and what follows it."""
        return self.gap_steps + self.post_gap_steps if self.post_gap_steps else self.prediction_horizon

    @property
    def universal(self) -> bool:
        """A gap-mode model whose forecast reaches past its hidden span: a 10-hour forecast beside the
        4-hour gap (Anton, 2026-10-23: "when I asked you to make 10h module I wanted it to be universal").

        One block serves both modes. A forecast window reads all of it in time order, as a
        forecast-only model of the same horizon does; a two-sided window reads the steps after the
        hidden span first and then the span, as every gap-mode model does. Which a window is, is read
        from what its block carries (:meth:`_two_sided`).
        """
        return bool(self.post_gap_steps) and self.prediction_horizon > self.gap_steps

    def _two_sided(self, values: torch.Tensor) -> torch.Tensor:
        """``(B,)``: which windows of a universal model's block are two-sided.

        Glucose after the hidden span, or an in-gap covariate a forecast window never carries (the
        trainer's forward blanking keeps only the announced channels there).
        """
        known = torch.isfinite(values[:, self.gap_steps:, 0]).any(dim=1)
        other = [c for c in self.gap_covariates if c not in self.announced_channels]
        if other:
            known = known | torch.isfinite(values[:, :, other]).flatten(1).any(dim=1)
        return known

    def _universal_order(self, values: torch.Tensor) -> torch.Tensor:
        """``(B, S)``: each window's block steps in reading order (see :attr:`universal`)."""
        steps = torch.arange(self.block_steps, device=values.device)
        two_sided = torch.cat([steps[self.gap_steps:], steps[:self.gap_steps]])
        return torch.where(self._two_sided(values).unsqueeze(1), two_sided.unsqueeze(0), steps.unsqueeze(0))

    def _block_delta(self, decoded: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """``(B, S, d)`` decoded block in reading order -> ``(B, horizon)`` change in time order."""
        if not self.universal:
            start = self.known_steps  # the forecast positions follow the known future in reading order
            return self.head(decoded[:, start:start + self.prediction_horizon]).squeeze(-1) * self.output_gain
        in_reading_order = self.head(decoded).squeeze(-1) * self.output_gain
        order = self._universal_order(values)
        return torch.empty_like(in_reading_order).scatter_(1, order, in_reading_order)[:, :self.prediction_horizon]

    @property
    def known_steps(self) -> int:
        """Block steps read ahead of the forecast positions: the steps after the hidden span, or none."""
        return self.post_gap_steps if self.decode_layout == LAYOUT_KNOWN_FUTURE_FIRST else 0

    def block_order(self, device: torch.device) -> torch.Tensor:
        """The block's step indices in the order the sequence reads them (see the module docstring)."""
        steps = torch.arange(self.block_steps, device=device)
        if not self.known_steps:
            return steps
        return torch.cat([steps[self.gap_steps:], steps[:self.gap_steps]])

    def decode_block(self, future: torch.Tensor | None, batch: int, like: torch.Tensor) -> torch.Tensor:
        """``(B, S, 4)`` known-future block or None -> ``(B, block_steps, 4)``: what is known of each step.

        The block always has :attr:`block_steps` steps, ``horizon`` for a forecast-only model and
        the hidden span plus the steps after it for a gap-mode model, so a window is decoded through
        the same layout in training and in serving whatever the caller hands in. What each step
        carries is read from ``future`` channel by channel: the announced channels over the horizon,
        the in-gap covariates over the whole block, and the glucose after the hidden span. Glucose
        inside the span is never read. The trainer's forward-mode blanking (``blank_for_forward``)
        has already removed what a forecast may not see, so reading these channels is reading
        exactly what the window's mode allows. NaN means "not known": a missing block, or a block
        shorter than the model's, is padded with it.
        """
        steps = self.block_steps
        values = torch.full((batch, steps, N_CHANNELS), float("nan"), device=like.device, dtype=like.dtype)
        if future is not None:
            if future.dim() != 3 or future.size(-1) != N_CHANNELS:
                raise ValueError(f"Expected a (batch, steps, {N_CHANNELS}) known-future block, got {tuple(future.shape)}")
            block = future[:, :steps].to(device=like.device, dtype=like.dtype)
            span = block.size(1)
            if self.announced_channels:
                ahead = min(span, self.prediction_horizon)
                values[:, :ahead, list(self.announced_channels)] = block[:, :ahead, list(self.announced_channels)]
            if self.plan_channels:
                ahead = min(span, self.prediction_horizon)
                values[:, :ahead, list(self.plan_channels)] = block[:, :ahead, list(self.plan_channels)]
            if self.post_gap_steps:
                if self.gap_covariates:
                    values[:, :span, list(self.gap_covariates)] = block[:, :, list(self.gap_covariates)]
                    if self.announced_channels:
                        # An announced channel inside the horizon keeps its announcement even when it
                        # is also an in-gap covariate: blanking keeps it there, and so does this read.
                        ahead = min(span, self.prediction_horizon)
                        idx = list(self.announced_channels)
                        values[:, :ahead, idx] = block[:, :ahead, idx]
                if span > self.gap_steps:
                    values[:, self.gap_steps:span, 0] = block[:, self.gap_steps:, 0]
            if self.training and self.future_dropout > 0:
                # One draw per window withholds its announcement, as t0 and Timestral do, so one
                # model serves strict forecasting and forecasting with announced carbohydrates. Only
                # a forecast-mode window is dropped: a two-sided window reads the in-gap covariates
                # anyway, because the annual file keeps them.
                ahead = min(span, self.prediction_horizon)
                if self.universal:
                    # The block is all horizon here, so "nothing after the horizon" cannot tell the modes apart.
                    forward = ~self._two_sided(values)
                elif self.history_rule is not None:
                    # A masked two-sided window may have no post-gap reading at all, but its block
                    # still carries the in-gap covariates; only a block with nothing after the
                    # horizon is a forecast window.
                    forward = ~torch.isfinite(values[:, self.prediction_horizon:]).flatten(1).any(1)
                else:
                    forward = ~torch.isfinite(values[:, :, 0]).any(dim=1)
                drop = forward & (torch.rand(batch, device=like.device) < self.future_dropout)
                idx = list(self.announced_channels)
                withheld = values[:, :ahead, idx].masked_fill(drop.view(-1, 1, 1), float("nan"))
                values[:, :ahead, idx] = withheld
        return values

    def block_tokens(self, values: torch.Tensor, hours: torch.Tensor | None = None,
                     static: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(B, S, 4)`` block values -> tokens, raw features and whether each step is read, in reading order.

        The raw features are the values with their given flags, which the decoder reads through
        ``skip_project``. A step after the hidden span is read only by a window that knows
        something there; in every other window those steps are left out of the recurrence and
        the decoder (``known_future_first``). The step-order layout reads every step. A static
        model (C79) adds ``static_project`` of each window's ``(B, static_features)`` vector to
        every block token; without one its person reads as unknown, the all-zero vector.
        """
        tokens, features, read = self._block_tokens(values, hours)
        if self.static_project is not None and static is not None:
            tokens = tokens + self.static_project(static.to(device=tokens.device, dtype=tokens.dtype)).unsqueeze(1)
        return tokens, features, read

    def _block_tokens(self, values: torch.Tensor, hours: torch.Tensor | None
                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.universal:
            order = self._universal_order(values)
            ordered = values.gather(1, order.unsqueeze(-1).expand(-1, -1, values.size(-1)))
            kind = (self._two_sided(values).unsqueeze(1) & (order >= self.gap_steps)).long()
            features = self.featurise(ordered, None if hours is None else hours.gather(1, order))
            tokens = self.block_project(features) + self.block_kind(kind) + self.block_lead(order)
            return tokens, features, torch.ones(values.shape[:2], dtype=torch.bool, device=values.device)
        order = self.block_order(values.device)
        kind = torch.zeros(self.block_steps, dtype=torch.long, device=values.device)
        if self.post_gap_steps:
            kind[self.gap_steps:] = 1
        features = self.featurise(values[:, order], None if hours is None else hours[:, order])
        tokens = self.block_project(features) + self.block_kind(kind[order]) + self.block_lead(order)
        read = torch.ones(values.shape[:2], dtype=torch.bool, device=values.device)
        if self.known_steps:
            after = torch.isfinite(values[:, self.gap_steps:]).flatten(1).any(dim=1)
            read[:, :self.known_steps] = after.unsqueeze(1)
        return tokens, features, read

    # ---- forward ---------------------------------------------------------------------------
    def _crop_mask(self, pad: torch.Tensor) -> torch.Tensor:
        """Training with ``min_input_steps``: mask a random left prefix so short histories are seen."""
        if not (self.training and self.min_input_steps):
            return pad
        batch, steps = pad.shape
        keep = torch.randint(self.min_input_steps, steps + 1, (batch, 1), device=pad.device)
        return pad | (torch.arange(steps, device=pad.device).unsqueeze(0) < steps - keep)

    def encode(self, x: torch.Tensor, future: torch.Tensor | None, hours: torch.Tensor | None = None,
               block_hours: torch.Tensor | None = None, static: torch.Tensor | None = None
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """-> the encoded sequence (history, then the block in reading order), the masked history,
        the block's values in step order, its raw features in reading order, and which steps were read."""
        if x.dim() != 3 or x.size(-1) < N_CHANNELS:
            raise ValueError(f"Expected (batch, time, {N_CHANNELS} [+ wearables]) input, got {tuple(x.shape)}")
        if future is not None and not (self.announced_channels or self.post_gap_steps):
            raise ValueError("this KronLSTM forecaster was built past-only and cannot be handed a known-future block")
        if future is None and self.announced_channels:
            raise ValueError("this KronLSTM forecaster was built with announced channels and was called without them")
        batch, steps, _ = x.shape
        # Padding is read from the four channels: a wearable value before the first reading is not history.
        pad = self._crop_mask(history_padding(x[..., :N_CHANNELS]))
        if self.history_rule is not None:
            pad = pad_before_first_glucose(x, pad)
        x = x.masked_fill(pad.unsqueeze(-1), float("nan"))
        history = self.history_tokens(x, hours).masked_fill(pad.unsqueeze(-1), 0.0)
        x = x[..., :N_CHANNELS]
        values = self.decode_block(future, batch, x)
        block, features, read = self.block_tokens(values, block_hours, static)
        h = torch.cat([history, block], dim=1)
        valid = torch.cat([~pad, read], dim=1)
        # The padding mask packs each window's real steps into one variable-length batch inside the
        # layer, which also takes the block steps a window does not read out of its sequence; a
        # batch with nothing to leave out skips the packing.
        mask = None if bool(valid.all()) else valid.to(torch.int32)
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda):
            for blk in self.blocks:
                h = blk(h, mask)
        h = self.final_norm(h.float()).masked_fill(~valid.unsqueeze(-1), 0.0)
        return h, x, values, features, valid

    def forward(self, x: torch.Tensor, future: torch.Tensor | None = None, hours: torch.Tensor | None = None,
                block_hours: torch.Tensor | None = None, static: torch.Tensor | None = None) -> torch.Tensor:
        steps = x.size(1) + self.block_steps
        if not self.training and x.size(0) * steps > INFERENCE_TOKENS:
            # The scoring paths hand over hundreds of weeks-long windows at once (the full-iLet
            # monitor scores 512 a call); in evaluation they are run in chunks of about
            # INFERENCE_TOKENS steps, which changes nothing but the peak memory.
            size = max(1, INFERENCE_TOKENS // steps)
            cut = lambda a, i: None if a is None else a[i:i + size]  # noqa: E731
            return torch.cat([
                self._forward(x[i:i + size], cut(future, i), cut(hours, i), cut(block_hours, i), cut(static, i))
                for i in range(0, x.size(0), size)
            ])
        return self._forward(x, future, hours, block_hours, static)

    def _forward(self, x: torch.Tensor, future: torch.Tensor | None, hours: torch.Tensor | None = None,
                 block_hours: torch.Tensor | None = None, static: torch.Tensor | None = None) -> torch.Tensor:
        h, x, values, features, valid = self.encode(x, future, hours, block_hours, static)
        steps = x.size(1)
        block, read = h[:, steps:], valid[:, steps:]
        if self.skip_project is not None:
            block = block + self.skip_project(features)
        decoded = self.decoder_norm(self.decoder(block, src_key_padding_mask=None if bool(read.all()) else ~read))
        delta = self._block_delta(decoded, values)
        forecast = self.starting_line(x[..., 0], values) + delta
        if self.training and self.aux_head is not None:
            anchor, has_anchor = last_glucose(x[..., 0])
            self._aux_loss = self.aux_weight * self.dense_history_loss(h[:, :steps], x[..., 0], anchor, has_anchor)
        return forecast

    # ---- streaming (C66) ---------------------------------------------------------------------
    def stream_history(self, x: torch.Tensor, states: LayerStates, valid: torch.Tensor | None = None,
                       cuts: Sequence[int] = (), checkpoint: bool = False,
                       hours: torch.Tensor | None = None) -> tuple[torch.Tensor, list[LayerStates]]:
        """Read ``(B, T, 4)`` scaled history (with a wearable model's channels after the four) from ``states`` on.

        The same projections and layers as :meth:`encode`, without the block: a stream reads a
        person's record piece by piece, and the memory every layer holds after a piece is where the
        next piece starts. ``valid`` ``(B, T)`` marks the steps that belong to each stream (a stream
        that ends inside the piece is padded after its end); padding is taken out of the recurrence
        as :meth:`encode` takes it out. ``cuts`` are step counts at which every layer's memory is
        copied out, each as a :class:`LayerStates` the decoder can start from. Returns the final-
        normed hidden sequence, zero on padding, for the dense auxiliary loss, and the cut states.

        ``checkpoint`` (whole-stretch training, ``kronlstm.stream_train.WholePlan``) recomputes each
        block in the backward pass instead of keeping its activations, so a stretch of up to a year
        fits on a card with the gradient through all of it. A recompute must start from the memory
        the block started from, so it reads from zero memory and leaves ``states`` as given: it is for
        streams that start here, and refuses states that carry a memory.
        """
        if x.dim() != 3 or x.size(-1) < N_CHANNELS:
            raise ValueError(f"Expected (batch, time, {N_CHANNELS} [+ wearables]) input, got {tuple(x.shape)}")
        if len(states) != len(self.blocks):
            raise ValueError(f"the states hold {len(states)} layers and the model has {len(self.blocks)}")
        if valid is None:
            valid = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        pad = ~valid
        x = x.masked_fill(pad.unsqueeze(-1), float("nan"))
        h = self.history_tokens(x, hours).masked_fill(pad.unsqueeze(-1), 0.0)
        mask = None if bool(valid.all()) else valid.to(torch.int32)
        snapshots: list[list[dict[str, torch.Tensor] | None]] = [[] for _ in cuts]
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda):
            recompute = checkpoint and torch.is_grad_enabled()
            if recompute and any(states[i] is not None for i in range(len(states))):
                raise ValueError("a checkpointed stream starts from zero memory; these states carry one")
            for blk in self.blocks:
                if recompute:
                    h, snaps = torch.utils.checkpoint.checkpoint(_read_block, blk, h, mask, len(self.blocks),
                                                                 tuple(int(c) for c in cuts), use_reentrant=False)
                    for j, snap in enumerate(snaps):
                        snapshots[j].append(snap)
                elif cuts:
                    h, snaps = blk.forward_pieces(h, mask, states, cuts)
                    for j, snap in enumerate(snaps):
                        snapshots[j].append(snap)
                else:
                    h = blk(h, mask, states)
        h = self.final_norm(h.float()).masked_fill(pad.unsqueeze(-1), 0.0)
        return h, [LayerStates(s) for s in snapshots]

    def decode_from_states(self, states: LayerStates, future: torch.Tensor | None,
                           glucose: torch.Tensor, block_hours: torch.Tensor | None = None,
                           static: torch.Tensor | None = None) -> torch.Tensor:
        """The forecast of ``N`` origins from the memory every layer holds at each of them.

        The block (:meth:`decode_block`, :meth:`block_tokens`) is read through the layers starting
        from ``states``, exactly as :meth:`encode` reads it after a window's history, then decoded
        as :meth:`_forward` decodes it. ``glucose`` ``(N, T)`` is scaled history glucose ending at
        each origin, NaN where there was no reading, from which the starting line is drawn; a few
        steps are enough, since an origin holds a reading. ``states`` is not changed.
        """
        batch = glucose.size(0)
        if future is not None and not (self.announced_channels or self.post_gap_steps):
            raise ValueError("this KronLSTM forecaster was built past-only and cannot be handed a known-future block")
        values = self.decode_block(future, batch, glucose)
        block, features, read = self.block_tokens(values, block_hours, static)
        mask = None if bool(read.all()) else read.to(torch.int32)
        own = states.clone()
        h = block
        with torch.autocast(device_type=glucose.device.type, dtype=torch.bfloat16, enabled=glucose.is_cuda):
            for blk in self.blocks:
                h = blk(h, mask, own)
        h = self.final_norm(h.float()).masked_fill(~read.unsqueeze(-1), 0.0)
        if self.skip_project is not None:
            h = h + self.skip_project(features)
        decoded = self.decoder_norm(self.decoder(h, src_key_padding_mask=None if bool(read.all()) else ~read))
        delta = self._block_delta(decoded, values)
        return self.starting_line(glucose, values) + delta

    def forward_streamed(self, x: torch.Tensor, future: torch.Tensor | None = None, pieces: int = 3,
                         hours: torch.Tensor | None = None, block_hours: torch.Tensor | None = None,
                         static: torch.Tensor | None = None) -> torch.Tensor:
        """:meth:`forward` computed the streaming way: the window's history in ``pieces`` with the
        memory carried between them, then the block decoded from the memory at the origin.

        It reads the window exactly as :meth:`encode` does, padding included, so it equals
        :meth:`forward` up to the kernels' rounding; the parity tests and the capped streaming
        scorer, which checks the streaming path against the windowed one, are built on that.
        """
        steps = x.size(1)
        pad = history_padding(x)
        if self.history_rule is not None:
            pad = pad_before_first_glucose(x, pad)
        cuts = sorted({round(steps * k / pieces) for k in range(1, pieces)} - {0, steps})
        states = LayerStates.empty(len(self.blocks))
        _, _ = self.stream_history(x, states, valid=~pad, cuts=cuts, hours=hours)
        return self.decode_from_states(states, future, x.masked_fill(pad.unsqueeze(-1), float("nan"))[..., 0],
                                       block_hours=block_hours, static=static)

    def stream_aux_loss(self, h: torch.Tensor, glucose: torch.Tensor, lookahead: torch.Tensor,
                        last_before: torch.Tensor) -> torch.Tensor | None:
        """The dense past-only auxiliary loss of a stream's piece, weighted, or ``None`` without the head.

        ``glucose`` ``(B, T)`` is the piece's scaled glucose (NaN where there was no reading or the
        stream does not reach), ``lookahead`` ``(B, aux_horizon)`` the glucose after the piece in
        the same stretch, NaN beyond it, and ``last_before`` ``(B,)`` the last reading before the
        piece, NaN when the stretch starts in it, so every step's forecast starts at the last
        reading at or before it as in a window and targets reach past the piece's end.
        """
        if self.aux_head is None:
            return None
        carried = torch.cat([last_before.unsqueeze(1), glucose], dim=1)
        anchor, has_anchor = last_glucose(carried)
        return self.aux_weight * self.dense_history_loss(h, glucose, anchor[:, 1:], has_anchor[:, 1:],
                                                         lookahead=lookahead)

    def starting_line(self, glucose: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """``(B, T)`` history glucose and ``(B, S, 4)`` block values -> ``(B, horizon)``: where the forecast starts.

        The forecast is a change from this line. It is flat at the last glucose reading, or at zero
        when a window has none. In a window with glucose after the hidden span (layout
        ``known_future_first``) it is the straight line from the last reading before the gap to the
        first reading after it, so an untrained model already interpolates across the gap and what
        it learns is the shape of the excursion rather than the level glucose returns to. Times are
        counted in steps from the history's last step: its last reading sits at or before 0, forecast
        step ``k`` at ``k + 1``, and the first reading after the span at ``gap_steps + j + 1``.
        """
        steps = glucose.size(1)
        position = torch.arange(steps, device=glucose.device)
        last = torch.where(torch.isfinite(glucose), position, torch.full_like(position, -1)).amax(dim=1)
        has_last = last >= 0
        level = torch.nan_to_num(glucose, nan=0.0).gather(1, last.clamp(min=0).unsqueeze(1)).squeeze(1)
        level = torch.where(has_last, level, torch.zeros_like(level))
        flat = level.unsqueeze(1).expand(-1, self.prediction_horizon)
        if not self.known_steps:
            return flat
        after = values[:, self.gap_steps:, 0]
        offset = torch.arange(after.size(1), device=after.device)
        first = torch.where(torch.isfinite(after), offset, torch.full_like(offset, after.size(1))).amin(dim=1)
        has_first = first < after.size(1)
        level_after = torch.nan_to_num(after, nan=0.0).gather(1, first.clamp(max=after.size(1) - 1).unsqueeze(1)).squeeze(1)
        t_before = (last - (steps - 1)).to(glucose.dtype).unsqueeze(1)
        t_after = (first + self.gap_steps + 1).to(glucose.dtype).unsqueeze(1)
        t = torch.arange(1, self.prediction_horizon + 1, device=glucose.device, dtype=glucose.dtype).unsqueeze(0)
        line = level.unsqueeze(1) + (level_after - level).unsqueeze(1) * (t - t_before) / (t_after - t_before)
        line = torch.where((has_last & has_first).unsqueeze(1), line, flat)
        return torch.where((~has_last & has_first).unsqueeze(1), level_after.unsqueeze(1).expand_as(flat), line)

    # ---- warm start ------------------------------------------------------------------------
    def load_initial_state(self, state: dict[str, torch.Tensor], source: object) -> None:
        """Warm start (``--init-from``), weights only.

        A parent trained in the step-order layout (C60's R.59 and R.60) lacks only
        ``skip_project``, which keeps this model's initialisation; every other tensor loads
        strictly. The parent's weights then run in this model's layout, so a warm-started run is a
        new run that starts from them, never a continuation of the parent.
        """
        mine = self.state_dict()
        new = {k for k in mine if k.startswith("skip_project.")}
        absent = new - set(state)
        if absent and absent != new:
            raise KeyError(f"{source} holds part of skip_project: {sorted(new - absent)}")
        if absent:
            state = {**state, **{k: mine[k] for k in absent}}
            print(f"  Warm start: {source} predates the decoder skip (step-order layout); "
                  "skip_project keeps its initialisation", flush=True)
        if self.static_project is not None and "static_project.weight" not in state:
            # A parent without statics (C79): the projection starts at zero, so the warm-started model
            # forecasts exactly as its parent until training moves it.
            state = {**state, "static_project.weight": mine["static_project.weight"]}
            print(f"  Warm start: {source} has no static inputs; static_project starts at zero", flush=True)
        if self.side_project is not None and "side_project.weight" not in state:
            # A parent without wearables (R.89): the projection starts at zero, so the warm-started model
            # forecasts exactly as its parent until training moves it.
            state = {**state, "side_project.weight": torch.zeros_like(mine["side_project.weight"])}
            print(f"  Warm start: {source} has no wearable inputs; side_project starts at zero", flush=True)
        for name in ("history_project.weight", "block_project.weight", "skip_project.weight"):
            given, want = state.get(name), mine.get(name)
            if given is not None and want is not None and given.shape[1] < want.shape[1]:
                # A parent without the clock (C74): its input columns are kept and the clock's start at
                # zero, so the warm-started model forecasts exactly as its parent until the clock helps.
                grown = torch.zeros_like(want)
                grown[:, :given.shape[1]] = given.to(grown)
                state = {**state, name: grown}
                print(f"  Warm start: {name} gains {want.shape[1] - given.shape[1]} clock inputs, which start at zero",
                      flush=True)
        lead = state.get("block_lead.weight")
        if lead is not None and lead.shape[0] < mine["block_lead.weight"].shape[0]:
            # A longer block than the parent's (a 10-hour forecast from a 4-hour-block parent): the
            # parent's lead positions are kept and the new ones keep this model's initialisation.
            grown = mine["block_lead.weight"].clone()
            grown[:lead.shape[0]] = lead.to(grown)
            state = {**state, "block_lead.weight": grown}
            print(f"  Warm start: the block grows from {lead.shape[0]} to {grown.shape[0]} steps; the "
                  f"{grown.shape[0] - lead.shape[0]} new lead positions keep their initialisation", flush=True)
        self.load_state_dict(state, strict=True)

    # ---- the dense past-only auxiliary loss -----------------------------------------------
    def dense_history_loss(self, h: torch.Tensor, glucose: torch.Tensor, anchor: torch.Tensor,
                           has_anchor: torch.Tensor, lookahead: torch.Tensor | None = None) -> torch.Tensor:
        """Forecast ``aux_horizon`` steps from every history position, scored on the history itself.

        ``lookahead`` (C66, a stream's piece) is the glucose after the history, ``(B, aux_horizon)``,
        so the last steps are scored too; a window has none, and its last steps reach only as far
        as its own history."""
        steps, ahead = glucose.size(1), self.aux_horizon
        extended = (F.pad(glucose, (0, ahead), value=float("nan")) if lookahead is None
                    else torch.cat([glucose, lookahead.to(glucose.dtype)], dim=1))
        target = extended.unfold(1, ahead, 1)[:, 1:steps + 1]
        ok = torch.isfinite(target) & has_anchor.unsqueeze(-1)
        pred = anchor.unsqueeze(-1) + self.aux_head(h) * self.output_gain
        target = torch.nan_to_num(target, nan=0.0)
        if self.glucose_mean is not None:
            # The shared task loss's definition (common.losses MARD): a fraction, not a percentage,
            # with the reference clamped at 40 mg/dL, so aux_weight compares like with like.
            true = target * self.glucose_scale + self.glucose_mean
            err = (pred - target).abs() * abs(self.glucose_scale) / true.abs().clamp(min=MARD_MIN_REFERENCE_MGDL)
        else:
            err = (pred - target).abs()
        ok = ok.to(err.dtype)
        return (err * ok).sum() / ok.sum().clamp(min=1.0)

    def get_auxiliary_loss(self) -> torch.Tensor | None:
        """The weighted auxiliary loss of the last training forward, once; the shared loop adds it."""
        loss, self._aux_loss = self._aux_loss, None
        return loss

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
