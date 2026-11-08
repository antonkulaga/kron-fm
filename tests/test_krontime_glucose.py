"""The glucose forecaster ported to krontime: the model tests of glucose_forecasting_metabonet's
``tests/test_kronlstm.py`` (commit 6e9bbbe91) that need nothing but the model.

The tests of the family's registration, metadata, scorers and masked-history rule stay in that
repository with the code they test. Every model test runs on the CPU with :class:`kron_core.testing.CausalMixer`
standing in for the KronLSTM layer and on a CUDA device with the vendored layer (``device`` fixture in
``conftest.py``).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch


# ---- the model, on the CPU stand-in and on the GPU ---------------------------------------------


GAP = dict(post_gap_steps=48, gap_steps=48, gap_covariates=(1, 2, 3))


def _model(device, **overrides):
    from krontime.glucose import KronLSTMForecaster

    kw = dict(n_time_steps=320, prediction_horizon=24, d_model=64, n_heads=2, n_layers=2, ff_units=128,
              dropout=0.0, decoder_layers=1)
    kw.update(overrides)
    torch.manual_seed(0)
    model = KronLSTMForecaster(**kw).to(device)
    # Both heads start at zero, so an untrained forecast is the starting line whatever the window
    # holds and every equality below would pass vacuously; random heads make the inputs matter.
    with torch.no_grad():
        torch.nn.init.normal_(model.head.weight, std=0.5)
        if model.aux_head is not None:
            torch.nn.init.normal_(model.aux_head.weight, std=0.5)
    return model


def _history(device, batch=3, steps=320):
    torch.manual_seed(1)
    return torch.randn(batch, steps, 4, device=device)


def _block(device, batch=3):
    torch.manual_seed(2)
    return torch.randn(batch, 96, 4, device=device)


def test_left_padding_is_the_short_history_itself(device):
    model = _model(device).eval()
    x = _history(device)
    nan = torch.full((3, 200, 4), float("nan"), device=device)
    with torch.no_grad():
        short = model(x[:, -120:])
        padded = model(torch.cat([nan, x[:, -120:]], dim=1))
        # What a leak would look like: the same 200 steps as zero-valued readings instead of padding.
        leaked = model(torch.cat([torch.zeros_like(nan), x[:, -120:]], dim=1))
    # The backbone runs in bf16 on the GPU, and a padded batch and a short one reach the matrix
    # kernels in different shapes, so they agree to bf16 rounding (about 0.4 per cent) there.
    scale = short.abs().max()
    assert (padded - short).abs().max() < 0.02 * scale
    assert (leaked - short).abs().max() > 5 * (padded - short).abs().max()


def test_missing_glucose_inside_the_history_stays_a_step(device):
    model = _model(device).eval()
    x = _history(device)
    gap = x.clone()
    gap[:, 100:130, 0] = float("nan")
    with torch.no_grad():
        full, holed = model(x), model(gap)
    assert torch.isfinite(holed).all()
    assert not torch.allclose(full, holed)


@pytest.mark.parametrize("layout", ["known_future_first", "step_order"])
def test_gap_mode_never_reads_glucose_inside_the_hidden_span(device, layout):
    model = _model(device, decode_layout=layout, **GAP).eval()
    x = _history(device)
    block = _block(device)
    leaked = block.clone()
    leaked[:, :48, 0] = 999.0
    with torch.no_grad():
        a, b = model(x, block), model(x, leaked)
    torch.testing.assert_close(a, b)
    after = block.clone()
    after[:, 60:, 0] += 2.0
    with torch.no_grad():
        assert not torch.allclose(a, model(x, after))


def test_announced_carbs_are_read_and_a_past_only_model_refuses_a_block(device):
    model = _model(device, announced_channels=(3,)).eval()
    x = _history(device)
    block = torch.full((3, 24, 4), float("nan"), device=device)
    meal = block.clone()
    meal[:, 3, 3] = 2.0
    with torch.no_grad():
        assert not torch.allclose(model(x, block), model(x, meal))
    with pytest.raises(ValueError, match="past-only"):
        _model(device)(x, block)


# ---- the decode layout (C63): the known future reaches the forecast --------------------------

def _forecast_states(model, x, block):
    """The recurrence's output at the forecast positions, before the decoder."""
    with torch.no_grad():
        h, x, _, _, _ = model.encode(x, block)
    start = x.size(1) + model.known_steps
    return h[:, start:start + model.prediction_horizon]


def test_the_recurrence_reads_the_glucose_after_the_gap_before_the_steps_it_fills(device):
    """The C60 fault in one assertion: in step order the forecast positions come before the
    post-gap glucose, so the recurrence cannot carry it to them; known-future-first can."""
    x, block = _history(device), _block(device)
    later = block.clone()
    later[:, 60:, 0] += 2.0  # readings after the first post-gap one, so the starting line is unchanged
    new = _model(device, **GAP).eval()
    assert not torch.allclose(_forecast_states(new, x, block), _forecast_states(new, x, later))
    old = _model(device, decode_layout="step_order", **GAP).eval()
    torch.testing.assert_close(_forecast_states(old, x, block), _forecast_states(old, x, later))


class _Silent(torch.nn.Module):
    """Replaces the final norm: the recurrence then hands the decoder nothing at all."""

    def forward(self, h):
        return torch.zeros_like(h)


@pytest.mark.parametrize("what", ["post-gap glucose", "announced carbs"])
def test_the_decoder_reads_the_raw_block_without_the_recurrence(device, what):
    """With the recurrence's output silenced, post-gap glucose and an announced meal still move the
    forecast, through skip_project alone; with skip_project zeroed as well, nothing does."""
    x = _history(device)
    if what == "post-gap glucose":
        model = _model(device, **GAP).eval()
        block = _block(device)
        other = block.clone()
        other[:, 60:, 0] += 2.0
    else:
        model = _model(device, announced_channels=(3,)).eval()
        block = torch.full((3, 24, 4), float("nan"), device=device)
        other = block.clone()
        other[:, 3, 3] = 2.0
    model.final_norm = _Silent()
    with torch.no_grad():
        assert not torch.allclose(model(x, block), model(x, other))
        model.skip_project.weight.zero_()
        torch.testing.assert_close(model(x, block), model(x, other))


def test_a_gap_forecast_starts_on_the_line_to_the_first_reading_after_the_gap(device):
    model = _model(device, **GAP).eval()
    with torch.no_grad():
        model.head.weight.zero_()  # an untrained head: the forecast is the starting line itself
    x = _history(device)
    x[:, -3:, 0] = float("nan")  # the last reading is 3 steps before the history's end
    block = torch.full((3, 96, 4), float("nan"), device=device)
    block[:, 53, 0] = 1.5  # the first reading after the gap: post-gap step 5
    block[:, 70, 0] = -4.0  # a later one, which the line ignores
    with torch.no_grad():
        line = model(x, block)
        flat = model(x, torch.full_like(block, float("nan")))
    before = x[:, -4, 0].unsqueeze(1)
    t = torch.arange(1, 25, device=device, dtype=x.dtype).unsqueeze(0)
    expected = before + (1.5 - before) * (t + 3) / (48 + 5 + 1 + 3)
    torch.testing.assert_close(line, expected)
    torch.testing.assert_close(flat, before.expand(-1, 24))
    old = _model(device, decode_layout="step_order", **GAP).eval()
    with torch.no_grad():
        old.head.weight.zero_()
        torch.testing.assert_close(old(x, block), before.expand(-1, 24))


def test_a_window_with_nothing_after_the_gap_leaves_those_steps_out(device):
    """A forecast-mode window does not read the steps after the gap at all, a gap window does,
    and a batch that mixes the two gives each window its own forecast."""
    model = _model(device, **GAP).eval()
    x = _history(device, batch=2)
    gap = _block(device, batch=2)
    forward = torch.full_like(gap, float("nan"))
    with torch.no_grad():
        f0, g0 = model(x, forward), model(x, gap)
        model.block_kind.weight[1] += 1.0  # changes every token after the gap and nothing else
        f1, g1 = model(x, forward), model(x, gap)
        mixed = model(x, torch.stack([forward[0], gap[1]]))
    torch.testing.assert_close(f0, f1)
    assert not torch.allclose(g0, g1)
    scale = g1.abs().max()
    assert (mixed[0] - f1[0]).abs().max() < 0.02 * scale and (mixed[1] - g1[1]).abs().max() < 0.02 * scale


def test_a_step_order_checkpoint_warm_starts_the_current_layout(device):
    old = _model(device, decode_layout="step_order", **GAP)
    new = _model(device, **GAP)
    own = {k: v.clone() for k, v in new.state_dict().items() if k.startswith("skip_project.")}
    assert own and old.skip_project is None
    parent = old.state_dict()
    new.load_initial_state(parent, "parent")
    after = new.state_dict()
    for key, value in parent.items():
        torch.testing.assert_close(after[key], value)
    for key, value in own.items():
        torch.testing.assert_close(after[key], value)
    with pytest.raises(KeyError, match="part of skip_project"):
        new.load_initial_state({**parent, "skip_project.weight": own["skip_project.weight"]}, "half")
    # The other way round is refused: a step-order model cannot hold the current layout's weights.
    with pytest.raises(RuntimeError, match="skip_project"):
        old.load_state_dict(new.state_dict(), strict=True)


# ---- the auxiliary loss and the rebuild ------------------------------------------------------

def test_auxiliary_loss_is_training_only_and_reads_the_history_alone(device):
    model = _model(device, aux_weight=0.5, glucose_mean=150.0, glucose_scale=55.0, **GAP)
    x = _history(device)
    block = _block(device)
    model.eval()
    model(x, block)
    assert model.get_auxiliary_loss() is None
    model.train()
    torch.manual_seed(5)
    model(x, block)
    first = model.get_auxiliary_loss()
    assert first is not None and torch.isfinite(first) and first.requires_grad
    assert model.get_auxiliary_loss() is None  # handed over once
    other = block.clone()
    other[:, 48:, 0] += 3.0
    torch.manual_seed(5)
    model(x, other)
    torch.testing.assert_close(model.get_auxiliary_loss(), first)


# ---- C64: physical inputs and outputs --------------------------------------------------------

#: R.60's MinMax ranges on the MetaboNet training frame (its scalers.json): glucose, basal, bolus, carbs.
R60_RANGES = ((1.8018, 500.9004), (0.0, 384.0), (0.0, 50.0), (0.0, 855.0))
R60_AFFINE = [(hi - lo, lo) for lo, hi in R60_RANGES]


def _scaled(raw: torch.Tensor) -> torch.Tensor:
    """mg/dL, U/h, U and g -> R.60's MinMax fractions, channel by channel."""
    return torch.stack([(raw[..., k] - lo) / (hi - lo) for k, (lo, hi) in enumerate(R60_RANGES)], dim=-1)


def _physical(device, **overrides):
    lo, hi = R60_RANGES[0]
    return _model(device, input_norm="physical", channel_affine=R60_AFFINE, glucose_mean=lo, glucose_scale=hi - lo,
                  **overrides)


def test_physical_inputs_carry_a_meal_and_a_bolus_in_their_own_units(device):
    """The R.62 fault in one assertion: on MinMax inputs a 60 g meal reaches the projections as 0.07,
    on physical ones as 2, and the extremes that set the MinMax range are clipped instead."""
    nan = float("nan")
    raw = torch.tensor([[140.0, 1.0, 5.0, 60.0], [90.0, 2.0, 0.0, 0.0], [nan, 1.0, 50.0, 855.0]], device=device)
    features = _physical(device).featurise(_scaled(raw)).cpu()
    expected = torch.tensor([[0.0, 0.0, 1.0, 2.0], [-1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 10.0, 10.0]])
    torch.testing.assert_close(features[:, :4], expected, atol=1e-4, rtol=1e-4)
    assert features[:, 4:].tolist() == [[1, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 1]]
    legacy = _model(device).featurise(_scaled(raw)).cpu()
    assert legacy[0, 3].item() == pytest.approx(60 / 855, rel=1e-4) and legacy[0, 1].item() < 0.003


def test_a_physical_head_speaks_fifty_mgdl_per_unit(device):
    lo, hi = R60_RANGES[0]
    model = _physical(device).eval()
    x = _scaled(torch.tensor([140.0, 1.0, 0.0, 0.0], device=device).expand(2, 320, 4).clone())
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.fill_(1.0)
        out = model(x)
    torch.testing.assert_close((out - x[:, -1:, 0]) * (hi - lo), torch.full_like(out, 50.0), rtol=1e-3, atol=1e-2)
    assert _model(device).output_gain == 1.0


def test_the_physical_auxiliary_head_is_read_in_the_same_units(device):
    model = _model(device, aux_weight=0.5, glucose_mean=40.0, glucose_scale=360.0, input_norm="physical",
                   channel_affine=[(360.0, 40.0), (10.0, 0.0), (50.0, 0.0), (300.0, 0.0)])
    with torch.no_grad():
        model.aux_head.weight.zero_()
        model.aux_head.bias.fill_(0.2)  # 0.2 x 50 mg/dL: every auxiliary forecast is 10 mg/dL too high
    glucose = torch.full((2, 50), (100.0 - 40.0) / 360.0, device=device)
    h = torch.zeros(2, 50, model.d_model, device=device)
    anchor, has_anchor = torch.full_like(glucose, glucose[0, 0].item()), torch.ones_like(glucose, dtype=torch.bool)
    loss = model.dense_history_loss(h, glucose, anchor, has_anchor)
    torch.testing.assert_close(loss, torch.tensor(0.1, device=device), rtol=1e-4, atol=1e-5)


def test_either_norm_loads_the_others_weights_and_saves_no_buffer(device):
    physical, legacy = _physical(device), _model(device)
    assert not [k for k in physical.state_dict() if k.startswith("feature_")]
    legacy.load_state_dict(physical.state_dict(), strict=True)
    physical.load_state_dict(legacy.state_dict(), strict=True)
    with pytest.raises(ValueError, match="channel_affine"):
        _model(device, input_norm="physical", glucose_mean=1.8, glucose_scale=499.1)
    with pytest.raises(ValueError, match="glucose affine"):
        _model(device, input_norm="physical", channel_affine=R60_AFFINE)
    with pytest.raises(ValueError, match="input_norm"):
        _model(device, input_norm="zscore")


# ---- the scorers and the shared loop around the family (no GPU needed) ---------------------

class _Recorder(torch.nn.Module):
    """Stands in for a gap-mode forecaster: records the history it is handed, forecasts zeros."""

    prediction_horizon = 24

    def __init__(self, pads: bool) -> None:
        super().__init__()
        self.pads_short_history = pads
        self.seen = None

    def forward(self, x, block):
        self.seen = x.clone()
        return torch.zeros(x.size(0), self.prediction_horizon)


# ---- C61: the masked-history path ------------------------------------------------------------

def test_pad_before_first_glucose_extends_padding_to_the_first_reading():
    from krontime.glucose import history_padding, pad_before_first_glucose

    nan = float("nan")
    x = torch.full((3, 8, 4), nan)
    # Row 0: covariates from step 1, glucose from step 4; a crop already pads steps 0 and 1.
    x[0, 1:, 1:] = 0.0
    x[0, 4:, 0] = 100.0
    # Row 1: padding before step 2 and the first kept step is a reading.
    x[1, 2:, :] = 1.0
    # Row 2: a reading on step 0, covariates throughout.
    x[2, :, :] = 2.0
    pad = history_padding(x)
    pad[0, :2] = True
    out = pad_before_first_glucose(x, pad)
    assert out[0].tolist() == [True, True, True, True, False, False, False, False]
    assert torch.equal(out[1], pad[1]) and torch.equal(out[2], pad[2])
    assert (out | pad).equal(out), "padding already in place is kept"


#: A universal model (Anton, 2026-10-23): a 10-hour forecast beside the 4-hour gap, in one 120-step block.
UNIVERSAL = dict(prediction_horizon=120, post_gap_steps=72, gap_steps=48, gap_covariates=(1, 2, 3),
                 announced_channels=(3,))


def _forecast_block(device, batch=3):
    """A forecast window's block: the announced carbs over the whole horizon and nothing else."""
    torch.manual_seed(3)
    block = torch.full((batch, 120, 4), float("nan"), device=device)
    block[:, :, 3] = torch.rand(batch, 120, device=device)
    return block


def _two_sided_block(device, batch=3):
    """A two-sided window's block: in-gap basal, bolus and carbs, and glucose after the 4-hour span."""
    block = _forecast_block(device, batch)
    block[:, :, 1:3] = torch.rand(batch, 120, 2, device=device)
    block[:, 48:, 0] = torch.rand(batch, 72, device=device)
    return block


def test_a_universal_model_builds_where_a_short_block_cannot(device):
    import pytest

    from krontime.glucose import KronLSTMForecaster

    model = _model(device, **UNIVERSAL)
    assert model.universal and model.block_steps == 120 and model.known_steps == 72
    assert not _model(device, **GAP).universal
    with pytest.raises(ValueError, match="hidden span"):
        KronLSTMForecaster(n_time_steps=320, prediction_horizon=120, d_model=64, n_heads=2, n_layers=2, ff_units=128,
                           decoder_layers=1, post_gap_steps=48, gap_steps=48)  # 96 steps cannot hold 120


def test_a_universal_forecast_window_decodes_as_the_forecast_only_model(device):
    """The warm start from R.66: a forecast window of a universal model, carrying only its announced
    carbs, is read exactly as a forecast-only model of the same horizon reads it, so the 10-hour
    forecasts it learned carry over unchanged."""
    only = _model(device, prediction_horizon=120, announced_channels=(3,)).eval()
    universal = _model(device, **UNIVERSAL).eval()
    universal.load_state_dict(only.state_dict())
    x, block = _history(device), _forecast_block(device)
    with torch.no_grad():
        a, b = only(x, block), universal(x, block)
    assert a.shape == b.shape == (3, 120)
    assert torch.equal(a, b)


def test_a_universal_two_sided_window_fills_the_span_from_what_follows_it(device):
    model = _model(device, **UNIVERSAL).eval()
    x, block = _history(device), _two_sided_block(device)
    moved = block.clone()
    moved[:, 48:, 0] += 1.0
    with torch.no_grad():
        base, after = model(x, block), model(x, moved)
        values = model.decode_block(block, 3, x)
    assert base.shape == (3, 120) and torch.isfinite(base).all()
    assert (after[:, :48] - base[:, :48]).abs().max() > 1e-3
    assert model._two_sided(values).all()
    assert not model._two_sided(model.decode_block(_forecast_block(device), 3, x)).any()


def test_a_universal_model_withholds_announcements_from_forecast_windows_only(device):
    """Announcement dropout tells the modes apart by what the block carries, since a universal block is
    all horizon and "nothing after the horizon" no longer marks a forecast window."""
    model = _model(device, future_dropout=1.0, **UNIVERSAL).train()
    x = _history(device)
    forecast = model.decode_block(_forecast_block(device), 3, x)
    two_sided = model.decode_block(_two_sided_block(device), 3, x)
    assert torch.isnan(forecast[:, :, 3]).all()
    assert torch.isfinite(two_sided[:, :, 3]).all()
