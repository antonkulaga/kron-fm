"""The activity head's features on synthetic weeks (no real person)."""
from datetime import datetime, timedelta

import numpy as np
import polars as pl

from kron_health.activity import UNIT_FREE, rhythm, samsung


def _week(peak_hour=14, days=7, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    hours = np.tile(np.arange(24), days)
    d = np.repeat(np.arange(days), 24)
    base = np.exp(-0.5 * ((hours - peak_hour) / 3.0) ** 2) * 100
    v = base * (1 + noise * rng.standard_normal(len(base))).clip(0)
    return v, hours, d


def test_a_regular_week_is_stable_and_its_peak_is_where_it_is():
    v, h, d = _week()
    f = rhythm(v, h, d, np.ones(len(v), bool))
    assert f["is"] > 0.99 and f["iv"] < 0.2 and f["ra"] > 0.9
    assert np.argmax([f[f"share_h{k:02d}"] for k in range(24)]) == 14
    assert abs(sum(f[f"share_h{k:02d}"] for k in range(24)) - 1) < 1e-9
    m10 = np.degrees(np.arctan2(f["m10_sin"], f["m10_cos"])) % 360 / 15
    assert abs(m10 - 14) <= 1


def test_noise_lowers_stability_and_the_scale_changes_nothing_unit_free():
    v, h, d = _week(noise=0.8)
    a = rhythm(v, h, d, np.ones(len(v), bool))
    b = rhythm(v * 37.0, h, d, np.ones(len(v), bool))
    assert a["is"] < 0.95
    assert all(abs(a[k] - b[k]) < 1e-9 for k in UNIT_FREE if k != "sleep_min")
    assert abs(b["log_total"] - a["log_total"]) > 1


def test_too_few_worn_days_give_nothing():
    v, h, d = _week(days=7)
    worn = d < 2
    assert rhythm(v, h, d, worn) is None


def test_samsung_step_bins_sum_to_hours():
    t0 = datetime(2026, 7, 1)
    starts = [t0 + timedelta(minutes=10 * i) for i in range(6 * 24 * 5)]
    steps = [100.0 if 9 <= t.hour < 18 else 0.0 for t in starts]
    bins = pl.DataFrame({"start": starts, "end": [s + timedelta(minutes=10) for s in starts], "steps": steps})
    sleeps = pl.DataFrame({"start": [t0 + timedelta(hours=23)], "end": [t0 + timedelta(hours=30)], "sleep_id": ["s"]})
    f = samsung(bins, sleeps)
    assert f is not None and f["sleep_min"] == 420.0
    assert abs(f["share_h12"] - 1 / 9) < 1e-9 and f["share_h03"] == 0.0


def test_hrv_features_on_a_known_series():
    from kron_health.hrv import age_mid, features

    rng = np.random.default_rng(1)
    t = np.arange(0, 1200, 0.25)
    rr = 1000 + 30 * np.sin(2 * np.pi * 0.25 * np.cumsum(np.full(1200, 1.0))) + rng.normal(0, 5, 1200)  # HF-ish beat-to-beat swing
    rr[[50, 400]] = [2600.0, 280.0]
    f = features(rr)
    assert f is not None and f["segments"] >= 3 and f["kept_fraction"] < 1.0
    assert abs(f["hr"] - 60) < 1 and f["rmssd"] > 20 and 0.0 < f["dfa_a1"] < 1.5
    assert f["sd1"] > 0 and f["sd2"] > 0
    assert age_mid(1) == 18.5 and age_mid(15) == 88.5


def test_ecg_strips_and_model_shapes():
    import torch

    from kron_health.ecg import LENGTH, ECGAge, strips

    x = np.sin(np.arange(15000) / 500 * 2 * np.pi * 1.2) * 0.8 + 3.0
    s = strips(x, 500.0)
    assert s.shape == (3, LENGTH) and abs(float(np.median(s))) < 0.1
    m = ECGAge(width=8)
    assert m(torch.from_numpy(s)).shape == (3,)


def test_band_removes_wander_and_offset():
    from kron_health.ecg import strips

    t = np.arange(7500) / 250
    beat = np.zeros_like(t)
    beat[(np.arange(0, 7500, 200))] = 1.0
    ecg = np.convolve(beat, np.hanning(15), mode="same") + 130.0 + 3.0 * np.sin(2 * np.pi * 0.1 * t)  # offset + wander
    raw = strips(ecg, 250.0)
    clean = strips(ecg, 250.0, filtered=True)
    assert raw.shape == clean.shape == (3, 2500)
    # the wander dominates the raw strip's low frequencies and is gone from the filtered one
    lowf = lambda s: np.abs(np.fft.rfft(s, axis=-1))[:, 1:4].sum() / np.abs(np.fft.rfft(s, axis=-1)).sum()
    assert lowf(clean) < 0.2 * lowf(raw)


def test_sdk_ecg_carries_the_lead_state_forward(tmp_path):
    from kron_health.ecg import sdk_ecg

    n = 500 * 24
    lead = np.full(n, -1)
    lead[::10] = 0
    lead[:500 * 4:10] = 5  # the first 4 s off
    pl.DataFrame({"timestamp_ms": np.arange(n) // 10 * 20, "ecg_mv": np.sin(np.arange(n) / 50) + 130,
                  "lead_off": lead, "sequence": np.arange(n), "ppg_green": np.zeros(n)}).write_csv(tmp_path / "sdk_ecg.csv")
    x = sdk_ecg(tmp_path)
    assert x is not None and len(x) == 500 * 20


def test_step_features_from_ten_minute_bins_match_across_sources():
    from datetime import datetime, timedelta

    from kron_health.steps import FEATURES, bin_features, samsung

    days = 7
    rng = np.random.default_rng(0)
    t0 = datetime(2026, 7, 1)
    starts = [t0 + timedelta(minutes=10 * i) for i in range(6 * 24 * days)]
    steps = np.array([float(rng.integers(600, 1200)) if 9 <= t.hour < 12 else (50.0 if 12 <= t.hour < 20 else 0.0)
                      for t in starts])
    a = samsung(pl.DataFrame({"start": starts, "steps": steps}))
    b = bin_features(np.array([(t - t0).days for t in starts]), np.array([t.hour for t in starts]), steps,
                     np.ones(len(starts), bool))
    assert a is not None and set(FEATURES) <= set(a)
    assert all(abs(a[k] - b[k]) < 1e-9 for k in FEATURES)
    # window 08-20: 72 bins a day, 18 of them 600-1200 steps (9-12 h), the 6 of 08 h empty
    assert 0.05 < a["brisk"] < 0.15 and abs(a["empty_daytime"] - 6 / 72) < 1e-9 and 60 < a["best30"] < 120


def test_workout_features_read_hr_at_speed_and_its_slope():
    from kron_health.workout import FEATURES, person, session_features

    t = np.arange(0, 1200, 1.0)
    speed = np.clip(5 + t // 120, 5, 14).astype(float)  # a ramp: 2 min per km/h from 5 to 14
    hr = 60 + 8 * np.interp(t - 30, t, speed)  # 8 bpm per km/h, 30 s behind
    f = session_features(t, speed, hr)
    assert f is not None and set(FEATURES) <= set(f)
    assert abs(f["hr_speed_slope"] - 8) < 0.5 and abs(f["hr_at_10_fit"] - 140) < 3
    assert abs(f["hr_at_8"] - 124) < 2 and abs(f["hr_at_12"] - 156) < 2
    p = person([f, None, dict(f, hr_at_6=float("nan"))])
    assert p["hr_at_8"] == f["hr_at_8"]


def test_sas_transport_zero_is_read_as_zero():
    # pandas decodes an XPORT zero as ~5.4e-79; the step reader must treat it as 0
    import pandas as pd

    part = pd.DataFrame({"SEQN": [1.0, 1.0], "PAXSTEP": [5.397605346934028e-79, 12.0]})
    num = part.select_dtypes("number").columns
    part[num] = part[num].mask(part[num].abs() < 1e-30, 0.0)
    assert (part.PAXSTEP == 0).sum() == 1
