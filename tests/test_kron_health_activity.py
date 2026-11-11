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
