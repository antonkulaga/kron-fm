"""The workout head: age from heart rate against walking and running speed, as a watch records a walk or a run.

Trained on the University of Malaga treadmill tests (PhysioNet; 992 graded tests to exhaustion of 857 people aged
10-63, heart rate and speed every 1-2 s). The features are those a free walk or run on a watch also gives:

* ``hr_at_6``, ``hr_at_8``, ``hr_at_10``, ``hr_at_12``: the heart rate held at 6, 8, 10 and 12 km/h (median of the
  samples after 30 s at that speed, within 0.75 km/h; NaN where the session never held it);
* ``hr_speed_slope``, ``hr_at_10_fit``: the slope and the 10 km/h value of a line of heart rate (30 s later) on speed
  over every moving sample (4-16 km/h);
* ``hr_peak``: the 98th percentile of heart rate over moving samples (robust to spikes), and ``hr_rest_start``:
  the heart rate of the first moving minute.

A person's features are the medians over their sessions. A treadmill test runs to exhaustion and a watch session
rarely does, so ``hr_peak`` is the feature least likely to carry over; the head is also reported without it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

SPEEDS = (6, 8, 10, 12)
FEATURES = [f"hr_at_{s}" for s in SPEEDS] + ["hr_speed_slope", "hr_at_10_fit", "hr_peak", "hr_rest_start"]
NO_PEAK = [f for f in FEATURES if f != "hr_peak"]


def session_features(t_s: np.ndarray, speed_kmh: np.ndarray, hr: np.ndarray) -> dict[str, float] | None:
    """Features of one walk, run or test: times (s), speed (km/h) and heart rate (bpm), any rate."""
    t, v, h = (np.asarray(a, float) for a in (t_s, speed_kmh, hr))
    ok = np.isfinite(t) & np.isfinite(v) & np.isfinite(h) & (h > 30) & (h < 230)
    t, v, h = t[ok], v[ok], h[ok]
    if len(t) < 60 or t[-1] - t[0] < 120:
        return None
    grid = np.arange(t[0], t[-1], 1.0)
    v = np.interp(grid, t, v)
    h = np.interp(grid, t, h)
    moving = (v >= 4) & (v <= 16)
    if moving.sum() < 120:
        return None
    out: dict[str, float] = {}
    for s in SPEEDS:
        near = np.abs(v - s) <= 0.75
        held = np.zeros(len(v), bool)  # 30 s or more inside the band, counted from entering it
        run = 0
        for i, n in enumerate(near):
            run = run + 1 if n else 0
            held[i] = run > 30
        out[f"hr_at_{s}"] = float(np.median(h[held])) if held.sum() >= 15 else float("nan")
    lag = 30
    if len(h) > lag + 60:
        vv, hh = v[:-lag], h[lag:]
        m = (vv >= 4) & (vv <= 16)
        if m.sum() >= 60 and np.ptp(vv[m]) >= 2:
            b, a = np.polyfit(vv[m], hh[m], 1)
            out["hr_speed_slope"], out["hr_at_10_fit"] = float(b), float(a + 10 * b)
    out.setdefault("hr_speed_slope", float("nan"))
    out.setdefault("hr_at_10_fit", float("nan"))
    out["hr_peak"] = float(np.percentile(h[moving], 98))
    first = np.flatnonzero(moving)[:60]
    out["hr_rest_start"] = float(np.median(h[first]))
    return out


def person(sessions: list[dict[str, float]]) -> dict[str, float] | None:
    """The median of a person's session features (a feature's NaNs ignored); None with no session."""
    rows = [s for s in sessions if s]
    if not rows:
        return None
    return {k: float(np.nanmedian([r.get(k, np.nan) for r in rows])) if any(np.isfinite(r.get(k, np.nan)) for r in rows)
            else float("nan") for k in FEATURES}


def treadmill(folder: Path) -> pl.DataFrame:
    """One row per Malaga test: ``subject`` (person), ``test``, ``age``, ``sex``, ``bmi`` and :data:`FEATURES`."""
    folder = Path(folder)
    root = next(p for p in folder.iterdir() if p.is_dir() and (p / "subject-info.csv").is_file())
    info = pl.read_csv(root / "subject-info.csv", infer_schema_length=None)
    meas = pl.read_csv(root / "test_measure.csv", infer_schema_length=None)
    rows = []
    for (test,), g in meas.group_by("ID_test"):
        f = session_features(g["time"].to_numpy(), g["Speed"].to_numpy(), g["HR"].to_numpy())
        if f is None:
            continue
        r = info.filter(pl.col("ID_test") == test)
        if r.is_empty():
            continue
        r = r.row(0, named=True)
        bmi = r["Weight"] / (r["Height"] / 100) ** 2 if r["Height"] else float("nan")
        rows.append({"subject": f"malaga/{r['ID']}", "test": test, "age": float(r["Age"]), "sex": float(r["Sex"]),
                     "bmi": float(bmi), **f})
    return pl.DataFrame(rows)


def samsung(export, min_moving_s: int = 120) -> tuple[dict[str, float] | None, int]:
    """The person's features from a Samsung Health export's walks and runs (live_data: speed in m/s, heart rate),
    and how many sessions gave features."""
    stream = "com.samsung.shealth.exercise"
    ex = export.table(stream)
    sessions = []
    for r in ex.iter_rows(named=True):
        if r.get("exercise_type") not in ("1001", "1002"):
            continue
        b = export.blob(stream, r.get("live_data"))
        if not b or len(b) < min_moving_s:
            continue
        t = np.array([p.get("start_time", np.nan) for p in b], float) / 1000.0
        v = np.array([p.get("speed", np.nan) for p in b], float) * 3.6
        h = np.array([p.get("heart_rate", np.nan) for p in b], float)
        f = session_features(t - np.nanmin(t), v, h)
        if f is not None:
            sessions.append(f)
    return person(sessions), len(sessions)
