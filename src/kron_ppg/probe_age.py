"""The wrist age clock on raw PPG recorded by the pulse-exploration watch probe (``hr_raw`` at 25 Hz).

The probe app (antonkulaga/pulse-exploration, ``watch/probe``) records ``com.samsung.sensor.hr_raw`` through
Android's SensorManager on a Galaxy Watch: one ``hr_raw.csv`` per run folder, ``event_ns`` (the sensor clock)
and sixteen integer channels at about 25 Hz. The channel meanings are the probe's own reading of one recording,
not Samsung's documentation (``pulse_exploration/watch_raw.py``):

* ``v0`` status, 2 on the wrist;
* ``v1``, ``v4`` raw light level of two PPG channels: it drops as blood volume rises (systole down);
* ``v5`` the pulsatile AC channel, signed 22-bit with an offset of 2**21; its sign is chosen per run so that it
  follows the inverted raw light level, i.e. systole up, the polarity every kron-ppg reader gives.

From there the run goes through the pipeline the training sets went through (:func:`kron_ppg.signal
.prepare_signal`, :func:`~kron_ppg.signal.cut_windows`: gaps filled, band-pass, 25 Hz, 30 s windows,
robust-scaled, the quality gate), and the deployable wrist clock (:mod:`kron_ppg.wrist_age`, a run folder with
``head.pt``) reads them in the band its head was fitted in. The person's age is the head over the mean window
embedding of every accepted window; the report says how many windows passed, against the readings-per-person
curve (16 readings of 15 s gave 7.35 y on the held-out cohort, one reading 10.3 y).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch

from kron_ppg.signal import bandpass, cut_windows, prepare_signal

FS = 25.0
OFFSET = 2 ** 21
ON_WRIST = 2


@dataclass
class ProbeRun:
    """One probe run's pulse on a uniform grid: ``x`` (systole up), ``valid`` (on the wrist, no gap)."""

    name: str
    x: np.ndarray
    valid: np.ndarray
    fs: float
    seconds_on_wrist: float
    sign: float  # +1 or -1: the sign put on the AC channel


def read_hr_raw(run: Path, fs: float = FS) -> ProbeRun:
    """The run's ``hr_raw.csv`` as one pulse signal on a ``fs`` grid of its own sensor clock."""
    run = Path(run)
    df = pl.read_csv(run / "hr_raw.csv", infer_schema_length=None)
    t = (df["event_ns"].to_numpy() - df["event_ns"][0]) / 1e9
    order = np.argsort(t, kind="stable")
    t = t[order]
    status = df["v0"].to_numpy()[order]
    light = df["v1"].to_numpy().astype(float)[order]
    ac = df["v5"].to_numpy().astype(float)[order] - OFFSET
    on = status == ON_WRIST
    grid = np.arange(0.0, t[-1], 1 / fs)
    # a grid point is valid when an on-wrist sample lies within one period of it
    idx = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
    near = np.where(np.abs(t[idx] - grid) < np.abs(t[idx - 1] - grid), idx, idx - 1)
    valid = on[near] & (np.abs(t[near] - grid) <= 1 / fs)
    if valid.sum() < 2 * fs:
        return ProbeRun(run.name, np.zeros(0), np.zeros(0, bool), fs, float(valid.sum() / fs), 1.0)
    x_ac = np.interp(grid, t, ac)
    x_light = np.interp(grid, t, light)
    ok = valid
    a = bandpass(-(x_light - np.median(x_light[ok])) * ok, fs)  # inverted light level: systole up
    b = bandpass((x_ac - np.median(x_ac[ok])) * ok, fs)
    sign = 1.0 if np.corrcoef(a[ok], b[ok])[0, 1] >= 0 else -1.0
    x = sign * x_ac
    x[~valid] = np.nan
    return ProbeRun(run.name, x, valid, fs, float(valid.sum() / fs), sign)


#: The probe's optical sources, best first: the SDK's on-demand 100 Hz green, its continuous 25 Hz green, the
#: Android hr_raw AC channel, and the green channel the SDK records beside its ECG.
SOURCES = ("sdk_ppg100", "sdk_ppg", "hr_raw", "sdk_ecg")


def _spread_duplicates(ts: np.ndarray) -> np.ndarray:
    """Per-sample times where the SDK stamps several samples with the same millisecond (as the probe does)."""
    if len(ts) < 3 or np.all(np.diff(ts) > 0):
        return ts
    idx = np.arange(len(ts))
    first = np.r_[True, np.diff(ts) != 0]
    ai, at = idx[first], ts[first]
    if len(ai) < 2:
        return ts
    step = float(np.median(np.diff(at) / np.diff(ai)))
    out = np.interp(idx, ai, at)
    tail = idx > ai[-1]
    out[tail] = at[-1] + (idx[tail] - ai[-1]) * step
    return out


def upright_sign(x: np.ndarray, fs: float) -> float:
    """+1 when systole points up already: the derivative's upper 2 % tail longer than its lower (fast rise)."""
    ok = np.isfinite(x)
    if ok.sum() < 4 * fs:
        return 1.0
    y = bandpass(np.where(ok, x - np.nanmedian(x), 0.0), fs)
    d = np.diff(y[ok])
    return 1.0 if np.percentile(d, 98) >= -np.percentile(d, 2) else -1.0


def read_sdk(run: Path, alias: str) -> ProbeRun | None:
    """An SDK green channel (``sdk_ppg``, ``sdk_ppg100``: ``green`` with ``green_status``; ``sdk_ecg``: ``ppg_green``)
    on a uniform grid at its own rate, systole up. None when the file is missing or holds no usable samples."""
    path = Path(run) / f"{alias}.csv"
    if not path.is_file() or path.stat().st_size == 0:
        return None
    df = pl.read_csv(path, infer_schema_length=None)
    col = "ppg_green" if alias == "sdk_ecg" else "green"
    if df.is_empty() or col not in df.columns or "timestamp_ms" not in df.columns:
        return None
    t = _spread_duplicates(df["timestamp_ms"].cast(pl.Float64).to_numpy())
    t = (t - t[0]) / 1000.0
    g = df[col].cast(pl.Float64, strict=False).to_numpy()
    ok = np.isfinite(g)
    if "green_status" in df.columns:
        ok &= df["green_status"].cast(pl.Float64, strict=False).fill_null(0).to_numpy() == 0
    if ok.sum() < 10:
        return None
    dt = float(np.median(np.diff(t[ok])))
    if not np.isfinite(dt) or dt <= 0:
        return None
    fs = float(round(1.0 / dt))
    grid = np.arange(0.0, t[-1], 1 / fs)
    idx = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
    near = np.where(np.abs(t[idx] - grid) < np.abs(t[idx - 1] - grid), idx, idx - 1)
    valid = ok[near] & (np.abs(t[near] - grid) <= 1.5 / fs)
    x = np.interp(grid, t[ok], g[ok])
    if alias == "sdk_ecg":
        # the ECG recording's green may repeat each optical sample over several ECG rows: keep its own rate
        changes = np.flatnonzero(np.diff(g[ok]) != 0)
        if len(changes) > 10:
            step = float(np.median(np.diff(t[ok][changes])))
            if step > 1.5 * dt:
                fs = float(round(1.0 / step))
                grid = np.arange(0.0, t[-1], 1 / fs)
                x = np.interp(grid, t[ok], g[ok])
                idx = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
                valid = ok[idx]
        sign = upright_sign(np.where(valid, x, np.nan), fs)
    else:
        sign = -1.0  # raw green light drops at systole, as the probe's own analysis reads it
    x = sign * x
    x[~valid] = np.nan
    return ProbeRun(f"{Path(run).name}:{alias}", x, valid, fs, float(valid.sum() / fs), sign)


def _valid_windows(valid: np.ndarray, size: int) -> list[int]:
    """Starts of non-overlapping ``size``-sample windows inside each run of valid samples, from the run's start."""
    starts, i, n = [], 0, len(valid)
    while i < n:
        if not valid[i]:
            i += 1
            continue
        j = i
        while j < n and valid[j]:
            j += 1
        starts += list(range(i, j - size + 1, size))
        i = j
    return starts


def probe_windows(run: ProbeRun, seconds: float = 30.0, target_fs: float = FS
                  ) -> tuple[np.ndarray, np.ndarray, dict]:
    """The run's quality-gated, robust-scaled windows ``(N, L)``, their starts (s), and the gate's tally."""
    if len(run.x) == 0:
        return np.zeros((0, int(seconds * target_fs)), np.float32), np.zeros(0), {"candidates": 0, "accepted": 0}
    from kron_ppg.signal import MAX_SCALED_ABS, quality, robust_scale

    y, ok = prepare_signal(np.nan_to_num(run.x), run.fs, target_fs)
    if len(y) == 0:
        return np.zeros((0, int(seconds * target_fs)), np.float32), np.zeros(0), {"candidates": 0, "accepted": 0}
    # the run's own invalid samples (off the wrist, status != 0) on the target grid
    src = np.minimum(np.round(np.arange(len(y)) * run.fs / target_fs).astype(int), len(run.valid) - 1)
    ok = ok & run.valid[src]
    size = int(round(seconds * target_fs))
    cand = _valid_windows(ok, size)
    w, starts, quals = [], [], []
    for st in cand:
        seg = robust_scale(y[st:st + size])
        q = quality(seg, target_fs, max_abs=MAX_SCALED_ABS)
        if q.ok:
            w.append(seg.astype(np.float32))
            starts.append(st / target_fs)
            quals.append(q)
    w = np.stack(w) if w else np.zeros((0, size), np.float32)
    starts = np.asarray(starts)
    tally = {"candidates": len(cand), "accepted": int(len(w)),
             "median_hr_bpm": float(np.median([q.hr_bpm for q in quals])) if quals else None,
             "median_periodicity": float(np.median([q.periodicity for q in quals])) if quals else None}
    return w, starts, tally


def estimate(runs: list[Path], model_run: Path, device: str = "cuda", seconds: float = 30.0) -> dict:
    """The person's age from every accepted window of ``runs``, with the clock of ``model_run``."""
    from kron_ppg.train import load_encoder
    from kron_ppg.wrist_age import embed, lowpass

    head = torch.load(Path(model_run) / "head.pt", map_location="cpu", weights_only=True)
    metrics = json.loads((Path(model_run) / "metrics.json").read_text())
    cutoff = metrics.get("lowpass_hz")
    encoder = load_encoder(head["encoder"], device)
    per_run, all_windows = [], []
    for r in runs:
        best = None
        for alias in SOURCES:
            if alias == "hr_raw":
                rec = read_hr_raw(Path(r)) if (Path(r) / "hr_raw.csv").is_file() else None
            else:
                rec = read_sdk(Path(r), alias)
            if rec is None:
                continue
            w, starts, tally = probe_windows(rec, seconds)
            entry = {"run": Path(r).name, "source": alias, "fs": rec.fs, "seconds_valid": rec.seconds_on_wrist,
                     "sign": rec.sign, **tally, "used": False}
            per_run.append(entry)
            if best is None and len(w):
                best = (entry, w)
        if best is not None:  # one source per run, the best that passed: the same seconds are never counted twice
            best[0]["used"] = True
            all_windows.append(lowpass(best[1], FS, cutoff) if cutoff else best[1])
    out = {"model": str(model_run), "band_hz": [0.5, cutoff or 8.0], "runs": per_run,
           "windows": int(sum(len(w) for w in all_windows))}
    if not all_windows:
        out["age"] = None
        out["note"] = "no window passed the quality gate: record more still-wrist readings"
        return out
    w = np.concatenate(all_windows)

    class _Pool:
        sample_rate = FS

        def window(self, i):
            return w[i]

    emb = embed(encoder, _Pool(), np.arange(len(w)), device=device)
    weight, bias = head["weight"].numpy().astype(np.float64), float(head["bias"])
    per_window = emb.astype(np.float64) @ weight + bias
    out["age"] = float(emb.mean(0).astype(np.float64) @ weight + bias)
    out["per_window_age"] = {"median": float(np.median(per_window)), "p10": float(np.percentile(per_window, 10)),
                             "p90": float(np.percentile(per_window, 90))}
    readings_15s = len(w) * seconds / 15
    out["reference"] = ("held-out wrist cohort, 1,220 people: MAE 7.50 y (7.16-7.85) from all readings; "
                        "16 readings of 15 s 7.35 y, 8 readings 7.7 y, 1 reading 10.3 y")
    out["readings_15s_equivalent"] = readings_15s
    return out
