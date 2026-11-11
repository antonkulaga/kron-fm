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


def probe_windows(run: ProbeRun, seconds: float = 30.0, target_fs: float = FS
                  ) -> tuple[np.ndarray, np.ndarray, dict]:
    """The run's quality-gated, robust-scaled windows ``(N, L)``, their starts (s), and the gate's tally."""
    if len(run.x) == 0:
        return np.zeros((0, int(seconds * target_fs)), np.float32), np.zeros(0), {"candidates": 0, "accepted": 0}
    y, ok = prepare_signal(run.x, run.fs, target_fs)
    size = int(round(seconds * target_fs))
    candidates = sum(1 for s in range(0, len(y) - size + 1, size) if ok[s:s + size].all())
    w, starts, quals = cut_windows(y, ok, target_fs, seconds, seconds)
    tally = {"candidates": candidates, "accepted": int(len(w)),
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
        rec = read_hr_raw(Path(r))
        w, starts, tally = probe_windows(rec, seconds)
        per_run.append({"run": rec.name, "seconds_on_wrist": rec.seconds_on_wrist, "ac_sign": rec.sign, **tally})
        if len(w):
            all_windows.append(lowpass(w, FS, cutoff) if cutoff else w)
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
