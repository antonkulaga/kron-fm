"""Raw PPG to the model's input: band-pass, resample, robust scaling, windows and a cheap quality gate.

Every source is brought to one form before it is stored in ``data/interim``:

1. **Band-pass 0.5-8 Hz** (zero-phase Butterworth): the band PaPaGei, AnyPPG and HiMAE keep. It removes
   baseline wander and respiration-driven drift below and sensor noise above, and it sits under the
   12.5 Hz Nyquist limit of the 25 Hz watch stream, so a 25 Hz model loses nothing the filter kept.
2. **Resample** to the model's rate with a polyphase filter (``resample_poly``), from any source rate
   (BIDMC 125 Hz, VitalDB 500 Hz, E4 64 Hz, Galaxy Watch 25 Hz).
3. **Windows** of ``seconds`` with ``hop``; each is **robust-scaled** on its own (median, then the
   interquartile range divided by 1.349, the normal-distribution IQR), as PaPaGei, Pulse-PPG and AI-PPG
   Age z-score each input. Amplitude differs by orders of magnitude between devices and placements;
   the shape is what is kept.
4. A window is kept when it passes :func:`quality`: finite, not flat, not clipped, no scaled sample
   beyond :data:`MAX_SCALED_ABS`, and periodic at a plausible heart rate. This is a gate against dead or detached sensors, not a full SQI.

A phone does steps 1-3 the same way (a causal band-pass is close enough for the gate) before feeding
the ONNX stream graph, which then reads the scaled samples patch by patch.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

#: The band the PPG foundation models keep (PaPaGei, AnyPPG, HiMAE): pulse and its harmonics.
BAND_HZ = (0.5, 8.0)
#: Heart rates the quality gate accepts, beats per minute.
HR_RANGE_BPM = (30.0, 200.0)
#: The largest robust-scaled sample a kept window may hold. A clean pulse stays under about 6 (VitalDB's
#: 99.9th percentile is 6.3); a quiet window with one motion spike has a tiny IQR, so the spike scales
#: into the thousands and past float16's 65,504 when the window is stored.
MAX_SCALED_ABS = 20.0
#: The normal distribution's interquartile range in standard deviations.
IQR_TO_SD = 1.349


def bandpass(x: np.ndarray, fs: float, band: tuple[float, float] = BAND_HZ, order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth band-pass along the last axis; the upper edge is capped below Nyquist."""
    from scipy.signal import butter, sosfiltfilt

    lo, hi = band
    hi = min(hi, 0.45 * fs)
    sos = butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    padlen = min(x.shape[-1] - 1, 3 * (2 * len(sos) + 1) * 4)
    return sosfiltfilt(sos, x, axis=-1, padlen=padlen)


def resample(x: np.ndarray, fs: float, target: float) -> np.ndarray:
    """``x`` at ``fs`` resampled to ``target`` Hz with a polyphase anti-aliasing filter."""
    from scipy.signal import resample_poly

    if abs(fs - target) < 1e-9:
        return np.asarray(x, dtype=np.float64)
    ratio = Fraction(target / fs).limit_denominator(1000)
    return resample_poly(np.asarray(x, dtype=np.float64), ratio.numerator, ratio.denominator, axis=-1)


def fill_gaps(x: np.ndarray, max_gap: int) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate NaN runs of at most ``max_gap`` samples; return the signal and its valid mask.

    Longer runs stay invalid (and are zero in the returned signal): a window touching one is dropped.
    """
    x = np.asarray(x, dtype=np.float64).copy()
    bad = ~np.isfinite(x)
    if not bad.any():
        return x, np.ones(len(x), dtype=bool)
    idx = np.arange(len(x))
    good = ~bad
    if good.sum() < 2:
        return np.zeros_like(x), np.zeros(len(x), dtype=bool)
    filled = np.interp(idx, idx[good], x[good])
    valid = good.copy()
    # runs of NaN: start and stop indices
    edges = np.diff(np.concatenate([[0], bad.astype(np.int8), [0]]))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if stop - start <= max_gap and start > 0 and stop < len(x):
            valid[start:stop] = True
    filled[~valid] = 0.0
    return filled, valid


def robust_scale(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """``(x - median) / (IQR / 1.349)`` along ``axis``; a flat input stays zero rather than dividing by zero."""
    med = np.median(x, axis=axis, keepdims=True)
    q75, q25 = np.percentile(x, [75, 25], axis=axis, keepdims=True)
    scale = (q75 - q25) / IQR_TO_SD
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (x - med) / scale


def windows(n: int, size: int, hop: int) -> np.ndarray:
    """Start indices of every whole window of ``size`` samples, ``hop`` apart, in ``n`` samples."""
    if n < size:
        return np.zeros(0, dtype=np.int64)
    return np.arange(0, n - size + 1, hop, dtype=np.int64)


@dataclass(frozen=True)
class Quality:
    ok: bool
    reason: str
    hr_bpm: float
    periodicity: float


def quality(seg: np.ndarray, fs: float, min_periodicity: float = 0.35, max_abs: float | None = None
            ) -> Quality:
    """A gate for one window: finite, not flat, not clipped, periodic at a heart rate in range, and, with
    ``max_abs`` (for a robust-scaled window), no sample beyond it (reason ``"outlier"``).

    Periodicity is the highest normalised autocorrelation at a lag inside :data:`HR_RANGE_BPM`; a clean
    pulse gives 0.7-0.95, motion and a detached sensor give less. The threshold is deliberately loose:
    pretraining should see imperfect wrist data, and the clock averages many windows per person.
    """
    if not np.all(np.isfinite(seg)):
        return Quality(False, "non-finite", float("nan"), 0.0)
    if max_abs is not None and np.max(np.abs(seg)) > max_abs:
        return Quality(False, "outlier", float("nan"), 0.0)
    spread = np.percentile(seg, 95) - np.percentile(seg, 5)
    if spread < 1e-6:
        return Quality(False, "flat", float("nan"), 0.0)
    top = np.max(seg)
    if np.mean(seg >= top - 1e-9) > 0.05 or np.mean(seg <= np.min(seg) + 1e-9) > 0.05:
        return Quality(False, "clipped", float("nan"), 0.0)
    x = seg - seg.mean()
    lo = int(fs * 60.0 / HR_RANGE_BPM[1])
    hi = min(int(fs * 60.0 / HR_RANGE_BPM[0]), len(x) // 2)
    if hi <= lo + 1:
        return Quality(False, "too short", float("nan"), 0.0)
    spec = np.fft.rfft(x, n=2 * len(x))
    ac = np.fft.irfft(spec * np.conj(spec))[:len(x)]
    ac = ac / max(ac[0], 1e-12)
    lag = lo + int(np.argmax(ac[lo:hi]))
    best = float(ac[lag])
    hr = 60.0 * fs / lag
    if best < min_periodicity:
        return Quality(False, "aperiodic", hr, best)
    return Quality(True, "ok", hr, best)


def prepare_signal(raw: np.ndarray, fs: float, target_fs: float, max_gap_s: float = 0.5
                   ) -> tuple[np.ndarray, np.ndarray]:
    """One recording to the model's rate: gaps filled, band-passed, resampled. Returns the signal and its
    valid mask at ``target_fs`` (a sample is valid when every source sample near it was)."""
    x, valid = fill_gaps(raw, max_gap=int(round(max_gap_s * fs)))
    if valid.sum() < 2 * fs:
        return np.zeros(0), np.zeros(0, dtype=bool)
    x = x - np.median(x[valid])
    x[~valid] = 0.0
    x = bandpass(x, fs)
    y = resample(x, fs, target_fs)
    # A target sample is valid when no invalid source sample lies within one target period of it.
    # (Resampling the mask itself rings around 1 and drops valid samples.)
    bad = ~valid
    reach = int(np.ceil(fs / target_fs))
    if bad.any():
        bad = np.convolve(bad.astype(np.int8), np.ones(2 * reach + 1, np.int8), mode="same") > 0
    src = np.minimum(np.round(np.arange(len(y)) * fs / target_fs).astype(np.int64), len(valid) - 1)
    return y, ~bad[src]


def cut_windows(x: np.ndarray, valid: np.ndarray, fs: float, seconds: float, hop_seconds: float,
                max_windows: int | None = None, min_periodicity: float = 0.35
                ) -> tuple[np.ndarray, np.ndarray, list[Quality]]:
    """Robust-scaled windows that are wholly valid and pass :func:`quality` within :data:`MAX_SCALED_ABS`.

    Returns ``(windows (N, L) float32, starts (N,) in seconds, qualities)``. With ``max_windows``, the
    accepted windows are thinned evenly over the recording rather than taken from its start, so a long
    surgical record contributes its whole course.
    """
    size, hop = int(round(seconds * fs)), max(1, int(round(hop_seconds * fs)))
    starts = [s for s in windows(len(x), size, hop) if valid[s:s + size].all()]
    out, kept, quals = [], [], []
    for s in starts:
        seg = robust_scale(x[s:s + size])
        q = quality(seg, fs, min_periodicity, max_abs=MAX_SCALED_ABS)
        if q.ok:
            out.append(seg.astype(np.float32))
            kept.append(s / fs)
            quals.append(q)
    if max_windows is not None and len(out) > max_windows:
        pick = np.unique(np.linspace(0, len(out) - 1, max_windows).round().astype(int))
        out = [out[i] for i in pick]
        kept = [kept[i] for i in pick]
        quals = [quals[i] for i in pick]
    if not out:
        return np.zeros((0, size), dtype=np.float32), np.zeros(0), []
    return np.stack(out), np.asarray(kept), quals
