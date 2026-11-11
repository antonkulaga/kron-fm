"""The HRV head: age from RR intervals, as a resting ECG gives them and a watch's sleep RR gives them.

:func:`features` takes one person's RR series (ms, with beat times) and returns short-term HRV in 5-minute
segments of normal-to-normal beats, medianed over the segments: mean HR, SDNN, RMSSD, pNN50, LF and HF power
(Welch on the RR series resampled at 4 Hz) and LF/HF, Poincare SD1/SD2, DFA alpha-1 (4-16 beats), and the
segment-to-segment spread of HR. The same function reads a Samsung Health export's sleep RR
(:func:`kron_ppg.samsung_health.sleep_rr` or the extractor's ``vitals``), so a head trained on a cohort's ECG reads a
watch's nights the same way (with a state mismatch to say aloud: a supine rest recording is not sleep).

:func:`autonomic_aging_record` turns one record of the Autonomic Aging dataset (two ECG leads at 1000 Hz) into RR
series per lead with wfdb's XQRS detector; the two leads are independent detections of the same beats, which is
the check a derived RR series gets before a result rests on it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from kron_ppg.samsung_health import clean_rr

#: Autonomic Aging's age groups (subject-info.csv ``Age_group``) and their mid-points in years.
AGE_GROUPS = {1: (18, 19), 2: (20, 24), 3: (25, 29), 4: (30, 34), 5: (35, 39), 6: (40, 44), 7: (45, 49), 8: (50, 54),
              9: (55, 59), 10: (60, 64), 11: (65, 69), 12: (70, 74), 13: (75, 79), 14: (80, 84), 15: (85, 92)}
FEATURES = ["hr", "sdnn", "rmssd", "pnn50", "log_lf", "log_hf", "lf_hf", "sd1", "sd2", "sd1_sd2", "dfa_a1", "hr_spread"]


def age_mid(group: int) -> float:
    lo, hi = AGE_GROUPS[int(group)]
    return (lo + hi) / 2


def _dfa_alpha1(rr: np.ndarray, scales=range(4, 17)) -> float:
    if len(rr) < 64:
        return float("nan")
    y = np.cumsum(rr - rr.mean())
    fl = []
    for n in scales:
        m = len(y) // n
        seg = y[: m * n].reshape(m, n)
        x = np.arange(n)
        coef = np.polyfit(x, seg.T, 1)
        trend = np.outer(coef[0], x) + coef[1][:, None]
        fl.append(np.sqrt(np.mean((seg - trend) ** 2)))
    return float(np.polyfit(np.log(list(scales)), np.log(fl), 1)[0])


def _spectrum(t_s: np.ndarray, rr: np.ndarray) -> tuple[float, float]:
    from scipy.signal import welch

    if len(rr) < 60 or t_s[-1] - t_s[0] < 120:
        return float("nan"), float("nan")
    grid = np.arange(t_s[0], t_s[-1], 0.25)
    x = np.interp(grid, t_s, rr)
    f, p = welch(x - x.mean(), fs=4.0, nperseg=min(len(x), 256))
    lf = np.trapezoid(p[(f >= 0.04) & (f < 0.15)], f[(f >= 0.04) & (f < 0.15)])
    hf = np.trapezoid(p[(f >= 0.15) & (f < 0.4)], f[(f >= 0.15) & (f < 0.4)])
    return float(lf), float(hf)


def features(rr_ms: np.ndarray, t_s: np.ndarray | None = None, segment_s: float = 300.0,
             min_beats: int = 150) -> dict[str, float] | None:
    """HRV of one series; ``t_s`` the beat times (default: the running sum of RR). None with no usable segment."""
    rr = np.asarray(rr_ms, float)
    t = np.cumsum(rr) / 1000.0 if t_s is None else np.asarray(t_s, float)
    nn = clean_rr(rr)
    seg = np.floor((t - t[0]) / segment_s).astype(int)
    rows = []
    for s in np.unique(seg):
        m = seg == s
        x, k, tt = rr[m], nn[m], t[m]
        if k.sum() < min_beats:
            continue
        both = k[1:] & k[:-1]
        d = np.diff(x)[both]
        good = x[k]
        lf, hf = _spectrum(tt[k], good)
        sd1 = float(np.std(d, ddof=1) / np.sqrt(2)) if len(d) > 2 else float("nan")
        sdnn = float(np.std(good, ddof=1))
        sd2 = float(np.sqrt(max(2 * sdnn ** 2 - sd1 ** 2, 0.0)))
        rows.append({"hr": 60000.0 / good.mean(), "sdnn": sdnn, "rmssd": float(np.sqrt(np.mean(d ** 2))),
                     "pnn50": float(np.mean(np.abs(d) > 50)), "log_lf": float(np.log(lf + 1)),
                     "log_hf": float(np.log(hf + 1)), "lf_hf": float(lf / hf) if hf > 0 else float("nan"),
                     "sd1": sd1, "sd2": sd2, "sd1_sd2": sd1 / sd2 if sd2 > 0 else float("nan"),
                     "dfa_a1": _dfa_alpha1(good)})
    if not rows:
        return None
    out = {k: float(np.nanmedian([r[k] for r in rows])) for k in rows[0]}
    out["hr_spread"] = float(np.std([r["hr"] for r in rows])) if len(rows) > 1 else 0.0
    out["segments"] = len(rows)
    out["kept_fraction"] = float(nn.mean())
    return out


def autonomic_aging_record(record: Path, fs_target: float = 250.0) -> dict[str, np.ndarray]:
    """RR intervals (ms) and beat times (s) of each ECG lead of one record, by XQRS at ``fs_target`` Hz."""
    import wfdb
    from scipy.signal import resample_poly
    from wfdb import processing

    rec = wfdb.rdrecord(str(record))
    out = {}
    for i, name in enumerate(rec.sig_name):
        if not name.upper().startswith("ECG"):
            continue
        x = rec.p_signal[:, i]
        x = np.nan_to_num(x - np.nanmedian(x))
        factor = int(round(rec.fs / fs_target))
        y = resample_poly(x, 1, factor) if factor > 1 else x
        fs = rec.fs / factor
        xq = processing.XQRS(sig=y, fs=fs)
        xq.detect(verbose=False)
        peaks = np.asarray(xq.qrs_inds, int)
        if len(peaks) < 3:
            continue
        # move each peak to the largest |deflection| within 50 ms, either polarity
        w = int(0.05 * fs)
        peaks = np.array([max(p - w, 0) + int(np.argmax(np.abs(y[max(p - w, 0):p + w + 1]))) for p in peaks])
        t = peaks / fs
        out[name] = {"rr_ms": np.diff(t) * 1000.0, "t_s": t[1:]}
    return out
