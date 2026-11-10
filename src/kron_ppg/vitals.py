"""Per-window vitals for VitalDB windows: what a Galaxy Watch can also measure, read from the other tracks.

``kron-ppg vitals vitaldb`` writes ``vitals.parquet`` next to a prepared variant's ``windows.parquet``,
one row per window keyed by ``(recording, start_s)``, from the tracks ``kron-ppg download vitaldb
--extras vitals --extras ecg`` fetched (:data:`kron_ppg.sources.VITALDB_EXTRAS`):

* the monitor's numbers over the window (median): ``hr`` (bpm), ``spo2`` (%), ``bt`` (body temperature,
  degC: a core temperature, not the watch's skin temperature), cuff pressure ``nibp_sbp/dbp/mbp`` (a
  cuff reading is held until the next one, every few minutes) and arterial-line pressure
  ``art_sbp/dbp/mbp`` (beat by beat, in about 55% of cases); ``sbp`` and ``dbp`` take the arterial value
  where there is one, else the cuff's, with ``bp_source`` saying which;
* from ECG lead II (500 Hz, the watch's ECG spot check is a single lead too): R peaks, ``ecg_hr``,
  ``rr_ms``, and the 30 s heart-rate variability ``sdnn_ms`` and ``rmssd_ms``;
* ``pat_ms``, the pulse arrival time: from each R peak to the steepest upstroke of the finger PPG that
  follows it before the next beat (80 ms to 1 s later; a maximum on the search's edge drops the beat),
  median over the window's beats. VitalDB's PLETH lags its ECG by a device delay, so values run long
  (often 0.5-0.8 s) and are comparable within a person and a monitor, not with published PATs. It
  shortens as arteries stiffen and as pressure rises, and the watch can measure it in its ECG spot check,
  which records green PPG alongside.

A value is NaN where its track is missing, out of range, or (ECG, PAT) too few clean beats are found.
"""
from __future__ import annotations

import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

from kron_core.paths import input_dir
from kron_ppg.data import DOMAIN, variant_dir
from kron_ppg.signal import bandpass, fill_gaps

FS_WAVE = 500.0
#: numeric track slug -> plausible range; values outside are dropped
NUMERIC_RANGES: dict[str, tuple[float, float]] = {
    "hr": (25, 220), "spo2": (50, 100), "bt": (30, 43),
    "nibp_sbp": (40, 260), "nibp_dbp": (15, 160), "nibp_mbp": (25, 200),
    "art_sbp": (40, 260), "art_dbp": (15, 160), "art_mbp": (25, 200),
}
ECG_COLUMNS = ("ecg_hr", "rr_ms", "sdnn_ms", "rmssd_ms", "pat_ms", "beats")
VITAL_COLUMNS = (*NUMERIC_RANGES, "sbp", "dbp", *ECG_COLUMNS)


def read_numeric(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """``(times, values)`` of a numeric VitalDB track, rows without a value dropped, sorted by time."""
    df = pl.read_csv(path, infer_schema=False)
    df = df.select(pl.col(df.columns[0]).cast(pl.Float64, strict=False).alias("t"),
                   pl.col(df.columns[-1]).str.strip_chars().cast(pl.Float64, strict=False).alias("v")
                   ).drop_nulls().sort("t")
    return df["t"].to_numpy(), df["v"].to_numpy()


def read_wave(path: Path) -> np.ndarray:
    """A 500 Hz VitalDB wave track (``Time`` only on its first and last rows) as float64, gaps as NaN."""
    df = pl.read_csv(path, infer_schema=False)
    return df.get_column(df.columns[-1]).str.strip_chars().cast(pl.Float64, strict=False).fill_null(np.nan).to_numpy()


def window_medians(times: np.ndarray, values: np.ndarray, starts: np.ndarray, seconds: float,
                   valid: tuple[float, float]) -> np.ndarray:
    """Median of the in-range values with time in ``[start, start + seconds)`` for each start; NaN if none."""
    keep = (values >= valid[0]) & (values <= valid[1])
    times, values = times[keep], values[keep]
    lo = np.searchsorted(times, starts, side="left")
    hi = np.searchsorted(times, starts + seconds, side="left")
    return np.array([np.median(values[a:b]) if b > a else np.nan for a, b in zip(lo, hi, strict=True)])


def r_peaks(ecg: np.ndarray, fs: float = FS_WAVE) -> np.ndarray:
    """R-peak sample indices of an ECG segment: band-pass 5-20 Hz, turn the larger deflection upright,
    peaks at least 270 ms apart and prominent against the segment's spread."""
    from scipy.signal import find_peaks

    y = bandpass(ecg, fs, (5.0, 20.0))
    if abs(np.percentile(y, 0.5)) > 1.5 * abs(np.percentile(y, 99.5)):
        y = -y
    spread = np.percentile(y, 99.5) - np.median(y)
    if not spread > 0:
        return np.zeros(0, dtype=np.int64)
    peaks, _ = find_peaks(y, distance=int(0.27 * fs), prominence=0.4 * spread)
    return peaks


#: an R-R interval further than this share from the window's median is not normal-to-normal (ectopic,
#: missed or extra beat); HRV is read on the rest, and a window with under 80% of them has none
NN_TOLERANCE = 0.2
#: ECG features are dropped when the ECG's heart rate and the PPG's pulse rate differ by more than this
MAX_RATE_MISMATCH = 0.2
#: ... or when more than this share of successive R-R intervals jump by 1.5x or more (T waves taken for R)
MAX_ALTERNATION = 0.4
#: pulse arrival time search, seconds after each R peak; the window ends at the next R peak plus ``PAT_MIN_S``
PAT_MIN_S, PAT_MAX_S = 0.08, 1.0
#: a beat's PAT is dropped when its steepest upstroke sits this close to either end of the search
PAT_EDGE_S = 0.01
#: the window's PAT is dropped when its beats' interquartile range exceeds this (ms)
PAT_MAX_IQR_MS = 60.0


def ppg_rate(ppg: np.ndarray, fs: float = FS_WAVE) -> float:
    """The PPG's pulse rate (bpm) from its autocorrelation, as the windows' quality gate reads it; NaN when
    the segment is not periodic."""
    from kron_ppg.signal import quality

    q = quality(bandpass(ppg, fs, (0.5, 8.0)), fs)
    return q.hr_bpm if q.ok else float("nan")


def pulse_arrival(peaks: np.ndarray, slope: np.ndarray, fs: float = FS_WAVE) -> np.ndarray:
    """Per beat, ms from the R peak to the steepest PPG upstroke in ``[R + 80 ms, min(next R + 80 ms, R + 1 s)]``.

    Bounding the search by the next R peak keeps each beat's upstroke its own, and a maximum on either
    edge of the search means the upstroke lies outside it, so that beat is dropped (NaN) rather than read
    as the edge. VitalDB's PLETH channel lags the ECG by a device delay, so PATs there run long (often
    0.5-0.8 s); within a person they still move with pressure and stiffness."""
    lo, edge = int(PAT_MIN_S * fs), int(PAT_EDGE_S * fs)
    out = np.full(max(0, len(peaks) - 1), np.nan)
    for i, (r, nxt) in enumerate(zip(peaks[:-1], peaks[1:], strict=True)):
        a, b = r + lo, min(nxt + lo, r + int(PAT_MAX_S * fs), len(slope))
        if b - a <= 2 * edge:
            continue
        k = int(np.argmax(slope[a:b]))
        if edge <= k < (b - a) - edge:
            out[i] = (lo + k) / fs * 1000.0
    return out


def ecg_features(ecg: np.ndarray, ppg: np.ndarray | None, fs: float = FS_WAVE, min_beats: int = 10
                 ) -> dict[str, float]:
    """``ecg_hr``, ``rr_ms``, ``sdnn_ms``, ``rmssd_ms``, ``pat_ms`` and ``beats`` of one window.

    SDNN and RMSSD are read on normal-to-normal intervals (:data:`NN_TOLERANCE`). The R-R features are
    NaN when the beats are too few or implausible, when R-R intervals alternate short
    and long (the detector took T waves for R peaks), or, with a PPG, when the ECG's rate and the PPG's
    pulse rate disagree by more than :data:`MAX_RATE_MISMATCH`; PAT then is NaN too."""
    out = dict.fromkeys(ECG_COLUMNS, float("nan"))
    if np.isnan(ecg).mean() > 0.05:
        return out
    ecg, _ = fill_gaps(ecg, int(0.2 * fs))
    peaks = r_peaks(ecg, fs)
    rr = np.diff(peaks) / fs * 1000.0
    ok = (rr >= 300) & (rr <= 2000)
    out["beats"] = float(len(peaks))
    if len(peaks) < min_beats or ok.mean() < 0.8:
        return out
    # normal-to-normal intervals, as HRV is defined: an ectopic, missed or extra beat is left out
    ok &= np.abs(rr - np.median(rr[ok])) <= NN_TOLERANCE * np.median(rr[ok])
    if ok.mean() < 0.8:
        return out
    jumps = np.abs(np.log(rr[1:] / rr[:-1])) >= np.log(1.5)
    if jumps.mean() > MAX_ALTERNATION:
        return out
    good = rr[ok]
    if ppg is not None and np.isnan(ppg).mean() <= 0.05:
        ppg, _ = fill_gaps(ppg, int(0.2 * fs))
    else:
        ppg = None
    ecg_hr = 60000.0 / float(np.median(good))
    if ppg is not None:
        rate = ppg_rate(ppg, fs)
        if not (np.isfinite(rate) and abs(ecg_hr - rate) <= MAX_RATE_MISMATCH * rate):
            return out
    pairs = ok[1:] & ok[:-1]
    out["rr_ms"] = float(np.median(good))
    out["ecg_hr"] = ecg_hr
    out["sdnn_ms"] = float(np.std(good))
    out["rmssd_ms"] = float(np.sqrt(np.mean(np.diff(rr)[pairs] ** 2))) if pairs.any() else float("nan")
    if ppg is not None:
        pats = pulse_arrival(peaks, np.gradient(bandpass(ppg, fs, (0.5, 8.0))), fs)
        pats = pats[np.isfinite(pats)]
        if len(pats) >= min_beats and np.subtract(*np.percentile(pats, [75, 25])) <= PAT_MAX_IQR_MS:
            out["pat_ms"] = float(np.median(pats))
    return out


def recording_vitals(folder: Path, recording: str, starts: np.ndarray, seconds: float) -> dict[str, np.ndarray]:
    """Every vital column for one recording's window starts."""
    cid = int(recording)
    n = len(starts)
    out = {c: np.full(n, np.nan) for c in VITAL_COLUMNS}
    for slug, valid in NUMERIC_RANGES.items():
        path = folder / "tracks" / slug / f"{cid:04d}.csv.gz"
        if path.is_file():
            t, v = read_numeric(path)
            out[slug] = window_medians(t, v, starts, seconds, valid)
    art = np.isfinite(out["art_sbp"]) & np.isfinite(out["art_dbp"])
    out["sbp"] = np.where(art, out["art_sbp"], out["nibp_sbp"])
    out["dbp"] = np.where(art, out["art_dbp"], out["nibp_dbp"])
    out["bp_source"] = np.where(art, "art", np.where(np.isfinite(out["nibp_sbp"]), "nibp", ""))
    ecg_path = folder / "tracks" / "ecg_ii" / f"{cid:04d}.csv.gz"
    if ecg_path.is_file():
        from kron_ppg.readers import read_vitaldb_track

        ecg = read_wave(ecg_path)
        ppg_path = folder / "pleth" / f"{cid:04d}.csv.gz"
        ppg = read_vitaldb_track(ppg_path) if ppg_path.is_file() else None
        size = int(round(seconds * FS_WAVE))
        for i, s in enumerate(starts):
            a = int(round(s * FS_WAVE))
            if a + size > len(ecg):
                continue
            seg_ppg = ppg[a:a + size] if ppg is not None and a + size <= len(ppg) else None
            for k, v in ecg_features(ecg[a:a + size], seg_ppg).items():
                out[k][i] = v
    return out


def _job(args):
    folder, recording, starts, seconds = args
    return recording, starts, recording_vitals(folder, recording, starts, seconds)


def prepare_vitals(source: str = "vitaldb", sample_rate: float = 25.0, seconds: float = 30.0,
                   folder: Path | None = None, variant: Path | None = None, workers: int | None = None) -> dict:
    """Write ``vitals.parquet`` for a prepared VitalDB variant; returns how many windows got each value."""
    if source != "vitaldb":
        raise ValueError("vitals come from VitalDB's other tracks; only the vitaldb source has them")
    folder = folder or input_dir(DOMAIN, source)
    variant = variant or variant_dir(source, sample_rate, seconds)
    if not (folder / "tracks").is_dir():
        raise FileNotFoundError(f"{folder}/tracks is missing: run `uv run kron-ppg download vitaldb "
                                "--extras vitals --extras ecg` first")
    windows = pl.read_parquet(variant / "windows.parquet", columns=["recording", "start_s"])
    groups = windows.group_by("recording", maintain_order=True).agg(pl.col("start_s"))
    jobs = [(folder, r, np.asarray(s, dtype=float), seconds)
            for r, s in zip(groups["recording"].to_list(), groups["start_s"].to_list(), strict=True)]
    workers = workers or min(16, len(os.sched_getaffinity(0)))
    parts, started = [], time.time()
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for k, (recording, starts, values) in enumerate(pool.map(_job, jobs, chunksize=4), 1):
            parts.append(pl.DataFrame({"recording": [recording] * len(starts), "start_s": starts,
                                       **{c: values[c] for c in VITAL_COLUMNS}, "bp_source": values["bp_source"]}))
            sys.stderr.write(f"\r  vitals: {k}/{len(jobs)} recordings, {time.time() - started:,.0f} s")
    sys.stderr.write("\n")
    table = pl.concat(parts) if parts else pl.DataFrame()
    table = table.with_columns(pl.selectors.float().fill_nan(None))
    table.write_parquet(variant / "vitals.parquet", compression="zstd")
    return {"windows": len(table), **{c: int(table[c].is_not_null().sum()) for c in VITAL_COLUMNS}}
