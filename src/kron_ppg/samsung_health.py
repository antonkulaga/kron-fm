"""Readers and per-person metrics for a Samsung Health "Download personal data" export.

The streams GlucoseDAO's ``samsung_health_extractor`` already gives (heart rate, sleep sessions and stages,
exercise, steps) are read from its parquet output; this module adds what it leaves out, from the export folder
itself:

* the watch's own binned summaries: HRV (SDNN and RMSSD per 30 s, Samsung's algorithm), respiratory rate,
  stress, skin temperature, raw SpO2, each a JSON blob named by the CSV row;
* **per-beat RR intervals in sleep** (``com.samsung.shealth.sleep_raw_data``): one record per 30 s epoch of a
  sleep, each with ``rri_info`` triples ``[flag, confidence 1-255, RR in ms]`` (checked on real exports: the RR
  of a night sums to the sleep's length within a few per cent);
* the ECG app's 30 s single-lead recordings (``chart_data``, 500 Hz).

An export is the phone app's database, **not the watch's sensor stream**: it holds no PPG waveform, so the
PPG age clock cannot read it. HRV here comes from RR intervals (clean, artefact-filtered) where the export
has them, and is labelled as Samsung's own values where it does not.

Export quirks (also handled by the extractor): every CSV opens with a metadata line; rows end with a trailing
comma; column names may carry a ``com.samsung.health.<stream>.`` prefix; stored times are UTC, and each row's
``time_offset`` (``UTC+0100``) gives the wearer's wall clock; blobs live at
``jsons/<stream>/<first character of the file name>/<file name>``.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl

_OFFSET = re.compile(r"UTC([+-])(\d{2})(\d{2})")


def offset_minutes(value: str | None) -> int:
    """``UTC+0130`` -> 90; anything else -> 0."""
    m = _OFFSET.fullmatch(value or "")
    if not m:
        return 0
    sign = 1 if m.group(1) == "+" else -1
    return sign * (int(m.group(2)) * 60 + int(m.group(3)))


@dataclass
class Export:
    """An unpacked export folder (``samsunghealth_<user>_<stamp>``)."""

    folder: Path

    def __post_init__(self) -> None:
        self.folder = Path(self.folder)

    def csv_path(self, stream: str) -> Path | None:
        """The stream's CSV, matched exactly (``...exercise`` never opens ``...exercise.weather``)."""
        pattern = re.compile(re.escape(stream) + r"\.\d+\.csv")
        hits = [p for p in self.folder.iterdir() if pattern.fullmatch(p.name)]
        return hits[0] if hits else None

    def table(self, stream: str) -> pl.DataFrame:
        """The stream's rows, every column a string, prefixes taken off the column names."""
        path = self.csv_path(stream)
        if path is None:
            return pl.DataFrame()
        text = path.read_text(encoding="utf-8-sig").splitlines()
        if len(text) < 2:
            return pl.DataFrame()
        frame = pl.read_csv("\n".join(text[1:]).encode(), infer_schema=False, truncate_ragged_lines=True)
        frame = frame.select([c for c in frame.columns if c and not c.startswith("_duplicated")])
        names = {c: c.rsplit(".", 1)[-1] if c.startswith("com.samsung.") else c for c in frame.columns}
        seen: set[str] = set()
        keep = []
        for c, n in names.items():  # a short name that clashes keeps its long form
            keep.append(pl.col(c).alias(n if n not in seen else c))
            seen.add(n)
        return frame.select(keep)

    def blob(self, stream: str, name: str | None):
        """The JSON a cell names, or None when the cell is empty or the file is missing."""
        if not name:
            return None
        path = self.folder / "jsons" / stream / name[0] / name
        if not path.is_file():
            return None
        return json.loads(path.read_text())


def _utc(col: str) -> pl.Expr:
    return pl.col(col).str.strptime(pl.Datetime("ms"), "%Y-%m-%d %H:%M:%S%.f", strict=False)


def binned(export: Export, stream: str, field: str = "binning_data") -> pl.DataFrame:
    """Each row's blob of bins, one output row per bin, with ``local`` time (the wearer's clock).

    Bins carry their own ``start_time`` (or ``time``) in UTC milliseconds; every numeric field of a bin is kept.
    """
    rows = export.table(stream)
    if rows.is_empty() or field not in rows.columns:
        return pl.DataFrame()
    out = []
    offsets = rows["time_offset"].to_list() if "time_offset" in rows.columns else [None] * len(rows)
    for name, off in zip(rows[field].to_list(), offsets, strict=True):
        bins = export.blob(stream, name)
        if isinstance(bins, dict):
            bins = next((v for v in bins.values() if isinstance(v, list)), None)
        if not bins:
            continue
        minutes = offset_minutes(off)
        for b in bins:
            if not isinstance(b, dict):
                continue
            t = b.get("start_time", b.get("time"))
            if t is None:
                continue
            row = {k: float(v) for k, v in b.items() if isinstance(v, (int, float)) and k not in ("start_time", "time",
                                                                                                    "end_time")}
            row["utc_ms"] = int(t)
            row["offset_min"] = minutes
            out.append(row)
    if not out:
        return pl.DataFrame()
    frame = pl.DataFrame(out, infer_schema_length=None)
    return frame.with_columns(
        (pl.from_epoch("utc_ms", time_unit="ms") + pl.duration(minutes="offset_min")).alias("local")
    ).sort("local")


def sleep_rr(export: Export) -> pl.DataFrame:
    """Per-beat RR intervals in sleep: ``sleep_id, night_start (local), epoch, t_s, rr_ms, confidence``.

    ``t_s`` is the beat's time from the sleep's start, from the 30 s epoch it sits in plus the RR sum
    within that epoch; ``night_start`` comes from the sleep table (``com.samsung.shealth.sleep``).
    """
    stream = "com.samsung.shealth.sleep_raw_data"
    raw = export.table(stream)
    if raw.is_empty():
        return pl.DataFrame()
    sleeps = export.table("com.samsung.shealth.sleep")
    starts: dict[str, tuple[str, str]] = {}
    if not sleeps.is_empty() and "datauuid" in sleeps.columns:
        for r in sleeps.select("datauuid", "start_time", "time_offset").iter_rows():
            starts[r[0]] = (r[1], r[2])
    rows = []
    for sleep_id, name in raw.select("sleep_uuid", "data").iter_rows():
        blob = export.blob(stream, name)
        if not blob:
            continue
        start, off = starts.get(sleep_id, (None, None))
        for epoch, record in enumerate(blob.get("data", [])):
            t = epoch * 30.0
            for flag, conf, rr in record.get("rri_info", []):
                rows.append({"sleep_id": sleep_id, "start_utc": start, "offset_min": offset_minutes(off),
                             "epoch": epoch, "t_s": t, "rr_ms": float(rr), "confidence": int(conf), "flag": int(flag)})
                t += rr / 1000.0
    if not rows:
        return pl.DataFrame()
    frame = pl.DataFrame(rows)
    return frame.with_columns(
        (_utc("start_utc") + pl.duration(minutes="offset_min")).alias("night_start")
    ).drop("start_utc")


def ecg_recordings(export: Export) -> list[dict]:
    """The ECG app's recordings: ``local`` start, ``fs``, ``samples`` (mV as stored), Samsung's mean HR."""
    stream = "com.samsung.health.ecg"
    rows = export.table(stream)
    out = []
    if rows.is_empty() or "chart_data" not in rows.columns:
        return out
    for r in rows.iter_rows(named=True):
        blob = export.blob(stream, r.get("chart_data"))
        if not blob or not blob.get("chart_values"):
            continue
        out.append({"start_utc": r.get("start_time"), "offset_min": offset_minutes(r.get("time_offset")),
                    "fs": float(r.get("sample_frequency") or 500), "samples": np.asarray(blob["chart_values"], float),
                    "samsung_mean_hr": float(r["mean_heart_rate"]) if r.get("mean_heart_rate") else float("nan"),
                    "classification": r.get("classification")})
    return out


# ---- metrics ------------------------------------------------------------------------------------------
def clean_rr(rr_ms: np.ndarray, lo: float = 300.0, hi: float = 2000.0, jump: float = 0.2,
             window: int = 11) -> np.ndarray:
    """A mask of normal-to-normal beats: within ``lo``-``hi`` ms and within ``jump`` of the local median."""
    rr = np.asarray(rr_ms, float)
    ok = (rr >= lo) & (rr <= hi)
    if len(rr) == 0:
        return ok
    pad = window // 2
    padded = np.pad(np.where(ok, rr, np.nan), pad, mode="edge")
    med = np.array([np.nanmedian(padded[i:i + window]) if np.isfinite(padded[i:i + window]).any() else np.nan
                    for i in range(len(rr))])
    return ok & (np.abs(rr - med) <= jump * med)


def hrv(rr_ms: np.ndarray, t_s: np.ndarray, segment_s: float = 300.0, min_beats: int = 150) -> dict:
    """Short-term HRV of a night: RMSSD and SDNN per ``segment_s`` window of NN beats, then their median.

    Successive differences are taken only between neighbouring beats both kept as NN. Also the mean NN
    heart rate and the share of beats kept.
    """
    rr = np.asarray(rr_ms, float)
    t = np.asarray(t_s, float)
    nn = clean_rr(rr)
    seg = np.floor(t / segment_s).astype(int)
    rmssd, sdnn, hr = [], [], []
    for s in np.unique(seg):
        m = seg == s
        if (nn & m).sum() < min_beats:
            continue
        x, k = rr[m], nn[m]
        both = k[1:] & k[:-1]
        d = np.diff(x)[both]
        if len(d) < min_beats // 2:
            continue
        rmssd.append(float(np.sqrt(np.mean(d ** 2))))
        sdnn.append(float(np.std(x[k], ddof=1)))
        hr.append(60000.0 / float(np.mean(x[k])))
    nan = float("nan")
    return {"segments": len(rmssd), "kept_fraction": float(nn.mean()) if len(nn) else nan,
            "rmssd_ms": float(np.median(rmssd)) if rmssd else nan, "sdnn_ms": float(np.median(sdnn)) if sdnn else nan,
            "hr_bpm": float(np.median(hr)) if hr else nan,
            "lowest_segment_hr_bpm": float(np.min(hr)) if hr else nan}


def r_peaks(ecg: np.ndarray, fs: float) -> np.ndarray:
    """R-peak sample indices of a short single-lead ECG (band-pass 5-15 Hz, squared slope, 250 ms refractory).

    **Not validated on real watch ECGs**: on 11 Samsung ECG-app recordings its heart rate differed from the app's
    own by a median of 2-6 bpm and up to 10 (missed and extra beats); a lower threshold made it worse. Use it on
    clean synthetic or chest ECG only until a proper detector replaces it.

    Polarity is taken from the larger of the positive and negative band-passed excursions, so an inverted
    lead still finds its R waves.
    """
    x = np.asarray(ecg, float) - np.mean(ecg)
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / fs)
    spec[(f < 5) | (f > 15)] = 0
    band = np.fft.irfft(spec, n=len(x))
    if np.percentile(-band, 99.5) > np.percentile(band, 99.5):
        band = -band
    energy = np.convolve(np.gradient(band) ** 2, np.ones(int(0.1 * fs)) / int(0.1 * fs), mode="same")
    thr = 0.3 * np.percentile(energy, 99)
    refractory = int(0.25 * fs)
    peaks, i = [], 0
    above = np.flatnonzero(energy > thr)
    while i < len(above):
        start = above[i]
        j = i
        while j + 1 < len(above) and above[j + 1] - above[j] <= 1:
            j += 1
        lo, hi = start, above[j] + 1
        p = lo + int(np.argmax(band[lo:hi]))
        if not peaks or p - peaks[-1] >= refractory:
            peaks.append(p)
        i = j + 1
    return np.asarray(peaks, int)


def nightly_resting_hr(heart_rate: pl.DataFrame, sleeps: pl.DataFrame, minutes: int = 5) -> pl.DataFrame:
    """Per sleep (the extractor's ``sleep_sessions``): the lowest ``minutes``-minute mean of the heart rate.

    ``heart_rate`` is the extractor's stream (``available``, ``heart_rate``; local naive times). A sleep with
    fewer than three readings gives no value.
    """
    hr = heart_rate.select(pl.col("available").alias("t"), pl.col("heart_rate")).drop_nulls().sort("t")
    out = []
    for start, end, sid in sleeps.select("start", "end", "sleep_id").iter_rows():
        if end - start < timedelta(hours=3):
            continue
        part = hr.filter((pl.col("t") >= start) & (pl.col("t") <= end))
        if len(part) < 3:
            continue
        roll = part.group_by_dynamic("t", every="1m", period=f"{minutes}m").agg(pl.col("heart_rate").mean())
        out.append({"sleep_id": sid, "night": (start - timedelta(hours=12)).date(), "start": start,
                    "hours": (end - start).total_seconds() / 3600, "readings": len(part),
                    "resting_hr_bpm": float(roll["heart_rate"].min())})
    return pl.DataFrame(out) if out else pl.DataFrame()
