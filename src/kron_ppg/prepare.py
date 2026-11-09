"""``data/input/ppg/<source>`` -> ``data/interim/ppg/<source>/fs<rate>-w<seconds>s``: windows ready to train on."""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import numpy as np

from kron_core.paths import input_dir
from kron_ppg.data import DOMAIN, variant_dir, write_windows
from kron_ppg.readers import READERS
from kron_ppg.signal import cut_windows, prepare_signal


def prepare_source(source: str, sample_rate: float = 25.0, seconds: float = 30.0, hop: float | None = None,
                   max_windows: int | None = 240, min_periodicity: float = 0.35, folder: Path | None = None,
                   out: Path | None = None) -> dict:
    """Read every recording of ``source``, cut quality-gated windows, and write the variant.

    ``max_windows`` caps the windows kept per recording (thinned evenly over it), so a three-hour
    surgical record (VitalDB) is not worth 360 eight-minute ICU ones (BIDMC). ``hop`` defaults to
    ``seconds`` (no overlap).
    """
    if source not in READERS:
        raise ValueError(f"no reader for {source!r}; one of {sorted(READERS)}")
    folder = folder or input_dir(DOMAIN, source)
    if not folder.exists():
        raise FileNotFoundError(f"{folder} is missing: run `uv run kron-data download {source}` first")
    out = out or variant_dir(source, sample_rate, seconds)
    data = READERS[source](folder)
    hop = hop or seconds
    windows, rows, reasons = [], [], {}
    started = time.time()
    n_rec = 0
    for rec in data.recordings:
        n_rec += 1
        x, valid = prepare_signal(rec.signal, rec.fs, sample_rate)
        w, starts, quals = cut_windows(x, valid, sample_rate, seconds, hop, max_windows, min_periodicity)
        windows.append(w)
        rows += [{"subject": rec.subject, "start_s": f"{s:.2f}", "hr_bpm": f"{q.hr_bpm:.1f}",
                  "periodicity": f"{q.periodicity:.3f}"} for s, q in zip(starts, quals, strict=True)]
        reasons[rec.recording] = len(w)
        sys.stderr.write(f"\r  {source}: {n_rec} recordings, {len(rows):,} windows, {time.time() - started:,.0f} s")
    sys.stderr.write("\n")
    manifest = folder / "SOURCE.json"
    info = {"source": source, "sample_rate": sample_rate, "seconds": seconds, "hop": hop,
            "max_windows_per_recording": max_windows, "min_periodicity": min_periodicity,
            "recordings": n_rec, "recordings_without_windows": sum(1 for v in reasons.values() if v == 0),
            "input": str(folder.resolve()),
            "input_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.is_file() else None,
            "steps": ["fill gaps <= 0.5 s", "band-pass 0.5-8 Hz", f"resample to {sample_rate:g} Hz",
                      f"{seconds:g} s windows", "robust scale per window", "quality gate"]}
    length = int(round(seconds * sample_rate))
    windows = [w if len(w) else np.zeros((0, length), np.float32) for w in windows]
    return write_windows(out, windows, rows, data.subjects, info)
