"""``data/input/ppg/<source>`` -> ``data/interim/ppg/<source>/fs<rate>-w<seconds>s``: windows ready to train on."""
from __future__ import annotations

import hashlib
import os
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from kron_core.paths import input_dir
from kron_ppg.data import DOMAIN, variant_dir, write_windows
from kron_ppg.readers import READERS, Recording
from kron_ppg.signal import cut_windows, prepare_signal


def _cut(rec: Recording, sample_rate: float, seconds: float, hop: float, max_windows: int | None,
         min_periodicity: float):
    x, valid = prepare_signal(rec.load(), rec.fs, sample_rate)
    w, starts, quals = cut_windows(x, valid, sample_rate, seconds, hop, max_windows, min_periodicity)
    return rec.subject, rec.recording, w, starts, quals


def _results(recordings, workers: int, *args):
    """``_cut`` over the recordings in their order, in ``workers`` processes with at most ``2 * workers``
    recordings in flight (each can hold hundreds of MB of samples)."""
    if workers <= 1:
        for rec in recordings:
            yield _cut(rec, *args)
        return
    with ProcessPoolExecutor(workers) as pool:
        pending: deque = deque()
        for rec in recordings:
            pending.append(pool.submit(_cut, rec, *args))
            if len(pending) >= 2 * workers:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def prepare_source(source: str, sample_rate: float = 25.0, seconds: float = 30.0, hop: float | None = None,
                   max_windows: int | None = 240, min_periodicity: float = 0.35, folder: Path | None = None,
                   out: Path | None = None, workers: int | None = None) -> dict:
    """Read every recording of ``source``, cut quality-gated windows, and write the variant.

    ``max_windows`` caps the windows kept per recording (thinned evenly over it), so a three-hour
    surgical record (VitalDB) is not worth 360 eight-minute ICU ones (BIDMC). ``hop`` defaults to
    ``seconds`` (no overlap). ``workers`` processes parse and filter recordings in parallel (default:
    the CPUs this process may use, at most 16); the output is the same in any number.
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
    if workers is None:
        workers = min(16, len(os.sched_getaffinity(0)))
    for subject, recording, w, starts, quals in _results(data.recordings, workers, sample_rate, seconds, hop,
                                                         max_windows, min_periodicity):
        n_rec += 1
        windows.append(w)
        rows += [{"subject": subject, "start_s": f"{s:.2f}", "hr_bpm": f"{q.hr_bpm:.1f}",
                  "periodicity": f"{q.periodicity:.3f}"} for s, q in zip(starts, quals, strict=True)]
        reasons[recording] = len(w)
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
