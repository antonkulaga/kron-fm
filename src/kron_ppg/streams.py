"""``data/input/ppg/<source>`` -> ``data/interim/ppg/<source>/streams-fs<rate>``: whole recordings, not windows.

KronLSTM is a streaming model: its memory carries a person's signal from one chunk to the next, so it can train
on hours of PPG with truncated backpropagation instead of on 30 s windows that each start from an empty memory.
This writes every recording of a source as one stream at the model's rate, the way :mod:`kron_ppg.prepare`
filters it (gaps up to 0.5 s filled, 0.5-8 Hz band-pass, resampled), but cuts nothing, caps nothing and drops
nothing: a gap or a motion artefact stays in the stream, marked, for the trainer to mask or weight.

Layout (all little-endian, read with ``numpy.memmap`` through :class:`Streams`):

* ``signal.f32``: every stream's samples, end to end, in the source's units after the band-pass (not scaled:
  scaling a stream is the trainer's choice, causally, e.g. by the running median of past ``scale`` values);
* ``valid.u8``: 1 where a sample is real signal, 0 inside a gap;
* ``blocks.f32``: per ``block_s`` block of each stream, ``(scale, periodicity, valid_fraction)``: the block's
  robust scale (IQR / 1.349 over its valid samples), the pulse periodicity the quality gate reads (highest
  normalised autocorrelation at a heart-rate lag), and how much of it is valid. Labels, not filters;
* ``streams.parquet``: one row per stream: subject, recording, offset and length in samples, block offset and
  count, valid hours; ``subjects.parquet`` and ``info.json`` as for windows.

Run ``python -m kron_ppg.streams <source>...``. The wrist readers then yield their recordings whole (a Vilnius
segment, a WildPPG day, a Galaxy Watch device over four weeks) instead of the 6 h pieces the window cap needs.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import sys
import time
import warnings
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

from kron_core.paths import input_dir, interim_dir
from kron_ppg.data import DOMAIN, _subject_frame
from kron_ppg.readers import READERS, Recording
from kron_ppg.signal import HR_RANGE_BPM, IQR_TO_SD, prepare_signal
from kron_ppg.watch_sources import owner_only

BLOCK_S = 10.0


def streams_dir(source: str, sample_rate: float = 25.0) -> Path:
    return interim_dir(DOMAIN, source) / f"streams-fs{sample_rate:g}"


def whole_recordings() -> None:
    """Make the wrist readers yield whole recordings: no 6 h pieces, a Galaxy Watch device as one stream."""
    from kron_ppg import watch_readers

    watch_readers.CHUNK_S = None
    watch_readers.GALAXY_ACTIVE2_UNIT = "device"


def block_stats(x: np.ndarray, valid: np.ndarray, fs: float, block_s: float = BLOCK_S) -> np.ndarray:
    """``(n_blocks, 3)`` float32: robust scale, periodicity and valid fraction of each whole ``block_s`` block
    (a trailing partial block is left out). Invalid samples count as zero for the autocorrelation."""
    size = int(round(block_s * fs))
    n = len(x) // size
    out = np.zeros((n, 3), np.float32)
    if n == 0:
        return out
    lo = int(fs * 60.0 / HR_RANGE_BPM[1])
    hi = min(int(fs * 60.0 / HR_RANGE_BPM[0]), size // 2)
    for start in range(0, n, 4096):  # bounded memory: 4096 blocks at a time
        stop = min(n, start + 4096)
        b = x[start * size:stop * size].reshape(-1, size).astype(np.float64)
        v = valid[start * size:stop * size].reshape(-1, size)
        frac = v.mean(axis=1)
        masked = np.where(v, b, np.nan)
        with warnings.catch_warnings():  # all-gap blocks: NaN, then 0 below
            warnings.simplefilter("ignore", RuntimeWarning)
            q75, q25 = np.nanpercentile(masked, [75, 25], axis=1)
            mean = np.nanmean(masked, axis=1, keepdims=True)
        scale = np.nan_to_num((q75 - q25) / IQR_TO_SD)
        z = np.where(v, b - mean, 0.0)
        z = np.nan_to_num(z)
        spec = np.fft.rfft(z, n=2 * size, axis=1)
        ac = np.fft.irfft(spec * np.conj(spec), axis=1)[:, :size]
        ac = ac / np.maximum(ac[:, :1], 1e-12)
        period = ac[:, lo:hi].max(axis=1) if hi > lo else np.zeros(len(b))
        out[start:stop] = np.stack([scale, np.where(frac > 0, period, 0.0), frac], axis=1)
    return out


def _stream(rec: Recording, sample_rate: float, block_s: float):
    y, valid = prepare_signal(rec.load(), rec.fs, sample_rate)
    return rec.subject, rec.recording, y.astype(np.float32), valid, block_stats(y, valid, sample_rate, block_s)


def _results(recordings, workers: int, *args):
    """``_stream`` over the recordings in order, in ``workers`` spawned processes (as :mod:`kron_ppg.prepare`)."""
    if workers <= 1:
        for rec in recordings:
            yield _stream(rec, *args)
        return
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=whole_recordings) as pool:
        pending: deque = deque()
        for rec in recordings:
            pending.append(pool.submit(_stream, rec, *args))
            if len(pending) >= 2 * workers:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def write_streams(source: str, sample_rate: float = 25.0, block_s: float = BLOCK_S, folder: Path | None = None,
                  out: Path | None = None, workers: int | None = None) -> dict:
    """Write every recording of ``source`` as a stream; returns what ``info.json`` holds.

    The wrist readers read whole recordings while it runs (:func:`whole_recordings`) and are set back after,
    so a later ``prepare`` in the same process still cuts its pieces."""
    from kron_ppg import watch_readers

    saved = watch_readers.CHUNK_S, watch_readers.GALAXY_ACTIVE2_UNIT
    try:
        return _write_streams(source, sample_rate, block_s, folder, out, workers)
    finally:
        watch_readers.CHUNK_S, watch_readers.GALAXY_ACTIVE2_UNIT = saved


def _write_streams(source: str, sample_rate: float, block_s: float, folder: Path | None, out: Path | None,
                   workers: int | None) -> dict:
    whole_recordings()
    if source not in READERS:
        raise ValueError(f"no reader for {source!r}; one of {sorted(READERS)}")
    owner_only(source)  # a restricted source's streams are written owner-only
    folder = folder or input_dir(DOMAIN, source)
    out = out or streams_dir(source, sample_rate)
    out.mkdir(parents=True, exist_ok=True)
    data = READERS[source](folder)
    if workers is None:
        workers = min(16, len(os.sched_getaffinity(0)))
    rows, offset, boffset, started = [], 0, 0, time.time()
    with (out / "signal.f32.part").open("wb") as fs, (out / "valid.u8.part").open("wb") as fv, \
            (out / "blocks.f32.part").open("wb") as fb:
        for subject, recording, y, valid, blocks in _results(data.recordings, workers, sample_rate, block_s):
            fs.write(y.astype("<f4").tobytes())
            fv.write(valid.astype(np.uint8).tobytes())
            fb.write(blocks.astype("<f4").tobytes())
            rows.append({"subject": str(subject), "recording": str(recording), "offset": offset, "length": len(y),
                         "block_offset": boffset, "blocks": len(blocks),
                         "valid_hours": round(float(valid.sum()) / sample_rate / 3600, 4)})
            offset += len(y)
            boffset += len(blocks)
            sys.stderr.write(f"\r  {source}: {len(rows)} streams, {offset / sample_rate / 3600:,.1f} h, "
                             f"{time.time() - started:,.0f} s")
    sys.stderr.write("\n")
    for name in ("signal.f32", "valid.u8", "blocks.f32"):
        (out / f"{name}.part").replace(out / name)
    table = pl.DataFrame(rows, schema={"subject": pl.String, "recording": pl.String, "offset": pl.Int64,
                                       "length": pl.Int64, "block_offset": pl.Int64, "blocks": pl.Int64,
                                       "valid_hours": pl.Float64})
    table.write_parquet(out / "streams.parquet")
    _subject_frame(data.subjects).write_parquet(out / "subjects.parquet", compression="zstd")
    manifest = folder / "SOURCE.json"
    info = {"source": source, "sample_rate": sample_rate, "block_s": block_s, "streams": len(rows),
            "samples": offset, "hours": round(offset / sample_rate / 3600, 2),
            "valid_hours": round(float(table["valid_hours"].sum()) if len(rows) else 0.0, 2),
            "subjects": len(data.subjects), "subjects_with_streams": table["subject"].n_unique() if len(rows) else 0,
            "input": str(folder.resolve()),
            "input_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.is_file() else None,
            "files": {"signal.f32": "float32", "valid.u8": "uint8", "blocks.f32": "float32 (n, 3): scale, "
                      "periodicity, valid_fraction"},
            "steps": ["fill gaps <= 0.5 s", "band-pass 0.5-8 Hz", f"resample to {sample_rate:g} Hz",
                      "no windows, no cap, no quality drop, no scaling"]}
    (out / "info.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


class Streams:
    """A written streams variant, memory-mapped: ``len()`` streams, ``[i]`` -> ``(signal, valid, blocks)``."""

    def __init__(self, folder: Path) -> None:
        self.folder = Path(folder)
        self.info = json.loads((self.folder / "info.json").read_text())
        self.table = pl.read_parquet(self.folder / "streams.parquet")
        self.subjects = pl.read_parquet(self.folder / "subjects.parquet")
        n = int(self.info["samples"])
        self.signal = np.memmap(self.folder / "signal.f32", dtype="<f4", mode="r", shape=(n,)) if n else np.zeros(0)
        self.valid = np.memmap(self.folder / "valid.u8", dtype=np.uint8, mode="r", shape=(n,)) if n else np.zeros(0)
        nb = int(self.table["blocks"].sum()) if len(self.table) else 0
        self.blocks = (np.memmap(self.folder / "blocks.f32", dtype="<f4", mode="r", shape=(nb, 3)) if nb
                       else np.zeros((0, 3), np.float32))

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r = self.table.row(i, named=True)
        s = slice(r["offset"], r["offset"] + r["length"])
        return self.signal[s], self.valid[s].astype(bool), self.blocks[r["block_offset"]:r["block_offset"] + r["blocks"]]


if __name__ == "__main__":  # python -m kron_ppg.streams <source>... [--workers N]
    args = sys.argv[1:]
    workers = None
    if "--workers" in args:
        i = args.index("--workers")
        workers = int(args[i + 1])
        del args[i:i + 2]
    if not args:
        sys.exit(f"usage: python -m kron_ppg.streams <source>... [--workers N]; sources: {', '.join(sorted(READERS))}")
    for name in args:
        print(json.dumps(write_streams(name, workers=workers)), flush=True)
