"""Readers for the wrist and smartwatch sets of :mod:`kron_ppg.watch_sources`: each yields one wrist PPG channel
per recording (green where the set has it) at a fixed rate, and a subject table with age where the set gives one.

As in :mod:`kron_ppg.readers`, nothing downloaded is executed or unpickled: CSVs are parsed as text, archives are
read member by member, MATLAB 7.3 files are read as HDF5 datasets.
"""
from __future__ import annotations

import re
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import polars as pl

from kron_ppg.readers import Recording, SourceData, _table, _values

#: A gap between two samples longer than this stays a gap (NaN) when a signal is put on a uniform grid.
MAX_GAP_S = 0.5


def uniform(t: np.ndarray, x: np.ndarray, fs: float) -> np.ndarray:
    """Samples ``x`` taken at times ``t`` (seconds, possibly jittered or unordered), linearly interpolated onto a
    grid at ``fs`` from the first time; grid points inside a gap longer than :data:`MAX_GAP_S`, or next to a
    NaN sample, are NaN."""
    keep = np.isfinite(t)
    t, x = t[keep], x[keep]
    order = np.argsort(t, kind="stable")
    t, x = t[order], x[order]
    t, first = np.unique(t, return_index=True)
    x = x[first]
    if len(t) < 2:
        return np.full(len(t), np.nan)
    grid = t[0] + np.arange(int(np.floor((t[-1] - t[0]) * fs)) + 1) / fs
    out = np.interp(grid, t, np.nan_to_num(x))
    right = np.clip(np.searchsorted(t, grid, side="left"), 1, len(t) - 1)
    left = right - 1
    bad = ((t[right] - t[left]) > MAX_GAP_S) | ~np.isfinite(x[left]) | ~np.isfinite(x[right])
    out[bad] = np.nan
    return out


def _zip_or_tree(folder: Path, archive: str) -> tuple[list[str], Callable[[str], bytes]]:
    """``(names, opener)`` over a set: the unpacked tree ``folder/unpacked`` when present, else the zip.
    macOS resource forks (``__MACOSX/``, ``._*``) are left out."""
    unpacked = folder / "unpacked"
    if unpacked.is_dir():
        names = [str(p.relative_to(unpacked)) for p in unpacked.rglob("*") if p.is_file()]
        opener = lambda name: (unpacked / name).read_bytes()  # noqa: E731
    else:
        z = zipfile.ZipFile(folder / archive)
        names, opener = z.namelist(), z.read
    names = [n for n in names if "__MACOSX/" not in n and not n.rsplit("/", 1)[-1].startswith("._")]
    return names, opener


# ---- WF-PPG ------------------------------------------------------------------------------------
#: WF-PPG's samples come at about 100 Hz with jittered timestamps; they are put on this grid.
WF_PPG_FS = 100.0


def read_wf_ppg(folder: Path) -> SourceData:
    """Wrist green PPG (``PPG_Wrist_G``, inverted: the raw value falls as blood volume rises) of each session,
    ``<subject>/<session>``. The subject is the participant folder, which numbers like ``additional_info.csv``'s
    ``ID``; the file names carry a second, recruitment number (folder 04 holds ``*_05.csv``). The finger channels
    (``PPG_Finger_*_AC``, band-passed by the authors) are left for the wrist-finger bridge."""
    names, opener = _zip_or_tree(folder, "WF-PPG.zip")
    info = next(n for n in names if n.endswith("additional_info.csv"))
    subjects = [{"subject": f"{int(r['ID']):02d}", "age": r.get("Age", ""), "sex": r.get("Gender", ""),
                 "height": r.get("H", ""), "weight": r.get("W", ""), "bmi": r.get("BMI", "")}
                for r in _table(opener(info)) if r.get("ID", "").strip()]
    pattern = re.compile(r"(?:^|/)(\d+)/(\d+)_preprocessed_(\d+)\.csv$")
    sessions = sorted((int(m[1]), int(m[2]), n) for n in names if (m := pattern.search(n)))

    def gen() -> Iterator[Recording]:
        for subject, session, name in sessions:
            df = pl.read_csv(opener(name), columns=["Time", "PPG_Wrist_G"], infer_schema=False)
            x = uniform(_values(df["Time"]), -_values(df["PPG_Wrist_G"]), WF_PPG_FS)
            yield Recording(f"{subject:02d}", f"{subject:02d}/{session}", WF_PPG_FS, x)

    return SourceData(subjects, gen())


READERS = {"wf_ppg": read_wf_ppg}
PARSERS: dict[str, Callable[[Path], np.ndarray]] = {}
