"""Reading each downloaded PPG source into recordings: a subject, a raw signal at its own rate, and an age.

Nothing here filters or resamples; :mod:`kron_ppg.prepare` does that the same way for every source.
Files are only parsed as text (CSV) or read from a zip: nothing downloaded is executed or unpickled.
"""
from __future__ import annotations

import csv
import io
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from kron_ppg.clock import OUTCOME_FLAGS
from kron_ppg.sources import open_text


@dataclass
class Recording:
    """One recording. ``signal`` is either held, or parsed on demand from ``path`` by the parser named
    ``parser`` (:data:`PARSERS`), so a worker process parses it rather than the reading process."""

    subject: str
    recording: str
    fs: float
    signal: np.ndarray | None = None
    path: Path | None = None
    parser: str = ""

    def load(self) -> np.ndarray:
        if self.signal is not None:
            return self.signal
        return PARSERS[self.parser](self.path)


@dataclass
class SourceData:
    subjects: list[dict] = field(default_factory=list)
    recordings: Iterator[Recording] = iter(())


def _float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _column(path: Path, name: str, delimiter: str = ",") -> np.ndarray:
    """One numeric column of a CSV (header names stripped), empty cells as NaN."""
    with open_text(path) as fh:
        reader = csv.reader(fh, delimiter=delimiter)
        header = [h.strip() for h in next(reader)]
        if name not in header:
            raise ValueError(f"{path} has no {name!r} column (columns: {header})")
        j = header.index(name)
        return np.array([_float(row[j]) if j < len(row) else np.nan for row in reader])


# ---- BIDMC -------------------------------------------------------------------------------------
def _bidmc_fix(path: Path) -> dict[str, str]:
    """``Age: 88`` and ``Gender: M`` lines of a ``_Fix.txt``; ``90+`` is kept as 90 with ``age_censored``."""
    out: dict[str, str] = {}
    for line in path.read_text(errors="replace").splitlines():
        key, _, value = line.partition(":")
        key, value = key.strip().lower(), value.strip()
        if key == "age":
            out["age_censored"] = "1" if value.endswith("+") else "0"
            age = _float(value.rstrip("+"))
            out["age"] = "" if np.isnan(age) else f"{age:g}"
        elif key in ("gender", "sex"):
            out["sex"] = value
    return out


def read_bidmc(folder: Path) -> SourceData:
    ids = sorted(p.name.split("_")[1] for p in folder.glob("bidmc_*_Signals.csv"))
    subjects = [{"subject": i, **_bidmc_fix(folder / f"bidmc_{i}_Fix.txt")} for i in ids]

    def gen() -> Iterator[Recording]:
        for i in ids:
            yield Recording(i, i, 125.0, path=folder / f"bidmc_{i}_Signals.csv", parser="bidmc")

    return SourceData(subjects, gen())


# ---- VitalDB -----------------------------------------------------------------------------------
def read_vitaldb_track(path: Path) -> np.ndarray:
    """The PLETH samples of an API track: one 500 Hz sample per row, ``Time`` only on the first and last rows.

    Millions of rows per case, so the value is cut off each line rather than going through ``csv``.
    """
    with open_text(path) as fh:
        header = fh.readline().strip().split(",")
        if header[-1] != "SNUADC/PLETH":
            raise ValueError(f"{path} is not a PLETH track (header {header})")
        values = [line.rstrip("\r\n").rpartition(",")[2] or "nan" for line in fh]
    return np.array(values, dtype=np.float64)


def read_vitaldb(folder: Path) -> SourceData:
    tracks = {int(p.name.split(".")[0]): p for p in (folder / "pleth").glob("*.csv.gz")}
    with open_text(folder / "cases.csv.gz") as fh:
        cases = {int(r["caseid"]): r for r in csv.DictReader(fh)}
    subjects, order = [], []
    for cid in sorted(tracks):
        r = cases.get(cid)
        if r is None:
            continue
        row = {"subject": r["subjectid"], "age": r.get("age", ""), "sex": r.get("sex", ""), "caseid": str(cid)}
        row.update({k: r.get(k, "") for k in OUTCOME_FLAGS})
        row.update({k: r.get(k, "") for k in ("department", "optype", "emop", "bmi")})
        subjects.append(row)
        order.append((r["subjectid"], cid))

    def gen() -> Iterator[Recording]:
        for subject, cid in order:
            yield Recording(subject, str(cid), 500.0, path=tracks[cid], parser="vitaldb")

    return SourceData(subjects, gen())


# ---- GalaxyPPG ---------------------------------------------------------------------------------
def _galaxy_open(folder: Path):
    """``(names, opener)`` over the dataset: the unpacked tree when present, else the zip."""
    unpacked = folder / "unpacked"
    if (unpacked / "Dataset" / "Meta.csv").is_file():
        names = [str(p.relative_to(unpacked)) for p in unpacked.rglob("*.csv")]
        return names, lambda name: (unpacked / name).open(encoding="utf-8-sig", newline="")
    archive = zipfile.ZipFile(folder / "GalaxyPPG.zip")
    names = archive.namelist()
    return names, lambda name: io.TextIOWrapper(archive.open(name), encoding="utf-8-sig", newline="")


def read_galaxyppg(folder: Path) -> SourceData:
    names, opener = _galaxy_open(folder)
    meta_name = next(n for n in names if n.endswith("Dataset/Meta.csv"))
    with opener(meta_name) as fh:
        meta = list(csv.DictReader(fh))
    subjects = [{"subject": r["UID"], "age": r["AGE"], "sex": r["GENDER"], "watch_wrist": r.get("GalaxyWatch", "")}
                for r in meta]
    ppg = {n.split("/")[-3]: n for n in names if n.endswith("GalaxyWatch/PPG.csv")}

    def gen() -> Iterator[Recording]:
        for s in subjects:
            name = ppg.get(s["subject"])
            if name is None:
                continue
            with opener(name) as fh:
                rows = list(csv.DictReader(fh))
            t = np.array([_float(r["timestamp"]) for r in rows]) / 1000.0
            x = np.array([_float(r["ppg"]) for r in rows])
            if len(t) < 100:
                continue
            fs = 1.0 / float(np.median(np.diff(t)))
            # the SDK's green PPG falls as blood volume rises: invert so the systolic peak points up
            yield Recording(s["subject"], s["subject"], round(fs, 3), -x)

    return SourceData(subjects, gen())


def _bidmc_pleth(path: Path) -> np.ndarray:
    return _column(path, "PLETH")


PARSERS = {"bidmc": _bidmc_pleth, "vitaldb": read_vitaldb_track}
READERS = {"bidmc": read_bidmc, "vitaldb": read_vitaldb, "galaxyppg": read_galaxyppg}
