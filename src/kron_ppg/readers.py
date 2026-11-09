"""Reading each downloaded PPG source into recordings: a subject, a raw signal at its own rate, and an age.

Nothing here filters or resamples; :mod:`kron_ppg.prepare` does that the same way for every source.
Tables are read with polars (gzip-compressed CSV directly); files are only parsed as CSV or read from a
zip: nothing downloaded is executed or unpickled.
"""
from __future__ import annotations

import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from kron_ppg.clock import OUTCOME_FLAGS
from kron_ppg.sources import VITALDB_TRACK


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


def _values(series: pl.Series) -> np.ndarray:
    """A numeric column as float64 (text cells stripped of spaces first), nulls and unparsable cells as NaN."""
    if series.dtype == pl.String:
        series = series.str.strip_chars()
    return series.cast(pl.Float64, strict=False).fill_null(float("nan")).to_numpy()


def _column(source: Path | bytes, name: str) -> np.ndarray:
    """One numeric column of a CSV (header names stripped of spaces), empty cells as NaN."""
    df = pl.read_csv(source, infer_schema=False)
    df = df.rename({c: c.strip() for c in df.columns})
    if name not in df.columns:
        raise ValueError(f"no {name!r} column (columns: {df.columns})")
    return _values(df.get_column(name))


def _table(source: Path | bytes) -> list[dict[str, str]]:
    """A CSV as rows of strings (empty cells as ""), the way the subject table keeps them."""
    if isinstance(source, bytes):
        source = source.removeprefix(b"\xef\xbb\xbf")
    return pl.read_csv(source, infer_schema=False).fill_null("").to_dicts()


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

    Millions of rows per case: polars parses the gzip-compressed CSV in native code (about 8x the
    pure-Python line parser it replaced).
    """
    df = pl.read_csv(path, schema_overrides={VITALDB_TRACK: pl.Float64})
    if df.columns[-1] != VITALDB_TRACK:
        raise ValueError(f"{path} is not a PLETH track (columns {df.columns})")
    return _values(df.get_column(VITALDB_TRACK))


#: preoperative labs kept from VitalDB's case table: haemoglobin, glucose, creatinine, albumin, urea
#: nitrogen, sodium (age-related, and things a person's health shows in)
VITALDB_LABS = ("preop_hb", "preop_gluc", "preop_cr", "preop_alb", "preop_bun", "preop_na")


def read_vitaldb(folder: Path) -> SourceData:
    tracks = {int(p.name.split(".")[0]): p for p in (folder / "pleth").glob("*.csv.gz")}
    cases = {int(r["caseid"]): r for r in _table(folder / "cases.csv.gz")}
    subjects, order = [], []
    for cid in sorted(tracks):
        r = cases.get(cid)
        if r is None:
            continue
        row = {"subject": r["subjectid"], "age": r.get("age", ""), "sex": r.get("sex", ""), "caseid": str(cid)}
        row.update({k: r.get(k, "") for k in OUTCOME_FLAGS})
        row.update({k: r.get(k, "") for k in ("department", "optype", "emop", "bmi", "height", "weight")})
        row.update({k: r.get(k, "") for k in VITALDB_LABS})
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
        return names, lambda name: (unpacked / name).read_bytes()
    archive = zipfile.ZipFile(folder / "GalaxyPPG.zip")
    return archive.namelist(), archive.read


def read_galaxyppg(folder: Path) -> SourceData:
    names, opener = _galaxy_open(folder)
    meta_name = next(n for n in names if n.endswith("Dataset/Meta.csv"))
    meta = _table(opener(meta_name))
    subjects = [{"subject": r["UID"], "age": r["AGE"], "sex": r["GENDER"], "watch_wrist": r.get("GalaxyWatch", "")}
                for r in meta]
    ppg = {n.split("/")[-3]: n for n in names if n.endswith("GalaxyWatch/PPG.csv")}

    def gen() -> Iterator[Recording]:
        for s in subjects:
            name = ppg.get(s["subject"])
            if name is None:
                continue
            data = opener(name).removeprefix(b"\xef\xbb\xbf")
            t = _column(data, "timestamp") / 1000.0
            x = _column(data, "ppg")
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
