"""PPG windows in ``data/interim/ppg/<source>/<variant>/`` and the batches training reads from them.

A prepared variant (``kron-ppg prepare``) holds:

* ``windows.parquet``: one row per window: ``subject``, ``recording``, ``start_s`` (in its recording),
  ``hr_bpm``, ``periodicity`` and ``signal``, the robust-scaled window at the variant's rate as a
  fixed-size ``Array(Float16, L)`` (zstd-compressed);
* ``subjects.parquet``: per subject ``subject``, ``age`` (null when unknown), ``sex`` and any outcome
  flags the source carries (VitalDB: ``preop_htn``, ``preop_dm``, ``death_inhosp``, ``asa``); a column
  whose every value is a whole number is stored as ``Int64`` (``age`` always as ``Float64``), one of
  other numbers as ``Float64``, any other as text;
* ``info.json``: the source, rate, window and hop, counts, and the input's ``SOURCE.json`` it was built from;
* optionally ``vitals.parquet`` (:mod:`kron_ppg.vitals`): per-window vitals keyed by ``(recording, start_s)``,
  loaded as :attr:`Windows.extra`.

Both tables are plain parquet, so polars, DuckDB, pyarrow or Hugging Face ``datasets`` read them as they are.

Splits are by subject (:func:`subject_split`), from a hash of the source and subject id, so a person's
windows never sit on both sides and a split does not move when other subjects are added.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import torch

from kron_core.paths import interim_dir
from kron_ppg.watch_sources import refuse_join

DOMAIN = "ppg"


def variant_name(sample_rate: float, seconds: float) -> str:
    return f"fs{sample_rate:g}-w{seconds:g}s"


def variant_dir(source: str, sample_rate: float, seconds: float) -> Path:
    return interim_dir(DOMAIN, source, variant_name(sample_rate, seconds))


@dataclass
class Windows:
    """One prepared variant, loaded: windows (memory-mapped), their subjects, and the subject table."""

    folder: Path
    x: np.ndarray
    subject: np.ndarray
    start_s: np.ndarray
    subjects: dict[str, dict[str, str]]
    info: dict
    extra: dict[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def load(cls, folder: str | Path) -> "Windows":
        folder = Path(folder)
        info = json.loads((folder / "info.json").read_text())
        table = pl.read_parquet(folder / "windows.parquet")
        length = int(round(float(info["seconds"]) * float(info["sample_rate"])))
        x = (table.get_column("signal").to_numpy() if len(table)
             else np.zeros((0, length), np.float16))
        subjects = {r["subject"]: r for r in pl.read_parquet(folder / "subjects.parquet").to_dicts()}
        extra = {}
        if (folder / "vitals.parquet").is_file():
            vitals = pl.read_parquet(folder / "vitals.parquet")
            numeric = [c for c, t in vitals.schema.items() if c not in ("recording", "start_s") and t.is_numeric()]
            joined = table.select("recording", "start_s").join(vitals.select("recording", "start_s", *numeric),
                                                               on=["recording", "start_s"], how="left",
                                                               maintain_order="left")
            extra = {c: joined[c].cast(pl.Float64).fill_null(np.nan).to_numpy() for c in numeric}
        return cls(folder, x, table.get_column("subject").to_numpy().astype(str),
                   table.get_column("start_s").to_numpy(), subjects, info, extra)

    @property
    def sample_rate(self) -> float:
        return float(self.info["sample_rate"])

    def age(self, subject: str) -> float:
        value = self.subjects.get(subject, {}).get("age")
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    def ages(self) -> np.ndarray:
        """Each window's subject's age, NaN where unknown."""
        lookup = {s: self.age(s) for s in self.subjects}
        return np.array([lookup.get(s, float("nan")) for s in self.subject])


def _subject_frame(subjects: list[dict]) -> pl.DataFrame:
    """The subject table with ``subject``, ``age`` and ``sex`` first; numeric columns typed as numbers."""
    names = ["subject", "age", "sex"] + sorted({k for s in subjects for k in s} - {"subject", "age", "sex"})
    raw = pl.DataFrame({n: [None if s.get(n) in (None, "") else str(s[n]) for s in subjects] for n in names},
                       schema={n: pl.String for n in names})
    columns = []
    for n in names:
        col = raw.get_column(n)
        number = col.cast(pl.Float64, strict=False)
        if n == "subject" or number.null_count() != col.null_count():
            columns.append(col)
        elif n != "age" and bool((number.drop_nulls() % 1 == 0).all()):
            columns.append(number.cast(pl.Int64))
        else:
            columns.append(number)
    return pl.DataFrame(columns)


def write_windows(folder: Path, windows: Sequence[np.ndarray], rows: list[dict], subjects: list[dict],
                  info: dict) -> dict:
    """Write a variant; ``windows`` is a list of ``(n_i, L)`` arrays matching ``rows`` in order, each row
    holding ``subject``, ``recording``, ``start_s``, ``hr_bpm`` and ``periodicity``."""
    folder.mkdir(parents=True, exist_ok=True)
    length = int(round(float(info["seconds"]) * float(info["sample_rate"])))
    x = np.concatenate(list(windows)) if len(windows) else np.zeros((0, length))
    if not np.isfinite(x).all() or np.abs(x).max(initial=0.0) > np.finfo(np.float16).max:
        raise ValueError("windows hold non-finite or out-of-float16-range samples; gate them with quality()")
    x = x.astype(np.float16)
    if len(x) != len(rows):
        raise ValueError(f"{len(x)} windows and {len(rows)} rows")
    table = pl.DataFrame({
        "subject": pl.Series([str(r["subject"]) for r in rows], dtype=pl.String),
        "recording": pl.Series([str(r["recording"]) for r in rows], dtype=pl.String),
        "start_s": pl.Series([r["start_s"] for r in rows], dtype=pl.Float64),
        "hr_bpm": pl.Series([r["hr_bpm"] for r in rows], dtype=pl.Float32),
        "periodicity": pl.Series([r["periodicity"] for r in rows], dtype=pl.Float32),
        "signal": pl.Series(x.reshape(len(x), length)).cast(pl.Array(pl.Float16, length)),
    })
    table.write_parquet(folder / "windows.parquet", compression="zstd", row_group_size=16_384)
    _subject_frame(subjects).write_parquet(folder / "subjects.parquet", compression="zstd")
    for stale in ("windows.npy", "windows.csv", "subjects.csv"):  # the format before parquet
        (folder / stale).unlink(missing_ok=True)
    info = {**info, "windows": int(len(x)), "subjects": len(subjects),
            "subjects_with_windows": len({r["subject"] for r in rows})}
    (folder / "info.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def subject_split(subjects: Sequence[str], source: str, val: float = 0.15, test: float = 0.15) -> dict[str, str]:
    """``subject -> "train" | "val" | "test"``, stable under adding subjects: a hash of source and id."""
    out = {}
    for s in subjects:
        u = int(hashlib.sha256(f"{source}/{s}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        out[s] = "test" if u < test else ("val" if u < test + val else "train")
    return out


class Pool:
    """The windows of one or more variants, indexed as one: subjects are keyed ``<source>/<subject>``."""

    def __init__(self, variants: Sequence[Windows]) -> None:
        if not variants:
            raise ValueError("no variants given")
        rates = {v.sample_rate for v in variants}
        lengths = {v.x.shape[1] for v in variants if len(v.x)}
        if len(rates) > 1 or len(lengths) > 1:
            raise ValueError(f"variants differ in rate {sorted(rates)} or window length {sorted(lengths)}")
        refuse_join(v.info["source"] for v in variants)  # a restricted source stays alone
        self.variants = list(variants)
        self.sample_rate = rates.pop()
        self.which = np.concatenate([np.full(len(v.x), k) for k, v in enumerate(variants)])
        self.row = np.concatenate([np.arange(len(v.x)) for v in variants])
        self.subject = np.concatenate([np.array([f"{v.info['source']}/{s}" for s in v.subject], dtype=object)
                                       for v in variants])
        self.age = np.concatenate([v.ages() for v in variants])
        self.table = {f"{v.info['source']}/{k}": row for v in variants for k, row in v.subjects.items()}
        names = sorted({c for v in variants for c in v.extra})
        #: per-window values from each variant's ``vitals.parquet``; NaN where a variant has none
        self.extra = {c: np.concatenate([v.extra.get(c, np.full(len(v.x), np.nan)) for v in variants]) for c in names}

    @classmethod
    def load(cls, folders: Sequence[str | Path]) -> "Pool":
        return cls([Windows.load(f) for f in folders])

    def __len__(self) -> int:
        return len(self.row)

    def window(self, i: int) -> np.ndarray:
        return np.asarray(self.variants[int(self.which[i])].x[int(self.row[i])], dtype=np.float32)

    def split(self, val: float = 0.15, test: float = 0.15) -> dict[str, np.ndarray]:
        """Window indices of each subject-level split."""
        keys = sorted(set(self.subject))
        side = {k: v for k, v in subject_split(keys, "kron-ppg", val, test).items()}
        label = np.array([side[s] for s in self.subject])
        return {name: np.flatnonzero(label == name) for name in ("train", "val", "test")}


class WindowSet(torch.utils.data.Dataset):
    """Windows at ``idx`` of a :class:`Pool` as ``(samples float32 (L,), subject int, age float)``."""

    def __init__(self, pool: Pool, idx: np.ndarray) -> None:
        self.pool = pool
        self.idx = np.asarray(idx)
        keys = sorted(set(pool.subject[self.idx]))
        self.subject_ids = {k: i for i, k in enumerate(keys)}

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        j = int(self.idx[i])
        return (torch.from_numpy(self.pool.window(j)), self.subject_ids[self.pool.subject[j]],
                float(self.pool.age[j]))


class SubjectPairs(torch.utils.data.Sampler[list[int]]):
    """Batches of ``batch_size`` windows made of ``batch_size // per_subject`` subjects, ``per_subject`` each.

    Every subject in a batch has positives for the participant contrast; a subject with fewer windows
    than ``per_subject`` is sampled with replacement. One epoch is ``len(dataset) // batch_size`` batches.
    """

    def __init__(self, subjects: np.ndarray, batch_size: int, per_subject: int = 2, seed: int = 0) -> None:
        if batch_size % per_subject:
            raise ValueError("batch_size must be a multiple of per_subject")
        self.by_subject: dict[str, np.ndarray] = {}
        for i, s in enumerate(subjects):
            self.by_subject.setdefault(str(s), []).append(i)  # type: ignore[arg-type]
        self.by_subject = {k: np.asarray(v) for k, v in self.by_subject.items()}
        self.keys = sorted(self.by_subject)
        self.batch_size, self.per_subject = batch_size, per_subject
        self.n_batches = max(1, len(subjects) // batch_size)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self) -> Iterator[list[int]]:
        n_subj = self.batch_size // self.per_subject
        for _ in range(self.n_batches):
            chosen = self.rng.choice(len(self.keys), size=n_subj, replace=len(self.keys) < n_subj)
            batch: list[int] = []
            for k in chosen:
                pool = self.by_subject[self.keys[k]]
                batch += self.rng.choice(pool, size=self.per_subject, replace=len(pool) < self.per_subject).tolist()
            yield batch
