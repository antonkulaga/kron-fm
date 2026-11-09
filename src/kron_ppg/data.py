"""PPG windows in ``data/interim/ppg/<source>/<variant>/`` and the batches training reads from them.

A prepared variant (``kron-ppg prepare``) holds:

* ``windows.npy``: ``float16`` ``(N, L)``, robust-scaled windows at the variant's rate;
* ``windows.csv``: per window ``subject``, ``start_s`` (in its recording), ``hr_bpm``, ``periodicity``;
* ``subjects.csv``: per subject ``subject``, ``age`` (empty when unknown), ``sex`` and any outcome
  flags the source carries (VitalDB: ``preop_htn``, ``preop_dm``, ``death_inhosp``, ``asa``);
* ``info.json``: the source, rate, window and hop, counts, and the input's ``SOURCE.json`` it was built from.

Splits are by subject (:func:`subject_split`), from a hash of the source and subject id, so a person's
windows never sit on both sides and a split does not move when other subjects are added.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from kron_core.paths import interim_dir

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

    @classmethod
    def load(cls, folder: str | Path) -> "Windows":
        folder = Path(folder)
        x = np.load(folder / "windows.npy", mmap_mode="r")
        with (folder / "windows.csv").open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        with (folder / "subjects.csv").open(newline="") as fh:
            subjects = {r["subject"]: r for r in csv.DictReader(fh)}
        info = json.loads((folder / "info.json").read_text())
        if len(rows) != len(x):
            raise ValueError(f"{folder}: {len(x)} windows and {len(rows)} rows in windows.csv")
        return cls(folder, x, np.array([r["subject"] for r in rows]),
                   np.array([float(r["start_s"]) for r in rows]), subjects, info)

    @property
    def sample_rate(self) -> float:
        return float(self.info["sample_rate"])

    def age(self, subject: str) -> float:
        value = self.subjects.get(subject, {}).get("age", "")
        try:
            return float(value)
        except ValueError:
            return float("nan")

    def ages(self) -> np.ndarray:
        """Each window's subject's age, NaN where unknown."""
        lookup = {s: self.age(s) for s in self.subjects}
        return np.array([lookup.get(s, float("nan")) for s in self.subject])


def write_windows(folder: Path, windows: Sequence[np.ndarray], rows: list[dict], subjects: list[dict],
                  info: dict) -> dict:
    """Write a variant; ``windows`` is a list of ``(n_i, L)`` arrays matching ``rows`` in order."""
    folder.mkdir(parents=True, exist_ok=True)
    x = np.concatenate(list(windows)) if len(windows) else np.zeros((0, 0))
    np.save(folder / "windows.npy", x.astype(np.float16))
    with (folder / "windows.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["subject", "start_s", "hr_bpm", "periodicity"])
        writer.writeheader()
        writer.writerows(rows)
    columns = ["subject", "age", "sex"] + sorted({k for s in subjects for k in s} - {"subject", "age", "sex"})
    with (folder / "subjects.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(subjects)
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
        self.variants = list(variants)
        self.sample_rate = rates.pop()
        self.which = np.concatenate([np.full(len(v.x), k) for k, v in enumerate(variants)])
        self.row = np.concatenate([np.arange(len(v.x)) for v in variants])
        self.subject = np.concatenate([np.array([f"{v.info['source']}/{s}" for s in v.subject], dtype=object)
                                       for v in variants])
        self.age = np.concatenate([v.ages() for v in variants])
        self.table = {f"{v.info['source']}/{k}": row for v in variants for k, row in v.subjects.items()}

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
