"""Where kron-fm keeps its data: ``data/input``, ``data/interim`` and ``data/output``.

* ``data/input/<domain>/<source>/``: downloads exactly as published, never edited, each folder with a
  ``SOURCE.json`` that records where every file came from, its size, SHA-256 and licence.
* ``data/interim/<domain>/<source>/``: what the code derives from an input, rebuildable from it (packed
  token rows, splits).
* ``data/output/``: what training and export produce: ``runs/<model>/<run>/`` (checkpoints, metrics,
  ONNX graphs) and ``eval/``.

The root is ``$KRON_FM_DATA`` when set, otherwise ``data/`` beside this repository's ``pyproject.toml``,
otherwise ``data/`` under the working directory, so the same command works from any folder and a machine
with a larger disk can point the whole tree elsewhere.
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path

DATA_ENV = "KRON_FM_DATA"


def repo_root() -> Path | None:
    """The kron-fm checkout this module is installed from (editable install), or ``None``."""
    root = Path(__file__).resolve().parents[2]
    return root if (root / "pyproject.toml").is_file() and (root / "src" / "kron_core").is_dir() else None


def data_root() -> Path:
    if os.environ.get(DATA_ENV):
        return Path(os.environ[DATA_ENV]).expanduser()
    root = repo_root()
    return (root if root is not None else Path.cwd()) / "data"


def input_dir(*parts: str) -> Path:
    return data_root().joinpath("input", *parts)


def interim_dir(*parts: str) -> Path:
    return data_root().joinpath("interim", *parts)


def output_dir(*parts: str) -> Path:
    return data_root().joinpath("output", *parts)


def runs_dir(model: str) -> Path:
    return output_dir("runs", model)


def run_name(*words: str) -> str:
    """``word-word-YYYYmmddTHHMMSSZ``: the words that say what the run is, then its UTC start time."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = "-".join(re.sub(r"[^A-Za-z0-9_.-]+", "_", w).strip("_") for w in words if w)
    return f"{slug}-{stamp}" if slug else stamp
