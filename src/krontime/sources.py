"""Time-series datasets krontime reads, into ``data/input/timeseries/<name>``.

MetaboNet sits behind the metabo-net.org login, and glucose_forecasting_metabonet already downloads it
(``uv run download-metabonet`` there, about 2.9 GB). Rather than copy that login flow, ``metabonet`` here
is a symbolic link to that checkout's ``data/input/metabonet``. The glucose checkout is found at
``$GLUCOSE_REPO``, else beside this repository.
"""
from __future__ import annotations

import os
from pathlib import Path

from kron_core.download import Source, link_folder
from kron_core.paths import repo_root

DOMAIN = "timeseries"
GLUCOSE_ENV = "GLUCOSE_REPO"


def glucose_repo() -> Path:
    if os.environ.get(GLUCOSE_ENV):
        return Path(os.environ[GLUCOSE_ENV]).expanduser()
    root = repo_root() or Path.cwd()
    return root.parent / "glucose_forecasting_metabonet"


def _link_metabonet(source: Source, folder: Path, force: bool) -> dict:
    # No SOURCE.json is written: the folder is the glucose checkout's, and the link itself is the record.
    target = glucose_repo() / "data" / "input" / "metabonet"
    return link_folder(source, folder, target,
                       f"Download it in that checkout first (`uv run download-metabonet`), or set ${GLUCOSE_ENV}.")


SOURCES: dict[str, Source] = {
    "metabonet": Source(
        "metabonet", DOMAIN,
        "MetaboNet 1.1: CGM glucose, insulin and carbohydrates on a 5-minute grid (train.parquet, test.parquet), "
        "linked from glucose_forecasting_metabonet",
        license="MetaboNet data use terms (metabo-net.org); not ours to republish",
        homepage="https://metabo-net.org/data",
        notes={"format": "parquet", "linked": True},
        fetch=_link_metabonet,
    ),
}
