"""Prepared PPG windows as a Hugging Face dataset: parquet shards per source and split, and a card.

``kron-ppg hub-export vitaldb bidmc galaxyppg`` writes ``data/output/hub/<name>/``:

* ``<source>/<split>-NNNNN-of-MMMMM.parquet``: the rows of ``windows.parquet`` (``subject``,
  ``recording``, ``start_s``, ``hr_bpm``, ``periodicity``, ``signal``) for that split, ``split`` being
  the person-level split kron-ppg trains on (:meth:`kron_ppg.data.Pool.split`), named ``train``,
  ``validation`` and ``test`` as the Hub expects;
* ``<source>/subjects.parquet``: the subject table with the same ``split`` column;
* ``README.md``: the dataset card, with each source's licence, citation and the changes made to it.

Every file is plain parquet, so the folder is uploaded as it is (``hf upload-large-folder``). The
sources are CC BY 4.0 (VitalDB, GalaxyPPG) and ODC-By 1.0 (BIDMC): both allow redistributing adapted
data with credit and forbid adding restrictions, so the dataset is ungated.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import polars as pl

from kron_ppg.data import subject_split, variant_dir

#: kron-ppg's split names -> the Hub's
HUB_SPLITS = {"train": "train", "val": "validation", "test": "test"}

#: per source: licence id on the Hub, what the licence asks for, citation, and where it came from
SOURCE_CARDS: dict[str, dict[str, str]] = {
    "vitaldb": {
        "license": "cc-by-4.0",
        "license_name": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "what": "VitalDB: finger PPG (`SNUADC/PLETH`, 500 Hz) recorded during surgery at Seoul National "
                "University Hospital, with age, sex, ASA class, `preop_htn`, `preop_dm`, `death_inhosp`, "
                "department, operation type, emergency flag and BMI per case. One case per person "
                "(the first by case id, of those with a PLETH track and an age).",
        "origin": "the VitalDB open API (https://vitaldb.net/dataset/), the same data as "
                  "https://physionet.org/content/vitaldb/1.0.0/",
        "citation": "Lee HC, Park Y, Yoon SB, Yang SM, Park D, Jung CW. VitalDB, a high-fidelity multi-parameter "
                    "vital signs database in surgical patients. Sci Data. 2022;9(1):279. "
                    "doi:10.1038/s41597-022-01411-5. Data: Lee HC, Jung CW. VitalDB (version 1.0.0). PhysioNet. "
                    "doi:10.13026/czw8-9p62.",
        "terms": "Credit to VitalDB is required (citation above). The vitaldb.net data use agreement also "
                 "asks users not to attempt to identify any person from the data.",
    },
    "bidmc": {
        "license": "odc-by",
        "license_name": "ODC-By 1.0",
        "license_url": "https://opendatacommons.org/licenses/by/1-0/",
        "what": "BIDMC PPG and Respiration: 53 eight-minute ICU recordings of finger PPG at 125 Hz with age "
                "and sex, from the MIMIC-II matched waveform database. Age 90+ is stored as 90 with "
                "`age_censored` = 1.",
        "origin": "https://physionet.org/content/bidmc/1.0.0/",
        "citation": "Pimentel MAF et al. Toward a robust estimation of respiratory rate from pulse oximeters. "
                    "IEEE Trans Biomed Eng. 2017;64(8):1914-1923. doi:10.1109/TBME.2016.2613124. "
                    "Goldberger AL et al. PhysioBank, PhysioToolkit, and PhysioNet. Circulation. "
                    "2000;101(23):e215-e220.",
        "terms": "Contains information from the BIDMC PPG and Respiration Dataset, which is made available "
                 "under the ODC Attribution License.",
    },
    "galaxyppg": {
        "license": "cc-by-4.0",
        "license_name": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "what": "GalaxyPPG: Samsung Galaxy Watch5 green PPG at 25 Hz from 24 people aged 20-29, worn "
                "through a protocol of rest and activity. The watch's signal is inverted so the systolic "
                "peak points up, as on a finger sensor.",
        "origin": "https://zenodo.org/records/14635823",
        "citation": "Park, Zheng & Lee (2025). GalaxyPPG: a multi-device PPG dataset [data set]. Zenodo. "
                    "doi:10.5281/zenodo.14635823.",
        "terms": "Credit to the GalaxyPPG authors is required (citation above).",
    },
}


def split_of(source: str, subjects: list[str]) -> dict[str, str]:
    """``subject -> train | validation | test``, exactly the split kron-ppg trains and scores on."""
    keys = {s: f"{source}/{s}" for s in subjects}
    side = subject_split(sorted(keys.values()), "kron-ppg")
    return {s: HUB_SPLITS[side[k]] for s, k in keys.items()}


def export_source(source: str, out: Path, sample_rate: float = 25.0, seconds: float = 30.0,
                  shard_rows: int = 200_000, folder: Path | None = None) -> dict:
    """Write one source's shards and subject table under ``out/<source>``; returns its counts."""
    folder = folder or variant_dir(source, sample_rate, seconds)
    if not (folder / "windows.parquet").is_file():
        raise FileNotFoundError(f"{folder} is not prepared: run `uv run kron-ppg prepare {source}`")
    info = json.loads((folder / "info.json").read_text())
    subjects = pl.read_parquet(folder / "subjects.parquet")
    split = split_of(source, subjects["subject"].to_list())
    target = out / source
    target.mkdir(parents=True, exist_ok=True)
    for stale in target.glob("*.parquet"):
        stale.unlink()
    subjects.with_columns(pl.col("subject").replace_strict(split).alias("split")).write_parquet(
        target / "subjects.parquet", compression="zstd")
    windows = pl.scan_parquet(folder / "windows.parquet").with_columns(
        pl.col("subject").replace_strict(split, return_dtype=pl.String).alias("split"))
    counts = {}
    for name in HUB_SPLITS.values():
        part = windows.filter(pl.col("split") == name).drop("split").collect()
        counts[name] = {"windows": len(part), "subjects": part["subject"].n_unique()}
        n_shards = max(1, math.ceil(len(part) / shard_rows))
        for k in range(n_shards):
            part.slice(k * shard_rows, shard_rows).write_parquet(
                target / f"{name}-{k:05d}-of-{n_shards:05d}.parquet", compression="zstd", row_group_size=10_000)
    return {"source": source, "sample_rate": info["sample_rate"], "seconds": info["seconds"],
            "windows": info["windows"], "subjects": info["subjects"], "splits": counts, "prepared": info}


def card(name: str, repo: str, exported: list[dict]) -> str:
    """The dataset card: YAML header (configs, licences) and the text."""
    sources = [e["source"] for e in exported]
    licenses = sorted({SOURCE_CARDS[s]["license"] for s in sources})
    total = sum(e["windows"] for e in exported)
    lines = ["---", "license:"] + [f"- {lic}" for lic in licenses] + [
        "task_categories:", "- time-series-forecasting", "- tabular-regression",
        "tags:", "- ppg", "- photoplethysmography", "- physiology", "- aging-clock", "- wearables", "- biosignals",
        "size_categories:", f"- {'1M<n<10M' if total >= 1_000_000 else '100K<n<1M' if total >= 100_000 else '1K<n<10K'}",
        "configs:"]
    for s in sources:
        lines += [f"- config_name: {s}", "  data_files:"]
        lines += [f"  - split: {h}\n    path: {s}/{h}-*.parquet" for h in HUB_SPLITS.values()]
        lines += [f"- config_name: {s}_subjects", "  data_files:", f"  - split: all\n    path: {s}/subjects.parquet"]
    lines += ["---", ""]
    e0 = exported[0]
    rate, secs = float(e0["sample_rate"]), float(e0["seconds"])
    length = int(round(rate * secs))
    steps = e0["prepared"]["steps"]
    body = f"""# {name}

Finger and wrist photoplethysmography (PPG) cut into quality-checked {secs:g}-second windows at {rate:g} Hz,
ready to train on, with each person's age, sex and, for VitalDB, outcome flags. It is the training data
of [kron-ppg](https://github.com/antonkulaga/kron-fm), a streaming KronLSTM PPG model whose first task
is an age clock, published so others can skip downloading about 18 GB of raw tracks and preparing them.

| config | windows | people | train / validation / test people |
|---|---|---|---|
""" + "\n".join(
        f"| `{e['source']}` | {e['windows']:,} | {e['subjects']:,} | "
        + " / ".join(f"{e['splits'][h]['subjects']:,}" for h in HUB_SPLITS.values()) + " |" for e in exported) + f"""

Each source also has a `<source>_subjects` config: one row per person with the same `split` column.

## Columns

Windows (`<source>` configs):

| column | type | meaning |
|---|---|---|
| `subject` | string | the person, as the source names them (VitalDB `subjectid`) |
| `recording` | string | the recording the window was cut from (VitalDB `caseid`) |
| `start_s` | float64 | where the window starts in its recording, seconds |
| `hr_bpm` | float32 | heart rate estimated from the window's autocorrelation |
| `periodicity` | float32 | the autocorrelation at that beat period (the quality gate keeps >= {e0['prepared']['min_periodicity']}) |
| `signal` | float16[{length}] | the window: {secs:g} s at {rate:g} Hz, robust-scaled (median 0, IQR/1.349 = 1) |

Subjects (`<source>_subjects` configs): `subject`, `age` (years, null when unknown), `sex`, the source's
other per-person fields, and `split`. VitalDB adds `caseid`, `asa`, `preop_htn`, `preop_dm`,
`death_inhosp`, `department`, `optype`, `emop` and `bmi` from its case table.

## How the windows were made (changes to the sources)

For each recording: {"; ".join(steps)}. A window is kept when it is finite, not flat, not clipped and its
autocorrelation at a heart rate of 30-200 bpm is at least {e0['prepared']['min_periodicity']}. At most
{e0['prepared']['max_windows_per_recording']} windows are kept per recording, spread evenly over it, so a
three-hour surgery does not outweigh an eight-minute ICU record. Windows do not overlap.

The split is by person, from a hash of `<source>/<subject>`, so a person's windows are never on two
sides and the split does not move when people are added.

## Loading

```python
import polars as pl
train = pl.read_parquet("hf://datasets/{repo}/vitaldb/train-*.parquet")
x = train["signal"].to_numpy()          # (n, {length}) float16
```

or `datasets.load_dataset("{repo}", "vitaldb", split="train")`.

## What it is not

Age from PPG is a calendar-age regression target. A predicted age, or the gap between it and the real
age, is not a biological age or a diagnosis. VitalDB people were having surgery and BIDMC people were in
intensive care, so neither is a healthy population.

## Sources, licences and credit

"""
    for s in sources:
        c = SOURCE_CARDS[s]
        body += f"""### `{s}` ({c['license_name']}, {c['license_url']})

{c['what']} From {c['origin']}.

{c['terms']}

> {c['citation']}

"""
    body += """This dataset adds no terms of its own: each source's rows are under that source's licence, with
the changes listed above.
"""
    return "\n".join(lines) + body


def export_hub(sources: list[str], out: Path, repo: str, name: str, sample_rate: float = 25.0,
               seconds: float = 30.0, folders: dict[str, Path] | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    exported = [export_source(s, out, sample_rate, seconds, folder=(folders or {}).get(s)) for s in sources]
    (out / "README.md").write_text(card(name, repo, exported))
    summary = {e["source"]: {"windows": e["windows"], "subjects": e["subjects"], "splits": e["splits"]} for e in exported}
    (out / "export.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
