"""PPG datasets kron-ppg can download, into ``data/input/ppg/<name>``, each with a ``SOURCE.json``.

Only sets that carry a per-subject age and download over plain HTTPS without credentials are here
(``reports/PPG foundation models and aging clocks 2026.md`` lists the rest and why they are not):

* ``bidmc``: 53 ICU recordings of 8 min, finger PPG at 125 Hz, ages 19-90+ (ODC-By 1.0). Small: the
  smoke test and an external check.
* ``vitaldb``: surgical finger PPG (``SNUADC/PLETH``, 500 Hz) from the VitalDB open API, with age, sex
  and outcome flags for 6,388 cases; 6,157 have the PLETH track, about 4 MB gzipped per case. The
  only open set large enough to train an age clock. A first download takes ``KRON_PPG_VITALDB_CASES``
  cases (default 200, one per person, spread over case ids); ``kron-ppg download vitaldb --cases N``
  takes more, and a later run only adds what is missing. CC BY 4.0, both through the API (the
  vitaldb.net data use agreement: share and adapt "for any purpose, even commercially", with credit
  to VitalDB, a licence link and a note of changes; no attempt to re-identify anyone) and as
  PhysioNet's full copy (95 GB of ``.vital`` files).
* ``galaxyppg``: Galaxy Watch5 25 Hz green PPG (plus E4 64 Hz) from 24 people aged 20-29 (CC BY 4.0).
  Narrow in age, so it is wrist pretraining data and a device check, not a clock set. When the
  pulse-exploration checkout already holds it, the folder is linked rather than downloaded again.
"""
from __future__ import annotations

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl

from kron_core.download import (RemoteFile, Source, download_file, link_folder, read_manifest, sha256,
                                write_manifest)
from kron_core.paths import repo_root

DOMAIN = "ppg"

BIDMC = "https://physionet.org/files/bidmc/1.0.0/bidmc_csv"
VITALDB_API = "https://api.vitaldb.net"
VITALDB_TRACK = "SNUADC/PLETH"
#: Further VitalDB tracks, by group, that match what a Galaxy Watch can also measure: its single-lead ECG
#: spot check (ECG lead II), and its heart rate, SpO2, blood pressure and temperature readings (the
#: Solar8000 monitor's numbers, with the arterial line's beat-by-beat pressure where there is one).
#: Each lands in ``tracks/<slug>/<caseid>.csv.gz``, for the same cases as the PLETH tracks.
VITALDB_EXTRAS: dict[str, dict[str, str]] = {
    "ecg": {"SNUADC/ECG_II": "ecg_ii"},
    "vitals": {"Solar8000/HR": "hr", "Solar8000/PLETH_SPO2": "spo2", "Solar8000/BT": "bt",
               "Solar8000/NIBP_SBP": "nibp_sbp", "Solar8000/NIBP_DBP": "nibp_dbp", "Solar8000/NIBP_MBP": "nibp_mbp",
               "Solar8000/ART_SBP": "art_sbp", "Solar8000/ART_DBP": "art_dbp", "Solar8000/ART_MBP": "art_mbp"},
}
VITALDB_CASES_ENV = "KRON_PPG_VITALDB_CASES"
VITALDB_DEFAULT_CASES = 200
GALAXYPPG_URL = "https://zenodo.org/records/14635823/files/GalaxyPPG.zip?download=1"
PULSE_ENV = "PULSE_REPO"


# ---- VitalDB ---------------------------------------------------------------------------------
def vitaldb_cases_wanted(n: int, trks: Path, cases: Path) -> list[tuple[int, str]]:
    """``n`` ``(caseid, tid)`` pairs with a PLETH track: one case per person (the first by case id),
    taken in a fixed pseudo-random order of case ids so any ``n`` spreads over the whole set."""
    tracks = (pl.read_csv(trks, infer_schema=False)
              .filter(pl.col("tname") == VITALDB_TRACK)
              .select(pl.col("caseid").cast(pl.Int64), "tid")
              .unique("caseid", keep="first"))
    first = (pl.read_csv(cases, infer_schema=False)
             .filter(pl.col("age").fill_null("") != "")
             .select(pl.col("caseid").cast(pl.Int64), "subjectid")
             .join(tracks, on="caseid", how="inner", maintain_order="left")
             .unique("subjectid", keep="first", maintain_order=True))
    tid = dict(zip(first["caseid"].to_list(), first["tid"].to_list(), strict=True))
    chosen = sorted(tid, key=lambda c: hashlib.sha256(f"vitaldb/{c}".encode()).hexdigest())
    return [(c, tid[c]) for c in chosen[:n]]


def vitaldb_extra_tracks(cases: list[int], trks: Path, groups: list[str]) -> list[tuple[int, str, str]]:
    """``(caseid, tid, slug)`` for every track of ``groups`` (keys of :data:`VITALDB_EXTRAS`) the cases have."""
    unknown = [g for g in groups if g not in VITALDB_EXTRAS]
    if unknown:
        raise ValueError(f"unknown VitalDB track groups {unknown}; one of {sorted(VITALDB_EXTRAS)}")
    slugs = {name: slug for g in groups for name, slug in VITALDB_EXTRAS[g].items()}
    rows = (pl.read_csv(trks, infer_schema=False)
            .filter(pl.col("tname").is_in(list(slugs)))
            .with_columns(pl.col("caseid").cast(pl.Int64))
            .filter(pl.col("caseid").is_in(cases))
            .unique(["caseid", "tname"], keep="first")
            .sort(["caseid", "tname"]))
    return [(c, t, slugs[n]) for c, t, n in zip(rows["caseid"], rows["tid"], rows["tname"], strict=True)]


def fetch_vitaldb(source: Source, folder: Path, force: bool, cases: int | None = None,
                  extras: tuple[str, ...] = (), workers: int = 1) -> dict:
    """The case table, the track list, ``cases`` PLETH tracks as ``pleth/<caseid>.csv.gz`` and, for the
    same cases, the tracks of each ``extras`` group as ``tracks/<slug>/<caseid>.csv.gz``; ``workers``
    downloads run at once (the API serves one track per request)."""
    n = cases if cases is not None else int(os.environ.get(VITALDB_CASES_ENV, VITALDB_DEFAULT_CASES))
    folder.mkdir(parents=True, exist_ok=True)
    old = {e["file"]: e for e in (read_manifest(folder) or {}).get("files", [])}

    def get(url: str, name: str) -> dict:
        dest = folder / name
        if dest.exists() and not force:
            if name in old and old[name].get("bytes") == dest.stat().st_size:
                return old[name]
            # A complete file from a run stopped before it wrote the manifest: download_file renames
            # the .part file only once the whole body has arrived, so a file under its own name is whole.
            return {"file": name, "url": url, "bytes": dest.stat().st_size, "sha256": sha256(dest),
                    "downloaded_at": None}
        return {**download_file(url, dest), "file": name}

    entries = [get(f"{VITALDB_API}/cases", "cases.csv.gz"), get(f"{VITALDB_API}/trks", "trks.csv.gz")]
    wanted = vitaldb_cases_wanted(n, folder / "trks.csv.gz", folder / "cases.csv.gz")
    jobs = [(f"{VITALDB_API}/{tid}", f"pleth/{cid:04d}.csv.gz") for cid, tid in wanted]
    jobs += [(f"{VITALDB_API}/{tid}", f"tracks/{slug}/{cid:04d}.csv.gz")
             for cid, tid, slug in vitaldb_extra_tracks([c for c, _ in wanted], folder / "trks.csv.gz", list(extras))]
    done = 0

    def fetch(job: tuple[str, str]) -> dict:
        nonlocal done
        entry = get(*job)
        done += 1
        if done % 100 == 0 or done == len(jobs):
            print(f"  [{done}/{len(jobs)}] {job[1]}", flush=True)
        return entry

    with ThreadPoolExecutor(max(1, workers)) as pool:
        entries += list(pool.map(fetch, jobs))
    # keep tracks a larger earlier download fetched
    have = {e["file"] for e in entries}
    entries += [e for name, e in old.items() if name not in have and (folder / name).exists()]
    return write_manifest(folder, source, entries, {"vitaldb_track": VITALDB_TRACK, "cases_requested": n,
                                                    "extras": sorted(set(extras))})


# ---- GalaxyPPG -------------------------------------------------------------------------------
def pulse_repo() -> Path:
    if os.environ.get(PULSE_ENV):
        return Path(os.environ[PULSE_ENV]).expanduser()
    return (repo_root() or Path.cwd()).parent / "pulse-exploration"


def fetch_galaxyppg(source: Source, folder: Path, force: bool) -> dict:
    local = pulse_repo() / "data" / "external" / "galaxyppg"
    if (local / "GalaxyPPG.zip").is_file() and not force:
        return link_folder(source, folder, local, "")
    folder.mkdir(parents=True, exist_ok=True)
    entry = download_file(GALAXYPPG_URL, folder / "GalaxyPPG.zip", source.files[0].size)
    return write_manifest(folder, source, [entry])


SOURCES: dict[str, Source] = {
    s.name: s for s in (
        Source("bidmc", DOMAIN,
               "BIDMC PPG and Respiration: 53 ICU recordings of 8 min, finger PPG 125 Hz, age and sex per record",
               tuple(RemoteFile(f"{BIDMC}/bidmc_{i:02d}_{kind}") for i in range(1, 54)
                     for kind in ("Signals.csv", "Fix.txt")),
               "ODC-By 1.0",
               "Pimentel et al. (2016) Toward a robust estimation of respiratory rate from pulse oximeters. "
               "IEEE Trans Biomed Eng 64(8):1914-1923. Goldberger et al. (2000) PhysioNet. Circulation 101:e215.",
               "https://physionet.org/content/bidmc/1.0.0/",
               {"format": "csv", "fs": 125, "signal_column": "PLETH", "age_file": "bidmc_NN_Fix.txt",
                "approx_mb": 70}),
        Source("vitaldb", DOMAIN,
               "VitalDB surgical finger PPG (SNUADC/PLETH 500 Hz) with age, sex and outcome flags; "
               f"{VITALDB_DEFAULT_CASES} cases by default (KRON_PPG_VITALDB_CASES, or kron-ppg download --cases)",
               (), "CC BY 4.0 (vitaldb.net data use agreement; the PhysioNet copy too)",
               "Lee et al. (2022) VitalDB, a high-fidelity multi-parameter vital signs database in surgical "
               "patients. Sci Data 9:279.",
               "https://vitaldb.net/dataset/",
               {"format": "csv.gz", "fs": 500, "track": VITALDB_TRACK, "cases_with_pleth": 6157,
                "approx_mb_per_case": 4.3, "approx_mb": round(4.3 * VITALDB_DEFAULT_CASES)},
               fetch=fetch_vitaldb),
        Source("galaxyppg", DOMAIN,
               "GalaxyPPG: Galaxy Watch5 25 Hz green PPG, E4 BVP 64 Hz and Polar H10 ECG from 24 people aged 20-29; "
               "linked from pulse-exploration when present",
               (RemoteFile(GALAXYPPG_URL, "GalaxyPPG.zip", 481_594_233),),
               "CC BY 4.0",
               "Park, Zheng & Lee (2025) GalaxyPPG: a multi-device PPG dataset. Sci Data. Zenodo 14635823.",
               "https://zenodo.org/records/14635823",
               {"format": "zip", "fs": 25, "invert": True, "approx_mb": 482},
               fetch=fetch_galaxyppg),
    )
}
