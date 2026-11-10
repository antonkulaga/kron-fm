"""Wrist and smartwatch PPG datasets, for pretraining kron-ppg on wrist data and bridging finger to wrist.

Each lands in ``data/input/ppg/<name>`` with a ``SOURCE.json``, like the sets in :mod:`kron_ppg.sources`,
and only sets that download without an account, a data-use agreement or a click-through licence are here
(``research_notes/Wrist PPG datasets 2026-11-10.md`` lists the rest and what each needs):

* ``galaxy_active2``: Samsung Galaxy Watch Active 2 raw green PPG (``raw_data/ppg.csv.gz`` in a 19.7 GB zip, with
  accelerometer, gyroscope, light and the watch's heart rate), four weeks free-living per person, 49 people aged
  21-43, about 33,600 h, with sleep diaries and surveys (Baigutanova et al. 2025). The largest open raw wrist set
  known here, and a Samsung one. CC BY 4.0.
* ``vilnius_af``: 39 patients with suspected atrial fibrillation, 5 to 8 days each (261 days), wrist green PPG
  (100 Hz, a research wristband) aligned to Holter ECG (500 Hz), with annotated AF episodes and age, sex,
  height, weight, comorbidities and medication per patient (``subject_info.xlsx``). Zenodo, newest version
  (2026-11-03). CC BY-NC-SA 4.0.
* ``multisite_ppg``: earring, ring, watch and necklace PPG (green and IR, 100 Hz) with Polar H10 ECG, 20 people
  aged 20-51; the raw recordings only (the windowed copy and the sample folder repeat them). CC BY-NC 4.0.
* ``wildppg``: long free-living recordings, PPG at forehead, sternum, wrist and ankle (128 Hz) with lead-I
  ECG, 16 people, about 216 h; one MATLAB file per person. CC BY-NC-SA 4.0.
* ``wf_ppg``: wrist and fingertip PPG on the same seconds (128 Hz), with ECG, SpO2 and blood pressure, under
  varied contact pressure; 27 people aged 20-32. CC BY 4.0.
* ``ptt_ppg``: PhysioNet pulse transit time PPG: green, red and IR PPG at two finger sites with ECG, sitting,
  walking and running, 22 people. ODbL 1.0.
* ``ppg_dalia``: Empatica E4 wrist BVP (64 Hz) and chest ECG over about 2.5 h of daily activities each,
  15 people, with age in each ``S*_quest.csv``. CC BY 4.0. The archive also holds Python pickles: never
  unpickle them; read the E4 CSVs and the questionnaire.
* ``utsa_ppg``: wrist green PPG and fingertip red and IR PPG on the same seconds (rate read from the files), with
  accelerometer and Holter ECG RR intervals, 12 people sitting, sleeping and at office work, 30 min or more each.
  The repository is MIT; it states no separate data licence.
* ``wesad``: Empatica E4 wrist BVP (64 Hz) and RespiBAN chest ECG in a stress study, 15 people, ages in each
  ``S*_readme.txt``; pickles again, never unpickled.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from kron_core.download import (USER_AGENT, RemoteFile, Source, download_file, hf_dataset_url, read_manifest,
                                write_manifest)

DOMAIN = "ppg"

GALAXY_ACTIVE2 = "https://ndownloader.figshare.com/files"
#: figshare 10.6084/m9.figshare.28509740 version 1: file id, name, size (listed 2026-11-10).
GALAXY_ACTIVE2_FILES = ((52669388, "README.docx", 13_558), (52669394, "survey.csv", 3_562),
                        (52669397, "sleep_diary.csv", 138_047), (52669391, "sensor_hrv.csv", 35_318_568),
                        (52669385, "sensor_hrv_filtered.csv", 17_288_962),
                        (52672928, "raw_data.zip", 19_740_546_963))
#: Zenodo record of the newest version (2026-11-03) of the Vilnius long-term ECG and wrist PPG AF set.
VILNIUS_AF_RECORD = 23921320
MULTISITE_REPO = "snowballlab/Multisite-PPG"
WILDPPG_DAV = "https://polybox.ethz.ch/public.php/dav/files/NWTuyNojU7aya1y/data"
#: The 16 participant files of the WildPPG share, with their sizes as listed on 2026-11-10.
WILDPPG_FILES = {
    "an0": 1_114_535_248, "e61": 1_116_745_592, "fex": 1_209_388_200, "k2s": 1_377_390_928,
    "kjd": 1_210_794_920, "l38": 1_204_565_328, "n31": 1_197_732_472, "ngh": 1_328_356_472,
    "p5d": 1_307_858_600, "p9p": 1_130_812_792, "qm9": 1_190_497_912, "ssx": 1_145_281_912,
    "trh": 1_301_829_752, "tz8": 1_313_284_472, "u7y": 1_198_536_312, "w4p": 1_257_015_672,
}
WF_PPG_URL = "https://ndownloader.figshare.com/files/50956401"
PTT_PPG = "https://physionet.org/files/pulse-transit-time-ppg/1.1.0"
DALIA_URL = "https://archive.ics.uci.edu/static/public/495/ppg+dalia.zip"
UTSA_URL = "https://drive.usercontent.google.com/download?id=1NP_mngn7YccQO09FdaqQ001wwTMmE0ks&export=download&confirm=t"
WESAD_URL = "https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download"


def _get_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode()


def fetch_tree(source: Source, folder: Path, force: bool, files: list[tuple[str, str, int | None]],
               extra: dict | None = None) -> dict:
    """Download ``(url, relative path, size)`` files into ``folder``, keeping sub-folders, skipping files the
    manifest already records at their size; write ``SOURCE.json`` after every file, so a stopped run resumes."""
    folder.mkdir(parents=True, exist_ok=True)
    old = {e["file"]: e for e in (read_manifest(folder) or {}).get("files", [])}
    entries: dict[str, dict] = {}
    for i, (url, rel, size) in enumerate(files, 1):
        dest = folder / rel
        previous = old.get(rel)
        if dest.exists() and not force and previous and previous.get("bytes") == dest.stat().st_size:
            entries[rel] = previous
            continue
        print(f"  [{i}/{len(files)}] {rel}", flush=True)
        entries[rel] = {**download_file(url, dest, size), "file": rel}
        write_manifest(folder, source, list(entries.values()), extra)
    return write_manifest(folder, source, list(entries.values()), extra)


def multisite_files() -> list[tuple[str, str, int | None]]:
    """The README, the Croissant record and every file under ``raw_data/`` of the Hugging Face repo."""
    tree = json.loads(_get_text(f"https://huggingface.co/api/datasets/{MULTISITE_REPO}/tree/main?recursive=true"))
    keep = [t for t in tree if t["type"] == "file"
            and (t["path"] in ("README.md", "croissant.json") or t["path"].startswith("raw_data/"))]
    return [(hf_dataset_url(MULTISITE_REPO, t["path"]), t["path"], t.get("size")) for t in keep]


def fetch_multisite(source: Source, folder: Path, force: bool) -> dict:
    return fetch_tree(source, folder, force, multisite_files(), {"subset": "README, croissant.json, raw_data/"})


def ptt_files() -> list[tuple[str, str, int | None]]:
    """Every file PhysioNet's ``SHA256SUMS.txt`` lists for the project (WFDB records and their CSV copies)."""
    names = [line.split(maxsplit=1)[1] for line in _get_text(f"{PTT_PPG}/SHA256SUMS.txt").splitlines() if line.strip()]
    return [(f"{PTT_PPG}/{n}", n, None) for n in ["SHA256SUMS.txt", *names]]


def fetch_ptt(source: Source, folder: Path, force: bool) -> dict:
    return fetch_tree(source, folder, force, ptt_files())


def zenodo_files(record: int) -> list[tuple[str, str, int | None]]:
    """Every file of a Zenodo record: tables and texts first, then PPG, then the rest (ECG), by name."""
    files = json.loads(_get_text(f"https://zenodo.org/api/records/{record}"))["files"]
    files.sort(key=lambda f: (f["key"].endswith(".mat"), "PPG" not in f["key"], f["key"]))
    return [(f"https://zenodo.org/records/{record}/files/{f['key']}?download=1", f["key"], f["size"]) for f in files]


def fetch_vilnius_af(source: Source, folder: Path, force: bool) -> dict:
    return fetch_tree(source, folder, force, zenodo_files(VILNIUS_AF_RECORD), {"zenodo_record": VILNIUS_AF_RECORD})


def fetch_files(source: Source, folder: Path, force: bool) -> dict:
    """The source's own file list, one by one, with ``SOURCE.json`` written after each (large single files)."""
    return fetch_tree(source, folder, force, [(f.url, f.filename, f.size) for f in source.files])


SOURCES: dict[str, Source] = {
    s.name: s for s in (
        Source("galaxy_active2", DOMAIN,
               "Galaxy Watch Active 2 raw green PPG with motion sensors, four weeks free-living, 49 people aged "
               "21-43, about 33,600 h, sleep diaries and surveys",
               tuple(RemoteFile(f"{GALAXY_ACTIVE2}/{i}", n, size) for i, n, size in GALAXY_ACTIVE2_FILES),
               "CC BY 4.0",
               "Baigutanova et al. (2025) A continuous real-world dataset comprising wearable-based heart rate "
               "variability alongside sleep diaries. Sci Data, doi 10.1038/s41597-025-05801-3. "
               "figshare 10.6084/m9.figshare.28509740.",
               "https://doi.org/10.6084/m9.figshare.28509740",
               {"format": "zip of csv.gz", "sites": ["wrist"], "device": "Samsung Galaxy Watch Active 2",
                "approx_mb": 19_793},
               fetch=fetch_files),
        Source("vilnius_af", DOMAIN,
               "Long-term wrist PPG (100 Hz green) with Holter ECG and annotated AF, 39 patients with suspected AF, "
               "5-8 days each; age, sex, comorbidities and medication per patient",
               (), "CC BY-NC-SA 4.0 (non-commercial, share-alike; research use is fine)",
               "Bacevicius, Plusciauskaite, Abramikas, Badaras, Butkuviene et al. Long-term electrocardiogram and "
               f"wrist-based photoplethysmogram recordings with annotated atrial fibrillation episodes. Zenodo "
               f"{VILNIUS_AF_RECORD} (version of 2026-11-03).",
               f"https://zenodo.org/records/{VILNIUS_AF_RECORD}",
               {"format": "mat", "fs": 100, "ecg_fs": 500, "sites": ["wrist"], "load": "scipy.io.loadmat",
                "approx_mb": 62_707},
               fetch=fetch_vilnius_af),
        Source("multisite_ppg", DOMAIN,
               "Multisite-PPG: earring, ring, watch and necklace PPG (green and IR, 100 Hz) with Polar H10 ECG, "
               "20 people aged 20-51; raw recordings (.npz) only",
               (), "CC BY-NC 4.0 (non-commercial; research use is fine)",
               "snowballlab, Multisite-PPG (2025), Hugging Face datasets snowballlab/Multisite-PPG.",
               f"https://huggingface.co/datasets/{MULTISITE_REPO}",
               {"format": "npz", "fs": 100, "sites": ["earring", "ring", "watch", "necklace"],
                "load": "numpy.load(path, allow_pickle=False)", "approx_mb": 10_000},
               fetch=fetch_multisite),
        Source("wildppg", DOMAIN,
               "WildPPG: free-living PPG at forehead, sternum, wrist and ankle (128 Hz) with lead-I ECG, "
               "16 people, about 216 h; one .mat per person",
               tuple(RemoteFile(f"{WILDPPG_DAV}/WildPPG_Part_{k}.mat", f"WildPPG_Part_{k}.mat", n)
                     for k, n in WILDPPG_FILES.items()),
               "CC BY-NC-SA 4.0 (non-commercial, share-alike; research use is fine)",
               "Meier, Demirel & Holz (2024) WildPPG: a real-world PPG dataset of long continuous recordings. "
               "NeurIPS Datasets and Benchmarks.",
               "https://siplab.org/projects/WildPPG",
               {"format": "mat", "fs": 128, "sites": ["forehead", "sternum", "wrist", "ankle"],
                "load": "scipy.io.loadmat", "approx_mb": round(sum(WILDPPG_FILES.values()) / 1e6)},
               fetch=fetch_files),
        Source("wf_ppg", DOMAIN,
               "WF-PPG: wrist and fingertip PPG on the same seconds (128 Hz) with ECG, SpO2 and blood pressure "
               "under varied contact pressure, 27 people aged 20-32",
               (RemoteFile(WF_PPG_URL, "WF-PPG.zip", 384_211_698),),
               "CC BY 4.0",
               "WF-PPG: a wrist-finger dual-channel dataset for studying the impact of contact pressure on PPG "
               "morphology (2025). Sci Data. figshare 10.6084/m9.figshare.27011998.",
               "https://doi.org/10.6084/m9.figshare.27011998",
               {"format": "zip", "fs": 128, "sites": ["wrist", "finger"], "approx_mb": 384},
               fetch=fetch_files),
        Source("ptt_ppg", DOMAIN,
               "PhysioNet pulse transit time PPG: green, red and IR PPG at two finger sites with ECG, sitting, "
               "walking and running, 22 people",
               (), "ODbL 1.0",
               "Mehrgardt, Khushi, Poon & Withana. Pulse Transit Time PPG Dataset (version 1.1.0). PhysioNet. "
               "Goldberger et al. (2000) PhysioNet. Circulation 101:e215.",
               "https://physionet.org/content/pulse-transit-time-ppg/1.1.0/",
               {"format": "wfdb+csv", "fs": 500, "sites": ["finger proximal", "finger distal"], "approx_mb": 2_900},
               fetch=fetch_ptt),
        Source("ppg_dalia", DOMAIN,
               "PPG-DaLiA: Empatica E4 wrist BVP (64 Hz) and chest ECG over daily activities, 15 people, age per "
               "person; the archive's pickles are never unpickled",
               (RemoteFile(DALIA_URL, "ppg_dalia.zip"),),
               "CC BY 4.0",
               "Reiss, Indlekofer, Schmidt & Van Laerhoven (2019) Deep PPG: large-scale heart rate estimation with "
               "convolutional neural networks. Sensors 19(14):3079. UCI 495, doi 10.24432/C53890.",
               "https://archive.ics.uci.edu/dataset/495/ppg+dalia",
               {"format": "zip", "fs": 64, "sites": ["wrist"], "never_unpickle": True, "approx_mb": 2_700},
               fetch=fetch_files),
        Source("utsa_ppg", DOMAIN,
               "UTSA-PPG: wrist green and fingertip red/IR PPG on the same seconds with accelerometer and Holter "
               "ECG RR, 12 people sitting, sleeping and at office work",
               (RemoteFile(UTSA_URL, "dataset.zip", 496_391_190),),
               "MIT (the repository's licence; no separate data licence stated)",
               "Xu, Zhang, Wang, Xie & Zhu (2025) A comprehensive PPG-based dataset for HR/HRV studies. IEEE ICHI, "
               "295-303, doi 10.1109/ICHI64645.2025.00042.",
               "https://github.com/utsanpb2-234/UTSA-PPG",
               {"format": "zip of csv", "sites": ["wrist", "finger"], "approx_mb": 496},
               fetch=fetch_files),
        Source("wesad", DOMAIN,
               "WESAD: Empatica E4 wrist BVP (64 Hz) and chest ECG in a stress study, 15 people, age per person; "
               "the archive's pickles are never unpickled",
               (RemoteFile(WESAD_URL, "WESAD.zip", 2_249_444_501),),
               "not yet verified: read the archive's readme (UCI dataset 465 lists it; the authors ask for citation)",
               "Schmidt, Reiss, Duerichen, Marberger & Van Laerhoven (2018) Introducing WESAD, a multimodal dataset "
               "for wearable stress and affect detection. ICMI. UCI 465, doi 10.24432/C57K5T.",
               "https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection",
               {"format": "zip", "fs": 64, "sites": ["wrist"], "never_unpickle": True, "approx_mb": 2_250},
               fetch=fetch_files),
    )
}
