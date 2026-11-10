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
* ``wrist_exercise``: PhysioNet Wrist PPG During Exercise: wrist green PPG with chest ECG on a treadmill and
  an exercise bike, 8 people aged 22-32; 50 MB. ODC-By 1.0.
* ``ppg_dalia``: Empatica E4 wrist BVP (64 Hz) and chest ECG over about 2.5 h of daily activities each,
  15 people, with age in each ``S*_quest.csv``. CC BY 4.0. The archive also holds Python pickles: never
  unpickle them; read the E4 CSVs and the questionnaire.
* ``utsa_ppg``: wrist green PPG and fingertip red and IR PPG on the same seconds (rate read from the files), with
  accelerometer and Holter ECG RR intervals, 12 people sitting, sleeping and at office work, 30 min or more each.
  The repository is MIT; it states no separate data licence.
* ``ppg_sport``: wrist PPG, IMU and ECG at 64 Hz from both wrists of 30 people across six activities (stationary,
  walking, running, badminton, table tennis, basketball), about 48 h, processed by the authors (IMWUT 2026);
  ``.npy`` files and ``participants.xlsx`` from the GitHub repository. CC BY-NC-ND 4.0 per the publication's
  repository record; the GitHub repository states no licence.
* ``wesad``: Empatica E4 wrist BVP (64 Hz) and RespiBAN chest ECG in a stress study, 15 people, ages in each
  ``S*_readme.txt``; pickles again, never unpickled.
* ``aurora_bp`` (restricted): Microsoft Aurora-BP, 1,221 people aged 21-85, 30 s measurements of wrist optical PPG,
  tonometry and ECG at 500 Hz with cuff blood pressure and cardiovascular history. **Under a data use agreement**
  the owner accepted by hand: research only, never joined with other data, never redistributed. Never downloaded
  by ``kron-data``: its fetch only checks the files the owner placed against the Zenodo record. ``notes.restricted``
  makes ``prepare`` and ``streams`` write owner-only and ``hub-export`` refuse it; ``notes.no_join`` makes the
  trainers refuse it beside any other source.

* ``pulsewatch`` (restricted): PulseWatch2025 (UMass / UConn), Samsung Gear S3 and Galaxy Watch 3 wrist PPG at 50 Hz
  in 30 s segments with accelerometer, a chest-patch ECG and adjudicated rhythm labels, stroke survivors aged 50+
  over about 14 days each. Shared by its authors on request (a form and a Google Drive folder); the owner was
  granted access and downloads it. Treated as restricted until its own terms are known: no terms were found in
  the shared folder (its code is MIT). Its fetch only records the files placed.

A source with ``notes["restricted"]`` true is never exported, uploaded or committed, and is written owner-only; one
with ``notes["no_join"]`` (Aurora-BP) is also never joined with another source in a training or probe set.
"""
from __future__ import annotations

import json
import os
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
WRIST_EXERCISE = "https://physionet.org/files/wrist/1.0.0"
DALIA_URL = "https://archive.ics.uci.edu/static/public/495/ppg+dalia.zip"
UTSA_URL = "https://drive.usercontent.google.com/download?id=1NP_mngn7YccQO09FdaqQ001wwTMmE0ks&export=download&confirm=t"
PPG_SPORT_REPO = "LaserHu/PPG-Sport"
WESAD_URL = "https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download"
AURORA_BP_RECORD = 19099166


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


def physionet_files(base: str) -> list[tuple[str, str, int | None]]:
    """Every file PhysioNet's ``SHA256SUMS.txt`` lists for a project (WFDB records and any CSV copies)."""
    names = [line.split(maxsplit=1)[1] for line in _get_text(f"{base}/SHA256SUMS.txt").splitlines() if line.strip()]
    return [(f"{base}/{n}", n, None) for n in ["SHA256SUMS.txt", *names]]


def fetch_ptt(source: Source, folder: Path, force: bool) -> dict:
    return fetch_tree(source, folder, force, physionet_files(PTT_PPG))


def fetch_wrist_exercise(source: Source, folder: Path, force: bool) -> dict:
    return fetch_tree(source, folder, force, physionet_files(WRIST_EXERCISE))


def zenodo_files(record: int) -> list[tuple[str, str, int | None]]:
    """Every file of a Zenodo record: tables and texts first, then PPG, then the rest (ECG), by name."""
    files = json.loads(_get_text(f"https://zenodo.org/api/records/{record}"))["files"]
    files.sort(key=lambda f: (f["key"].endswith(".mat"), "PPG" not in f["key"], f["key"]))
    return [(f"https://zenodo.org/records/{record}/files/{f['key']}?download=1", f["key"], f["size"]) for f in files]


def fetch_vilnius_af(source: Source, folder: Path, force: bool) -> dict:
    return fetch_tree(source, folder, force, zenodo_files(VILNIUS_AF_RECORD), {"zenodo_record": VILNIUS_AF_RECORD})


def github_files(repo: str, prefixes: tuple[str, ...], branch: str = "main") -> list[tuple[str, str, int | None]]:
    """The files of a GitHub repository under ``prefixes`` (or named exactly so), from its git tree."""
    tree = json.loads(_get_text(f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1"))
    if tree.get("truncated"):
        raise RuntimeError(f"{repo}: the git tree listing is truncated")
    return [(f"https://raw.githubusercontent.com/{repo}/{branch}/{t['path']}", t["path"], t.get("size"))
            for t in tree["tree"] if t["type"] == "blob" and t["path"].startswith(prefixes)]


def fetch_ppg_sport(source: Source, folder: Path, force: bool) -> dict:
    return fetch_tree(source, folder, force, github_files(PPG_SPORT_REPO, ("dataset/", "participants.xlsx",
                                                                           "readme.md", "cleaned_gt/")))


def is_restricted(name: str) -> bool:
    """Whether a source is held under terms that forbid redistribution or joining (``notes["restricted"]``)."""
    from kron_ppg.sources import SOURCES as ALL

    return bool(ALL.get(name) and ALL[name].notes.get("restricted"))


def refuse_join(sources) -> None:
    """Raise when a source whose terms forbid joining (``notes["no_join"]``) would be joined with any other
    source in one training or probe set."""
    from kron_ppg.sources import SOURCES as ALL

    names = set(sources)
    alone = sorted(n for n in names if ALL.get(n) and ALL[n].notes.get("no_join"))
    if len(names) > 1 and alone:
        raise PermissionError(f"these sources are never joined with other data: {', '.join(alone)} "
                              f"with {', '.join(sorted(names - set(alone)))}")


def owner_only(name: str) -> None:
    """For a restricted source, make everything this process writes from now on owner-only (umask 077)."""
    if is_restricted(name):
        os.umask(0o077)


def fetch_aurora_bp(source: Source, folder: Path, force: bool) -> dict:
    """Never downloads: the owner downloads Aurora-BP under its agreement. Checks each file the Zenodo record lists
    (its ``zenodo_record.json``, saved beside them) by size and md5, and writes an owner-only ``SOURCE.json``."""
    import hashlib

    os.umask(0o077)
    record = folder / "zenodo_record.json"
    if not record.exists():
        raise FileNotFoundError(f"{folder}: Aurora-BP is obtained by its owner under a data use agreement, not "
                                "downloaded here; place its files and zenodo_record.json there first")
    entries = []
    for f in json.loads(record.read_text())["files"]:
        path = folder / f["key"]
        md5 = hashlib.md5()
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 24), b""):
                md5.update(block)
        if path.stat().st_size != f["size"] or f"md5:{md5.hexdigest()}" != f["checksum"]:
            raise ValueError(f"{path.name}: size or md5 differs from the Zenodo record")
        entries.append({"file": f["key"], "bytes": f["size"], "md5": md5.hexdigest(), "verified": "zenodo md5"})
    return write_manifest(folder, source, entries, {"zenodo_record": AURORA_BP_RECORD})


def fetch_pulsewatch(source: Source, folder: Path, force: bool) -> dict:
    """Never downloads (the shared folder is by request): records the per-subject tars and label files the owner
    placed, with their sizes, in an owner-only ``SOURCE.json``."""
    os.umask(0o077)
    found = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix in (".tar", ".csv")
                   and p.name != "SOURCE.json")
    if not found:
        raise FileNotFoundError(f"{folder}: PulseWatch2025 is shared by its authors on request; place it there first")
    entries = [{"file": str(p.relative_to(folder)), "bytes": p.stat().st_size} for p in found]
    return write_manifest(folder, source, entries)


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
        Source("wrist_exercise", DOMAIN,
               "PhysioNet Wrist PPG During Exercise: wrist PPG with chest ECG, treadmill and bike, 8 people aged 22-32",
               (), "ODC-By 1.0",
               "Jarchi & Casson (2016) Description of a database containing wrist PPG signals recorded during "
               "physical exercise with both accelerometer and gyroscope measures of motion. Data 2(1):1. "
               "Goldberger et al. (2000) PhysioNet. Circulation 101:e215.",
               "https://physionet.org/content/wrist/1.0.0/",
               {"format": "wfdb", "fs": 256, "sites": ["wrist"], "approx_mb": 50},
               fetch=fetch_wrist_exercise),
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
        Source("ppg_sport", DOMAIN,
               "PPG-Sport: both-wrist PPG, IMU and ECG at 64 Hz, 30 people, six sports and daily activities, ~48 h",
               (), "CC BY-NC-ND 4.0 per the publication's repository record (non-commercial, no derivatives; "
                   "research use); the GitHub repository states no licence",
               "PPG-Sport: a dataset for reliable heart rate monitoring from wrist PPG under dynamic sports "
               "conditions (2026). Proc. ACM IMWUT 10(2), doi 10.1145/3810216.",
               f"https://github.com/{PPG_SPORT_REPO}",
               {"format": "npy", "fs": 64, "sites": ["left wrist", "right wrist"],
                "load": "numpy.load(path, allow_pickle=False)", "approx_mb": 400},
               fetch=fetch_ppg_sport),
        Source("wesad", DOMAIN,
               "WESAD: Empatica E4 wrist BVP (64 Hz) and chest ECG in a stress study, 15 people, age per person; "
               "the archive's pickles are never unpickled",
               (RemoteFile(WESAD_URL, "WESAD.zip", 2_249_444_501),),
               "none stated in the archive's readme, which asks for citation of Schmidt et al. 2018; distributed openly "
               "through UCI (dataset 465), whose licence was not read",
               "Schmidt, Reiss, Duerichen, Marberger & Van Laerhoven (2018) Introducing WESAD, a multimodal dataset "
               "for wearable stress and affect detection. ICMI. UCI 465, doi 10.24432/C57K5T.",
               "https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection",
               {"format": "zip", "fs": 64, "sites": ["wrist"], "never_unpickle": True, "approx_mb": 2_250},
               fetch=fetch_files),
        Source("aurora_bp", DOMAIN,
               "Aurora-BP: wrist optical PPG, tonometry and ECG at 500 Hz in 30 s measurements with cuff blood "
               "pressure, 1,221 people aged 21-85, age and cardiovascular history; restricted (data use agreement)",
               (), "Microsoft Aurora-BP Data Use Agreement: research only, no joining with other data, no redistribution",
               "Mieloszyk et al. (2022) A comparison of wearable tonometry, photoplethysmography, and "
               "electrocardiography for cuffless measurement of blood pressure in an ambulatory setting. IEEE JBHI "
               "26(7):2864-2875.",
               "https://github.com/microsoft/aurorabp-sample-data",
               {"format": "zip of tsv", "fs": 500, "sites": ["wrist"], "restricted": True, "no_join": True,
                "terms": "research only; never joined with other data (no pretraining or training set with any "
                         "other source; a frozen encoder from elsewhere may be probed on it); never published, "
                         "uploaded or committed; owner-only files; no diagnostic or product use; publications "
                         "report results only and cite Mieloszyk et al. 2022",
                "approx_mb": 6_700},
               fetch=fetch_aurora_bp),
        Source("pulsewatch", DOMAIN,
               "PulseWatch2025: Samsung Gear S3 / Galaxy Watch 3 wrist PPG (50 Hz, 30 s segments) with accelerometer, "
               "chest-patch ECG and adjudicated NSR/AF/PAC-PVC labels, stroke survivors aged 50+, ~14 days each; "
               "restricted (shared on request)",
               (), "shared by the authors on request (Google Form and Drive access); no data licence found in the "
                   "shared folder (code MIT): treated as restricted, research only, never redistributed",
               "The Pulsewatch study (UMass / UConn), Synapse syn23565056; Dong Han et al., ICASSP 2025 "
               "(PulsewatchRelease code, MIT).",
               "https://www.synapse.org/Synapse:syn23565056",
               {"format": "tar of txt", "fs": 50, "sites": ["wrist"], "device": "Samsung Gear S3 / Galaxy Watch 3",
                "restricted": True, "terms": "unknown until the authors' terms are read; treated like a DUA set",
                "approx_mb": 946_000},
               fetch=fetch_pulsewatch),
    )
}
