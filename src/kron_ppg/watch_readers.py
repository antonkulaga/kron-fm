"""Readers for the wrist and smartwatch sets of :mod:`kron_ppg.watch_sources`: each yields one wrist PPG channel
per recording (green where the set has it) at a fixed rate, and a subject table with age where the set gives one.

As in :mod:`kron_ppg.readers`, nothing downloaded is executed or unpickled: CSVs are parsed as text, archives are
read member by member, MATLAB 7.3 files are read as HDF5 datasets.
"""
from __future__ import annotations

import re
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import polars as pl

from kron_ppg import readers as _readers
from kron_ppg.readers import Recording, SourceData, _table, _values

#: A gap between two samples longer than this stays a gap (NaN) when a signal is put on a uniform grid.
MAX_GAP_S = 0.5
#: Long recordings (a Vilnius segment of up to 35 h, a WildPPG day, a Multisite span) are cut into recordings of
#: at most this long, so the per-recording window cap of ``prepare`` samples all of them, not only their start.
CHUNK_S: float | None = 6 * 3600


def pieces(x: np.ndarray, fs: float) -> Iterator[tuple[int, np.ndarray]]:
    """``(j, part)`` of a recording cut into pieces of at most :data:`CHUNK_S`, or ``(0, whole)`` when
    ``CHUNK_S`` is None (streams for a streaming trainer, :mod:`kron_ppg.streams`)."""
    if CHUNK_S is None:
        yield 0, x
        return
    step = int(CHUNK_S * fs)
    for j, start in enumerate(range(0, len(x), step)):
        yield j, x[start:start + step]


#: For whole recordings (``CHUNK_S`` None), timestamped sets are split where no sample comes for this long, so a
#: stream does not carry hours (or, from a stray clock, years) of empty grid. No signal is cut or dropped.
SESSION_GAP_S = 3600.0


def sessions(t: np.ndarray, x: np.ndarray, fs: float) -> Iterator[tuple[int, np.ndarray]]:
    """``(j, uniform piece)`` of timestamped samples: one piece per stretch without a gap over
    :data:`SESSION_GAP_S` (when ``CHUNK_S`` is set, the whole span as one grid, cut by :func:`pieces`)."""
    if CHUNK_S is not None:
        yield from pieces(uniform(t, x, fs), fs)
        return
    order = np.argsort(t, kind="stable")
    t, x = t[order], x[order]
    breaks = np.flatnonzero(np.diff(t) > SESSION_GAP_S) + 1
    for j, (a, b) in enumerate(zip(np.r_[0, breaks], np.r_[breaks, len(t)], strict=True)):
        if b - a >= 2:
            yield j, uniform(t[a:b], x[a:b], fs)


def uniform(t: np.ndarray, x: np.ndarray, fs: float) -> np.ndarray:
    """Samples ``x`` taken at times ``t`` (seconds, possibly jittered or unordered), linearly interpolated onto a
    grid at ``fs`` from the first time; samples sharing a time (a batch stamped once) are averaged. Grid points
    inside a gap longer than :data:`MAX_GAP_S`, or next to a NaN sample, are NaN."""
    keep = np.isfinite(t)
    t, x = t[keep], np.asarray(x, np.float64)[keep]
    t, inverse, counts = np.unique(t, return_inverse=True, return_counts=True)
    x = np.bincount(inverse, weights=x, minlength=len(t)) / counts
    if len(t) < 2:
        return np.full(len(t), np.nan)
    grid = t[0] + np.arange(int(np.floor((t[-1] - t[0]) * fs)) + 1) / fs
    out = np.interp(grid, t, np.nan_to_num(x))
    right = np.clip(np.searchsorted(t, grid, side="left"), 1, len(t) - 1)
    left = right - 1
    bad = ((t[right] - t[left]) > MAX_GAP_S) | ~np.isfinite(x[left]) | ~np.isfinite(x[right])
    out[bad] = np.nan
    return out


def _zip_or_tree(folder: Path, archive: str) -> tuple[list[str], Callable[[str], bytes]]:
    """``(names, opener)`` over a set: the unpacked tree ``folder/unpacked`` when present, else the zip.
    macOS resource forks (``__MACOSX/``, ``._*``) are left out."""
    unpacked = folder / "unpacked"
    if unpacked.is_dir():
        names = [str(p.relative_to(unpacked)) for p in unpacked.rglob("*") if p.is_file()]
        opener = lambda name: (unpacked / name).read_bytes()  # noqa: E731
    else:
        z = zipfile.ZipFile(folder / archive)
        names, opener = z.namelist(), z.read
    names = [n for n in names if "__MACOSX/" not in n and not n.rsplit("/", 1)[-1].startswith("._")]
    return names, opener


def xlsx_rows(path: Path, sheet: int = 1) -> list[list[str]]:
    """The cells of one sheet of an ``.xlsx`` as rows of text, placed by their references (``C7``), so empty
    cells keep their columns. The workbook is read as the zip of XML it is; nothing in it is evaluated."""
    import xml.etree.ElementTree as ET

    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            shared = ["".join(si.itertext()) for si in ET.fromstring(z.read("xl/sharedStrings.xml")).iter(f"{ns}si")]
        root = ET.fromstring(z.read(f"xl/worksheets/sheet{sheet}.xml"))
    rows = []
    for row in root.iter(f"{ns}row"):
        cells: dict[int, str] = {}
        for c in row.iter(f"{ns}c"):
            letters = re.match(r"[A-Z]+", c.get("r", "")).group(0)
            col = 0
            for ch in letters:
                col = col * 26 + ord(ch) - 64
            v = c.find(f"{ns}v")
            text = "".join(c.find(f"{ns}is").itertext()) if c.get("t") == "inlineStr" else (v.text if v is not None else "")
            cells[col - 1] = shared[int(text)] if c.get("t") == "s" and text else (text or "")
        rows.append([cells.get(i, "") for i in range(max(cells, default=-1) + 1)])
    return rows


# ---- WF-PPG ------------------------------------------------------------------------------------
#: WF-PPG's samples come at about 100 Hz with jittered timestamps; they are put on this grid.
WF_PPG_FS = 100.0


def read_wf_ppg(folder: Path) -> SourceData:
    """Wrist green PPG (``PPG_Wrist_G``, inverted: the raw value falls as blood volume rises) of each session,
    ``<subject>/<session>``. The subject is the participant folder, which numbers like ``additional_info.csv``'s
    ``ID``; the file names carry a second, recruitment number (folder 04 holds ``*_05.csv``). The finger channels
    (``PPG_Finger_*_AC``, band-passed by the authors) are left for the wrist-finger bridge."""
    names, opener = _zip_or_tree(folder, "WF-PPG.zip")
    info = next(n for n in names if n.endswith("additional_info.csv"))
    subjects = [{"subject": f"{int(r['ID']):02d}", "age": r.get("Age", ""), "sex": r.get("Gender", ""),
                 "height": r.get("H", ""), "weight": r.get("W", ""), "bmi": r.get("BMI", "")}
                for r in _table(opener(info)) if r.get("ID", "").strip()]
    pattern = re.compile(r"(?:^|/)(\d+)/(\d+)_preprocessed_(\d+)\.csv$")
    sessions = sorted((int(m[1]), int(m[2]), n) for n in names if (m := pattern.search(n)))

    def gen() -> Iterator[Recording]:
        for subject, session, name in sessions:
            df = pl.read_csv(opener(name), columns=["Time", "PPG_Wrist_G"], infer_schema=False)
            x = uniform(_values(df["Time"]), -_values(df["PPG_Wrist_G"]), WF_PPG_FS)
            yield Recording(f"{subject:02d}", f"{subject:02d}/{session}", WF_PPG_FS, x)

    return SourceData(subjects, gen())


# ---- Multisite-PPG -----------------------------------------------------------------------------
MULTISITE_FS = 100.0


def multisite_watch(path: Path) -> np.ndarray:
    """A ``P*_Watch_raw.npz`` (read without pickles) as inverted green PPG on a 100 Hz grid from its millisecond
    timestamps; the hours between sessions are NaN. The raw value falls as blood volume rises, at every site."""
    with np.load(path, allow_pickle=False) as z:
        return uniform(z["timestamp"] / 1000.0, -z["ppg_green"].astype(np.float64), MULTISITE_FS)


def read_multisite_ppg(folder: Path) -> SourceData:
    """The watch (wrist) green channel of each participant, in recordings ``<id>/watch.<chunk>`` of at most
    :data:`CHUNK_S` (the sessions span about 27 h, mostly gaps between them). The ring, earring and
    necklace (``P*_<Site>_raw.npz``, same layout) and the Polar H10 ECG are left for the bridge."""
    raw = folder / "raw_data"
    subjects = [{"subject": r["Participant Index"], "age": r.get("Age", ""), "sex": r.get("Gender", ""),
                 "ethnicity": r.get("Race/Ethnicity", ""), "fitzpatrick": r.get("Fitzpatrick Skin Index", "")}
                for r in _table(raw / "participants_demographics.csv")]
    files = {s["subject"]: raw / s["subject"] / f"{s['subject']}_Watch_raw.npz" for s in subjects}

    def gen() -> Iterator[Recording]:
        for s in subjects:
            path = files[s["subject"]]
            if path.is_file():
                with np.load(path, allow_pickle=False) as z:
                    t, x = z["timestamp"] / 1000.0, -z["ppg_green"].astype(np.float64)
                for j, part in sessions(t, x, MULTISITE_FS):
                    yield Recording(s["subject"], f"{s['subject']}/watch.{j}", MULTISITE_FS, part)

    return SourceData(subjects, gen())


# ---- Galaxy Watch Active 2 (Baigutanova et al. 2025) ---------------------------------------------
#: The watch's green PPG comes every 100 ms (often two samples under one batch time stamp).
GALAXY_ACTIVE2_FS = 10.0
GALAXY_ACTIVE2_MEMBER = "raw_data/ppg.csv.gz"
DAY_MS = 86_400_000


def unpack_galaxy_active2(folder: Path, block: int = 1 << 26) -> dict:
    """Split the 4.7 GB ``raw_data/ppg.csv.gz`` inside ``raw_data.zip`` into
    ``unpacked/ppg/<deviceId>/<UTC day>/part-<n>.parquet`` (``ts`` in ms, ``ppg``), streaming: the zip member is
    gunzipped block by block and each block parsed as CSV text. Returns rows and files written."""
    import gzip

    out = folder / "unpacked" / "ppg"
    out.mkdir(parents=True, exist_ok=True)
    schema = {"deviceId": pl.String, "ts": pl.Int64, "ppg": pl.Float64}
    rows = files = 0
    with zipfile.ZipFile(folder / "raw_data.zip") as z, z.open(GALAXY_ACTIVE2_MEMBER) as member, \
            gzip.open(member) as text:
        header = text.readline().strip().decode()
        if header.split(",") != list(schema):
            raise ValueError(f"unexpected header {header!r} in {GALAXY_ACTIVE2_MEMBER}")
        leftover, part = b"", 0
        while True:
            chunk = text.read(block)
            data = leftover + chunk
            if not chunk:
                leftover = b""
            else:
                cut = data.rfind(b"\n") + 1
                data, leftover = data[:cut], data[cut:]
            if data.strip():
                df = pl.read_csv(data, has_header=False, new_columns=list(schema), schema=schema)
                df = df.with_columns((pl.col("ts") // DAY_MS).alias("day"))
                for (device, day), g in df.group_by(["deviceId", "day"]):
                    stamp = np.datetime64(int(day), "D").astype(str)
                    dest = out / str(device) / stamp / f"part-{part:05d}.parquet"
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    g.select("ts", "ppg").write_parquet(dest)
                    files += 1
                rows += len(df)
                part += 1
            if not chunk:
                break
    return {"rows": rows, "files": files, "out": str(out)}


#: ``"day"``: one recording per device and UTC day (spreads the window cap); ``"device"``: one per device over its
#: four weeks, the hours between days as gaps (a stream, :mod:`kron_ppg.streams`).
GALAXY_ACTIVE2_UNIT = "day"


def galaxy_active2_device(path: Path) -> np.ndarray:
    """All days of one device folder of the unpacked set on one 10 Hz grid (raw value inverted)."""
    df = pl.concat([pl.read_parquet(p) for p in sorted(path.glob("*/part-*.parquet"))])
    return uniform(df["ts"].to_numpy() / 1000.0, -df["ppg"].to_numpy(), GALAXY_ACTIVE2_FS)


def galaxy_active2_day(path: Path) -> np.ndarray:
    """One device-day folder of the unpacked set as green PPG on a 10 Hz grid (raw value inverted)."""
    df = pl.concat([pl.read_parquet(p) for p in sorted(path.glob("part-*.parquet"))])
    return uniform(df["ts"].to_numpy() / 1000.0, -df["ppg"].to_numpy(), GALAXY_ACTIVE2_FS)


def read_galaxy_active2(folder: Path) -> SourceData:
    """One recording per device and UTC day (``<deviceId>/<YYYY-MM-DD>``), so the per-recording window cap
    spreads over the four weeks. Needs ``unpack_galaxy_active2`` first. Subjects come from ``survey.csv``
    (age, sex coded 1/2, height, weight and the sleep, mood and chronotype scores)."""
    root = folder / "unpacked" / "ppg"
    if not root.is_dir():
        raise FileNotFoundError(f"{root} is missing: run `python -m kron_ppg.watch_readers unpack {folder}`")
    keep = ("age", "sex", "height", "weight", "smartwatch", "exercise", "coffee", "smoking", "drinking",
            "ISI_1", "PHQ9_1", "GAD7_1", "MEQ")
    subjects = [{"subject": r["deviceId"], **{k: r.get(k, "") for k in keep}} for r in _table(folder / "survey.csv")]
    days = sorted((d.parent.name, d.name, d) for d in root.glob("*/*") if d.is_dir())

    def gen() -> Iterator[Recording]:
        if GALAXY_ACTIVE2_UNIT == "device":  # whole: a device's four weeks, split only where it went silent
            for device in sorted({d for d, _, _ in days}):
                df = pl.concat([pl.read_parquet(p) for p in sorted((root / device).glob("*/part-*.parquet"))])
                for j, part in sessions(df["ts"].to_numpy() / 1000.0, -df["ppg"].to_numpy(), GALAXY_ACTIVE2_FS):
                    yield Recording(device, f"{device}/{j}", GALAXY_ACTIVE2_FS, part)
            return
        for device, day, path in days:
            yield Recording(device, f"{device}/{day}", GALAXY_ACTIVE2_FS, path=path, parser="galaxy_active2_day")

    return SourceData(subjects, gen())


# ---- Vilnius long-term wrist PPG with AF (Zenodo) ------------------------------------------------
VILNIUS_FS = 100.0
#: Columns 0-29 of ``subject_info.xlsx`` (the last two, free-text medication lists, are left out); ``?`` = unknown.
VILNIUS_COLUMNS = (
    "subject", "sex_female", "age", "height", "weight", "alcohol", "smoking", "physical_inactivity",
    "excessive_physical_activity", "familial_af", "sleep_apnea", "hyperlipidaemia", "hyperglycaemia",
    "impaired_fasting_glycaemia", "hyperthyroidism", "adrenal_hyperfunction", "hfref", "hfpef",
    "coronary_artery_disease", "chronic_kidney_disease", "copd_asthma", "stroke_tia", "metabolic_syndrome",
    "bp_over_140_80", "triglycerides_high", "hdl_low", "waist_high", "glucose_intolerance_diabetes",
    "hypertension_stage", "antihypertensive_treatment")


def read_vilnius_af(folder: Path) -> SourceData:
    """Wrist green PPG (``PPG_GREEN``, inverted) of each ECG-synchronised segment of each ``NNN_PPG.mat``
    (MATLAB 7.3, read with h5py as HDF5 datasets), cut into recordings ``<NNN>/<segment>.<chunk>`` of at most
    :data:`CHUNK_S`. Subjects: age, sex, body size, risk factors and comorbidities per patient."""
    import h5py

    rows = xlsx_rows(folder / "subject_info.xlsx")
    subjects = []
    for row in rows:
        if not row or not row[0].strip().isdigit():
            continue  # the two header rows
        values = [(row[i] if i < len(row) else "").strip() for i in range(len(VILNIUS_COLUMNS))]
        rec = {k: ("" if v == "?" else v) for k, v in zip(VILNIUS_COLUMNS, values, strict=True)}
        rec["subject"] = f"{int(rec['subject']):03d}"
        rec["sex"] = {"0": "M", "1": "F"}.get(rec["sex_female"], "")
        subjects.append(rec)
    files = sorted(folder.glob("[0-9][0-9][0-9]_PPG.mat"))
    def gen() -> Iterator[Recording]:
        for path in files:
            subject = path.name[:3]
            with h5py.File(path, "r") as f:
                for k, ref in enumerate(f["PPG_GREEN"][:, 0]):
                    x = -np.asarray(f[ref][()], np.float64).ravel()
                    for j, part in pieces(x, VILNIUS_FS):
                        yield Recording(subject, f"{subject}/{k}.{j}", VILNIUS_FS, part)

    return SourceData(subjects, gen())


# ---- UTSA-PPG --------------------------------------------------------------------------------------
UTSA_FS = 100.0


def read_utsa_ppg(folder: Path) -> SourceData:
    """Wrist green (``green``, inverted: the authors' filtered channels fall as blood volume rises) of each
    ``subject<N>_<scene>_signals.csv`` (sit, sleep, office), ``<N>/<scene>``. The set gives no ages; the
    fingertip ``ir`` and ``red`` channels are left for the bridge."""
    names, opener = _zip_or_tree(folder, "dataset.zip")
    pattern = re.compile(r"(?:^|/)subject(\d+)_([a-z]+)_signals\.csv$")
    found = sorted((int(m[1]), m[2], n) for n in names if (m := pattern.search(n)))
    subjects = [{"subject": f"{i:02d}", "age": ""} for i in sorted({i for i, _, _ in found})]

    def gen() -> Iterator[Recording]:
        for i, scene, name in found:
            df = pl.read_csv(opener(name), columns=["time", "green"], infer_schema=False)
            x = uniform(_values(df["time"]), -_values(df["green"]), UTSA_FS)
            yield Recording(f"{i:02d}", f"{i:02d}/{scene}", UTSA_FS, x)

    return SourceData(subjects, gen())


# ---- Empatica E4 sets: PPG-DaLiA ---------------------------------------------------------------------
def e4_bvp(archive: bytes) -> tuple[np.ndarray, float]:
    """``BVP.csv`` of an Empatica E4 export (a zip): its first line is the start time, its second the rate,
    then one sample per line."""
    with zipfile.ZipFile(__import__("io").BytesIO(archive)) as z:
        lines = z.read("BVP.csv").decode().split()
    fs = float(lines[1].split(",")[0])
    return np.array([_float(v.split(",")[0]) for v in lines[2:]]), fs


def _float(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _quest(text: str) -> dict[str, str]:
    """``# AGE, 55`` lines of a PPG-DaLiA ``S*_quest.csv``."""
    out = {}
    for line in text.splitlines():
        key, _, value = line.lstrip("# ").partition(",")
        if key.strip():
            out[key.strip().lower()] = value.strip()
    return out


def read_ppg_dalia(folder: Path) -> SourceData:
    """The E4's wrist BVP (64 Hz) per participant, read from ``S*_E4.zip`` inside the nested ``data.zip``;
    age, sex, height, weight, skin type and sport level from ``S*_quest.csv``. The ``S*.pkl`` files are
    Python pickles and are never opened."""
    outer = zipfile.ZipFile(folder / "ppg_dalia.zip")
    inner = zipfile.ZipFile(outer.open("data.zip"))
    names = inner.namelist()
    ids = sorted({int(m[1]) for n in names if (m := re.search(r"/S(\d+)/S\d+_E4\.zip$", n))})
    subjects = []
    for i in ids:
        q = _quest(inner.read(f"PPG_FieldStudy/S{i}/S{i}_quest.csv").decode(errors="replace"))
        subjects.append({"subject": f"S{i}", "age": q.get("age", ""), "sex": q.get("gender", "").upper(),
                         "height": q.get("height", ""), "weight": q.get("weight", ""), "skin": q.get("skin", ""),
                         "sport": q.get("sport", "")})

    def gen() -> Iterator[Recording]:
        for i in ids:
            x, fs = e4_bvp(inner.read(f"PPG_FieldStudy/S{i}/S{i}_E4.zip"))
            yield Recording(f"S{i}", f"S{i}", fs, x)

    return SourceData(subjects, gen())


# ---- WESAD -------------------------------------------------------------------------------------------
def _wesad_readme(text: str) -> dict[str, str]:
    """``Age: 28``, ``Gender: male``, ``Height (cm): 178`` ... lines of a WESAD ``S*_readme.txt``."""
    keys = {"age": "age", "gender": "sex", "height (cm)": "height", "weight (kg)": "weight",
            "dominant hand": "dominant_hand"}
    out = {}
    for line in text.splitlines():
        key, _, value = line.partition(":")
        if key.strip().lower() in keys:
            value = value.strip()
            out[keys[key.strip().lower()]] = value[:1].upper() if key.strip().lower() == "gender" else value
    return out


def read_wesad(folder: Path) -> SourceData:
    """The E4's wrist BVP (64 Hz, worn on the non-dominant wrist) per participant from ``S*_E4_Data.zip``, age
    and body size from ``S*_readme.txt``; the ``S*.pkl`` files are pickles and are never opened."""
    z = zipfile.ZipFile(folder / "WESAD.zip")
    names = z.namelist()
    ids = sorted({int(m[1]) for n in names if (m := re.search(r"/S(\d+)/S\d+_E4_Data\.zip$", n))})
    subjects = [{"subject": f"S{i}", **_wesad_readme(z.read(f"WESAD/S{i}/S{i}_readme.txt").decode(errors="replace"))}
                for i in ids]

    def gen() -> Iterator[Recording]:
        for i in ids:
            x, fs = e4_bvp(z.read(f"WESAD/S{i}/S{i}_E4_Data.zip"))
            yield Recording(f"S{i}", f"S{i}", fs, x)

    return SourceData(subjects, gen())


# ---- WildPPG ---------------------------------------------------------------------------------------
WILDPPG_FS = 128.0


def read_wildppg(folder: Path) -> SourceData:
    """The wrist device's green PPG (``wrist.ppg_g.v``, 530 nm, 128 Hz, already systole-up) of each
    ``WildPPG_Part_<id>.mat`` (MATLAB 5, read with scipy.io.loadmat: arrays and structs only), cut into
    recordings ``<id>/<chunk>`` of at most :data:`CHUNK_S`. The files carry no ages; the forehead,
    sternum and ankle sites, IR and red, and the ECG are left for the bridge."""
    import scipy.io as sio

    files = sorted(folder.glob("WildPPG_Part_*.mat"))
    subjects = [{"subject": p.stem.removeprefix("WildPPG_Part_"), "age": ""} for p in files]
    def gen() -> Iterator[Recording]:
        for path, s in zip(files, subjects, strict=True):
            wrist = sio.loadmat(path, variable_names=["wrist"], squeeze_me=True, struct_as_record=False)["wrist"]
            fs = float(wrist.ppg_g.fs)
            if fs != WILDPPG_FS:
                raise ValueError(f"{path.name}: wrist green at {fs} Hz, expected {WILDPPG_FS}")
            x = np.asarray(wrist.ppg_g.v, np.float64).ravel()
            for j, part in pieces(x, WILDPPG_FS):
                yield Recording(s["subject"], f"{s['subject']}/{j}", WILDPPG_FS, part)

    return SourceData(subjects, gen())


# ---- PPG-Sport ---------------------------------------------------------------------------------------
PPG_SPORT_FS = 64.0


def ppg_sport_npy(path: Path) -> np.ndarray:
    """One ``S<NN><L|R>_<activity>_ppg.npy`` (read without pickles), inverted: at rest it rises slowly and falls
    fast, as the raw wrist channels of the other sets do."""
    return -np.load(path, allow_pickle=False).astype(np.float64).ravel()


def read_ppg_sport(folder: Path) -> SourceData:
    """Each wrist's PPG per activity, ``<subject>/<L|R>/<activity>`` at 64 Hz; age, sex, height, weight and
    the sports experience levels from ``participants.xlsx``."""
    rows = xlsx_rows(folder / "participants.xlsx")
    subjects = [{"subject": r[0], "sex": r[1], "age": r[2], "height": r[3], "weight": r[4],
                 "badminton": r[5], "table_tennis": r[6], "basketball": r[7]}
                for r in rows[1:] if len(r) >= 8 and r[0].startswith("S")]
    pattern = re.compile(r"^(S\d+)([LR])_([a-z]+)_ppg\.npy$")
    found = sorted((m[1], m[2], m[3], p) for p in (folder / "dataset").glob("*_ppg.npy") if (m := pattern.match(p.name)))

    def gen() -> Iterator[Recording]:
        for subject, wrist, activity, path in found:
            yield Recording(subject, f"{subject}/{wrist}/{activity}", PPG_SPORT_FS, path=path, parser="ppg_sport_npy")

    return SourceData(subjects, gen())


# ---- Aurora-BP (restricted) -------------------------------------------------------------------
AURORA_FS = 500.0
AURORA_PROTOCOLS = ("oscillometric", "auscultatory")


def aurora_optical(data: bytes) -> np.ndarray:
    """The optical (wrist PPG) column of one measurement TSV, on its 500 Hz grid. Upright as recorded: its
    band-passed derivative is right-skewed, like the corrected wrist channels of the other sets."""
    df = pl.read_csv(data, separator="\t", columns=["t", "optical"], infer_schema=False)
    return uniform(_values(df["t"]), _values(df["optical"]), AURORA_FS)


def read_aurora_bp(folder: Path) -> SourceData:
    """Every measurement with waveforms of both protocols, ``<pid>/<file stem>``, 30 s each at 500 Hz; age, sex,
    height, weight and the cardiovascular history per person from ``participants.tsv``. Restricted: the data use
    agreement forbids joining it with any other source, so it is prepared, trained and probed on alone, and
    everything written from it is owner-only (:func:`kron_ppg.watch_sources.owner_only`)."""
    keep = ("age", "gender", "height", "weight", "self_report_htn", "high_blood_pressure", "coronary_artery_disease",
            "diabetes", "arrythmia", "previous_heart_attack", "previous_stroke", "heart_failure", "aortic_stenosis",
            "valvular_heart_disease", "other_cv_diseases", "cvd_meds", "fitzpatrick_scale")
    subjects = [{"subject": r["pid"], **{("sex" if k == "gender" else k): r.get(k, "") for k in keep}}
                for r in _table_tsv(folder / "participants.tsv") if r.get("pid", "").strip()]
    measurements = []
    for protocol in AURORA_PROTOCOLS:
        table = folder / f"measurements_{protocol}.tsv"
        if not table.exists():
            continue
        measurements += [(protocol, r["pid"], r["waveform_file_path"]) for r in _table_tsv(table)
                         if r.get("waveforms_generated", "").strip() in ("1", "1.0") and r.get("waveform_file_path")]

    def gen() -> Iterator[Recording]:
        for protocol in AURORA_PROTOCOLS:
            archive = folder / f"measurements_{protocol}.zip"
            if not archive.exists():
                continue
            with zipfile.ZipFile(archive) as z:
                members = set(z.namelist())
                for proto, pid, name in measurements:
                    if proto == protocol and name in members:
                        stem = name.rsplit("/", 1)[-1].removesuffix(".tsv")
                        yield Recording(pid, f"{pid}/{stem}", AURORA_FS, aurora_optical(z.read(name)))

    return SourceData(subjects, gen())


def _table_tsv(path: Path) -> list[dict[str, str]]:
    return pl.read_csv(path, separator="\t", infer_schema=False).to_dicts()


UNPACKERS = {"galaxy_active2": unpack_galaxy_active2}
READERS = {"wf_ppg": read_wf_ppg, "multisite_ppg": read_multisite_ppg, "galaxy_active2": read_galaxy_active2,
           "vilnius_af": read_vilnius_af, "utsa_ppg": read_utsa_ppg,
           "ppg_dalia": read_ppg_dalia, "wesad": read_wesad,
           "wildppg": read_wildppg, "ppg_sport": read_ppg_sport, "aurora_bp": read_aurora_bp}
PARSERS: dict[str, Callable[[Path], np.ndarray]] = {"multisite_watch": multisite_watch,
                                                    "galaxy_active2_day": galaxy_active2_day,
                                                    "galaxy_active2_device": galaxy_active2_device,
                                                    "ppg_sport_npy": ppg_sport_npy}
# Registered into kron_ppg.readers whichever of the two modules is imported first.
_readers.READERS.update(READERS)
_readers.PARSERS.update(PARSERS)


if __name__ == "__main__":  # python -m kron_ppg.watch_readers unpack <source> [folder]
    import json
    import sys

    from kron_core.paths import input_dir

    if len(sys.argv) < 3 or sys.argv[1] != "unpack" or sys.argv[2] not in UNPACKERS:
        sys.exit(f"usage: python -m kron_ppg.watch_readers unpack {{{','.join(UNPACKERS)}}} [folder]")
    target = Path(sys.argv[3]) if len(sys.argv) > 3 else input_dir("ppg", sys.argv[2])
    print(json.dumps(UNPACKERS[sys.argv[2]](target)))
