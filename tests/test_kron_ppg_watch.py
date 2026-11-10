"""Readers of the wrist and smartwatch sets, on small synthetic copies of each set's layout."""
from __future__ import annotations

import zipfile

import numpy as np
import pytest

pytest.importorskip("scipy")

from kron_ppg.prepare import prepare_source  # noqa: E402
from kron_ppg.readers import READERS  # noqa: E402
from kron_ppg.watch_readers import uniform  # noqa: E402


def synthetic_ppg(seconds: float, fs: float, hr: float = 70.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * fs)) / fs
    phase = 2 * np.pi * hr / 60 * t
    return np.sin(phase) + 0.4 * np.sin(2 * phase + 0.6) + 0.02 * rng.standard_normal(len(t))


def test_uniform_interpolates_jittered_times_and_keeps_long_gaps_and_nans():
    t = np.array([0.0, 0.011, 0.019, 0.03, 1.5, 1.51, 1.52])
    x = np.array([0.0, 1.1, 1.9, 3.0, 150.0, np.nan, 152.0])
    y = uniform(t, x, 100.0)
    assert len(y) == 153 and y[0] == 0.0 and y[1] == pytest.approx(1.0) and y[3] == pytest.approx(3.0)
    assert np.isnan(y[4:150]).all()  # 1.47 s without a sample
    assert np.isnan(y[151]) and np.isnan(y[152 - 1])  # beside the NaN sample
    assert np.array_equal(uniform(t[::-1].copy(), x[::-1].copy(), 100.0), y, equal_nan=True)


def fake_wf_ppg(folder, ages=(23, 61)):
    """WF-PPG's zip: ``WF-PPG (3rd)/<folder>/<session>_preprocessed_<recruitment no>.csv`` and the subject table,
    with a macOS resource fork beside each, at about 100 Hz with jittered times, the raw green inverted."""
    folder.mkdir(parents=True)
    root = "WF-PPG (3rd)"
    rng = np.random.default_rng(0)
    with zipfile.ZipFile(folder / "WF-PPG.zip", "w") as z:
        z.writestr(f"{root}/additional_info.csv",
                   "ID,H,W,Age,Gender,BMI\n" + "".join(f"{i},170,60,{a},F,20.7\n" for i, a in enumerate(ages, 1)))
        for i in range(1, len(ages) + 1):
            for session in (0, 1):
                n = 6000
                t = 360 + np.cumsum(rng.uniform(0.002, 0.018, n))
                ppg = 1e5 - 800 * synthetic_ppg(70, 1000.0, hr=60 + 10 * i, seed=i)[((t - 360) * 1000).astype(int)]
                rows = "".join(f"{k},{tk:.6f},{v:.3f},1,2,0,0,0,0,4.5,0.9,101,61,98\n"
                               for k, (tk, v) in enumerate(zip(t, ppg, strict=True)))
                name = f"{root}/{i:02d}/{session}_preprocessed_{i + 2:02d}.csv"
                z.writestr(name, ",Time,PPG_Wrist_G,PPG_Wrist_IR,PPG_Wrist_R,PPG_Finger_G_AC,PPG_Finger_IR_AC,"
                                 "PPG_Finger_R_AC,PPG_Wrist_G_AC,Pressure,ECG,BP_Sys,BP_Dia,SpO2\n" + rows)
                z.writestr(f"__MACOSX/{root}/{i:02d}/._{session}_preprocessed_{i + 2:02d}.csv", "junk")


def test_wf_ppg_reads_wrist_green_upright_on_a_uniform_grid(tmp_path):
    fake_wf_ppg(tmp_path / "wf_ppg")
    data = READERS["wf_ppg"](tmp_path / "wf_ppg")
    assert [(s["subject"], s["age"]) for s in data.subjects] == [("01", "23"), ("02", "61")]
    recs = list(data.recordings)
    assert [r.recording for r in recs] == ["01/0", "01/1", "02/0", "02/1"]
    x = recs[0].load()
    assert recs[0].fs == 100.0 and 5000 < len(x) < 6000 and np.isfinite(x).all()
    assert x.mean() == pytest.approx(-1e5, rel=0.01)  # inverted: systole up


def test_wf_ppg_prepares_windows_for_every_subject(tmp_path):
    fake_wf_ppg(tmp_path / "wf_ppg")
    info = prepare_source("wf_ppg", folder=tmp_path / "wf_ppg", out=tmp_path / "out", seconds=10.0, workers=1)
    assert info["subjects_with_windows"] == 2 and info["windows"] >= 8


@pytest.mark.parametrize("first", ["kron_ppg.watch_readers", "kron_ppg.readers"])
def test_wrist_readers_register_whichever_module_is_imported_first(first):
    import subprocess
    import sys

    code = f"import {first}; from kron_ppg.readers import READERS; assert 'wf_ppg' in READERS and 'bidmc' in READERS"
    subprocess.run([sys.executable, "-c", code], check=True)


def fake_multisite(folder):
    raw = folder / "raw_data"
    (raw / "P1").mkdir(parents=True)
    (raw / "participants_demographics.csv").write_text(
        "Participant Index,Age,Gender,Race/Ethnicity,Fitzpatrick Skin Index\nP1,51,Female,White / Caucasian,1\n"
        "P2,25,Male,Asian / Pacific Islander,3\n")
    t0 = 1.7716e12
    t = np.concatenate([t0 + np.arange(6000) * 10.0, t0 + 3_600_000 + np.arange(6000) * 10.0])
    green = 2e4 - 300 * np.concatenate([synthetic_ppg(60, 100.0, seed=1), synthetic_ppg(60, 100.0, seed=2)])
    np.savez(raw / "P1" / "P1_Watch_raw.npz", timestamp=t, ppg_green=green, ppg_ir=green * 10,
             datetime_str=np.array(["2026-02-21 15:37:16"] * len(t)))


def test_multisite_reads_the_watch_green_inverted_without_pickles_and_keeps_the_gap(tmp_path):
    fake_multisite(tmp_path / "multisite_ppg")
    data = READERS["multisite_ppg"](tmp_path / "multisite_ppg")
    assert [(s["subject"], s["age"]) for s in data.subjects] == [("P1", "51"), ("P2", "25")]
    recs = list(data.recordings)  # P2 has no watch file; one 6 h chunk holds the whole 61 min span
    assert [r.recording for r in recs] == ["P1/watch.0"]
    x = recs[0].load()
    assert recs[0].fs == 100.0 and len(x) == 366_000
    assert np.isfinite(x[:6000]).all() and np.isnan(x[6001:360_000]).all()
    assert np.nanmean(x) == pytest.approx(-2e4, rel=0.01)


def fake_galaxy_active2(folder):
    """raw_data.zip holding raw_data/ppg.csv.gz (deviceId, ts in ms, ppg) at 10 Hz with a doubled batch stamp,
    two devices, one crossing a UTC midnight, and survey.csv."""
    import gzip

    folder.mkdir(parents=True)
    (folder / "survey.csv").write_text("deviceId,sex,age,height,weight,ISI_1,PHQ9_1,GAD7_1,MEQ\n"
                                       "ab12,1,36,178.0,76.0,8,1,1,46\ncd34,2,31,159.0,50.0,11,3,1,28\n")
    lines = ["deviceId,ts,ppg"]
    for device, start in (("ab12", 86_400_000 * 20_000 - 60_000), ("cd34", 86_400_000 * 20_001 + 3_600_000)):
        x = 2.6e6 - 2e4 * synthetic_ppg(120, 10.0, seed=len(lines))
        ts = start + np.arange(len(x)) * 100
        ts[5] = ts[4]  # two samples under one batch stamp
        lines += [f"{device},{t},{int(v)}" for t, v in zip(ts, x, strict=True)]
    with zipfile.ZipFile(folder / "raw_data.zip", "w") as z:
        z.writestr("raw_data/ppg.csv.gz", gzip.compress(("\n".join(lines) + "\n").encode()))
        z.writestr("__MACOSX/raw_data/._ppg.csv.gz", "junk")


def test_galaxy_active2_unpacks_by_device_and_day_and_reads_inverted_10_hz(tmp_path):
    from kron_ppg.watch_readers import unpack_galaxy_active2

    folder = tmp_path / "galaxy_active2"
    fake_galaxy_active2(folder)
    with pytest.raises(FileNotFoundError):
        READERS["galaxy_active2"](folder)
    info = unpack_galaxy_active2(folder, block=5000)  # many blocks: lines cut across block edges
    assert info["rows"] == 2400
    data = READERS["galaxy_active2"](folder)
    assert [(s["subject"], s["age"], s["sex"]) for s in data.subjects] == [("ab12", "36", "1"), ("cd34", "31", "2")]
    recs = list(data.recordings)
    assert [r.recording for r in recs] == ["ab12/2024-10-03", "ab12/2024-10-04", "cd34/2024-10-05"]
    lengths = [len(r.load()) for r in recs]
    assert lengths == [600, 600, 1200]  # the doubled stamp is averaged; the grid spans the times
    x = recs[2].load()
    assert recs[2].fs == 10.0 and np.isfinite(x).all() and x.mean() == pytest.approx(-2.6e6, rel=0.01)


def write_xlsx(path, rows):
    """A minimal one-sheet workbook with inline strings, cells placed by reference (empty ones left out)."""
    def col(i):
        name = ""
        i += 1
        while i:
            i, r = divmod(i - 1, 26)
            name = chr(65 + r) + name
        return name

    cells = "".join(
        f'<row r="{r + 1}">' + "".join(f'<c r="{col(c)}{r + 1}" t="inlineStr"><is><t>{v}</t></is></c>'
                                       for c, v in enumerate(row) if v != "") + "</row>"
        for r, row in enumerate(rows))
    sheet = ('<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
             f"<sheetData>{cells}</sheetData></worksheet>")
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("xl/worksheets/sheet1.xml", sheet)


def fake_vilnius(folder, chunk_s):
    h5py = pytest.importorskip("h5py")
    folder.mkdir(parents=True)
    header = ["Patient ID", "Gender ( 0 - male, 1 - female)", "Age, years"] + [""] * 19 + ["Metabolic syndrome"]
    second = [""] * 23 + ["Arterial blood pressure >140/80 mmHg"]
    row = ["06", "1", "77", "162", "85"] + ["0"] * 17 + ["1", "?"] + ["0"] * 4 + ["2", "1", "1 x Metoprololi"]
    write_xlsx(folder / "subject_info.xlsx", [header, second, row])
    with h5py.File(folder / "006_PPG.mat", "w") as f:
        refs = f.create_group("#refs#")
        lengths = (int(2.5 * chunk_s * 100), int(0.5 * chunk_s * 100))
        made = []
        for k, n in enumerate(lengths):
            made.append(refs.create_dataset(f"s{k}", data=(1e5 - 1e3 * synthetic_ppg(n / 100, 100.0, seed=k))[None, :]).ref)
        f.create_dataset("PPG_GREEN", data=np.array(made, dtype=h5py.ref_dtype)[:, None])
    (folder / "007_PPG.mat.part").write_bytes(b"partial")


def test_vilnius_reads_the_subject_table_and_cuts_inverted_green_segments(tmp_path, monkeypatch):
    from kron_ppg import watch_readers

    monkeypatch.setattr(watch_readers, "CHUNK_S", 20)
    fake_vilnius(tmp_path / "vilnius_af", 20)
    data = READERS["vilnius_af"](tmp_path / "vilnius_af")
    (s,) = data.subjects
    assert (s["subject"], s["age"], s["sex"], s["height"]) == ("006", "77", "F", "162")
    assert s["bp_over_140_80"] == "" and s["metabolic_syndrome"] == "1" and s["hypertension_stage"] == "2"
    recs = list(data.recordings)
    assert [r.recording for r in recs] == ["006/0.0", "006/0.1", "006/0.2", "006/1.0"]
    assert [len(r.load()) for r in recs] == [2000, 2000, 1000, 1000]
    assert recs[0].fs == 100.0 and recs[0].load().mean() == pytest.approx(-1e5, rel=0.01)


def test_utsa_reads_wrist_green_inverted_per_scene(tmp_path):
    folder = tmp_path / "utsa_ppg"
    folder.mkdir()
    with zipfile.ZipFile(folder / "dataset.zip", "w") as z:
        for i, scene in ((2, "sleep"), (10, "sit")):
            x = -5 * synthetic_ppg(30, 100.0, seed=i)
            rows = "".join(f"{k / 100:.2f},1,2,{v:.4f},0,0,0\n" for k, v in enumerate(x))
            z.writestr(f"dataset/subject{i}_{scene}_signals.csv", "time,ir,red,green,x,y,z\n" + rows)
        z.writestr("dataset/subject2_sleep_ecg.csv", "time,rr,hr\n0,1,60\n")
    data = READERS["utsa_ppg"](folder)
    assert [s["subject"] for s in data.subjects] == ["02", "10"]
    recs = list(data.recordings)
    assert [r.recording for r in recs] == ["02/sleep", "10/sit"] and recs[0].fs == 100.0
    assert np.allclose(recs[0].load(), 5 * synthetic_ppg(30, 100.0, seed=2), atol=1e-3)


def e4_export(x, fs=64.0):
    import io

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("BVP.csv", "1531385666.00\n" + f"{fs:f}\n" + "".join(f"{v:.2f}\n" for v in x))
        z.writestr("HR.csv", "1531385666.00\n1.000000\n70\n")
    return buf.getvalue()


def test_ppg_dalia_reads_e4_bvp_from_the_nested_archive_and_never_the_pickles(tmp_path, monkeypatch):
    import io
    import pickle

    folder = tmp_path / "ppg_dalia"
    folder.mkdir()
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        for i, age in ((1, 34), (10, 55)):
            z.writestr(f"PPG_FieldStudy/S{i}/S{i}_E4.zip", e4_export(50 * synthetic_ppg(60, 64.0, seed=i)))
            z.writestr(f"PPG_FieldStudy/S{i}/S{i}_quest.csv",
                       f"# SUBJECT_ID, S{i}\n# AGE, {age}\n# GENDER, f\n# HEIGHT, 164\n# WEIGHT, 56\n# SKIN, 4\n")
            z.writestr(f"PPG_FieldStudy/S{i}/S{i}.pkl", b"not to be opened")
    with zipfile.ZipFile(folder / "ppg_dalia.zip", "w", zipfile.ZIP_STORED) as z:
        z.writestr("data.zip", inner.getvalue())
    monkeypatch.setattr(pickle, "load", lambda *a, **k: pytest.fail("a pickle was opened"))
    monkeypatch.setattr(pickle, "loads", lambda *a, **k: pytest.fail("a pickle was opened"))
    data = READERS["ppg_dalia"](folder)
    assert [(s["subject"], s["age"], s["sex"]) for s in data.subjects] == [("S1", "34", "F"), ("S10", "55", "F")]
    recs = list(data.recordings)
    assert [r.recording for r in recs] == ["S1", "S10"] and recs[0].fs == 64.0 and len(recs[0].load()) == 3840


def test_wesad_reads_e4_bvp_and_the_readme_ages(tmp_path):
    folder = tmp_path / "wesad"
    folder.mkdir()
    with zipfile.ZipFile(folder / "WESAD.zip", "w") as z:
        z.writestr("WESAD/S10/S10_E4_Data.zip", e4_export(40 * synthetic_ppg(60, 64.0, seed=3)))
        z.writestr("WESAD/S10/S10_readme.txt", "### Personal information ###\nAge: 28\nHeight (cm): 178\n"
                                               "Weight (kg): 76\nGender: male\nDominant hand: right\n")
        z.writestr("WESAD/S10/S10.pkl", b"not to be opened")
    data = READERS["wesad"](folder)
    assert data.subjects == [{"subject": "S10", "age": "28", "height": "178", "weight": "76", "sex": "M",
                              "dominant_hand": "right"}]
    (rec,) = list(data.recordings)
    assert rec.fs == 64.0 and len(rec.load()) == 3840


def test_wildppg_reads_the_wrist_green_in_chunks(tmp_path, monkeypatch):
    import scipy.io as sio

    from kron_ppg import watch_readers

    monkeypatch.setattr(watch_readers, "CHUNK_S", 20)
    folder = tmp_path / "wildppg"
    folder.mkdir()
    x = synthetic_ppg(50, 128.0)
    channel = {"fs": 128, "descr": "reflective PPG at 530 nm wavelength", "v": x}
    sio.savemat(folder / "WildPPG_Part_an0.mat", {"id": "an0", "notes": "", "wrist": {"ppg_g": channel,
                                                                                         "ppg_ir": channel}})
    data = READERS["wildppg"](folder)
    assert data.subjects == [{"subject": "an0", "age": ""}]
    recs = list(data.recordings)
    assert [r.recording for r in recs] == ["an0/0", "an0/1", "an0/2"] and recs[0].fs == 128.0
    assert np.allclose(np.concatenate([r.load() for r in recs]), x)


def test_ppg_sport_reads_each_wrist_and_activity_inverted(tmp_path):
    folder = tmp_path / "ppg_sport"
    (folder / "dataset").mkdir(parents=True)
    write_xlsx(folder / "participants.xlsx", [
        ["Participants", "Gender", "Age", "HeighT (cm)", "Weight (kg)", "Badminton", "Table Tennis", "Basketball"],
        ["S01", "M", "27", "175", "90", "3", "2", "2"]])
    x = synthetic_ppg(60, 64.0)
    for wrist in "LR":
        np.save(folder / "dataset" / f"S01{wrist}_running_ppg.npy", -x[:, None])
    np.save(folder / "dataset" / "S01_running_ecg.npy", x[:, None])
    data = READERS["ppg_sport"](folder)
    assert data.subjects[0]["age"] == "27" and data.subjects[0]["basketball"] == "2"
    recs = list(data.recordings)
    assert [r.recording for r in recs] == ["S01/L/running", "S01/R/running"] and recs[0].fs == 64.0
    assert np.allclose(recs[0].load(), x)


def test_streams_keep_every_recording_whole_with_gaps_and_noise_marked_not_dropped(tmp_path, monkeypatch):
    from kron_ppg import readers, watch_readers
    from kron_ppg.readers import Recording, SourceData
    from kron_ppg.streams import Streams, write_streams

    rng = np.random.default_rng(0)
    clean = synthetic_ppg(120, 100.0)
    gappy = synthetic_ppg(120, 100.0, seed=1)
    gappy[3000:5000] = np.nan  # 20 s gap: stays in the stream, marked invalid
    noise = rng.standard_normal(60 * 100)  # aperiodic: a window gate would drop it, a stream keeps it

    def fake(folder):
        recs = [Recording("a", "a/0", 100.0, clean), Recording("a", "a/1", 100.0, gappy),
                Recording("b", "b/0", 100.0, noise)]
        return SourceData([{"subject": "a", "age": "61"}, {"subject": "b", "age": "35"}], iter(recs))

    monkeypatch.setitem(readers.READERS, "fake", fake)
    before = watch_readers.CHUNK_S, watch_readers.GALAXY_ACTIVE2_UNIT
    (tmp_path / "in").mkdir()
    info = write_streams("fake", folder=tmp_path / "in", out=tmp_path / "out", workers=1)
    assert info["streams"] == 3 and info["samples"] == 25 * (120 + 120 + 60)
    assert (watch_readers.CHUNK_S, watch_readers.GALAXY_ACTIVE2_UNIT) == before  # set back for a later prepare
    s = Streams(tmp_path / "out")
    assert len(s) == 3 and list(s.table["recording"]) == ["a/0", "a/1", "b/0"]
    x, valid, blocks = s[1]
    assert len(x) == 3000 and not valid[30 * 25 + 10:50 * 25 - 10].any() and valid[:25 * 29].all()
    assert blocks.shape == (12, 3) and blocks[4, 2] == 0.0 and blocks[0, 2] == 1.0
    _, _, clean_blocks = s[0]
    _, noise_valid, noise_blocks = s[2]
    assert noise_valid.all() and noise_blocks[:, 1].mean() < clean_blocks[:, 1].mean() - 0.3
    assert (clean_blocks[:, 0] > 0).all() and dict(s.subjects.select("subject", "age").iter_rows())["a"] == 61


def test_wrist_readers_give_whole_recordings_for_streams(tmp_path, monkeypatch):
    from kron_ppg import watch_readers

    monkeypatch.setattr(watch_readers, "CHUNK_S", None)
    fake_vilnius(tmp_path / "vilnius_af", 20)
    recs = list(READERS["vilnius_af"](tmp_path / "vilnius_af").recordings)
    assert [r.recording for r in recs] == ["006/0.0", "006/1.0"] and len(recs[0].load()) == 5000
    monkeypatch.setattr(watch_readers, "GALAXY_ACTIVE2_UNIT", "device")
    folder = tmp_path / "galaxy_active2"
    fake_galaxy_active2(folder)
    watch_readers.unpack_galaxy_active2(folder, block=5000)
    recs = list(READERS["galaxy_active2"](folder).recordings)
    assert [r.recording for r in recs] == ["ab12/0", "cd34/0"] and len(recs[0].load()) == 1200


def test_whole_recordings_split_only_where_the_signal_stops_for_an_hour(monkeypatch):
    from kron_ppg import watch_readers

    monkeypatch.setattr(watch_readers, "CHUNK_S", None)
    t = np.r_[np.arange(600) / 10, 1e5 + np.arange(600) / 10, 1e5 + 60 + 1800 + np.arange(600) / 10]  # a stray
    x = np.arange(len(t), dtype=float)  # clock jump, then a 30 min gap that stays inside its session
    parts = list(watch_readers.sessions(t, x, 10.0))
    assert [j for j, _ in parts] == [0, 1] and len(parts[0][1]) == 600
    assert abs(len(parts[1][1]) - 19_200) <= 1 and np.isnan(parts[1][1][700])  # 60 + 1800 + 60 s at 10 Hz


# ---- Aurora-BP (restricted) ---------------------------------------------------------------------
def fake_aurora_bp(folder, seconds=32.0):
    """Aurora-BP's layout: participants and measurement tables (TSV) and a zip of one TSV per measurement at
    500 Hz, the optical channel upright; one measurement without waveforms."""
    import os

    folder.mkdir(parents=True)
    (folder / "participants.tsv").write_text("pid\tage\tgender\theight\tweight\tarrythmia\n"
                                             "o000\t33.0\tM\t68.0\t200.0\t0\no001\t71.0\tF\t62.0\t150.0\t1\n")
    rows = ["pid\tphase\tmeasurement\twaveform_file_path\twaveforms_generated"]
    fs = 500.0
    t = np.arange(int(seconds * fs)) / fs
    with zipfile.ZipFile(folder / "measurements_oscillometric.zip", "w") as z:
        for k, (pid, stem) in enumerate([("o000", "o000.initial.Supine_1"), ("o001", "o001.ambulatory.measurement_3")]):
            name = f"measurements_oscillometric/{pid}/{stem}.tsv"
            rows.append(f"{pid}\tinitial\tSupine 1\t{name}\t1")
            x = synthetic_ppg(seconds, fs, seed=k)
            body = "\n".join(f"{a:.3f}\t133.5\t{v:.5f}\t2355.5\t0.1\t0.2\t0.3" for a, v in zip(t, x, strict=True))
            z.writestr(name, "t\tekg\toptical\tpressure\taccel_x\taccel_y\taccel_z\n" + body + "\n")
    rows.append("o001\tinitial\tSupine 2\tmeasurements_oscillometric/o001/missing.tsv\t0")
    (folder / "measurements_oscillometric.tsv").write_text("\n".join(rows) + "\n")
    os.chmod(folder, 0o700)


def test_aurora_bp_is_read_upright_and_prepared_owner_only(tmp_path):
    import os
    import stat

    fake_aurora_bp(tmp_path / "aurora_bp")
    data = READERS["aurora_bp"](tmp_path / "aurora_bp")
    assert [s["age"] for s in data.subjects] == ["33.0", "71.0"] and data.subjects[1]["sex"] == "F"
    recs = list(data.recordings)
    assert [r.recording for r in recs] == ["o000/o000.initial.Supine_1", "o001/o001.ambulatory.measurement_3"]
    assert recs[0].fs == 500.0 and np.allclose(recs[0].signal, synthetic_ppg(32.0, 500.0), atol=1e-4)
    saved = os.umask(0o022)
    try:
        info = prepare_source("aurora_bp", 25.0, 30.0, folder=tmp_path / "aurora_bp", out=tmp_path / "out" / "w",
                              workers=1)
    finally:
        os.umask(saved)
    assert info["windows"] == 2
    for p in [tmp_path / "out", *(tmp_path / "out").rglob("*")]:
        mode = stat.S_IMODE(p.stat().st_mode)
        assert mode & 0o077 == 0, f"{p} is {oct(mode)}"


def test_restricted_sources_are_never_exported_or_joined(tmp_path):
    from kron_ppg.hub import export_hub
    from kron_ppg.watch_sources import is_restricted, refuse_join

    assert is_restricted("aurora_bp") and not is_restricted("vitaldb") and not is_restricted("galaxy_active2")
    with pytest.raises(PermissionError, match="aurora_bp"):
        export_hub(["bidmc", "aurora_bp"], tmp_path / "hub", "x/y", "t")
    assert not (tmp_path / "hub").exists()  # refused before anything is written
    with pytest.raises(PermissionError, match="never joined"):
        refuse_join(["aurora_bp", "vitaldb"])
    refuse_join(["aurora_bp"])  # alone is allowed
    refuse_join(["vitaldb", "galaxy_active2", "wildppg"])


def test_aurora_fetch_only_verifies_what_the_owner_placed(tmp_path):
    import hashlib
    import json
    import os

    from kron_ppg.sources import SOURCES

    saved = os.umask(0o022)
    try:
        _check_aurora_fetch(SOURCES["aurora_bp"], tmp_path / "aurora_bp", hashlib, json)
    finally:
        os.umask(saved)


def _check_aurora_fetch(source, folder, hashlib, json):
    folder.mkdir()
    with pytest.raises(FileNotFoundError, match="data use agreement"):
        source.fetch(source, folder, False)
    (folder / "README.md").write_text("terms")
    record = {"files": [{"key": "README.md", "size": 5, "checksum": "md5:" + hashlib.md5(b"terms").hexdigest()}]}
    (folder / "zenodo_record.json").write_text(json.dumps(record))
    manifest = source.fetch(source, folder, False)
    assert manifest["notes"]["restricted"] is True and manifest["files"][0]["verified"] == "zenodo md5"
    (folder / "README.md").write_text("other")
    with pytest.raises(ValueError, match="md5"):
        source.fetch(source, folder, False)
