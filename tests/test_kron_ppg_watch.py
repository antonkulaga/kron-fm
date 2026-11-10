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
    recs = list(data.recordings)  # P2 has no watch file
    assert [r.recording for r in recs] == ["P1/watch"]
    x = recs[0].load()
    assert recs[0].fs == 100.0 and len(x) == 366_000
    assert np.isfinite(x[:6000]).all() and np.isnan(x[6001:360_000]).all()
    assert np.nanmean(x) == pytest.approx(-2e4, rel=0.01)
