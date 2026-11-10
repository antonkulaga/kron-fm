"""Samsung Health export readers and metrics, on a synthetic export (no real rows)."""
import json
from datetime import datetime

import numpy as np
import polars as pl

from kron_ppg.samsung_health import (Export, binned, clean_rr, ecg_recordings, hrv, nightly_resting_hr,
                                     offset_minutes, r_peaks, sleep_rr)

STAMP = "20990101000000"


def _csv(folder, stream, header, rows):
    lines = [f"{stream},1,1", ",".join(header) + ","] + [",".join(r) + "," for r in rows]
    (folder / f"{stream}.{STAMP}.csv").write_text("﻿" + "\n".join(lines) + "\n")


def _blob(folder, stream, name, obj):
    d = folder / "jsons" / stream / name[0]
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps(obj))


def _export(tmp_path):
    f = tmp_path / f"samsunghealth_test_{STAMP}"
    f.mkdir()
    s = "com.samsung.health.hrv"
    _csv(f, s, ["start_time", "time_offset", "binning_data", "datauuid"],
         [["2099-01-01 00:00:00.000", "UTC+0100", "a1.binning_data.json", "a1"]])
    _blob(f, s, "a1.binning_data.json", [{"start_time": 4070908800000 + 30000 * i, "end_time": 0, "sdnn": 50.0 + i,
                                          "rmssd": 40.0} for i in range(4)])
    # a stream whose name is a prefix of another must not open the other
    _csv(f, "com.samsung.health.hrv.extra", ["start_time"], [["x"]])
    sleep = "com.samsung.shealth.sleep"
    _csv(f, sleep, ["com.samsung.health.sleep.start_time", "com.samsung.health.sleep.time_offset",
                    "com.samsung.health.sleep.datauuid"], [["2099-01-01 22:00:00.000", "UTC+0200", "s1"]])
    raw = "com.samsung.shealth.sleep_raw_data"
    _csv(f, raw, ["sleep_uuid", "data", "datauuid"], [["s1", "r1.data.json", "r1"]])
    _blob(f, raw, "r1.data.json", {"data": [{"acc": [0] * 8, "rri_info": [[1, 200, 1000]] * 30},
                                             {"acc": [0] * 8, "rri_info": [[1, 200, 1200]] * 25}], "version": 1})
    ecg = "com.samsung.health.ecg"
    fs, n = 500, 15000
    t = np.arange(n) / fs
    trace = np.zeros(n)
    for beat in np.arange(0.5, 30, 0.8):  # 75 bpm, sharp R waves
        trace += np.exp(-((t - beat) / 0.012) ** 2)
    _csv(f, ecg, ["sample_frequency", "start_time", "time_offset", "mean_heart_rate", "chart_data", "classification"],
         [["500", "2099-01-01 08:00:00.000", "UTC+0000", "75.0", "e1.chart_data.json", "1"]])
    _blob(f, ecg, "e1.chart_data.json", {"chart_values": trace.tolist()})
    return Export(f)


def test_offset_and_exact_stream_match(tmp_path):
    assert offset_minutes("UTC+0130") == 90 and offset_minutes("UTC-0500") == -300 and offset_minutes(None) == 0
    e = _export(tmp_path)
    assert e.csv_path("com.samsung.health.hrv").name.startswith("com.samsung.health.hrv.2099")
    assert e.table("com.samsung.shealth.sleep").columns == ["start_time", "time_offset", "datauuid"]


def test_binned_reads_each_bin_on_the_wearers_clock(tmp_path):
    b = binned(_export(tmp_path), "com.samsung.health.hrv")
    assert len(b) == 4 and b["sdnn"].to_list() == [50.0, 51.0, 52.0, 53.0]
    assert b["local"][0] == datetime(2099, 1, 1, 1, 0, 0)  # UTC midnight at UTC+0100


def test_sleep_rr_places_beats_in_their_epochs(tmp_path):
    rr = sleep_rr(_export(tmp_path))
    assert len(rr) == 55 and rr["epoch"].unique().sort().to_list() == [0, 1]
    second = rr.filter(pl.col("epoch") == 1)
    assert second["t_s"][0] == 30.0 and second["t_s"][1] == 31.2
    assert rr["night_start"][0] == datetime(2099, 1, 2, 0, 0, 0)


def test_clean_rr_and_hrv_on_a_known_series():
    rng = np.random.default_rng(0)
    rr = 1000 + rng.normal(0, 30, 2000)
    rr[[100, 500, 900]] = [300.0, 2500.0, 1600.0]  # an extra beat, a gap, a missed beat
    keep = clean_rr(rr)
    assert not keep[[100, 500, 900]].any() and keep.mean() > 0.98
    out = hrv(rr, np.cumsum(rr) / 1000)
    # white noise of sd 30: RMSSD = sqrt(2) * 30, SDNN = 30
    assert abs(out["rmssd_ms"] - 42.4) < 4 and abs(out["sdnn_ms"] - 30) < 4 and abs(out["hr_bpm"] - 60) < 1


def test_r_peaks_find_every_beat_either_polarity(tmp_path):
    rec = ecg_recordings(_export(tmp_path))[0]
    for sign in (1, -1):
        p = r_peaks(sign * rec["samples"], rec["fs"])
        hr = 60 / np.median(np.diff(p) / rec["fs"])
        assert abs(len(p) - 37) <= 1 and abs(hr - 75) < 1


def test_nightly_resting_hr_takes_the_lowest_five_minutes():
    t0 = datetime(2099, 1, 1, 23, 0)
    times = pl.datetime_range(t0, datetime(2099, 1, 2, 6, 0), "1m", eager=True)
    hr = np.full(len(times), 60.0)
    hr[200:206] = 48.0
    stream = pl.DataFrame({"available": times, "heart_rate": hr})
    sleeps = pl.DataFrame({"start": [t0], "end": [datetime(2099, 1, 2, 6, 0)], "sleep_id": ["s"]})
    out = nightly_resting_hr(stream, sleeps)
    assert out["resting_hr_bpm"][0] == 48.0 and abs(out["hours"][0] - 7.0) < 1e-9
