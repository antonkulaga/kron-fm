"""kron-ppg's signal processing, readers, interim windows, clock scoring and a tiny end-to-end run."""
from __future__ import annotations

import gzip
import json
import math

import numpy as np
import polars as pl
import pytest
import torch

pytest.importorskip("scipy")

from kron_ppg.clock import auc, fit_bias, gap_associations, json_safe, per_subject, scores  # noqa: E402
from kron_ppg.data import Pool, SubjectPairs, WindowSet, subject_split, write_windows  # noqa: E402
from kron_ppg.model import KronPPGConfig  # noqa: E402
from kron_ppg.prepare import prepare_source  # noqa: E402
from kron_ppg.readers import read_bidmc, read_vitaldb  # noqa: E402
from kron_ppg.signal import cut_windows, fill_gaps, prepare_signal, quality, robust_scale  # noqa: E402

SMALL = dict(d_model=32, n_heads=2, n_layers=2, ff_units=64, dropout=0.0, patch=5, horizon=3, proj_dim=16)


def synthetic_ppg(seconds: float, fs: float, hr: float = 70.0, seed: int = 0) -> np.ndarray:
    """A pulse with a systolic peak and a later, smaller diastolic wave, on a drifting baseline."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * fs)) / fs
    phase = (t * hr / 60.0) % 1.0
    beat = np.exp(-((phase - 0.2) / 0.07) ** 2) + 0.35 * np.exp(-((phase - 0.5) / 0.08) ** 2)
    return 1000 + 50 * beat + 20 * np.sin(2 * np.pi * 0.1 * t) + 0.5 * rng.standard_normal(len(t))


# ---- signal ------------------------------------------------------------------------------------
def test_short_gaps_are_filled_and_long_ones_stay_invalid():
    x = np.arange(20, dtype=float)
    x[3:5] = np.nan
    x[10:16] = np.nan
    filled, valid = fill_gaps(x, max_gap=3)
    assert np.allclose(filled[3:5], [3, 4]) and valid[3:5].all()
    assert not valid[10:16].any() and (filled[10:16] == 0).all()


def test_robust_scale_ignores_a_spike():
    x = np.random.default_rng(0).standard_normal(1000)
    x[5] = 1e6
    z = robust_scale(x)
    assert abs(np.median(z)) < 1e-9 and 0.8 < np.percentile(z, 75) - np.percentile(z, 25) < 1.6


def test_the_quality_gate_reads_the_heart_rate_and_refuses_noise_and_flat_lines():
    fs = 25.0
    x, valid = prepare_signal(synthetic_ppg(60, 125.0, hr=72), 125.0, fs)
    assert valid.all() and len(x) == 60 * 25
    good = quality(robust_scale(x[:750]), fs)
    assert good.ok and abs(good.hr_bpm - 72) < 4
    assert not quality(np.random.default_rng(1).standard_normal(750), fs).ok
    assert quality(np.zeros(750), fs).reason == "flat"


def test_windows_are_cut_only_where_the_signal_is_valid_and_thinned_evenly():
    fs = 25.0
    x, valid = prepare_signal(synthetic_ppg(300, 500.0), 500.0, fs)
    valid[3000:3100] = False  # 4 s missing inside the 120-150 s window
    w, starts, _ = cut_windows(x, valid, fs, 30, 30)
    assert w.shape[1] == 750 and 120.0 not in starts.tolist() and len(w) >= 7
    thin, thin_starts, _ = cut_windows(x, valid, fs, 30, 30, max_windows=3)
    assert len(thin) == 3 and thin_starts[0] == starts[0] and thin_starts[-1] == starts[-1]


# ---- readers and prepare -----------------------------------------------------------------------
def fake_bidmc(folder, ages=("88", "65", "90+", "NaN")):
    folder.mkdir(parents=True)
    for i, age in enumerate(ages, 1):
        x = synthetic_ppg(120, 125.0, hr=60 + 5 * i, seed=i)
        rows = "\n".join(f"{t / 125:.3f}, 0.1, {v:.4f}, 0.2, 0.3, 0.4" for t, v in enumerate(x))
        (folder / f"bidmc_{i:02d}_Signals.csv").write_text("Time [s], RESP, PLETH, V, AVR, II\n" + rows + "\n")
        (folder / f"bidmc_{i:02d}_Fix.txt").write_text(f"Fixed parameters\nAge: {age}\nGender: F\n")


def test_bidmc_reads_ages_with_ninety_plus_censored_and_nan_unknown(tmp_path):
    fake_bidmc(tmp_path / "bidmc")
    data = read_bidmc(tmp_path / "bidmc")
    by = {s["subject"]: s for s in data.subjects}
    assert by["01"]["age"] == "88" and by["03"]["age"] == "90" and by["03"]["age_censored"] == "1"
    assert by["04"]["age"] == ""
    rec = next(data.recordings)
    assert rec.fs == 125.0 and len(rec.load()) == 120 * 125


def test_vitaldb_tracks_read_one_sample_per_row_and_subjects_come_from_subjectid(tmp_path):
    folder = tmp_path / "vitaldb"
    (folder / "pleth").mkdir(parents=True)
    with gzip.open(folder / "cases.csv.gz", "wt") as fh:
        fh.write("caseid,subjectid,age,sex,asa,preop_htn,preop_dm,death_inhosp\n7,4151,73,M,3,1,0,0\n")
    with gzip.open(folder / "pleth" / "0007.csv.gz", "wt") as fh:
        fh.write("Time,SNUADC/PLETH\n0,\n0.002,1.5\n,2.5\n,\n10.0,3.0\n")
    data = read_vitaldb(folder)
    assert data.subjects[0]["subject"] == "4151" and data.subjects[0]["asa"] == "3"
    rec = next(data.recordings)
    assert rec.subject == "4151" and rec.fs == 500.0
    assert np.array_equal(rec.load(), [np.nan, 1.5, 2.5, np.nan, 3.0], equal_nan=True)


def test_prepare_writes_a_variant_the_pool_loads(tmp_path):
    fake_bidmc(tmp_path / "in" / "bidmc")
    info = prepare_source("bidmc", 25.0, 30.0, folder=tmp_path / "in" / "bidmc", out=tmp_path / "out")
    assert info["windows"] >= 8 and info["subjects_with_windows"] == 4
    pool = Pool.load([tmp_path / "out"])
    assert pool.window(0).shape == (750,) and pool.window(0).dtype == np.float32
    assert set(pool.subject) == {"bidmc/01", "bidmc/02", "bidmc/03", "bidmc/04"}
    assert np.isnan(pool.age[pool.subject == "bidmc/04"]).all()


def test_a_variant_is_typed_parquet(tmp_path):
    fake_bidmc(tmp_path / "in" / "bidmc")
    prepare_source("bidmc", 25.0, 30.0, folder=tmp_path / "in" / "bidmc", out=tmp_path / "out")
    windows = pl.read_parquet(tmp_path / "out" / "windows.parquet")
    assert windows.schema["signal"] == pl.Array(pl.Float16, 750)
    assert windows.schema["start_s"] == pl.Float64 and windows.schema["subject"] == pl.String
    assert set(windows["recording"]) <= {"01", "02", "03", "04"}
    subjects = pl.read_parquet(tmp_path / "out" / "subjects.parquet")
    assert subjects.columns[:3] == ["subject", "age", "sex"]
    assert subjects.schema["age"] == pl.Float64 and subjects.schema["sex"] == pl.String
    assert subjects.schema["age_censored"] == pl.Int64
    assert subjects.filter(pl.col("subject") == "04")["age"].is_null().all()
    assert not (tmp_path / "out" / "windows.npy").exists()


def test_parallel_preparation_writes_what_one_process_writes(tmp_path):
    fake_bidmc(tmp_path / "in" / "bidmc")
    for workers in (1, 2):
        prepare_source("bidmc", folder=tmp_path / "in" / "bidmc", out=tmp_path / f"w{workers}", workers=workers)
    one, two = (pl.read_parquet(tmp_path / f"w{w}" / "windows.parquet") for w in (1, 2))
    assert len(one) >= 8 and one.equals(two)


# ---- splits and batches ------------------------------------------------------------------------
def test_the_subject_split_is_stable_when_subjects_are_added():
    a = subject_split([str(i) for i in range(100)], "src")
    b = subject_split([str(i) for i in range(300)], "src")
    assert all(b[k] == v for k, v in a.items())
    assert 0.6 < sum(v == "train" for v in b.values()) / 300 < 0.8


def test_subject_pairs_give_every_subject_in_a_batch_a_positive():
    subjects = np.array(["a"] * 5 + ["b"] * 1 + ["c"] * 3 + ["d"] * 4)
    for batch in SubjectPairs(subjects, batch_size=6, per_subject=2, seed=0):
        chosen = subjects[batch]
        assert len(batch) == 6 and all((chosen == s).sum() >= 2 for s in chosen)


# ---- clock scoring -----------------------------------------------------------------------------
def test_auc_counts_ties_as_half():
    assert auc(np.array([1, 2, 3, 4.0]), np.array([0, 0, 1, 1], bool)) == 1.0
    assert auc(np.array([1, 1, 1, 1.0]), np.array([0, 1, 0, 1], bool)) == 0.5
    assert math.isnan(auc(np.array([1, 2.0]), np.array([1, 1], bool)))


def test_the_bias_correction_removes_regression_to_the_mean():
    age = np.linspace(20, 80, 61)
    pred = 50 + 0.5 * (age - 50)  # a clock that shrinks toward the mean
    a, b = fit_bias(pred, age)
    s = scores(pred, age, (a, b))
    assert abs(b - 0.5) < 1e-9 and s["gap_sd"] < 1e-9 and s["mae"] > 0


def test_gap_associations_read_outcome_flags_and_skip_unknown_values():
    gaps = {"x/1": 5.0, "x/2": 4.0, "x/3": -3.0, "x/4": -4.0, "x/5": 0.0}
    table = {"x/1": {"preop_htn": "1", "asa": "3"}, "x/2": {"preop_htn": "1", "asa": "2"},
             "x/3": {"preop_htn": "0", "asa": "1"}, "x/4": {"preop_htn": "0", "asa": ""},
             "x/5": {"preop_htn": ""}}
    out = gap_associations(gaps, table, ages={"x/1": 70.0, "x/2": 30.0, "x/3": 60.0, "x/4": 40.0, "x/5": 50.0})
    assert out["preop_htn"]["auc_of_gap"] == 1.0 and out["preop_htn"]["positive"] == 2
    assert out["preop_htn"]["auc_of_age"] == 0.5
    assert out["asa"]["positive"] == 1 and out["asa"]["negative"] == 2
    assert "death_inhosp" not in out
    assert json.loads(json.dumps(json_safe({"a": float("nan")}))) == {"a": None}


def test_per_subject_averages_windows():
    out = per_subject(["a", "a", "b"], [40.0, 50.0, 70.0], [44.0, 44.0, 71.0])
    assert out == {"a": (45.0, 44.0, 2), "b": (70.0, 71.0, 1)}


# ---- end to end --------------------------------------------------------------------------------
def tiny_variant(folder, n_subjects=24, per_subject=4, seconds=10, fs=25.0):
    rng = np.random.default_rng(0)
    windows, rows, subjects = [], [], []
    for s in range(n_subjects):
        age = 20 + 60 * s / (n_subjects - 1)
        subjects.append({"subject": f"s{s:02d}", "age": f"{age:.0f}", "sex": "F",
                         "preop_htn": str(int(age > 50)), "preop_dm": "0", "asa": str(1 + (s % 3 == 0))})
        x, _ = prepare_signal(synthetic_ppg(per_subject * seconds + 4, 100.0, hr=55 + s, seed=s), 100.0, fs)
        for k in range(per_subject):
            seg = x[int((2 + k * seconds) * fs):int((2 + (k + 1) * seconds) * fs)]
            windows.append(robust_scale(seg + 0.01 * rng.standard_normal(len(seg)))[None].astype(np.float32))
            rows.append({"subject": f"s{s:02d}", "recording": f"s{s:02d}", "start_s": k * seconds, "hr_bpm": 60.0,
                         "periodicity": 0.9})
    write_windows(folder, windows, rows, subjects, {"source": "tiny", "sample_rate": fs, "seconds": seconds})
    return folder


def test_pretrain_then_clock_runs_end_to_end_and_writes_its_records(device, tmp_path):
    from kron_ppg.train import ClockArgs, PretrainArgs, fit_clock, load_clock, pretrain

    variant = tiny_variant(tmp_path / "tiny")
    config = KronPPGConfig(**SMALL)
    result = pretrain(PretrainArgs([variant], tmp_path / "pre", config, "next+subject", batch_size=8, steps=4,
                                   eval_every=2, device=str(device), num_workers=0))
    assert math.isfinite(result["best_val_next"]) and (tmp_path / "pre" / "checkpoint.pt").is_file()
    clock = fit_clock(ClockArgs([variant], tmp_path / "clock", tmp_path / "pre" / "checkpoint.pt", config,
                                batch_size=8, steps=4, eval_every=2, device=str(device), num_workers=0))
    metrics = json.loads((tmp_path / "clock" / "metrics.json").read_text())
    assert metrics["test"]["subjects"] >= 2 and "bias" in metrics and "preop_htn" in metrics["outcomes"]["test"]
    assert clock["best_step"] in (2, 4)
    predictions = pl.read_parquet(tmp_path / "clock" / "predictions.parquet")
    assert predictions.columns == ["split", "subject", "age", "predicted", "gap_corrected", "windows"]
    assert len(predictions) == 24 and predictions["windows"].sum() == 96
    assert pl.read_parquet(tmp_path / "clock" / "metrics.parquet")["step"].to_list() == [2, 4]
    model = load_clock(tmp_path / "clock" / "checkpoint.pt", device)
    with torch.no_grad():
        assert model(torch.from_numpy(Pool.load([variant]).window(0))[None].to(device)).shape == (1,)


def test_a_clock_refuses_windows_at_another_rate(tmp_path):
    from kron_ppg.train import ClockArgs, fit_clock

    variant = tiny_variant(tmp_path / "tiny")
    with pytest.raises(ValueError, match="prepare them at the model's rate"):
        fit_clock(ClockArgs([variant], tmp_path / "c", None, KronPPGConfig(**{**SMALL, "sample_rate": 50.0}),
                            steps=1, device="cpu", num_workers=0))


def test_window_set_items_are_samples_subject_and_age(tmp_path):
    pool = Pool.load([tiny_variant(tmp_path / "tiny", n_subjects=3, per_subject=2)])
    ds = WindowSet(pool, np.arange(len(pool)))
    x, subject, age = ds[3]
    assert x.shape == (250,) and subject == 1 and age == pytest.approx(50.0)


def test_the_hub_export_keeps_the_training_split(tmp_path):
    from kron_ppg.hub import export_hub

    fake_bidmc(tmp_path / "in" / "bidmc")
    folders = {}
    for rate in (25.0, 125.0):
        folders[("bidmc", rate)] = tmp_path / f"prep{rate:g}"
        prepare_source("bidmc", rate, 30.0, folder=tmp_path / "in" / "bidmc", out=folders[("bidmc", rate)])
    summary = export_hub(["bidmc"], tmp_path / "hub", "someone/ppg", "PPG", rates=(25.0, 125.0, 250.0),
                         folders=folders)
    assert set(summary) == {"bidmc_25hz", "bidmc_125hz"}  # 250 Hz is above BIDMC's own 125 Hz
    for rate, length in ((25.0, 750), (125.0, 3750)):
        pool = Pool.load([folders[("bidmc", rate)]])
        sides = pool.split()
        for ours, hub in (("train", "train"), ("val", "validation"), ("test", "test")):
            shards = sorted((tmp_path / "hub" / "bidmc" / f"{rate:g}hz").glob(f"{hub}-*.parquet"))
            rows = pl.concat([pl.read_parquet(p) for p in shards])
            assert len(rows) == len(sides[ours]) == summary[f"bidmc_{rate:g}hz"]["splits"][hub]["windows"]
            if len(rows):
                assert set("bidmc/" + rows["subject"]) == set(pool.subject[sides[ours]])
                assert rows.schema["signal"] == pl.Array(pl.Float16, length)
    subjects = pl.read_parquet(tmp_path / "hub" / "bidmc" / "subjects.parquet")
    assert set(subjects["split"]) <= {"train", "validation", "test"} and len(subjects) == 4
    card = (tmp_path / "hub" / "README.md").read_text()
    assert card.startswith("---\nlicense:\n- odc-by\n") and "config_name: bidmc_125hz" in card
    assert "path: bidmc/125hz/test-*.parquet" in card and "config_name: bidmc_subjects" in card
    assert "hf://datasets/someone/ppg/bidmc/25hz" in card and "ODC Attribution License" in card


def test_trait_values_from_the_subject_table():
    from kron_ppg.traits import TRAITS, trait_value

    row = {"age": 61.0, "bmi": None, "preop_htn": 1, "preop_dm": 0, "asa": 2, "sex": "F"}
    assert trait_value(TRAITS["age"], row) == 61.0 and math.isnan(trait_value(TRAITS["bmi"], row))
    assert trait_value(TRAITS["preop_htn"], row) == 1.0 and trait_value(TRAITS["preop_dm"], row) == 0.0
    assert trait_value(TRAITS["asa_ge3"], row) == 0.0 and trait_value(TRAITS["asa_ge3"], {"asa": "3"}) == 1.0
    assert trait_value(TRAITS["sex_male"], row) == 0.0 and math.isnan(trait_value(TRAITS["sex_male"], {"sex": "?"}))


def test_the_logistic_fit_separates_a_shifted_class():
    from kron_ppg.traits import fit_logistic

    rng = np.random.default_rng(0)
    x = np.r_[rng.normal(-1, 1, 200), rng.normal(1, 1, 200)][:, None]
    y = np.r_[np.zeros(200), np.ones(200)]
    w = fit_logistic(x, y)
    assert w[1] > 1.0 and abs(w[0]) < 0.3
    assert auc(np.column_stack([np.ones(400), x]) @ w, y.astype(bool)) > 0.85


def test_traits_train_and_score_end_to_end(tmp_path, device):
    from kron_ppg.traits import TraitArgs, fit_traits

    variant = tiny_variant(tmp_path / "tiny")
    result = fit_traits(TraitArgs([variant], tmp_path / "traits", None, ("age", "preop_htn", "sex_male"),
                                  KronPPGConfig(**SMALL), batch_size=8, steps=4, eval_every=2, device=str(device),
                                  num_workers=0))
    s = result["scores"]
    assert s["age"]["kind"] == "regression" and math.isfinite(s["age"]["test"]["mae"])
    assert s["preop_htn"]["test"]["positive"] + s["preop_htn"]["test"]["negative"] == result["subjects"]["test"]
    assert "auc_age_plus_ppg" in s["preop_htn"]["test"] and "auc_age" in s["preop_htn"]["val"]
    assert s["sex_male"]["test"]["positive"] == 0  # every tiny subject is F: the AUC is undefined, not a crash
    assert "preop_htn" in s["age"]["gap_outcomes_test"]
    predictions = pl.read_parquet(tmp_path / "traits" / "predictions.parquet")
    assert {"age", "age_predicted", "preop_htn_predicted"} <= set(predictions.columns)
    assert len(predictions) == result["subjects"]["val"] + result["subjects"]["test"]
    assert (tmp_path / "traits" / "checkpoint.pt").is_file()



# ---- vitals and the traits they feed -----------------------------------------------------------
def synthetic_ecg_ppg(seconds=30, fs=500.0, rr=0.9, pat=0.21, seed=0):
    """Lead II-like spikes every ``rr`` s and a PPG whose upstroke is steepest ``pat`` s after each."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * fs)) / fs
    beats = np.arange(0.3, seconds - 0.7, rr)
    ecg = sum(np.exp(-((t - b) / 0.008) ** 2) for b in beats) + 0.02 * rng.standard_normal(len(t))
    ppg = sum(1 / (1 + np.exp(np.clip(-(t - b - pat) / 0.03, -50, 50))) * np.exp(-np.clip(t - b - pat, 0, None) / 0.35)
              for b in beats) + 0.005 * rng.standard_normal(len(t))
    return ecg, ppg, len(beats)


def test_ecg_features_find_beats_and_the_pulse_arrival_time():
    from kron_ppg.vitals import ecg_features

    ecg, ppg, n = synthetic_ecg_ppg()
    f = ecg_features(ecg, ppg)
    assert abs(f["beats"] - n) <= 1 and abs(f["rr_ms"] - 900) < 5 and abs(f["ecg_hr"] - 66.7) < 1
    assert f["rmssd_ms"] < 10 and abs(f["pat_ms"] - 210) < 30
    assert np.isnan(ecg_features(np.full(15000, np.nan), ppg)["rr_ms"])


def test_window_medians_respect_ranges_and_gaps():
    from kron_ppg.vitals import window_medians

    t = np.arange(0, 100, 2.0)
    v = np.where(t < 50, 120.0, 140.0)
    v[3] = 999  # out of range: dropped
    m = window_medians(t, v, np.array([0.0, 60.0, 200.0]), 30.0, (40, 260))
    assert m[0] == 120 and m[1] == 140 and np.isnan(m[2])


def test_window_level_traits_read_vitals_and_score_within_person(tmp_path, device):
    from kron_ppg.traits import TraitArgs, fit_traits

    variant = tiny_variant(tmp_path / "tiny")
    w = pl.read_parquet(variant / "windows.parquet", columns=["recording", "start_s"])
    rng = np.random.default_rng(1)
    pl.DataFrame({"recording": w["recording"], "start_s": w["start_s"],
                  "sbp": rng.normal(120, 15, len(w)), "pat_ms": rng.normal(220, 20, len(w)),
                  "bp_source": ["art"] * len(w)}).write_parquet(variant / "vitals.parquet")
    pool = Pool.load([variant])
    assert set(pool.extra) == {"sbp", "pat_ms"} and np.isfinite(pool.extra["sbp"]).all()
    result = fit_traits(TraitArgs([variant], tmp_path / "t", None, ("age", "sbp", "healthy", "preop_hb"),
                                  KronPPGConfig(**SMALL), batch_size=8, steps=4, eval_every=2, device=str(device),
                                  num_workers=0))
    s = result["scores"]
    assert {"mae", "r", "within_person_r"} <= set(s["sbp"]["test_windows"])
    assert s["healthy"]["test"]["positive"] + s["healthy"]["test"]["negative"] > 0
    assert s["preop_hb"]["test"]["subjects"] == 0  # no labels anywhere: skipped, not a crash


def test_healthy_trait_and_healthy_only_clock(tmp_path, device):
    from kron_ppg.traits import TRAITS, trait_value
    from kron_ppg.train import ClockArgs, fit_clock

    h = TRAITS["healthy"]
    assert trait_value(h, {"asa": 1, "preop_htn": 0, "preop_dm": 0}) == 1.0
    assert trait_value(h, {"asa": 2, "preop_htn": 0, "preop_dm": 0}) == 0.0
    assert trait_value(h, {"asa": 1, "preop_htn": 1}) == 0.0
    assert np.isnan(trait_value(h, {"asa": 1, "preop_htn": 0}))  # diabetes unknown
    variant = tiny_variant(tmp_path / "tiny")
    fit_clock(ClockArgs([variant], tmp_path / "c", None, KronPPGConfig(**SMALL), batch_size=8, steps=2,
                        eval_every=2, device=str(device), num_workers=0, healthy_only=True))
    config = json.loads((tmp_path / "c" / "config.json").read_text())
    every = Pool.load([variant])
    assert config["healthy_only"] and config["subjects"]["test"] == len(set(every.subject[every.split()["test"]]))


def test_extra_vitaldb_tracks_follow_the_chosen_cases(tmp_path):
    from kron_ppg.sources import vitaldb_extra_tracks

    trks = tmp_path / "trks.csv.gz"
    with gzip.open(trks, "wt") as fh:
        fh.write("caseid,tname,tid\n1,SNUADC/PLETH,a\n1,SNUADC/ECG_II,b\n1,Solar8000/NIBP_SBP,c\n"
                 "2,SNUADC/ECG_II,d\n3,Solar8000/HR,e\n")
    assert vitaldb_extra_tracks([1, 3], trks, ["ecg"]) == [(1, "b", "ecg_ii")]
    assert vitaldb_extra_tracks([1, 3], trks, ["vitals"]) == [(1, "c", "nibp_sbp"), (3, "e", "hr")]
    with pytest.raises(ValueError):
        vitaldb_extra_tracks([1], trks, ["accelerometer"])
