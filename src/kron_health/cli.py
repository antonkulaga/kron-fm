"""``kron-health``: fit the summary-signal heads and run every head on one person's data.

    uv run kron-health fit-activity                    # NHANES hourly rhythm -> runs/kron-health/act-gbm
    uv run kron-health fit-hrv                         # Autonomic Aging resting HRV -> runs/kron-health/hrv-gbm
    uv run kron-health estimate <export folder> [--probe <run>]... --out <file.json>

``estimate`` reads a Samsung Health export (unpacked; the extractor's ``streams`` output beside it in
``<person>/out/streams`` when present, else computed) and, optionally, pulse-exploration probe runs, and gives each
head's estimate under its own name with the head's own error on its own test data and how many inputs it read. It
never combines them into one number and never calls any of them the person's age: they are separate readings with
wide error bands. Personal data stay where they are; write ``--out`` to a private folder.
"""
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import typer

from kron_core.paths import runs_dir

app = typer.Typer(add_completion=False, no_args_is_help=True)
MODEL = "kron-health"

#: Each head's error on its own held-out test data (see docs/plans/samsung-health-models.md).
REFERENCE = {
    "activity_age": "NHANES 2011-2014 wrist monitor, 8,924 adults 18-80, 10-fold CV: MAE 11.9 y (11.7-12.1), r 0.61 "
                    "(median age 15.8); never checked on a watch's step counts",
    "step_age": "NHANES 2005-2006, 4,467 adults 18-85, step counts in 10-minute bins (hip pedometer), 10-fold CV: "
                "MAE 12.5 y (12.2-12.8), r 0.62 (median age 17.0)",
    "resting_hrv_age": "Autonomic Aging, 1,093 people 18-92 at supine rest: MAE 8.7 y (8.2-9.2), r 0.54 (median age 9.7): "
                       "barely past the median; read from sleep RR here, a different state (sleep raises HRV, so it reads young): "
                       "not to be read as a person's age",
    "ecg_age": "CODE-15% + PTB-XL lead I, band-passed: PTB-XL test 9.0 y (8.7-9.4), r 0.75; CODE-15% test 9.3 y; "
               "clinical ECGs, never a watch's",
    "ppg_age": "run-4 encoder + open-wrist head, 5 Hz band: 1,220 people of a held-out wrist cohort, MAE 7.5 y "
               "(7.2-7.9), r 0.60, from 16+ readings; worse with fewer",
}


def _gbm():
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
                                         l2_regularization=1.0, random_state=0)


@app.command("fit-activity")
def fit_activity() -> None:
    """Fit the activity head (unit-free features) on every NHANES adult and save it."""
    import joblib
    import polars as pl

    from kron_core.paths import interim_dir
    from kron_health.activity import UNIT_FREE

    f = pl.read_parquet(interim_dir() / "health" / "nhanes_pam" / "activity.parquet")
    x = f.select(UNIT_FREE).to_numpy().astype(float)
    m = _gbm().fit(x, f["age"].to_numpy())
    out = runs_dir(MODEL) / "act-gbm"
    out.mkdir(parents=True, exist_ok=True)
    # the training range of each feature: an input outside it is not read (step counts sit outside the monitor's)
    lo, hi = np.nanpercentile(x, 0.5, axis=0), np.nanpercentile(x, 99.5, axis=0)
    joblib.dump({"model": m, "features": UNIT_FREE, "people": f.height, "lo": lo, "hi": hi}, out / "model.joblib")
    typer.echo(f"{out}: fitted on {f.height} people")


@app.command("fit-hrv")
def fit_hrv() -> None:
    """Fit the resting HRV head on every Autonomic Aging person with an age (one lead per record) and save it."""
    import joblib
    import polars as pl

    from kron_core.paths import interim_dir
    from kron_health.hrv import FEATURES

    f = pl.read_parquet(interim_dir() / "health" / "autonomic_aging" / "hrv.parquet")
    f = f.filter(pl.col("lead").is_in(["ECG1", "ECG"])).sort("lead").unique("record", keep="first")
    f = f.filter(pl.col("age").is_not_null() & pl.col("rmssd").is_not_null())
    x = f.select(FEATURES).to_numpy().astype(float)
    med = np.nanmedian(x, axis=0)
    x = np.where(np.isfinite(x), x, med)
    from sklearn.ensemble import HistGradientBoostingRegressor

    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
                                      random_state=0).fit(x, f["age"].to_numpy())
    out = runs_dir(MODEL) / "hrv-gbm"
    out.mkdir(parents=True, exist_ok=True)
    lo, hi = np.nanpercentile(x, 0.5, axis=0), np.nanpercentile(x, 99.5, axis=0)
    joblib.dump({"model": m, "features": FEATURES, "fill": med, "people": f.height, "lo": lo, "hi": hi},
                out / "model.joblib")
    typer.echo(f"{out}: fitted on {f.height} people")


def _activity(person_out: Path, export, days: int) -> dict:
    import joblib
    import polars as pl

    from kron_health.activity import samsung

    path = runs_dir(MODEL) / "act-gbm" / "model.joblib"
    if not path.is_file():
        return {"note": "not fitted: run `kron-health fit-activity`"}
    streams = person_out / "streams"
    if (streams / "step_bins.parquet").is_file():
        steps = pl.read_parquet(streams / "step_bins.parquet")
        sleeps = pl.read_parquet(streams / "sleep_sessions.parquet")
    else:
        from samsung_health.streams import sleep_sessions, step_bins  # GlucoseDAO's extractor, when installed

        steps, sleeps = step_bins(export), sleep_sessions(export)
    if steps.is_empty():
        return {"note": "no step bins"}
    end = steps["start"].max()
    start = end - timedelta(days=days)
    feat = samsung(steps.filter(pl.col("start") >= start), sleeps.filter(pl.col("start") >= start))
    if feat is None:
        return {"note": "fewer than 3 days worn 20 h or more"}
    m = joblib.load(path)
    x = np.array([[feat[k] for k in m["features"]]], float)
    outside = [k for k, v, lo, hi in zip(m["features"], x[0], m["lo"], m["hi"]) if np.isfinite(v) and not lo <= v <= hi]
    days_in = int(steps.filter(pl.col("start") >= start)["start"].dt.date().n_unique())
    if len(outside) >= 2:
        return {"estimate": None, "days": days, "inputs": days_in, "outside_training_range": outside,
                "note": "not read: the input lies outside the range the head was trained on (watch step counts are "
                        "zero most hours, a research monitor's activity never is), so its number would be an artefact"}
    return {"estimate": float(m["model"].predict(x)[0]), "days": days, "inputs": days_in}


def _steps(person_out: Path, days: int) -> dict:
    import joblib
    import polars as pl

    from kron_health.steps import samsung

    path = runs_dir(MODEL) / "steps-gbm" / "model.joblib"
    if not path.is_file():
        return {"note": "not fitted: scripts/kron_health/steps_eval.py"}
    bins = person_out / "streams" / "step_bins.parquet"
    if not bins.is_file():
        return {"note": "no step bins (run the extractor's streams first)"}
    steps = pl.read_parquet(bins)
    start = steps["start"].max() - timedelta(days=days)
    steps = steps.filter(pl.col("start") >= start)
    feat = samsung(steps)
    if feat is None:
        return {"note": "fewer than 4 days with 10 worn hours"}
    m = joblib.load(path)
    x = np.array([[feat[k] for k in m["features"]]], float)
    outside = [k for k, v, lo, hi in zip(m["features"], x[0], m["lo"], m["hi"]) if np.isfinite(v) and not lo <= v <= hi]
    days_in = int(steps["start"].dt.date().n_unique())
    if len(outside) >= 2:
        return {"estimate": None, "days": days, "inputs": days_in, "outside_training_range": outside,
                "note": "not read: the input lies outside the range the head was trained on"}
    return {"estimate": float(m["model"].predict(x)[0]), "days": days, "inputs": days_in,
            "outside_training_range": outside}


def _hrv(export) -> dict:
    import joblib

    from kron_health.hrv import features
    from kron_ppg.samsung_health import sleep_rr

    path = runs_dir(MODEL) / "hrv-gbm" / "model.joblib"
    if not path.is_file():
        return {"note": "not fitted: run `kron-health fit-hrv`"}
    rr = sleep_rr(export)
    if rr.is_empty():
        return {"note": "no per-beat RR intervals in the export"}
    m = joblib.load(path)
    rows, refused = [], []
    for (_,), night in rr.group_by("sleep_id"):
        f = features(night["rr_ms"].to_numpy(), night["t_s"].to_numpy())
        if f is None:
            continue
        x = np.array([[f.get(k, np.nan) for k in m["features"]]], float)
        out_ = [k for k, v, lo, hi in zip(m["features"], x[0], m["lo"], m["hi"]) if np.isfinite(v) and not lo <= v <= hi]
        if len(out_) >= 2:
            refused.append(out_)
            continue
        rows.append(float(m["model"].predict(np.where(np.isfinite(x), x, m["fill"]))[0]))
    if not rows:
        return {"estimate": None, "nights_refused": len(refused),
                "note": "no night inside the head's training range (it learnt from supine rest, not sleep)"}
    return {"estimate": float(np.median(rows)), "inputs": len(rows), "nights_refused": len(refused), "per_night": rows}


def _ecg(export, probes: list[Path]) -> dict:
    import torch

    from kron_health.ecg import ECGAge, sdk_ecg, strips
    from kron_ppg.samsung_health import ecg_recordings

    path = runs_dir(MODEL) / "ecg4-code15-ptbxl-lead1-combined-filtered" / "checkpoint.pt"
    if not path.is_file():
        return {"note": "no ECG head checkpoint"}
    ck = torch.load(path, map_location="cpu", weights_only=True)
    m = ECGAge(mean=ck["mean"], scale=ck["scale"])
    m.load_state_dict(ck["state"])
    m.eval()
    filt = bool(ck.get("filtered", False))

    def read(x, fs):
        s = torch.from_numpy(strips(x, fs, filtered=filt))
        if not len(s):
            return []
        with torch.no_grad():
            return ((m(s) + m(-s)) / 2).numpy().tolist()

    app_ = [v for r in ecg_recordings(export) for v in read(r["samples"], r["fs"])]
    sdk = [v for p in probes if (x := sdk_ecg(p)) is not None for v in read(x, 500.0)]
    out = {}
    if app_:
        out["ecg_app"] = {"estimate": float(np.mean(app_)), "inputs": len(app_), "unit": "10 s strips"}
    if sdk:
        out["watch_sdk"] = {"estimate": float(np.mean(sdk)), "inputs": len(sdk), "unit": "10 s strips"}
    return out or {"note": "no ECG recording"}


@app.command()
def estimate(export_folder: Path = typer.Argument(..., help="the unpacked samsunghealth_* folder"),
             probe: list[Path] = typer.Option([], help="pulse-exploration probe run folders (raw PPG, SDK ECG)"),
             out: Path | None = typer.Option(None, help="write the result here too (a private folder)"),
             activity_days: int = 90, device: str = "cuda") -> None:
    """Every head on one person's data, each under its own name with its own error."""
    from kron_ppg.samsung_health import Export
    from kron_ppg.wrist_age import json_safe

    export = Export(export_folder)
    person_out = export_folder.parent / "out"
    result = {"activity_age": _activity(person_out, export, activity_days),
              "step_age": _steps(person_out, activity_days), "resting_hrv_age": _hrv(export),
              "ecg_age": _ecg(export, probe)}
    if probe:
        from kron_ppg.probe_age import estimate as ppg

        p = ppg(probe, runs_dir("kron-ppg") / "wa4-open-r4-lp5", device)
        result["ppg_age"] = {"estimate": p.get("age"), "inputs": p.get("windows"),
                             "readings_15s_equivalent": p.get("readings_15s_equivalent"), "runs": p.get("runs")}
    for k in result:
        result[k]["reference"] = REFERENCE[k]
    result["note"] = ("Separate readings, each with its own error band; not one age, not a diagnosis. "
                      "No PPG waveform is in a Samsung Health export: ppg_age needs the watch probe's raw recordings.")
    text = json.dumps(json_safe(result), indent=2)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
    typer.echo(text)


if __name__ == "__main__":
    app()
