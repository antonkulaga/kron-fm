"""The step-count head: age from a week or more of step counts, as a watch reports them (10-minute bins).

Trained on NHANES 2005-2006 (ActiGraph 7164 on the hip, per-minute steps, ~7,000 people). NHANES minutes are summed
into 10-minute bins first, so every feature is computed from the same thing a Samsung Health export holds
(``step_bins``: steps per 10 minutes). The hip monitor was worn while awake and taken off for the evening and the
night, a watch is worn day and night, so every feature reads only the hours both cover, :data:`HOURS` (08:00-19:59),
and only worn time in them:

* ``share_h08`` ... ``share_h19``: each hour's share of the window's steps (the day's shape);
* ``daily_cv``: day-to-day variation of the window's steps;
* ``log_daily_steps``: mean steps per kept day in the window (log);
* ``empty_daytime``: share of worn 10-minute bins with no step;
* ``brisk``, ``moderate``: share of worn bins with >= 1,000 and >= 500 steps (about 100 and 50 steps a minute);
* ``best30``: median over days of the best 30 minutes (three consecutive bins), steps per minute.

A kept day has at least :data:`MIN_HOURS` worn hours in the window (NHANES: an hour with fewer than 60 minutes of
zero intensity counts as worn; the watch: an hour with any bin). Two or more features outside the training range
refuse the input, as for the activity head.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

HOURS = range(8, 20)
MIN_HOURS = 8
STEP_FEATURES = ["log_daily_steps", "empty_daytime", "brisk", "moderate", "best30"]
FEATURES = [f"share_h{h:02d}" for h in HOURS] + ["daily_cv"] + STEP_FEATURES


def bin_features(day: np.ndarray, hour: np.ndarray, steps10: np.ndarray, worn_hour: np.ndarray,
                 min_days: int = 4) -> dict[str, float] | None:
    """Features of one person's 10-minute step bins: ``day`` index, clock ``hour``, steps per bin, and whether the
    bin's hour was worn (one entry per bin). None with fewer than ``min_days`` kept days."""
    day, hour, s, worn = (np.asarray(a) for a in (day, hour, steps10, worn_hour))
    s = s.astype(float)
    win = np.isin(hour, list(HOURS)) & worn.astype(bool)
    days = [d for d in np.unique(day[win]) if len(np.unique(hour[win & (day == d)])) >= MIN_HOURS]
    if len(days) < min_days:
        return None
    keep = win & np.isin(day, days)
    dk, hk, sk = day[keep], hour[keep], s[keep]
    total = sk.sum()
    if total <= 0:
        return None
    # each hour's mean steps per worn bin, then shares, so a day missing an hour does not bend the shape
    profile = np.array([sk[hk == h].mean() if (hk == h).any() else np.nan for h in HOURS])
    profile = np.where(np.isfinite(profile), profile, np.nanmean(profile))
    out = {f"share_h{h:02d}": float(profile[i] / profile.sum()) for i, h in enumerate(HOURS)}
    daily = np.array([sk[dk == d].sum() for d in days])
    out["daily_cv"] = float(daily.std() / max(daily.mean(), 1e-12))
    out["log_daily_steps"] = float(np.log1p(daily.mean()))
    out["empty_daytime"] = float(np.mean(sk == 0))
    out["brisk"] = float(np.mean(sk >= 1000))
    out["moderate"] = float(np.mean(sk >= 500))
    best = [np.max(np.convolve(sk[dk == d], np.ones(3), mode="valid")) / 30.0 for d in days if (dk == d).sum() >= 3]
    out["best30"] = float(np.median(best)) if best else float("nan")
    return out


def nhanes_2005(folder: Path, min_age: float = 18.0, chunk: int = 2_000_000) -> pl.DataFrame:
    """One row per NHANES 2005-2006 adult with four or more worn days: ``subject``, ``age``, ``sex`` and :data:`FEATURES`.

    PAXRAW_D is read in chunks (tens of millions of minutes); minutes are summed into 10-minute bins per person.
    A worn hour has fewer than 60 minutes of zero intensity (Troiano's non-wear rule, applied per hour).
    """
    import zipfile

    import pandas as pd

    folder = Path(folder)
    demo = pd.read_sas(folder / "DEMO_D.xpt", format="xport", encoding="latin-1")
    age = dict(zip(demo.SEQN.astype(int), demo.RIDAGEYR))
    sex = dict(zip(demo.SEQN.astype(int), demo.RIAGENDR))
    xpt = folder / "PAXRAW_D.xpt"
    if not xpt.is_file():
        with zipfile.ZipFile(folder / "PAXRAW_D.zip") as z:
            name = next(n for n in z.namelist() if n.lower().endswith(".xpt"))
            z.extract(name, folder)
            (folder / name).rename(xpt)
    rows, pending = [], None

    def finish(g: pd.DataFrame) -> None:
        seqn = int(g.SEQN.iloc[0])
        a = age.get(seqn, np.nan)
        if not np.isfinite(a) or a < min_age:
            return
        if "PAXSTAT" in g and (g.PAXSTAT != 1).mean() > 0.5:  # mostly unreliable
            return
        if "PAXCAL" in g and (g.PAXCAL == 2).any():  # out of calibration
            return
        n = g.PAXN.to_numpy().astype(int)
        day = (n - 1) // 1440
        hour = g.PAXHOUR.to_numpy().astype(int)
        b = (n - 1) // 10
        steps = np.nan_to_num(g.PAXSTEP.to_numpy().astype(float))
        zero = (np.nan_to_num(g.PAXINTEN.to_numpy()) == 0)
        f = pd.DataFrame({"b": b, "day": day, "hour": hour, "steps": steps, "zero": zero, "hk": day * 24 + hour})
        worn_h = f.groupby("hk").zero.sum() < 60
        bins = f.groupby("b").agg(day=("day", "first"), hour=("hour", "first"), steps=("steps", "sum"),
                                  hk=("hk", "first"))
        feat = bin_features(bins.day.to_numpy(), bins.hour.to_numpy(), bins.steps.to_numpy(),
                            worn_h.reindex(bins.hk).fillna(False).to_numpy())
        if feat is not None:
            rows.append({"subject": f"nhanes05/{seqn}", "age": float(a), "sex": float(sex.get(seqn, np.nan)), **feat})

    for part in pd.read_sas(xpt, format="xport", encoding="latin-1", chunksize=chunk):
        part = part.assign(SEQN=part.SEQN.astype(int))
        # SAS transport stores zero as an IBM float that pandas decodes as ~5.4e-79, not 0: every "== 0" test
        # (a minute without steps, a minute of zero intensity) would fail on it
        num = part.select_dtypes("number").columns
        part[num] = part[num].mask(part[num].abs() < 1e-30, 0.0)
        if pending is not None:
            part = pd.concat([pending, part])
        last = part.SEQN.iloc[-1]
        done, pending = part[part.SEQN != last], part[part.SEQN == last]
        for _, g in done.groupby("SEQN", sort=False):
            finish(g)
    if pending is not None and len(pending):
        finish(pending)
    return pl.DataFrame(rows)


def samsung(step_bins: pl.DataFrame) -> dict[str, float] | None:
    """The same features from a Samsung Health export's ``step_bins`` (10-minute bins, local time)."""
    s = step_bins.with_columns(pl.col("start").dt.truncate("10m").alias("b")).group_by("b").agg(
        pl.col("steps").sum()).sort("b")
    if s.is_empty():
        return None
    full = pl.DataFrame({"b": pl.datetime_range(s["b"].min(), s["b"].max(), "10m", eager=True)})
    s = full.join(s, on="b", how="left")
    worn_bin = s["steps"].is_not_null()
    s = s.with_columns(worn_bin.alias("w"), pl.col("b").dt.truncate("1h").alias("h"))
    worn_hour = s.group_by("h").agg(pl.col("w").any().alias("wh"))
    s = s.join(worn_hour, on="h", how="left")
    return bin_features(s["b"].dt.date().cast(pl.Int32).to_numpy(), s["b"].dt.hour().to_numpy(),
                        s["steps"].fill_null(0.0).to_numpy(), s["wh"].to_numpy())
