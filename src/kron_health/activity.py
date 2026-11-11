"""The activity head: age from a week of hourly activity, as NHANES measures it and a watch's step bins give it.

Features are computed from one hourly series per person (clock hour, activity, whether the hour was worn), so
the NHANES 2011-2014 wrist monitor (hourly MIMS, ActiGraph) and a Samsung Health export (10-minute step bins summed
to hours) are described the same way. Most are **unit-free**, since MIMS and steps do not share a scale:

* the 24-hour profile: each clock hour's share of the mean day's activity;
* rest-activity rhythm (van Someren): interdaily stability (IS), intradaily variability (IV), relative amplitude
  RA = (M10 - L5) / (M10 + L5), and the clock times of M10 (the most active 10 hours) and L5 (the least active 5);
* day-to-day variation of the daily total (coefficient of variation) and the share of hours with almost nothing;
* sleep minutes per day where the source classifies them (NHANES's sleep-wear minutes; a watch's sleep sessions).

``log_total`` (the mean daily activity in the source's own unit) is kept apart: it helps within one device and
does not transfer to another.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

UNIT_FREE = [f"share_h{h:02d}" for h in range(24)] + ["is", "iv", "ra", "m10_sin", "m10_cos", "l5_sin", "l5_cos",
                                                         "daily_cv", "quiet_share", "sleep_min"]
ALL = UNIT_FREE + ["log_total"]


def _circ(hour: float) -> tuple[float, float]:
    a = 2 * np.pi * hour / 24
    return float(np.sin(a)), float(np.cos(a))


def rhythm(values: np.ndarray, hours: np.ndarray, days: np.ndarray, worn: np.ndarray,
           sleep_min: np.ndarray | None = None, min_days: int = 3) -> dict[str, float] | None:
    """Features of one person's hourly series (``hours`` 0-23, ``days`` an index per calendar day).

    Only days with at least 20 worn hours count, and at least ``min_days`` of them; otherwise None.
    """
    v = np.asarray(values, float)
    h = np.asarray(hours, int)
    d = np.asarray(days)
    w = np.asarray(worn, bool)
    good_days = [x for x in np.unique(d) if w[d == x].sum() >= 20]
    if len(good_days) < min_days:
        return None
    keep = np.isin(d, good_days) & w
    v, h, d = v[keep], h[keep], d[keep]
    if v.sum() <= 0:
        return None
    profile = np.array([v[h == k].mean() if (h == k).any() else np.nan for k in range(24)])
    profile = np.where(np.isfinite(profile), profile, np.nanmean(profile))
    out = {f"share_h{k:02d}": float(profile[k] / profile.sum()) for k in range(24)}
    mean = v.mean()
    n = len(v)
    out["is"] = float(n * np.sum((profile - mean) ** 2) / (24 * np.sum((v - mean) ** 2))) if np.var(v) > 0 else 0.0
    out["iv"] = float(n * np.sum(np.diff(v) ** 2) / ((n - 1) * np.sum((v - mean) ** 2))) if np.var(v) > 0 else 0.0
    ring = np.concatenate([profile, profile])
    m10 = np.array([ring[k:k + 10].mean() for k in range(24)])
    l5 = np.array([ring[k:k + 5].mean() for k in range(24)])
    m10_at, l5_at = int(np.argmax(m10)), int(np.argmin(l5))
    out["ra"] = float((m10[m10_at] - l5[l5_at]) / max(m10[m10_at] + l5[l5_at], 1e-12))
    out["m10_sin"], out["m10_cos"] = _circ(m10_at + 5)
    out["l5_sin"], out["l5_cos"] = _circ(l5_at + 2.5)
    daily = np.array([v[d == x].sum() for x in good_days])
    out["daily_cv"] = float(daily.std() / max(daily.mean(), 1e-12))
    out["quiet_share"] = float(np.mean(v < 0.02 * profile.max()))
    out["sleep_min"] = float(np.nanmean(sleep_min)) if sleep_min is not None and len(sleep_min) else float("nan")
    out["log_total"] = float(np.log1p(daily.mean()))
    return out


def nhanes(folder: Path, min_age: float = 18.0) -> pl.DataFrame:
    """One row per NHANES 2011-2014 participant with an accepted week: ``subject``, ``age``, ``sex`` and :data:`ALL`."""
    import pandas as pd

    rows = []
    for cycle in ("G", "H"):
        demo = pd.read_sas(Path(folder) / f"DEMO_{cycle}.xpt", format="xport", encoding="latin-1")
        head = pd.read_sas(Path(folder) / f"PAXHD_{cycle}.xpt", format="xport", encoding="latin-1")
        hr = pd.read_sas(Path(folder) / f"PAXHR_{cycle}.xpt", format="xport", encoding="latin-1")
        day = pd.read_sas(Path(folder) / f"PAXDAY_{cycle}.xpt", format="xport", encoding="latin-1")
        age = dict(zip(demo.SEQN.astype(int), demo.RIDAGEYR))
        sex = dict(zip(demo.SEQN.astype(int), demo.RIAGENDR))
        start = {int(s): int(str(t).split(":")[0]) for s, t in zip(head.SEQN, head.PAXFTIME)
                 if isinstance(t, str) and ":" in t}
        ok = set(head.SEQN[head.PAXSTS == 1].astype(int))
        sleep = day.assign(SEQN=day.SEQN.astype(int)).groupby("SEQN").PAXSWMD.apply(
            lambda s: s[s > 1e-6].to_numpy())
        hr = hr.assign(SEQN=hr.SEQN.astype(int)).sort_values(["SEQN", "PAXSSNHP"])
        for seqn, g in hr.groupby("SEQN", sort=False):
            if seqn not in ok or seqn not in start or not np.isfinite(age.get(seqn, np.nan)) or age[seqn] < min_age:
                continue
            k = np.arange(len(g))
            clock = start[seqn] + k  # the first row is the partial hour the recording starts in
            worn = (g.PAXWWMH.to_numpy() + g.PAXSWMH.to_numpy()) >= 45
            f = rhythm(g.PAXMTSH.to_numpy(), clock % 24, clock // 24, worn,
                       sleep_min=sleep.get(seqn, np.array([])))
            if f is None:
                continue
            rows.append({"subject": f"nhanes/{cycle}{seqn}", "age": float(age[seqn]), "sex": float(sex[seqn]), **f})
    return pl.DataFrame(rows)


def samsung(step_bins: pl.DataFrame, sleeps: pl.DataFrame | None = None) -> dict[str, float] | None:
    """The same features from a Samsung Health export (the extractor's ``step_bins`` and ``sleep_sessions``).

    Step bins are summed per local clock hour; an hour counts as worn when any bin covers it (the export has no
    wear flag: a day without the watch has no bins). Sleep minutes per day come from sessions of 3 h or more.
    """
    s = (step_bins.with_columns(pl.col("start").dt.truncate("1h").alias("hour"))
         .group_by("hour").agg(pl.col("steps").sum()).sort("hour"))
    if s.is_empty():
        return None
    full = pl.DataFrame({"hour": pl.datetime_range(s["hour"].min(), s["hour"].max(), "1h", eager=True)})
    s = full.join(s, on="hour", how="left")
    worn = s["steps"].is_not_null().to_numpy()
    values = s["steps"].fill_null(0.0).to_numpy()
    hours = s["hour"].dt.hour().to_numpy()
    days = s["hour"].dt.date().cast(pl.Int32).to_numpy()
    sleep_min = None
    if sleeps is not None and not sleeps.is_empty():
        nights = sleeps.with_columns(((pl.col("end") - pl.col("start")).dt.total_minutes()).alias("m")) \
            .filter(pl.col("m") >= 180)
        sleep_min = nights["m"].to_numpy().astype(float)
    return rhythm(values, hours, days, worn, sleep_min)
