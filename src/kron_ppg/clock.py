"""Scoring a PPG age clock the way aging clocks are judged: per person, bias-corrected, against outcomes.

* **Per person.** Window predictions are averaged per subject before scoring, as Apple's PpgAge and the
  OPPO heart-age protocol average many checks per person; window-level MAE overstates what a single
  check can say.
* **Bias correction.** Every regression clock regresses toward the training mean (young people read
  old, old people young). The *age gap* used downstream is the residual after regressing prediction on
  age, ``gap = pred - (a + b * age)``, the standard correction (Le Goallec & Patel; Apple's PpgAge uses the
  same residual). It is fitted on the **validation** subjects, never the training ones: a model that fits
  its training subjects closely gives a slope near 1 there, which leaves the held-out gap strongly
  anti-correlated with age (r -0.3 to -0.6 in kron-ppg's first run), and every age-linked outcome then
  reads as a *younger* gap. The test gap is the one to read.
* **Outcome evidence over MAE.** A clock is worth something only if its gap carries information beyond
  age: AI-PPG Age's gap > 9 y gave a MACCE hazard ratio of 2.37 in UK Biobank, PpgAge's 6-year gap a
  hazard ratio of 1.40 for cardiac events. Where a source carries outcome or disease flags (VitalDB's
  ``preop_htn``, ``preop_dm``, ``death_inhosp``; ASA class >= 3), :func:`gap_associations` reports the
  AUC of the corrected gap for each, on held-out subjects, beside the AUC of age alone; 0.5 means the gap
  carries nothing, and a gap AUC far below age's says the clock adds little to the calendar.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import numpy as np

#: Subject-table columns read as binary outcomes when present, and how a value counts as positive.
OUTCOME_FLAGS: dict[str, str] = {
    "preop_htn": "== 1", "preop_dm": "== 1", "death_inhosp": "== 1", "asa": ">= 3",
}


def json_safe(value):
    """``value`` with every NaN or infinity replaced by ``None``, so the result is strict JSON."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def per_subject(subjects: Iterable[str], preds: Iterable[float], ages: Iterable[float]
                ) -> dict[str, tuple[float, float, int]]:
    """``subject -> (mean prediction, age, windows)``."""
    acc: dict[str, list] = {}
    for s, p, a in zip(subjects, preds, ages, strict=True):
        entry = acc.setdefault(str(s), [0.0, float(a), 0])
        entry[0] += float(p)
        entry[2] += 1
    return {s: (total / n, age, n) for s, (total, age, n) in acc.items()}


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def fit_bias(pred: np.ndarray, age: np.ndarray) -> tuple[float, float]:
    """``(a, b)`` of ``pred ~ a + b * age`` by least squares; the identity when it cannot be fitted."""
    if len(age) < 3 or np.std(age) == 0:
        return 0.0, 1.0
    b, a = np.polyfit(age, pred, 1)
    return float(a), float(b)


def scores(pred: np.ndarray, age: np.ndarray, bias: tuple[float, float]) -> dict[str, float]:
    """MAE, RMSE and Pearson r of per-subject predictions, the age-only baseline's MAE (predicting the
    median age), and the corrected gap's spread and its residual correlation with age."""
    pred, age = np.asarray(pred, float), np.asarray(age, float)
    a, b = bias
    gap = pred - (a + b * age)
    return {
        "subjects": int(len(age)),
        "mae": float(np.mean(np.abs(pred - age))) if len(age) else float("nan"),
        "rmse": float(np.sqrt(np.mean((pred - age) ** 2))) if len(age) else float("nan"),
        "r": pearson(pred, age),
        "baseline_mae_median_age": float(np.mean(np.abs(age - np.median(age)))) if len(age) else float("nan"),
        "gap_sd": float(np.std(gap)) if len(age) else float("nan"),
        "gap_age_r": pearson(gap, age),
    }


def auc(score: np.ndarray, positive: np.ndarray) -> float:
    """Mann-Whitney AUC of ``score`` for ``positive`` (ties count half); NaN without both classes."""
    score, positive = np.asarray(score, float), np.asarray(positive, bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    sorted_scores = score[order]
    i = 0
    while i < len(score):
        j = i
        while j + 1 < len(score) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _flag(value: str, rule: str) -> bool | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(v):
        return None
    op, threshold = rule.split()
    return v >= float(threshold) if op == ">=" else v == float(threshold)


def gap_associations(gaps: Mapping[str, float], table: Mapping[str, Mapping[str, str]],
                     flags: Mapping[str, str] = OUTCOME_FLAGS,
                     ages: Mapping[str, float] | None = None) -> dict[str, dict[str, float]]:
    """For each outcome column present: positives, negatives, the AUC of the gap, the mean gap of each class
    and, with ``ages``, the AUC of age alone for comparison."""
    out = {}
    for column, rule in flags.items():
        rows = [(s, g, _flag(table.get(s, {}).get(column, ""), rule)) for s, g in gaps.items()]
        rows = [r for r in rows if r[2] is not None]
        if not rows:
            continue
        g = np.array([r[1] for r in rows])
        f = np.array([r[2] for r in rows])
        entry = {"positive": int(f.sum()), "negative": int((~f).sum()), "auc_of_gap": auc(g, f),
                 "mean_gap_positive": float(g[f].mean()) if f.any() else float("nan"),
                 "mean_gap_negative": float(g[~f].mean()) if (~f).any() else float("nan")}
        if ages is not None:
            entry["auc_of_age"] = auc(np.array([ages[r[0]] for r in rows]), f)
        out[column] = entry
    return out
