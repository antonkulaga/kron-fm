"""Per-person age from wrist PPG with a frozen encoder: embeddings, a ridge head, and honest cross-validation.

The wrist sets with ages are small (tens of people each, a few hundred in all), so one hashed 70/15/15 split
leaves a test set of a few dozen people and an MAE whose interval is wider than the effect. This module
instead:

1. **Embeds** at most ``cap`` windows per person (spread over the record) with a frozen encoder and averages
   them per person: one vector per person, the same pooled embedding the age head reads.
2. **Fits a ridge head** (standardised features, the penalty chosen by generalised cross-validation inside
   the training fold) in repeated grouped K-fold: every person is predicted only by a head that never saw
   them; predictions are averaged over the repeats.
3. **Scores against baselines**, per person, with intervals from a bootstrap over people:
   the training fold's median age (the constant baseline) and, where several sources are pooled, the
   training fold's median age *of the person's own source*. The second matters: sources differ in age
   (Vilnius patients are 37-79, the lab sets 20-35), and a head that only recognises the device beats the
   constant baseline without reading any age from the pulse. A claim needs the source-median baseline beaten.
4. **A permutation null**: the same cross-validation with ages shuffled within each source; the real MAE is
   compared with that distribution.
5. **Leave one source out**: each source predicted by a head fitted on the others, the cross-device number.

The final head (fitted on everyone) folds into a linear layer over the pooled embedding, so the deployed
clock is the encoder's stream graph plus one dot product (:func:`linear_head`).
"""
from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from kron_ppg.data import Pool
from kron_ppg.model import KronPPG, patchify


# ---- embeddings ---------------------------------------------------------------------------------------
def spread_index(subjects: np.ndarray, cap: int, seed: int = 0) -> np.ndarray:
    """At most ``cap`` window indices per subject, evenly spread over each subject's windows in order."""
    keep = []
    for s in np.unique(subjects):
        mine = np.flatnonzero(subjects == s)
        if len(mine) > cap:
            mine = mine[np.unique(np.linspace(0, len(mine) - 1, cap).round().astype(int))]
        keep.append(mine)
    return np.sort(np.concatenate(keep)) if keep else np.zeros(0, dtype=int)


def lowpass(x: np.ndarray, fs: float, cutoff: float) -> np.ndarray:
    """Windows ``(n, L)`` with every frequency above ``cutoff`` Hz removed (an FFT brick wall).

    What a watch sampling at ``2 * cutoff`` Hz keeps: Galaxy Watch Active 2 records at 10 Hz, so ``cutoff`` 5
    shows what a model loses on its signal.
    """
    spec = np.fft.rfft(x.astype(np.float64), axis=1)
    spec[:, np.fft.rfftfreq(x.shape[1], 1 / fs) > cutoff] = 0
    return np.fft.irfft(spec, n=x.shape[1], axis=1).astype(np.float32)


@torch.no_grad()
def embed(encoder: KronPPG, pool: Pool, idx: np.ndarray, batch_size: int = 256,
          device: torch.device | str = "cpu", cutoff: float | None = None) -> np.ndarray:
    """The pooled embedding ``(len(idx), d_model)`` of each window at ``idx`` (low-passed at ``cutoff`` Hz)."""
    encoder.eval()
    dev = torch.device(device)
    out = []
    for start in range(0, len(idx), batch_size):
        chunk = idx[start:start + batch_size]
        w = np.stack([pool.window(int(i)) for i in chunk])
        if cutoff:
            w = lowpass(w, pool.sample_rate, cutoff)
        x = torch.from_numpy(w).to(dev)
        patches, ok = patchify(x, encoder.config.patch)
        out.append(encoder.embedding(encoder.encode(patches, ok), ok).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, encoder.config.d_model), np.float32)


def handcrafted(x: np.ndarray, fs: float) -> np.ndarray:
    """A baseline without any encoder: per window, the pulse rate and the signal's simple statistics.

    Heart rate (autocorrelation peak in 40-180 bpm) and its peak height (regularity), the power in four
    bands of the spectrum, the skewness and kurtosis of the signal and of its derivative (pulse shape),
    and the second-derivative energy. ``x`` is ``(n, L)`` scaled windows; returns ``(n, 12)``.
    """
    x = x.astype(np.float64)
    x = x - x.mean(1, keepdims=True)
    n, length = x.shape
    spec = np.abs(np.fft.rfft(x, axis=1)) ** 2
    freqs = np.fft.rfftfreq(length, 1 / fs)
    total = spec[:, (freqs > 0.5) & (freqs < 8)].sum(1) + 1e-12
    bands = [spec[:, (freqs >= lo) & (freqs < hi)].sum(1) / total for lo, hi in ((0.5, 1.0), (1.0, 2.0),
                                                                                (2.0, 4.0), (4.0, 8.0))]
    ac = np.fft.irfft(np.abs(np.fft.rfft(x, 2 * length, axis=1)) ** 2, axis=1)[:, :length]
    ac = ac / np.maximum(ac[:, :1], 1e-12)
    lo, hi = int(fs * 60 / 180), int(fs * 60 / 40)
    lag = lo + ac[:, lo:hi + 1].argmax(1)
    peak = ac[np.arange(n), lag]

    def moments(v):
        v = v - v.mean(1, keepdims=True)
        sd = v.std(1) + 1e-12
        return (v ** 3).mean(1) / sd ** 3, (v ** 4).mean(1) / sd ** 4

    d1 = np.diff(x, axis=1)
    d2 = np.diff(d1, axis=1)
    s0, k0 = moments(x)
    s1, k1 = moments(d1)
    return np.stack([60 * fs / lag, peak, *bands, s0, k0, s1, k1,
                     (d2 ** 2).mean(1) / ((d1 ** 2).mean(1) + 1e-12), (d1 ** 2).mean(1) / ((x ** 2).mean(1) + 1e-12)],
                    axis=1)


@dataclass
class People:
    """One row per person: the mean window embedding, age, source and window count."""

    subject: np.ndarray
    source: np.ndarray
    age: np.ndarray
    x: np.ndarray
    windows: np.ndarray

    def save(self, path: Path) -> None:
        np.savez(path, subject=self.subject.astype(str), source=self.source.astype(str), age=self.age,
                 x=self.x, windows=self.windows)

    @classmethod
    def load(cls, path: Path) -> "People":
        z = np.load(path)
        return cls(z["subject"], z["source"], z["age"], z["x"], z["windows"])


def per_person(subjects: np.ndarray, ages: np.ndarray, emb: np.ndarray) -> People:
    keys = np.unique(subjects)
    rows = [np.flatnonzero(subjects == k) for k in keys]
    return People(subject=keys, source=np.array([k.split("/", 1)[0] for k in keys]),
                  age=np.array([ages[r[0]] for r in rows], dtype=float),
                  x=np.stack([emb[r].mean(0) for r in rows]).astype(np.float64),
                  windows=np.array([len(r) for r in rows]))


# ---- ridge with the penalty chosen by GCV -------------------------------------------------------------
ALPHAS = np.logspace(-3, 4, 29)


@dataclass
class Ridge:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    alpha: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.scale) @ self.coef + self.intercept


def fit_ridge(x: np.ndarray, y: np.ndarray, alphas: np.ndarray = ALPHAS) -> Ridge:
    """Ridge on standardised features; ``alpha`` (per sample, so it scales with n) by generalised CV."""
    mean, scale = x.mean(0), x.std(0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    z = (x - mean) / scale
    y0 = y.mean()
    u, s, vt = np.linalg.svd(z, full_matrices=False)
    uty = u.T @ (y - y0)
    n = len(y)
    best = None
    for alpha in alphas:
        lam = alpha * n
        d = s ** 2 / (s ** 2 + lam)
        fit = u @ (d * uty)
        resid = (y - y0) - fit
        dof = d.sum() + 1.0
        gcv = float((resid ** 2).mean() / max(1.0 - dof / n, 1e-6) ** 2)
        if best is None or gcv < best[0]:
            best = (gcv, alpha, vt.T @ (s / (s ** 2 + lam) * uty))
    _, alpha, coef = best
    return Ridge(mean, scale, coef, float(y0), float(alpha))


def grouped_folds(n: int, k: int, rng: np.random.Generator, strata: np.ndarray | None = None) -> np.ndarray:
    """A fold number per person, ``k`` folds, balanced within each stratum (source) when given."""
    fold = np.empty(n, dtype=int)
    strata = np.zeros(n, dtype=int) if strata is None else strata
    offset = 0
    for s in np.unique(strata):
        mine = rng.permutation(np.flatnonzero(strata == s))
        fold[mine] = (np.arange(len(mine)) + offset) % k
        offset += len(mine)
    return fold


def cross_validate(p: People, folds: int = 10, repeats: int = 5, seed: int = 0,
                   age: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Out-of-fold predictions per person (mean over repeats) and both baselines, fold for fold."""
    y = p.age if age is None else age
    rng = np.random.default_rng(seed)
    pred = np.zeros((repeats, len(y)))
    base = np.zeros((repeats, len(y)))
    base_src = np.zeros((repeats, len(y)))
    for r in range(repeats):
        fold = grouped_folds(len(y), folds, rng, p.source)
        for f in range(folds):
            test, train = fold == f, fold != f
            if not test.any():
                continue
            pred[r, test] = fit_ridge(p.x[train], y[train]).predict(p.x[test])
            base[r, test] = np.median(y[train])
            for s in np.unique(p.source[test]):
                own = train & (p.source == s)
                base_src[r, test & (p.source == s)] = np.median(y[own]) if own.any() else np.median(y[train])
    return {"pred": pred.mean(0), "base": base.mean(0), "base_source": base_src.mean(0)}


def leave_source_out(p: People) -> dict[str, dict]:
    """Each source predicted by a head fitted on all the others (needs at least two sources)."""
    out = {}
    for s in np.unique(p.source):
        test = p.source == s
        if test.all():
            continue
        pred = fit_ridge(p.x[~test], p.age[~test]).predict(p.x[test])
        out[str(s)] = {"people": int(test.sum()), "mae": float(np.abs(pred - p.age[test]).mean()),
                       "mae_constant": float(np.abs(np.median(p.age[~test]) - p.age[test]).mean()),
                       "r": _r(pred, p.age[test])}
    return out


# ---- scoring ------------------------------------------------------------------------------------------
def _r(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def bootstrap(err: dict[str, np.ndarray], n: int = 2000, seed: int = 0) -> dict[str, list[float]]:
    """95 % intervals of each MAE and of each paired difference to the model, resampling people."""
    rng = np.random.default_rng(seed)
    m = len(next(iter(err.values())))
    draws = rng.integers(0, m, size=(n, m))
    out = {}
    for k, e in err.items():
        means = e[draws].mean(1)
        out[k] = [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]
    for k in err:
        if k != "model":
            d = (err["model"] - err[k])[draws].mean(1)
            out[f"model_minus_{k}"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    return out


def score(age: np.ndarray, cv: dict[str, np.ndarray], seed: int = 0) -> dict:
    err = {"model": np.abs(cv["pred"] - age), "constant": np.abs(cv["base"] - age),
           "source_median": np.abs(cv["base_source"] - age)}
    return {"people": int(len(age)), "mae": float(err["model"].mean()),
            "mae_constant": float(err["constant"].mean()), "mae_source_median": float(err["source_median"].mean()),
            "r": _r(cv["pred"], age), "interval95": bootstrap(err, seed=seed)}


def permutation_null(p: People, n: int = 20, folds: int = 10, repeats: int = 2, seed: int = 0) -> dict:
    """MAE of the same cross-validation with ages shuffled within each source, ``n`` times."""
    rng = np.random.default_rng(seed + 1)
    maes = []
    for _ in range(n):
        shuffled = p.age.copy()
        for s in np.unique(p.source):
            mine = np.flatnonzero(p.source == s)
            shuffled[mine] = rng.permutation(shuffled[mine])
        cv = cross_validate(p, folds, repeats, int(rng.integers(1 << 31)), age=shuffled)
        maes.append(float(np.abs(cv["pred"] - shuffled).mean()))
    return {"n": n, "mae": maes, "mean": float(np.mean(maes)), "min": float(np.min(maes))}


def evaluate(p: People, folds: int = 10, repeats: int = 5, null: int = 20, seed: int = 0) -> tuple[dict, dict]:
    """The whole report for one set of people, and the out-of-fold predictions."""
    folds = min(folds, len(p.age))
    cv = cross_validate(p, folds, repeats, seed)
    result = {"pooled": score(p.age, cv, seed)}
    result["per_source"] = {}
    for s in np.unique(p.source):
        mine = p.source == s
        if mine.sum() >= 3:
            result["per_source"][str(s)] = {**score(p.age[mine], {k: v[mine] for k, v in cv.items()}, seed),
                                            "age_range": [float(p.age[mine].min()), float(p.age[mine].max())]}
    if null:
        result["null"] = permutation_null(p, null, folds, 2, seed)
        result["null"]["p_value"] = float((1 + sum(m <= result["pooled"]["mae"] for m in result["null"]["mae"]))
                                          / (1 + null))
    if len(np.unique(p.source)) > 1:
        result["leave_source_out"] = leave_source_out(p)
    final = fit_ridge(p.x, p.age)
    result["final_head"] = {"alpha": final.alpha, "people": int(len(p.age))}
    return result, {**cv, "final": final}


def linear_head(r: Ridge) -> tuple[np.ndarray, float]:
    """The ridge folded into ``age = w @ pooled + b`` for the deployed graph."""
    w = r.coef / r.scale
    return w, float(r.intercept - (r.mean / r.scale) @ r.coef)


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (np.floating, np.integer)):
        return json_safe(value.item())
    return value


def run(folders: Sequence[Path], encoder_path: Path | str | None, out: Path, cap: int = 200, folds: int = 10,
        repeats: int = 5, null: int = 20, batch_size: int = 256, device: str = "cuda", seed: int = 0,
        cutoff: float | None = None) -> dict:
    """Embed (``encoder_path`` a checkpoint, ``"random"`` for an untrained encoder, None for hand-made
    features), cross-validate and write ``people.npz``, ``predictions.parquet``, ``metrics.json``, ``head.pt``."""
    import polars as pl

    from kron_ppg.train import load_encoder

    out.mkdir(parents=True, exist_ok=True)
    cache = out / "people.npz"
    if cache.is_file():
        people = People.load(cache)
    else:
        pool = Pool.load(folders)
        aged = np.flatnonzero(np.isfinite(pool.age))
        idx = aged[spread_index(pool.subject[aged], cap, seed)]
        if encoder_path is None:  # the hand-made baseline
            def windows(part):
                w = np.stack([pool.window(int(i)) for i in part])
                return lowpass(w, pool.sample_rate, cutoff) if cutoff else w

            emb = np.concatenate([handcrafted(windows(idx[k:k + 4096]), pool.sample_rate)
                                  for k in range(0, len(idx), 4096)])
        else:
            if str(encoder_path) == "random":  # an untrained encoder of the default shape
                from kron_ppg.model import KronPPGConfig

                torch.manual_seed(seed)
                encoder = KronPPG(KronPPGConfig(sample_rate=pool.sample_rate)).to(device)
            else:
                encoder = load_encoder(encoder_path, device)
            emb = embed(encoder, pool, idx, batch_size, device, cutoff)
        people = per_person(pool.subject[idx], pool.age[idx], emb)
        people.save(cache)
    result, cv = evaluate(people, folds, repeats, null, seed)
    result = {"encoder": str(encoder_path), "data": [str(f) for f in folders], "cap": cap, "lowpass_hz": cutoff,
              "folds": folds,
              "repeats": repeats, **result}
    (out / "metrics.json").write_text(json.dumps(json_safe(result), indent=2) + "\n")
    pl.DataFrame({"subject": people.subject.astype(str), "source": people.source.astype(str), "age": people.age,
                  "predicted": cv["pred"], "constant": cv["base"], "source_median": cv["base_source"],
                  "windows": people.windows}).write_parquet(out / "predictions.parquet")
    w, b = linear_head(cv["final"])
    torch.save({"encoder": str(encoder_path), "weight": torch.tensor(w, dtype=torch.float32), "bias": b,
                "alpha": cv["final"].alpha}, out / "head.pt")
    return result


def transfer(train: Sequence[People], test: People, calibrate: People | None = None, seed: int = 0) -> dict:
    """A head fitted on ``train`` people (other sources), scored on ``test`` people it never saw.

    ``calibrate``: people of the target kind (for example open wrist sets) on which a linear map
    ``age = a + b * pred`` is fitted after the head, to move a finger-fitted scale onto the wrist; the test
    people are never used for it. Scores against the test set's own median (an oracle constant that knows the
    target's age distribution, so a harder baseline than any constant the head could have learnt).
    """
    x = np.concatenate([p.x for p in train])
    y = np.concatenate([p.age for p in train])
    head = fit_ridge(x, y)
    pred = head.predict(test.x)
    out = {"train_people": int(len(y)), "test_people": int(len(test.age)), "alpha": head.alpha}
    if calibrate is not None:
        c = head.predict(calibrate.x)
        b, a = np.polyfit(c, calibrate.age, 1)
        pred = a + b * pred
        out["calibration"] = {"people": int(len(calibrate.age)), "a": float(a), "b": float(b)}
    err = {"model": np.abs(pred - test.age), "oracle_median": np.abs(np.median(test.age) - test.age),
           "train_median": np.abs(np.median(y) - test.age)}
    out.update({"mae": float(err["model"].mean()), "mae_oracle_median": float(err["oracle_median"].mean()),
                "mae_train_median": float(err["train_median"].mean()), "r": _r(pred, test.age),
                "mean_error": float((pred - test.age).mean()), "interval95": bootstrap(err, seed=seed)})
    # the ranking alone, with the scale refitted on the test people (an upper bound, not a deployable number)
    b, a = np.polyfit(pred, test.age, 1)
    out["mae_rescaled_on_test"] = float(np.abs(a + b * pred - test.age).mean())
    return out
