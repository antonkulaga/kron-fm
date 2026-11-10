"""Linear probes of frozen kron-ppg encoders on a set's people, split by person.

A frozen encoder embeds up to ``per_subject`` windows of each person (the pooled hidden state of each window,
averaged over the person's windows); a ridge model reads a per-person target from that embedding. People, not
windows, are the unit throughout: K-fold cross-validation over people gives every person an out-of-fold
prediction (the ridge penalty chosen by an inner K-fold on the training people only), and the intervals are a
bootstrap over people. A binary target is fitted as 0/1 and scored by AUC (only the ranking matters); a
regression target by MAE and Pearson r, beside the MAE of predicting the training median.

Encoders compared side by side on the same people and folds: any pretraining or clock checkpoint, and
``untrained:<checkpoint>``, a randomly initialised twin with that checkpoint's configuration.

``python -m kron_ppg.wrist_probe OUT.json SOURCE[,SOURCE] --target any_af:binary --target age:regression
--encoder r4=CKPT --encoder twin=untrained:CKPT [--per-subject 400 --folds 5 --boot 2000 --device cuda]``

A restricted source's results are written owner-only; a ``no_join`` source (Aurora-BP) is probed alone (the
window pool refuses to mix it), which is what its agreement allows for a frozen encoder from elsewhere.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from kron_ppg.clock import auc, json_safe, pearson
from kron_ppg.data import Pool, variant_dir
from kron_ppg.model import KronPPG, KronPPGConfig, patchify
from kron_ppg.watch_sources import owner_only

ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)


def load_frozen(spec: str, device: torch.device | str = "cpu", seed: int = 0) -> KronPPG:
    """A checkpoint's encoder, or with ``untrained:<checkpoint>`` a random twin of its configuration."""
    from kron_ppg.train import load_encoder

    if spec.startswith("untrained:"):
        ckpt = torch.load(spec.split(":", 1)[1], map_location="cpu", weights_only=True)
        torch.manual_seed(seed)
        encoder = KronPPG(KronPPGConfig(**ckpt["config"])).to(device)
    else:
        encoder = load_encoder(spec, device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


@torch.no_grad()
def embed_people(encoder: KronPPG, pool: Pool, per_subject: int = 400, seed: int = 0, batch: int = 256,
                 device: torch.device | str = "cpu") -> tuple[list[str], np.ndarray, np.ndarray]:
    """``(keys, embeddings (n, d_model), windows used per person)``: the mean pooled embedding of up to
    ``per_subject`` windows per person, drawn at random without replacement."""
    rng = np.random.default_rng(seed)
    keys = sorted(set(pool.subject))
    chosen = []
    for k in keys:
        idx = np.flatnonzero(pool.subject == k)
        chosen.append(np.sort(rng.choice(idx, min(per_subject, len(idx)), replace=False)))
    flat = np.concatenate(chosen)
    owner = np.concatenate([np.full(len(c), j) for j, c in enumerate(chosen)])
    out = np.zeros((len(keys), encoder.config.d_model))
    for start in range(0, len(flat), batch):
        part = flat[start:start + batch]
        x = torch.from_numpy(np.stack([pool.window(int(i)) for i in part])).to(device)
        patches, ok = patchify(x, encoder.config.patch)
        z = encoder.embedding(encoder.encode(patches, ok), ok).float().cpu().numpy()
        np.add.at(out, owner[start:start + batch], z)
    counts = np.array([len(c) for c in chosen])
    return keys, out / counts[:, None], counts


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """``(mean, sd, w, b)`` of ridge regression on standardised ``x``."""
    mu, sd = x.mean(0), x.std(0) + 1e-8
    z = (x - mu) / sd
    b = float(y.mean())
    w = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ (y - b))
    return mu, sd, w, b


def ridge_predict(model: tuple, x: np.ndarray) -> np.ndarray:
    mu, sd, w, b = model
    return ((x - mu) / sd) @ w + b


def folds_of(n: int, k: int, seed: int) -> np.ndarray:
    """A fold number per item, shuffled, as even as ``n`` allows."""
    return np.random.default_rng(seed).permutation(np.arange(n) % k)


def best_alpha(x: np.ndarray, y: np.ndarray, k: int, seed: int) -> float:
    """The penalty with the lowest inner K-fold squared error (inside the training people only)."""
    k = max(2, min(k, len(y)))
    fold = folds_of(len(y), k, seed + 1)
    err = []
    for a in ALPHAS:
        e = 0.0
        for f in range(k):
            tr, te = fold != f, fold == f
            e += float(((ridge_predict(ridge_fit(x[tr], y[tr], a), x[te]) - y[te]) ** 2).sum())
        err.append(e)
    return ALPHAS[int(np.argmin(err))]


def cv_predict(x: np.ndarray, y: np.ndarray, k: int = 5, seed: int = 0) -> tuple[np.ndarray, list[float]]:
    """Out-of-fold predictions for every person (K-fold over people) and the penalty each fold chose."""
    fold = folds_of(len(y), k, seed)
    pred, alphas = np.full(len(y), np.nan), []
    for f in range(k):
        tr, te = fold != f, fold == f
        a = best_alpha(x[tr], y[tr], k, seed)
        alphas.append(a)
        pred[te] = ridge_predict(ridge_fit(x[tr], y[tr], a), x[te])
    return pred, alphas


def bootstrap(stat, n: int, boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% percentile interval of ``stat(idx)`` over people resampled with replacement (NaN draws dropped)."""
    rng = np.random.default_rng(seed)
    vals = np.array([stat(rng.integers(0, n, n)) for _ in range(boot)])
    vals = vals[np.isfinite(vals)]
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if len(vals) else (math.nan, math.nan)


def target_values(pool: Pool, keys: list[str], column: str, kind: str) -> np.ndarray:
    """A per-person target from the subject table: a number, or for ``binary`` 1 when >= 1 (or M / yes)."""
    out = np.full(len(keys), np.nan)
    for j, k in enumerate(keys):
        value = pool.table.get(k, {}).get(column)
        raw = "" if value is None else str(value).strip()
        if kind == "binary" and raw.lower() in ("m", "male", "yes", "true"):
            out[j] = 1.0
        elif kind == "binary" and raw.lower() in ("f", "female", "no", "false"):
            out[j] = 0.0
        else:
            try:
                v = float(raw)
            except ValueError:
                continue
            out[j] = float(v >= 1) if kind == "binary" else v
    return out


def score(y: np.ndarray, pred: np.ndarray, kind: str, boot: int, seed: int) -> dict:
    n = len(y)
    if kind == "binary":
        pos = y > 0.5
        return {"people": n, "positive": int(pos.sum()), "auc": auc(pred, pos),
                "auc_ci95": bootstrap(lambda i: auc(pred[i], pos[i]), n, boot, seed)}
    med = float(np.median(y))
    return {"people": n, "mae": float(np.abs(pred - y).mean()), "r": pearson(pred, y),
            "mae_ci95": bootstrap(lambda i: float(np.abs(pred[i] - y[i]).mean()), n, boot, seed),
            "r_ci95": bootstrap(lambda i: pearson(pred[i], y[i]), n, boot, seed),
            "baseline_mae_median": float(np.abs(y - med).mean())}


def probe(sources: list[str], targets: list[tuple[str, str]], encoders: dict[str, str], per_subject: int = 400,
          k: int = 5, boot: int = 2000, seed: int = 0, device: str = "cpu", sample_rate: float = 25.0,
          seconds: float = 30.0, folders: list[Path] | None = None) -> dict:
    """Every (encoder, target) on the same people and folds. Paired AUC / MAE differences against the first
    encoder are bootstrapped on the same resampled people."""
    for s in sources:
        owner_only(s)
    pool = Pool.load(folders or [variant_dir(s, sample_rate, seconds) for s in sources])
    result: dict = {"sources": sources, "per_subject": per_subject, "folds": k, "boot": boot, "seed": seed,
                    "encoders": encoders, "targets": {}}
    embeddings = {}
    for name, spec in encoders.items():
        enc = load_frozen(spec, device, seed)
        keys, z, counts = embed_people(enc, pool, per_subject, seed, device=device)
        embeddings[name] = z
        result.setdefault("people", len(keys))
        result.setdefault("windows_per_person_median", float(np.median(counts)))
    first = next(iter(encoders))
    for column, kind in targets:
        y_all = target_values(pool, keys, column, kind)
        have = np.isfinite(y_all)
        entry: dict = {"kind": kind, "labelled_people": int(have.sum())}
        if have.sum() < 2 * k or (kind == "binary" and len(set(y_all[have])) < 2):
            entry["skipped"] = "too few labelled people or one class only"
            result["targets"][column] = entry
            continue
        y = y_all[have]
        preds = {}
        for name, z in embeddings.items():
            pred, alphas = cv_predict(z[have], y, k, seed)
            preds[name] = pred
            entry[name] = {**score(y, pred, kind, boot, seed), "alphas": alphas}
        for name in embeddings:
            if name == first:
                continue
            a, b = preds[name], preds[first]
            if kind == "binary":
                pos = y > 0.5
                stat = lambda i, a=a, b=b, pos=pos: auc(a[i], pos[i]) - auc(b[i], pos[i])
                entry[name]["auc_minus_" + first] = stat(np.arange(len(y)))
            else:
                stat = lambda i, a=a, b=b, y=y: float(np.abs(a[i] - y[i]).mean() - np.abs(b[i] - y[i]).mean())
                entry[name]["mae_minus_" + first] = stat(np.arange(len(y)))
            entry[name]["diff_ci95"] = bootstrap(stat, len(y), boot, seed)
        result["targets"][column] = entry
    return result


def _cli(argv: list[str]) -> None:
    if len(argv) < 2:
        sys.exit(__doc__)
    out, sources = Path(argv[0]), argv[1].split(",")
    targets, encoders, opts = [], {}, {}
    i = 2
    while i < len(argv):
        key, value = argv[i], argv[i + 1]
        if key == "--target":
            column, kind = value.split(":")
            targets.append((column, kind))
        elif key == "--encoder":
            name, spec = value.split("=", 1)
            encoders[name] = spec
        else:
            opts[key[2:].replace("-", "_")] = value
        i += 2
    result = probe(sources, targets, encoders, per_subject=int(opts.get("per_subject", 400)),
                   k=int(opts.get("folds", 5)), boot=int(opts.get("boot", 2000)), seed=int(opts.get("seed", 0)),
                   device=opts.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
                   seconds=float(opts.get("seconds", 30.0)))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(json_safe(result), indent=2) + "\n")
    print(json.dumps(json_safe(result["targets"]), indent=1)[:4000])


if __name__ == "__main__":
    _cli(sys.argv[1:])
