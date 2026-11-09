"""Age-related traits from PPG: one multi-task head on the kron-ppg encoder.

Run 1 showed that an age clock's *gap* carried no outcome signal (AUC 0.50-0.56 for hypertension,
diabetes and ASA >= 3), so this asks the direct question instead: does a PPG window say whether a person
has an age-related condition, and does it say more than their age does?

The head predicts, from the encoder's pooled embedding, every trait the subject table carries:

* regression: ``age`` (years), ``bmi`` (kg/m2);
* binary: ``preop_htn`` (hypertension), ``preop_dm`` (diabetes), ``asa_ge3`` (ASA class >= 3: severe
  systemic disease), and ``sex_male`` as a control the literature says PPG predicts well.

Missing labels are masked per trait. Regression targets are learned in units of their training spread;
binary ones as logits starting at the training prevalence.

Scoring is per person, averaging the window outputs, on the validation and test people:

* regression: MAE, Pearson r and the median-prediction baseline's MAE;
* binary: the AUC of the PPG score, the AUC of age alone, and the AUC of a logistic model on
  ``age + PPG score`` fitted on the validation people and scored on the test people. A PPG score that
  only re-learns age adds nothing over age; the combined AUC above age's says it does.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch import nn

from kron_ppg.clock import auc, fit_bias, gap_associations, json_safe, pearson
from kron_ppg.data import Pool, WindowSet
from kron_ppg.model import KronPPG, KronPPGConfig, patchify


@dataclass(frozen=True)
class Trait:
    name: str
    kind: str  # "regression" | "binary"
    column: str
    positive: Sequence[str] = ()  # binary from text: the values that count as 1
    negative: Sequence[str] = ()  # ... and as 0
    at_least: float | None = None  # binary from a number: >= this is 1


TRAITS: dict[str, Trait] = {t.name: t for t in (
    Trait("age", "regression", "age"),
    Trait("bmi", "regression", "bmi"),
    Trait("preop_htn", "binary", "preop_htn", at_least=1),
    Trait("preop_dm", "binary", "preop_dm", at_least=1),
    Trait("asa_ge3", "binary", "asa", at_least=3),
    Trait("sex_male", "binary", "sex", positive=("M", "m", "male", "Male"), negative=("F", "f", "female", "Female")),
)}
DEFAULT_TRAITS = tuple(TRAITS)


def trait_value(trait: Trait, row: Mapping) -> float:
    """The trait's target for one subject-table row: a number, 0/1, or NaN when unknown."""
    value = row.get(trait.column)
    if value is None or value == "":
        return float("nan")
    if trait.positive or trait.negative:
        text = str(value).strip()
        return 1.0 if text in trait.positive else 0.0 if text in trait.negative else float("nan")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(number):
        return float("nan")
    if trait.kind == "binary":
        return float(number >= (trait.at_least if trait.at_least is not None else 1))
    return number


def targets(pool: Pool, traits: Sequence[str]) -> np.ndarray:
    """``(windows, traits)`` float32 targets, NaN where the subject's value is unknown."""
    per_subject = {s: [trait_value(TRAITS[t], pool.table.get(s, {})) for t in traits] for s in set(pool.subject)}
    return np.array([per_subject[s] for s in pool.subject], dtype=np.float32).reshape(len(pool.subject), len(traits))


class TraitHead(nn.Module):
    """Pooled embedding -> one output per trait: ``offset + scale * MLP(LayerNorm(z))``.

    For a regression trait ``offset``/``scale`` are the training mean and spread, so the output is in
    the trait's units; for a binary one they are ``logit(prevalence)`` and 1, so the output is a logit.
    """

    def __init__(self, d_model: int, traits: Sequence[str], hidden: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.traits = list(traits)
        self.norm = nn.LayerNorm(d_model)
        self.net = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, len(self.traits)))
        self.register_buffer("offset", torch.zeros(len(self.traits)))
        self.register_buffer("scale", torch.ones(len(self.traits)))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def calibrate(self, y: np.ndarray) -> None:
        """Set offsets and scales from per-subject training targets ``(subjects, traits)``."""
        for j, name in enumerate(self.traits):
            col = y[:, j][np.isfinite(y[:, j])]
            if not len(col):
                continue
            if TRAITS[name].kind == "regression":
                self.offset[j] = float(col.mean())
                self.scale[j] = max(float(col.std()), 1e-3)
            else:
                p = float(np.clip(col.mean(), 1e-3, 1 - 1e-3))
                self.offset[j] = math.log(p / (1 - p))

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.offset + self.scale * self.net(self.norm(pooled)).float()


class PPGTraits(nn.Module):
    """Encoder + trait head: scaled PPG samples ``(B, L)`` -> ``(B, traits)``."""

    def __init__(self, encoder: KronPPG, head: TraitHead) -> None:
        super().__init__()
        self.encoder, self.head = encoder, head

    def forward(self, samples: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        patches, ok = patchify(samples, self.encoder.config.patch, valid)
        return self.head(self.encoder.embedding(self.encoder.encode(patches, ok), ok))


def trait_loss(out: torch.Tensor, y: torch.Tensor, head: TraitHead) -> torch.Tensor:
    """Mean over traits of each trait's masked loss: smooth L1 in units of the training spread for a
    regression trait, binary cross-entropy for a binary one. Traits with no label in the batch are skipped."""
    losses = []
    for j, name in enumerate(head.traits):
        known = torch.isfinite(y[:, j])
        if not known.any():
            continue
        o, t = out[known, j], y[known, j]
        if TRAITS[name].kind == "regression":
            losses.append(F.smooth_l1_loss(o / head.scale[j], t / head.scale[j], beta=1.0))
        else:
            losses.append(F.binary_cross_entropy_with_logits(o, t))
    return torch.stack(losses).mean() if losses else out.sum() * 0.0


def fit_logistic(x: np.ndarray, y: np.ndarray, ridge: float = 1e-3, iters: int = 50) -> np.ndarray:
    """Weights ``[bias, w...]`` of a ridge-penalised logistic regression by Newton steps (standardised
    inputs are the caller's job; the ridge keeps a separable set finite)."""
    a = np.column_stack([np.ones(len(x)), x])
    w = np.zeros(a.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(a @ w, -30, 30)))
        grad = a.T @ (p - y) + ridge * np.r_[0, w[1:]]
        hess = (a * (p * (1 - p))[:, None]).T @ a + ridge * np.diag(np.r_[0, np.ones(len(w) - 1)])
        step = np.linalg.solve(hess + 1e-9 * np.eye(len(w)), grad)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def score_traits(preds: Mapping[str, dict[str, np.ndarray]], truth: Mapping[str, dict[str, np.ndarray]],
                 traits: Sequence[str]) -> dict:
    """Per-trait scores on ``val`` and ``test`` from per-subject predictions and targets (arrays aligned
    by subject within each split). ``truth[split]["age"]`` must exist for the age comparisons."""
    out: dict = {}
    for name in traits:
        kind = TRAITS[name].kind
        entry: dict = {"kind": kind}
        for split in ("val", "test"):
            p, t = preds[split][name], truth[split][name]
            known = np.isfinite(t)
            p, t = p[known], t[known]
            if kind == "regression":
                entry[split] = {"subjects": int(len(t)),
                                "mae": float(np.mean(np.abs(p - t))) if len(t) else float("nan"),
                                "r": pearson(p, t),
                                "baseline_mae_median": float(np.mean(np.abs(t - np.median(t)))) if len(t) else float("nan")}
            else:
                age = truth[split]["age"][known]
                has_age = np.isfinite(age)
                entry[split] = {"positive": int(t.sum()), "negative": int((1 - t).sum()),
                                "auc_ppg": auc(p, t.astype(bool)),
                                "auc_age": auc(age[has_age], t[has_age].astype(bool))}
        if kind == "binary":
            entry["test"].update(_beyond_age(preds, truth, name))
        out[name] = entry
    return out


def _beyond_age(preds, truth, name: str) -> dict:
    """AUC on test people of logistic models fitted on the validation people: age alone, and age plus
    the PPG score. Both use the same people (those with the trait and an age known)."""
    def xy(split):
        t, age, p = truth[split][name], truth[split]["age"], preds[split][name]
        ok = np.isfinite(t) & np.isfinite(age)
        return age[ok], p[ok], t[ok]

    va, vp, vt = xy("val")
    ta, tp, tt = xy("test")
    if len(np.unique(vt)) < 2 or len(np.unique(tt)) < 2:
        return {"auc_age_plus_ppg": float("nan")}
    mu, sd = np.array([va.mean(), vp.mean()]), np.array([va.std() or 1.0, vp.std() or 1.0])
    w = fit_logistic((np.column_stack([va, vp]) - mu) / sd, vt)
    combined = np.column_stack([np.ones(len(ta)), (np.column_stack([ta, tp]) - mu) / sd]) @ w
    return {"auc_age_plus_ppg": auc(combined, tt.astype(bool)), "logistic_val": [float(v) for v in w]}


@dataclass
class TraitArgs:
    folders: list[Path]
    out: Path
    encoder: Path | None = None
    traits: tuple[str, ...] = DEFAULT_TRAITS
    config: KronPPGConfig = field(default_factory=KronPPGConfig)
    freeze: bool = False
    batch_size: int = 128
    steps: int = 6_000
    lr: float = 3e-4
    head_lr: float = 1e-3
    warmup_fraction: float = 0.05
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    eval_every: int = 500
    eval_windows_per_subject: int = 20
    seed: int = 0
    device: str = "cuda"
    num_workers: int = 2


class _Labelled(torch.utils.data.Dataset):
    """``WindowSet`` items with the window's trait targets instead of its age."""

    def __init__(self, windows: WindowSet, y: np.ndarray) -> None:
        self.windows, self.y = windows, y

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int):
        x, _, _ = self.windows[i]
        return x, torch.from_numpy(self.y[self.windows.idx[i]])


def _per_subject(pool: Pool, idx: np.ndarray, values: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Subjects (sorted) and the mean of ``values`` rows over each subject's windows."""
    subjects = pool.subject[idx]
    keys, inverse = np.unique(subjects, return_inverse=True)
    sums = np.zeros((len(keys), values.shape[1]))
    np.add.at(sums, inverse, values)
    counts = np.bincount(inverse, minlength=len(keys))[:, None]
    return [str(k) for k in keys], sums / counts


def fit_traits(a: TraitArgs) -> dict:
    """Fine-tune (or, with ``freeze``, probe) the encoder with a trait head; writes the run folder."""
    from kron_ppg.train import _check_rate, _eval_index, _lr_at, _param_groups, load_encoder

    unknown = [t for t in a.traits if t not in TRAITS]
    if unknown or "age" not in a.traits:
        raise ValueError(f"traits must include age and come from {sorted(TRAITS)}; unknown: {unknown}")
    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    pool = Pool.load(a.folders)
    encoder = load_encoder(a.encoder, dev) if a.encoder else KronPPG(a.config).to(dev)
    config = encoder.config
    _check_rate(pool, config)
    traits = list(a.traits)
    y = targets(pool, traits)
    labelled = np.isfinite(y).any(axis=1)
    split = {k: v[labelled[v]] for k, v in pool.split().items()}
    if min(len(set(pool.subject[v])) for v in split.values()) < 2:
        raise ValueError("need at least two labelled subjects in each of train, val and test")
    head = TraitHead(config.d_model, traits).to(dev)
    _, y_train_subjects = _per_subject(pool, split["train"], np.nan_to_num(y[split["train"]], nan=np.nan))
    head.calibrate(y_train_subjects)
    model = PPGTraits(encoder, head).to(dev)
    if a.freeze:
        for p in encoder.parameters():
            p.requires_grad_(False)
    a.out.mkdir(parents=True, exist_ok=True)
    eval_idx = {k: _eval_index(pool, v, a.eval_windows_per_subject, a.seed) for k, v in split.items()}
    prevalence = {t: float(np.nanmean(y_train_subjects[:, j])) for j, t in enumerate(traits)
                  if TRAITS[t].kind == "binary"}
    record = {"task": "traits", "traits": traits, "model": config.to_dict(),
              "encoder": str(a.encoder) if a.encoder else None, "freeze": a.freeze,
              "data": [str(f) for f in a.folders], "lr": a.lr, "head_lr": a.head_lr, "batch_size": a.batch_size,
              "steps": a.steps, "seed": a.seed, "windows": {k: int(len(v)) for k, v in split.items()},
              "subjects": {k: int(len(set(pool.subject[v]))) for k, v in split.items()},
              "labelled_subjects": {t: int(np.isfinite(y_train_subjects[:, j]).sum()) for j, t in enumerate(traits)},
              "train_prevalence": prevalence}
    (a.out / "config.json").write_text(json.dumps(json_safe(record), indent=2) + "\n")

    train_subjects = pool.subject[split["train"]]
    _, inverse, counts = np.unique(train_subjects, return_inverse=True, return_counts=True)
    sampler = torch.utils.data.WeightedRandomSampler(torch.as_tensor(1.0 / counts[inverse], dtype=torch.double),
                                                     num_samples=a.batch_size * a.steps,
                                                     generator=torch.Generator().manual_seed(a.seed))
    loader = torch.utils.data.DataLoader(_Labelled(WindowSet(pool, split["train"]), y), batch_size=a.batch_size,
                                         sampler=sampler, drop_last=True, num_workers=a.num_workers)
    groups = [{"params": list(head.parameters()), "lr": a.head_lr, "weight_decay": 0.0, "base": a.head_lr}]
    if not a.freeze:
        groups += [{**g, "lr": a.lr, "base": a.lr} for g in _param_groups(encoder, a.weight_decay)]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95))
    warmup = max(1, int(a.warmup_fraction * a.steps))

    def evaluate() -> dict:
        model.eval()
        preds, truth, subjects, val_loss = {}, {}, {}, 0.0
        with torch.no_grad():
            for k in ("val", "test"):
                idx = eval_idx[k]
                outs = []
                for start in range(0, len(idx), a.batch_size):
                    chunk = idx[start:start + a.batch_size]
                    x = torch.from_numpy(np.stack([pool.window(int(i)) for i in chunk])).to(dev)
                    outs.append(model(x).float().cpu().numpy())
                window_out = np.concatenate(outs)
                if k == "val":
                    val_loss = float(trait_loss(torch.from_numpy(window_out), torch.from_numpy(y[idx]), head.cpu()))
                    head.to(dev)
                keys, mean_out = _per_subject(pool, idx, window_out)
                _, mean_y = _per_subject(pool, idx, y[idx])
                subjects[k] = keys
                preds[k] = {t: mean_out[:, j] for j, t in enumerate(traits)}
                truth[k] = {t: mean_y[:, j] for j, t in enumerate(traits)}
        model.train()
        if a.freeze:
            encoder.eval()
        scores = score_traits(preds, truth, traits)
        bias = fit_bias(preds["val"]["age"], truth["val"]["age"])
        gaps = {s: p - (bias[0] + bias[1] * g)
                for s, p, g in zip(subjects["test"], preds["test"]["age"], truth["test"]["age"], strict=True)
                if np.isfinite(g)}
        ages = {s: g for s, g in zip(subjects["test"], truth["test"]["age"], strict=True) if np.isfinite(g)}
        scores["age"]["gap_outcomes_test"] = gap_associations(gaps, pool.table, ages=ages)
        return {"val_loss": val_loss, "scores": scores, "subjects": subjects, "preds": preds, "truth": truth}

    log: list[dict] = []
    best, best_eval, step = float("inf"), None, 0
    model.train()
    if a.freeze:
        encoder.eval()
    for x, target in loader:
        if step >= a.steps:
            break
        x, target = x.to(dev, non_blocking=True), target.to(dev)
        for group in opt.param_groups:
            group["lr"] = _lr_at(step, a.steps, warmup, group["base"])
        loss = trait_loss(model(x), target, head)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], a.grad_clip)
        opt.step()
        step += 1
        if step % a.eval_every == 0 or step == a.steps:
            ev = evaluate()
            s = ev["scores"]
            row = {"step": step, "train_loss": loss.item(), "val_loss": ev["val_loss"],
                   "test_age_mae": s["age"]["test"]["mae"]}
            row.update({f"test_auc_{t}": s[t]["test"]["auc_ppg"] for t in traits if TRAITS[t].kind == "binary"})
            log.append(row)
            pl.DataFrame(log).write_parquet(a.out / "metrics.parquet")
            print(f"step {step}: train {loss.item():.3f}, val {ev['val_loss']:.3f}, test age MAE "
                  f"{s['age']['test']['mae']:.2f}, " + ", ".join(
                      f"{t} AUC {s[t]['test']['auc_ppg']:.3f} (age {s[t]['test']['auc_age']:.3f}, "
                      f"age+ppg {s[t]['test']['auc_age_plus_ppg']:.3f})"
                      for t in traits if TRAITS[t].kind == "binary"), flush=True)
            if ev["val_loss"] < best:
                best, best_eval = ev["val_loss"], {**ev, "step": step}
                torch.save({"encoder": encoder.state_dict(), "head": head.state_dict(), "traits": traits,
                            "config": config.to_dict(), "step": step}, a.out / "checkpoint.pt")
    assert best_eval is not None
    result = {**record, "best_step": best_eval["step"], "val_loss": best_eval["val_loss"],
              "scores": best_eval["scores"]}
    (a.out / "metrics.json").write_text(json.dumps(json_safe(result), indent=2) + "\n")
    rows = []
    for k in ("val", "test"):
        for i, s in enumerate(best_eval["subjects"][k]):
            row = {"split": k, "subject": s}
            for t in traits:
                row[t] = float(best_eval["truth"][k][t][i])
                row[f"{t}_predicted"] = float(best_eval["preds"][k][t][i])
            rows.append(row)
    pl.DataFrame(rows).with_columns(pl.selectors.float().fill_nan(None)).write_parquet(a.out / "predictions.parquet")
    return result
