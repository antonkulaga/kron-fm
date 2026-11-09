"""Pretraining the kron-ppg encoder and fitting the age clock on prepared windows.

Both write a run folder under ``data/output/runs/kron-ppg/`` by default: ``config.json``, ``metrics.parquet``
and ``checkpoint.pt``; the clock also writes ``metrics.json`` (per-split scores, the bias correction,
the gap's outcome associations) and ``predictions.parquet`` (per subject).
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from kron_ppg.clock import fit_bias, gap_associations, json_safe, per_subject, scores
from kron_ppg.data import Pool, SubjectPairs, WindowSet
from kron_ppg.model import OBJECTIVES, AgeHead, KronPPG, KronPPGConfig, PPGAgeClock, patchify


def _lr_at(step: int, steps: int, warmup: int, lr: float) -> float:
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, steps - warmup)
    return lr * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def _param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        exempt = getattr(p, "_no_weight_decay", False) or p.dim() < 2 or "norm" in name
        (no_decay if exempt else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def _check_rate(pool: Pool, config: KronPPGConfig) -> None:
    if abs(pool.sample_rate - config.sample_rate) > 1e-6:
        raise ValueError(f"the windows are at {pool.sample_rate} Hz and the model reads {config.sample_rate} Hz; "
                         "prepare them at the model's rate")


def load_encoder(path: str | Path, device: torch.device | str = "cpu") -> KronPPG:
    """An encoder from a pretraining or clock checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    config = KronPPGConfig(**ckpt["config"])
    encoder = KronPPG(config)
    state = ckpt["model"] if "model" in ckpt else ckpt["encoder"]
    encoder.load_state_dict(state)
    return encoder.to(device)


def load_clock(path: str | Path, device: torch.device | str = "cpu") -> PPGAgeClock:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    encoder = KronPPG(KronPPGConfig(**ckpt["config"]))
    encoder.load_state_dict(ckpt["encoder"])
    head = AgeHead(encoder.config.d_model, **ckpt.get("head_args", {}))
    head.load_state_dict(ckpt["head"])
    return PPGAgeClock(encoder, head).to(device)


@dataclass
class PretrainArgs:
    folders: list[Path]
    out: Path
    config: KronPPGConfig = field(default_factory=KronPPGConfig)
    objective: str = "next"
    subject_weight: float = 0.1
    batch_size: int = 64
    steps: int = 5_000
    lr: float = 1e-3
    warmup_fraction: float = 0.05
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    eval_every: int = 250
    eval_batches: int = 20
    seed: int = 0
    device: str = "cuda"
    num_workers: int = 2


def pretrain(a: PretrainArgs) -> dict:
    """Self-supervised pretraining on every window (age not needed), validated on held-out subjects."""
    if a.objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}")
    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    pool = Pool.load(a.folders)
    _check_rate(pool, a.config)
    split = pool.split()
    train_set, val_set = WindowSet(pool, split["train"]), WindowSet(pool, split["val"])
    if not len(train_set) or not len(val_set):
        raise ValueError(f"need windows on both sides of the split: {len(train_set)} train, {len(val_set)} val")
    if a.objective == "next+subject":
        sampler = SubjectPairs(pool.subject[split["train"]], a.batch_size, 2, a.seed)
        loader = torch.utils.data.DataLoader(train_set, batch_sampler=sampler, num_workers=a.num_workers)
    else:
        loader = torch.utils.data.DataLoader(train_set, batch_size=a.batch_size, shuffle=True, drop_last=True,
                                             num_workers=a.num_workers)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=a.batch_size)
    model = KronPPG(a.config).to(dev)
    a.out.mkdir(parents=True, exist_ok=True)
    record = {"task": "pretrain", "model": a.config.to_dict(), "objective": a.objective,
              "subject_weight": a.subject_weight, "data": [str(f) for f in a.folders], "lr": a.lr,
              "batch_size": a.batch_size, "steps": a.steps, "seed": a.seed,
              "windows": {k: int(len(v)) for k, v in split.items()},
              "subjects": {k: int(len(set(pool.subject[v]))) for k, v in split.items()},
              "parameters": model.n_parameters()}
    (a.out / "config.json").write_text(json.dumps(record, indent=2) + "\n")
    opt = torch.optim.AdamW(_param_groups(model, a.weight_decay), lr=a.lr, betas=(0.9, 0.95))
    warmup = max(1, int(a.warmup_fraction * a.steps))
    patch = a.config.patch

    def evaluate() -> float:
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for i, (x, _, _) in enumerate(val_loader):
                if i >= a.eval_batches:
                    break
                patches, ok = patchify(x.to(dev), patch)
                total += float(model.next_patch_loss(model.encode(patches, ok), patches, ok))
                n += 1
        model.train()
        return total / max(n, 1)

    log: list[dict] = []
    best = float("inf")
    step, t0, seen = 0, time.time(), 0
    model.train()
    while step < a.steps:
        for x, subject, _ in loader:
            if step >= a.steps:
                break
            x = x.to(dev, non_blocking=True)
            patches, ok = patchify(x, patch)
            for group in opt.param_groups:
                group["lr"] = _lr_at(step, a.steps, warmup, a.lr)
            h = model.encode(patches, ok)
            next_loss = model.next_patch_loss(h, patches, ok)
            subj_loss = (model.subject_loss(model.embedding(h, ok), subject.to(dev))
                         if a.objective == "next+subject" else next_loss.new_zeros(()))
            loss = next_loss + a.subject_weight * subj_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
            opt.step()
            step += 1
            seen += len(x)
            if step % a.eval_every == 0 or step == a.steps:
                val = evaluate()
                rate = seen / (time.time() - t0)
                log.append({"step": step, "lr": opt.param_groups[0]["lr"], "train_next": next_loss.item(),
                            "train_subject": subj_loss.item(), "val_next": val, "windows_per_s": round(rate, 1)})
                pl.DataFrame(log).write_parquet(a.out / "metrics.parquet")
                print(f"step {step}: next {next_loss.item():.4f}, subject {subj_loss.item():.4f}, "
                      f"val next {val:.4f}, {rate:,.0f} windows/s", flush=True)
                if val < best:
                    best = val
                    torch.save({"model": model.state_dict(), "config": a.config.to_dict(), "step": step,
                                "objective": a.objective, "val_next": val}, a.out / "checkpoint.pt")
    return {**record, "best_val_next": best}


@dataclass
class ClockArgs:
    folders: list[Path]
    out: Path
    encoder: Path | None = None
    config: KronPPGConfig = field(default_factory=KronPPGConfig)
    freeze: bool = False
    batch_size: int = 64
    steps: int = 3_000
    lr: float = 3e-4
    head_lr: float = 1e-3
    warmup_fraction: float = 0.05
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    eval_every: int = 250
    eval_windows_per_subject: int = 20
    seed: int = 0
    device: str = "cuda"
    num_workers: int = 2
    #: learn age from healthy people only (ASA 1, no hypertension or diabetes; :mod:`kron_ppg.traits`), the
    #: usual aging-clock design: train and validation (which fits the bias) are healthy, test is everyone,
    #: so the test gap says whether the unhealthy read older than healthy aging predicts
    healthy_only: bool = False


def _eval_index(pool: Pool, idx: np.ndarray, per_subject_cap: int, seed: int) -> np.ndarray:
    """At most ``per_subject_cap`` windows per subject, spread over each subject's windows."""
    rng = np.random.default_rng(seed)
    keep = []
    subjects = pool.subject[idx]
    for s in np.unique(subjects):
        mine = idx[subjects == s]
        keep.append(mine if len(mine) <= per_subject_cap else rng.choice(mine, per_subject_cap, replace=False))
    return np.sort(np.concatenate(keep)) if keep else idx[:0]


def predict(clock: PPGAgeClock, pool: Pool, idx: np.ndarray, batch_size: int, dev: torch.device
            ) -> dict[str, tuple[float, float, int]]:
    """Per-subject mean predictions over the windows at ``idx``."""
    clock.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(idx), batch_size):
            chunk = idx[start:start + batch_size]
            x = torch.from_numpy(np.stack([pool.window(int(i)) for i in chunk])).to(dev)
            preds.append(clock(x).float().cpu().numpy())
    clock.train()
    flat = np.concatenate(preds) if preds else np.zeros(0)
    return per_subject(pool.subject[idx], flat, pool.age[idx])


def fit_clock(a: ClockArgs) -> dict:
    """Fine-tune (or, with ``freeze``, probe) the encoder with an age head on subjects with a known age."""
    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    pool = Pool.load(a.folders)
    encoder = load_encoder(a.encoder, dev) if a.encoder else KronPPG(a.config).to(dev)
    config = encoder.config
    _check_rate(pool, config)
    has_age = np.isfinite(pool.age)
    split = {k: v[has_age[v]] for k, v in pool.split().items()}
    if a.healthy_only:
        from kron_ppg.traits import TRAITS, trait_value

        healthy = {s: trait_value(TRAITS["healthy"], pool.table.get(s, {})) == 1.0 for s in set(pool.subject)}
        for k in ("train", "val"):
            split[k] = split[k][np.array([healthy[s] for s in pool.subject[split[k]]], dtype=bool)]
    if min(len(set(pool.subject[v])) for v in split.values()) < 2:
        raise ValueError("need at least two subjects with an age in each of train, val and test: "
                         f"{ {k: len(set(pool.subject[v])) for k, v in split.items()} }")
    head = AgeHead(config.d_model).to(dev)
    train_ages = np.array([pool.age[split["train"]][pool.subject[split["train"]] == s][0]
                           for s in np.unique(pool.subject[split["train"]])])
    head.set_scale(torch.tensor(train_ages))
    clock = PPGAgeClock(encoder, head).to(dev)
    if a.freeze:
        for p in encoder.parameters():
            p.requires_grad_(False)
    a.out.mkdir(parents=True, exist_ok=True)
    eval_idx = {k: _eval_index(pool, v, a.eval_windows_per_subject, a.seed) for k, v in split.items()}
    record = {"task": "clock", "model": config.to_dict(), "encoder": str(a.encoder) if a.encoder else None,
              "freeze": a.freeze, "healthy_only": a.healthy_only, "data": [str(f) for f in a.folders], "lr": a.lr, "head_lr": a.head_lr,
              "batch_size": a.batch_size, "steps": a.steps, "seed": a.seed,
              "windows": {k: int(len(v)) for k, v in split.items()},
              "subjects": {k: int(len(set(pool.subject[v]))) for k, v in split.items()},
              "train_age": {"mean": float(train_ages.mean()), "sd": float(train_ages.std()),
                            "min": float(train_ages.min()), "max": float(train_ages.max())}}
    (a.out / "config.json").write_text(json.dumps(record, indent=2) + "\n")

    # Balance subjects: each window is drawn with weight 1 / (its subject's windows), so a long surgical
    # record does not outweigh a short one.
    train_subjects = pool.subject[split["train"]]
    _, inverse, counts = np.unique(train_subjects, return_inverse=True, return_counts=True)
    weights = torch.as_tensor(1.0 / counts[inverse], dtype=torch.double)
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=a.batch_size * a.steps,
                                                     generator=torch.Generator().manual_seed(a.seed))
    loader = torch.utils.data.DataLoader(WindowSet(pool, split["train"]), batch_size=a.batch_size,
                                         sampler=sampler, drop_last=True, num_workers=a.num_workers)
    groups = [{"params": [p for p in head.parameters()], "lr": a.head_lr, "weight_decay": 0.0, "base": a.head_lr}]
    if not a.freeze:
        groups += [{**g, "lr": a.lr, "base": a.lr} for g in _param_groups(encoder, a.weight_decay)]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95))
    warmup = max(1, int(a.warmup_fraction * a.steps))

    def evaluate() -> dict:
        subj = {k: predict(clock, pool, v, a.batch_size, dev) for k, v in eval_idx.items()}
        arrays = {k: (np.array([p for p, _, _ in d.values()]), np.array([g for _, g, _ in d.values()]))
                  for k, d in subj.items()}
        bias = fit_bias(*arrays["val"])  # held out: see kron_ppg.clock
        out = {"bias": {"a": bias[0], "b": bias[1]},
               **{k: scores(p, g, bias) for k, (p, g) in arrays.items()}}
        gaps = {k: {s: p - (bias[0] + bias[1] * g) for s, (p, g, _) in d.items()} for k, d in subj.items()}
        out["outcomes"] = {k: gap_associations(gaps[k], pool.table, ages={s: g for s, (_, g, _) in subj[k].items()})
                           for k in ("val", "test")}
        return {"scores": out, "subjects": subj, "gaps": gaps}

    log: list[dict] = []
    best, best_eval, step = float("inf"), None, 0
    clock.train()
    if a.freeze:
        encoder.eval()  # no dropout in a frozen encoder
    for x, _, age in loader:
        if step >= a.steps:
            break
        x, age = x.to(dev, non_blocking=True), age.to(dev, dtype=torch.float32)
        for group in opt.param_groups:
            group["lr"] = _lr_at(step, a.steps, warmup, group["base"])
        pred = clock(x)
        loss = F.smooth_l1_loss(pred.float(), age, beta=2.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], a.grad_clip)
        opt.step()
        step += 1
        if step % a.eval_every == 0 or step == a.steps:
            ev = evaluate()
            s = ev["scores"]
            log.append({"step": step, "train_l1": loss.item(), "val_mae": s["val"]["mae"], "val_r": s["val"]["r"],
                        "test_mae": s["test"]["mae"], "test_r": s["test"]["r"]})
            pl.DataFrame(log).write_parquet(a.out / "metrics.parquet")
            print(f"step {step}: train l1 {loss.item():.2f}, val MAE {s['val']['mae']:.2f} (r {s['val']['r']:.2f}), "
                  f"test MAE {s['test']['mae']:.2f} (r {s['test']['r']:.2f})", flush=True)
            if s["val"]["mae"] < best:
                best, best_eval = s["val"]["mae"], {**ev, "step": step}
                torch.save({"encoder": encoder.state_dict(), "head": head.state_dict(), "config": config.to_dict(),
                            "step": step, "bias": s["bias"]}, a.out / "checkpoint.pt")
    assert best_eval is not None
    result = {**record, "best_step": best_eval["step"], **best_eval["scores"]}
    (a.out / "metrics.json").write_text(json.dumps(json_safe(result), indent=2) + "\n")
    pl.DataFrame(
        [{"split": k, "subject": s, "age": g, "predicted": p, "gap_corrected": best_eval["gaps"][k][s], "windows": n}
         for k, d in best_eval["subjects"].items() for s, (p, g, n) in sorted(d.items())],
        schema={"split": pl.String, "subject": pl.String, "age": pl.Float64, "predicted": pl.Float64,
                "gap_corrected": pl.Float64, "windows": pl.Int64},
    ).write_parquet(a.out / "predictions.parquet")
    return result
