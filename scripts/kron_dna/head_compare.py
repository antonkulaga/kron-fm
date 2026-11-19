"""Which output head reads a frozen trunk best? Several head variants trained side by side on the same features.

    uv run python scripts/kron_dna/head_compare.py CHECKPOINT OUT_DIR --tracks STORE --labels STORE \
        [--name hg38] [--split SPLIT.u8] [--window 131072] [--batch 2] [--hours 3 | --steps N] [--eval-windows 128]

The trunk (a ``genome-train`` checkpoint) is frozen and run once per batch without gradient on the clean sequence;
its decoder states, the input every per-base head reads, go to all variants, so each sees identical data in the same
order. Every variant has one head per task with its own AdamW and the same schedule:

    linear       one 1x1 map, no hidden layer
    mlp32        the current head's form with 32 hidden units (what a 1 Mb run falls back to for memory)
    mlp128       the current head: RMS norm, 1x1 to 128, GELU, 1x1            (the reference)
    mlp512       the same with 512 hidden units
    mlp3         three layers: norm, 1x1 to 256, GELU, 1x1 to 256, GELU, 1x1
    conv9        norm, a 9-base convolution to 128, GELU, 1x1: the only variant that reads neighbouring bases
    kan          norm, 1x1 down to 32 (to fit), then two KAN layers (32 -> 64 -> outputs), each a sum of learned
                 one-variable functions on a basis of 8 Gaussians plus a SiLU branch
    kron         a per-base linear readout plus one bidirectional KronLSTM layer (width 128) over the 32-base means
                 of the states, its outputs added back at every base: the only variant that mixes distant positions
                 (dropped, with a line saying so, if the layer cannot be built or run on the device)
    kron2        the same with two KronLSTM layers
    kron128      one KronLSTM layer of width 256 over 128-base means, the core's own resolution

Tasks: the measured-track store (``--tracks``: Poisson-multinomial loss, scored by Pearson of log1p signal at 128 bp
per track) and the label store (``--labels``: weighted BCE, scored by average precision per channel). Training
windows come from the training split (code 1 of ``--split``); scoring from a fixed set of windows on the held-out
chromosomes (code 4), the same for every variant and every evaluation.

Output in ``OUT_DIR``: ``progress.csv`` (each evaluation), ``results.json`` (per variant: parameters, its own memory
and seconds per step measured alone, the final scores per channel, and from a bootstrap over the held-out windows a
95 % interval of its mean score and of its paired difference to ``mlp128``), ``heads.pt``.
A difference counts when its interval excludes 0; one seed, one trunk: it says nothing about other trunks.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import typer

from kron_core.paths import interim_dir
from kron_dna import genome as G
from kron_dna import xheads as X

REFERENCE = "mlp128"


class KANLayer(nn.Module):
    """A Kolmogorov-Arnold layer over channels of ``[B, C, L]``: each output is a sum over inputs of a learned
    one-variable function, here a mix of ``k`` Gaussians on a fixed grid plus a SiLU branch."""

    def __init__(self, c_in: int, c_out: int, k: int = 8, lo: float = -2.5, hi: float = 2.5) -> None:
        super().__init__()
        self.register_buffer("grid", torch.linspace(lo, hi, k).view(1, 1, k, 1))
        self.width = (hi - lo) / (k - 1)
        self.mix = nn.Conv1d(c_in * k, c_out, 1)
        self.base = nn.Conv1d(c_in, c_out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, n = x.shape
        phi = torch.exp(-((x.unsqueeze(2) - self.grid) / self.width) ** 2).reshape(b, -1, n)
        return self.mix(phi) + self.base(F.silu(x))


class KronHead(nn.Module):
    """A per-base linear readout plus one bidirectional KronLSTM layer over ``bin``-base means of the states (each
    window its own document), whose outputs are added back at every base of the bin."""

    def __init__(self, c: int, t: int, bin: int = 32, d: int = 128, layers: int = 1) -> None:
        super().__init__()
        from kron_core.bidirectional import BiKronLSTMBlock

        self.bin = bin
        self.norm, self.local, self.down = G.ChannelRMSNorm(c), nn.Conv1d(c, t, 1), nn.Conv1d(c, d, 1)
        self.blocks = nn.ModuleList(BiKronLSTMBlock(d, 4, 2 * d, 0.0, i) for i in range(layers))
        self.out = nn.Conv1d(d, t, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm(x)
        p = F.avg_pool1d(self.down(z), self.bin)                       # [B, d, n]
        b, d, n = p.shape
        cu = torch.arange(0, b * n + 1, n, device=x.device, dtype=torch.int32)
        h = p.transpose(1, 2).reshape(1, b * n, d)
        for block in self.blocks:
            h = block(h, cu)
        h = h.reshape(b, n, d).transpose(1, 2)
        wide = torch.repeat_interleave(self.out(h), self.bin, dim=-1)
        return self.local(z) + F.pad(wide, (0, x.shape[-1] - wide.shape[-1]))


def make_head(kind: str, c: int, t: int) -> nn.Module:
    """A head ``[B, C, L] -> [B, T, L]`` of the named variant."""
    norm = G.ChannelRMSNorm(c)
    if kind == "linear":
        return nn.Conv1d(c, t, 1)
    if kind in ("mlp32", "mlp128", "mlp512"):
        return G.MLPHead(c, int(kind[3:]), t)
    if kind == "mlp3":
        return nn.Sequential(norm, nn.Conv1d(c, 256, 1), nn.GELU(), nn.Conv1d(256, 256, 1), nn.GELU(), nn.Conv1d(256, t, 1))
    if kind == "conv9":
        return nn.Sequential(norm, nn.Conv1d(c, 128, 9, padding=4), nn.GELU(), nn.Conv1d(128, t, 1))
    if kind == "kan":
        return nn.Sequential(norm, nn.Conv1d(c, 32, 1), G.ChannelRMSNorm(32), KANLayer(32, 64), G.ChannelRMSNorm(64),
                             KANLayer(64, t))
    if kind in KRON:
        return KronHead(c, t, **KRON[kind])
    raise ValueError(kind)


KRON = {"kron": dict(bin=32, d=128, layers=1), "kron2": dict(bin=32, d=128, layers=2),
        "kron128": dict(bin=128, d=256, layers=1)}
VARIANTS = ("linear", "mlp32", "mlp128", "mlp512", "mlp3", "conv9", "kan", "kron", "kron2", "kron128")


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# ----------------------------------------------------------------------------------------------- scoring pieces

def pearson_stats(pred: torch.Tensor, want: torch.Tensor, bin: int = 128) -> torch.Tensor:
    """Sufficient statistics of one window, ``[6, T]``: n, sum x, sum y, sum xy, sum xx, sum yy over its 128 bp bins
    (``pred`` and ``want`` ``[L, T]`` log1p signal)."""
    n, t = pred.shape
    keep = n - n % bin
    x = pred[:keep].double().reshape(-1, bin, t).mean(1)
    y = want[:keep].double().reshape(-1, bin, t).mean(1)
    return torch.stack([torch.full((t,), float(x.shape[0]), dtype=torch.float64, device=x.device), x.sum(0), y.sum(0),
                        (x * y).sum(0),
                        (x * x).sum(0), (y * y).sum(0)]).cpu()


def pearson_from(stats: torch.Tensor) -> np.ndarray:
    """Per-track Pearson from summed statistics ``[..., 6, T]``."""
    n, sx, sy, sxy, sxx, syy = (stats[..., k, :] for k in range(6))
    cov = sxy / n - sx * sy / n ** 2
    vx, vy = sxx / n - (sx / n) ** 2, syy / n - (sy / n) ** 2
    r = cov / torch.sqrt((vx * vy).clamp_min(1e-30))
    return torch.where((vx > 1e-12) & (vy > 1e-12), r, torch.full_like(r, float("nan"))).numpy()


def ap_piece(pred: torch.Tensor, want: torch.Tensor, keep: float = 0.02) -> list[tuple]:
    """One window, per channel: (positives' logits, the top ``keep`` share of negatives' logits, negatives in all)."""
    out = []
    for k in range(pred.shape[-1]):
        y = want[:, k] > 0
        p, q = pred[y, k].float().cpu(), pred[~y, k].float()
        top = max(1, int(q.numel() * keep))
        out.append((p, torch.topk(q, min(top, q.numel())).values.cpu(), int(q.numel())))
    return out


def ap_pooled(pieces: list[list[tuple]]) -> np.ndarray:
    """Average precision per channel over the given windows' pieces, pooled. A positive below the highest of the
    windows' lowest kept negatives is ranked after every negative (a lower bound on its precision)."""
    t = len(pieces[0])
    out = np.full(t, np.nan)
    for k in range(t):
        p = torch.cat([w[k][0] for w in pieces])
        if not p.numel():
            continue
        q = torch.cat([w[k][1] for w in pieces])
        total = sum(w[k][2] for w in pieces)
        floor = max((float(w[k][1].min()) for w in pieces if w[k][1].numel() < w[k][2]), default=float("-inf"))
        p = torch.sort(p, descending=True).values
        above = torch.searchsorted(-torch.sort(q, descending=True).values, -p).double()
        above = torch.where(p >= floor, above, torch.full_like(above, float(total)))
        rank = torch.arange(1, p.numel() + 1, dtype=torch.float64)
        out[k] = float((rank / (rank + above)).mean())
    return out


def interval(values: list[float]) -> list[float]:
    v = np.asarray([x for x in values if not math.isnan(x)])
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [float("nan")] * 2


# ------------------------------------------------------------------------------------------------------ the run

def run(checkpoint: Path, out: Path, tracks: Path, labels: Path, name: str = "hg38", split: Path | None = None,
        window: int = 131072, batch: int = 2, hours: float = 3.0, steps: int = 0, eval_windows: int = 128,
        eval_every: int = 1000, lr: float = 4e-4, warmup: int = 200, center: float = 0.75, seed: int = 0,
        device: str = "cuda", variants: tuple[str, ...] = VARIANTS, n_boot: int = 1000, n_boot_ap: int = 200,
        say=print) -> dict:
    dev = torch.device(device)
    out.mkdir(parents=True, exist_ok=True)
    src = interim_dir("genome", name)
    bases = np.memmap(src / "bases.u8", dtype=np.uint8, mode="r")
    split_mm = np.memmap(split or src / "ntv3" / "split.u8", dtype=np.uint8, mode="r")
    stores = {"dnase": X.open_store(tracks), "splice": X.open_store(labels)}
    for k, st in stores.items():
        if st.dense.shape[0] != len(bases) or len(split_mm) != len(bases):
            raise typer.BadParameter(f"{k}: the store or the split does not cover the genome's {len(bases):,} bases")
    index = pl.read_parquet(src / "contigs.parquet").with_columns((pl.col("end") - pl.col("start")).alias("len"))
    index = index.filter(pl.col("len") >= window)
    held = index.filter(pl.col("chrom").is_in(["chr8", "chr9"]))
    train = index.filter(~pl.col("chrom").is_in(["chr8", "chr9"]))
    rng = np.random.default_rng(seed)

    def draw(table: pl.DataFrame, n: int, gen) -> list[int]:
        p = table["len"].to_numpy() / table["len"].sum()
        rows = gen.choice(table.height, size=n, p=p)
        return [int(table["offset"][int(r)]) + int(gen.integers(0, int(table["len"][int(r)]) - window + 1)) for r in rows]

    model, state = G.load_checkpoint(checkpoint)
    model = model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    unit = 2 ** model.config.n_down
    width = model.head.fc1.in_channels
    grabbed: dict = {}
    model.head.norm.register_forward_hook(lambda mod, inp, res: grabbed.__setitem__("x", inp[0]))

    def features(starts: list[int]) -> torch.Tensor:
        """Decoder states ``[B, C, window]`` (float32) of the windows at ``starts``: the trunk, clean input."""
        seq = torch.from_numpy(np.stack([np.asarray(bases[a:a + window]).astype(np.int64) for a in starts])).to(dev)
        seg = torch.zeros(len(starts), window // unit, dtype=torch.long, device=dev)
        ctx = torch.zeros(len(starts), dtype=torch.long, device=dev) if model.context is not None else None
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            model(seq, seg, ctx)
        return grabbed.pop("x").float()

    def targets(starts: list[int]) -> tuple[dict, torch.Tensor]:
        want = {k: torch.from_numpy(np.stack([st.window(a, window, window) for a in starts])).to(dev)
                for k, st in stores.items()}
        sp = torch.from_numpy(np.stack([np.asarray(split_mm[a:a + window]) for a in starts]))
        return want, sp

    consts = {k: st.consts(dev) for k, st in stores.items()}
    torch.manual_seed(seed)
    heads, dropped = {}, {}
    for v in variants:
        try:                                          # a variant that cannot be built or run here is left out
            h = nn.ModuleDict({k: make_head(v, width, len(st.ids)) for k, st in stores.items()}).to(dev)
            with torch.no_grad():
                for k in stores:
                    h[k](torch.zeros(1, width, min(window, 4096), device=dev))
            heads[v] = h
        except Exception as err:                      # noqa: BLE001  (the comparison goes on without it)
            dropped[v] = f"{type(err).__name__}: {err}"[:300]
            say(f"variant {v} dropped: {dropped[v]}")
    variants = tuple(heads)
    opts = {v: torch.optim.AdamW(heads[v].parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95)) for v in variants}
    params = {v: {k: n_params(heads[v][k]) for k in stores} for v in variants}

    def losses(v: str, x: torch.Tensor, want: dict, ok: torch.Tensor) -> dict:
        return {k: X.head_loss(heads[v][k](x).transpose(1, 2), want[k], ok, consts[k], stores[k].kind) for k in stores}

    # each variant alone on one batch: its own memory above the features and its seconds per step
    probe_starts = draw(train, batch, np.random.default_rng([seed, 1]))
    x0, (w0, s0) = features(probe_starts), targets(probe_starts)
    ok0 = X.supervised(s0, center).to(dev)
    def alone(v: str) -> dict:
        """Training steps of variant ``v`` alone on one batch: its peak memory over all of them, and its seconds per
        step over the last three; the first step is a warm-up, since it holds kernel compilation and autotuning."""
        twin = nn.ModuleDict({k: make_head(v, width, len(st.ids)) for k, st in stores.items()}).to(dev)
        o = torch.optim.AdamW(twin.parameters(), lr=0.0)
        if dev.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(dev)
            before = torch.cuda.memory_allocated(dev)

        def step() -> None:
            o.zero_grad()
            sum(X.head_loss(twin[k](x0).transpose(1, 2), w0[k], ok0, consts[k], stores[k].kind) for k in stores).backward()
            o.step()

        step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(3):
            step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        got = dict(seconds_per_step=(time.time() - t0) / 3,
                       peak_gib_above_features=((torch.cuda.max_memory_allocated(dev) - before) / 2 ** 30
                                                if dev.type == "cuda" else float("nan")))
        del twin, o
        return got

    cost = {}
    for v in list(variants):
        try:
            cost[v] = alone(v)
        except Exception as err:                      # noqa: BLE001  (as above: the comparison goes on without it)
            dropped[v] = f"{type(err).__name__}: {err}"[:300]
            heads.pop(v)
            say(f"variant {v} dropped at its first training steps: {dropped[v]}")
            continue
        say(f"{v}: {sum(params[v].values()):,} parameters, {cost[v]['seconds_per_step']:.3f} s/step alone, "
            f"{cost[v]['peak_gib_above_features']:.2f} GiB above the features (batch {batch} x {window:,})")
    variants = tuple(heads)
    del x0, w0

    eval_starts = draw(held, eval_windows, np.random.default_rng([seed, 2]))

    def evaluate() -> dict:
        """Per variant: per-window Pearson statistics ``[W, 6, T]`` and per-window AP pieces."""
        got = {v: dict(stats=[], pieces=[]) for v in variants}
        for h in heads.values():
            h.eval()
        with torch.no_grad():
            for a in eval_starts:
                x, (want, _) = features([a]), targets([a])
                for v in variants:
                    pd = heads[v]["dnase"](x).transpose(1, 2)[0]
                    p_log, w_log = X.eval_pair(stores["dnase"], pd, want["dnase"][0], consts["dnase"])
                    got[v]["stats"].append(pearson_stats(p_log, w_log))
                    got[v]["pieces"].append(ap_piece(heads[v]["splice"](x).transpose(1, 2)[0], want["splice"][0]))
        for h in heads.values():
            h.train()
        return {v: dict(stats=torch.stack(g["stats"]), pieces=g["pieces"]) for v, g in got.items()}

    def means(ev: dict) -> dict:
        return {v: dict(dnase=pearson_from(e["stats"].sum(0)), splice=ap_pooled(e["pieces"])) for v, e in ev.items()}

    progress = out / "progress.csv"
    progress.write_text("step,hours,variant,train_dnase,train_splice,heldout_dnase_pearson,heldout_splice_ap\n")
    t_start, step, total = time.time(), 0, steps
    recent = {v: {k: [] for k in stores} for v in variants}
    ev = None
    while True:
        elapsed = (time.time() - t_start) / 3600
        done = step >= steps if steps else elapsed >= hours
        if step and (step % eval_every == 0 or done):
            ev = evaluate()
            m = means(ev)
            with progress.open("a") as fh:
                for v in variants:
                    fh.write(f"{step},{elapsed:.3f},{v},{np.mean(recent[v]['dnase']):.5f},{np.mean(recent[v]['splice']):.5f},"
                             f"{np.nanmean(m[v]['dnase']):.5f},{np.nanmean(m[v]['splice']):.5f}\n")
                    recent[v] = {k: [] for k in stores}
            say(f"step {step} ({elapsed:.2f} h): " + "; ".join(
                f"{v} r {np.nanmean(m[v]['dnase']):.4f} ap {np.nanmean(m[v]['splice']):.4f}" for v in variants))
        if done:
            break
        # the schedule: warmup, then a cosine over the planned steps, or over the hours when the run is timed
        frac = step / steps if steps else elapsed / hours
        rate = lr * min(1.0, (step + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(frac, 1.0)))
        starts = draw(train, batch, rng)
        x, (want, sp) = features(starts), targets(starts)
        ok = X.supervised(sp, center).to(dev)
        if float(ok.sum()) == 0:
            continue
        for v in variants:
            for g in opts[v].param_groups:
                g["lr"] = rate
            opts[v].zero_grad()
            ls = losses(v, x, want, ok)
            sum(ls.values()).backward()
            torch.nn.utils.clip_grad_norm_(heads[v].parameters(), 1.0)
            opts[v].step()
            for k in stores:
                recent[v][k].append(float(ls[k].detach()))
        step += 1

    # the verdict's numbers: each variant's mean over channels, and its paired difference to the reference, with
    # intervals from resampling the held-out windows
    m = means(ev)
    boot = np.random.default_rng([seed, 3])
    w = len(eval_starts)
    idx = boot.integers(0, w, (n_boot, w))
    r_boot = {v: np.nanmean(pearson_from(torch.stack([ev[v]["stats"][torch.from_numpy(i)].sum(0) for i in idx])), -1)
              for v in variants}
    idx_ap = idx[:n_boot_ap]
    ap_boot = {v: np.array([np.nanmean(ap_pooled([ev[v]["pieces"][j] for j in i])) for i in idx_ap]) for v in variants}
    ref = REFERENCE if REFERENCE in variants else variants[0]
    result = dict(
        checkpoint=str(checkpoint), trunk_step=state.get("step"), window=window, batch=batch, steps=step,
        dropped=dropped,
        hours=(time.time() - t_start) / 3600, eval_windows=w, seed=seed, lr=lr, center=center, reference=ref,
        stores={k: st.identity for k, st in stores.items()},
        variants={v: dict(
            parameters=params[v], **cost[v],
            dnase_pearson_mean=float(np.nanmean(m[v]["dnase"])), dnase_pearson_ci=interval(list(r_boot[v])),
            dnase_minus_reference=float(np.nanmean(m[v]["dnase"]) - np.nanmean(m[ref]["dnase"])),
            dnase_minus_reference_ci=interval(list(r_boot[v] - r_boot[ref])),
            dnase_pearson_per_track=[float(x) for x in m[v]["dnase"]],
            splice_ap_mean=float(np.nanmean(m[v]["splice"])), splice_ap_ci=interval(list(ap_boot[v])),
            splice_minus_reference=float(np.nanmean(m[v]["splice"]) - np.nanmean(m[ref]["splice"])),
            splice_minus_reference_ci=interval(list(ap_boot[v] - ap_boot[ref])),
            splice_ap_per_channel=[float(x) for x in m[v]["splice"]]) for v in variants})
    (out / "results.json").write_text(json.dumps(result, indent=1))
    torch.save({v: heads[v].state_dict() for v in variants}, out / "heads.pt")
    for v in variants:
        r = result["variants"][v]
        say(f"{v:8s} DNase r {r['dnase_pearson_mean']:.4f} {r['dnase_pearson_ci']}  vs {ref} "
            f"{r['dnase_minus_reference']:+.4f} {r['dnase_minus_reference_ci']} | splice AP {r['splice_ap_mean']:.4f} "
            f"vs {ref} {r['splice_minus_reference']:+.4f} {r['splice_minus_reference_ci']}")
    return result


def main(checkpoint: Path, out: Path, tracks: Path = typer.Option(...), labels: Path = typer.Option(...),
         name: str = "hg38", split: Path = typer.Option(None), window: int = 131072, batch: int = 2,
         hours: float = 3.0, steps: int = 0, eval_windows: int = 128, eval_every: int = 1000, lr: float = 4e-4,
         seed: int = 0, device: str = "cuda") -> None:
    run(checkpoint, out, tracks, labels, name=name, split=split, window=window, batch=batch, hours=hours, steps=steps,
        eval_windows=eval_windows, eval_every=eval_every, lr=lr, seed=seed, device=device)


if __name__ == "__main__":
    typer.run(main)
