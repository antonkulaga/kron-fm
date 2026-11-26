"""Fine-tuning harness for the paper's claim C4 (a pretrained kron-dna model adapts to a new task on one card):
lentiMPRA activity (Agarwal et al. 2025, the store ``interim/eval/lentimpra-v1``), one cell type, the paper's ten folds.

    python scripts/kron_dna/finetune_mpra.py CKPT OUT.jsonl --cell K562 --test-fold 1 --mode full
        [--random-init] [--form genomic --window 2048 | --form insert] [--share 1.0] [--seed 0]
        [--mem-gib 11] [--epochs 3] [--batch 32]

Folds are 1..10; validation fold = the test fold + 1 (10 -> 1), training = the other eight (``--share`` keeps that part of them). The model reads
each element as an input form:

* ``genomic``: ``--window`` bases of hg38 centred on the element (the element in its own genomic context);
* ``insert``: the element alone, padded with N to a multiple of the model's bin (short input, out of distribution
  for a 1 Mb model: that is what the arm measures).

The head reads the decoder's 1-bp features (what every track head reads) averaged over the element's bases:
LayerNorm, Linear(C, 256), GELU, Linear(256, 1). ``--mode``: ``frozen`` (trunk fixed, head only), ``decoder`` (the
decoder's up-sampling blocks and skips train, encoder and core fixed), ``full`` (everything). ``--random-init``: the same
configuration from random weights, the control the claim needs. Model selection on the validation fold (best epoch);
the test fold is read once, at the end. Each run appends one JSON row: Pearson and Spearman on test and validation,
wall time, peak memory, card, batch, trainable parameters.
"""
from __future__ import annotations

import argparse
import json
import math
import socket
import time

import numpy as np
import polars as pl
import torch
from scipy.stats import pearsonr, spearmanr

from kron_core.paths import interim_dir
from kron_dna import genome as G


def hg38_reader():
    src = interim_dir("genome", "hg38")
    bases = np.memmap(src / "bases.u8", dtype=np.uint8, mode="r")
    contigs = pl.read_parquet(src / "contigs.parquet")
    by_chrom = {c: contigs.filter(pl.col("chrom") == c) for c in contigs["chrom"].unique().to_list()}

    def codes(chrom: str, lo: int, hi: int) -> np.ndarray:
        out = np.full(hi - lo, 4, dtype=np.int64)                      # N outside every contig
        for r in by_chrom.get(chrom, contigs.head(0)).filter((pl.col("start") < hi) & (pl.col("end") > lo)).iter_rows(named=True):
            s, e = max(lo, r["start"]), min(hi, r["end"])
            out[s - lo:e - lo] = bases[r["offset"] + s - r["start"]:r["offset"] + e - r["start"]]
        return out
    return codes


def encode_elements(t: pl.DataFrame, form: str, window: int, unit: int, codes) -> tuple[np.ndarray, np.ndarray]:
    """``[N, L]`` base codes and ``[N, 2]`` element spans (start, end) inside each row."""
    n = t.height
    if form == "genomic":
        length = window
    else:
        longest = int((t["end"] - t["start"]).max())
        length = math.ceil(longest / unit) * unit
    x = np.full((n, length), 4, dtype=np.int64); span = np.zeros((n, 2), dtype=np.int64)
    for i, r in enumerate(t.iter_rows(named=True)):
        s, e = int(r["start"]), int(r["end"])
        if form == "genomic":
            lo = (s + e) // 2 - length // 2
            x[i] = codes(r["chrom"], lo, lo + length); span[i] = (s - lo, e - lo)
        else:
            c = codes(r["chrom"], s, e); x[i, :len(c)] = c; span[i] = (0, len(c))
        if r["strand"] == "-":                                           # read the element as it was cloned
            x[i] = np.where(x[i] < 4, 3 - x[i], x[i])[::-1]
            a, b = span[i]; span[i] = (length - b, length - a)
    return x, span


class Head(torch.nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.LayerNorm(c), torch.nn.Linear(c, 256), torch.nn.GELU(), torch.nn.Linear(256, 1))

    def forward(self, feats: torch.Tensor, span: torch.Tensor) -> torch.Tensor:   # feats [B, C, L]
        pos = torch.arange(feats.shape[-1], device=feats.device)
        m = ((pos[None] >= span[:, :1]) & (pos[None] < span[:, 1:])).to(feats.dtype)
        pooled = (feats * m[:, None]).sum(-1) / m.sum(-1, keepdim=True).clamp_min(1)
        return self.net(pooled.float()).squeeze(-1)


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("ckpt"); a.add_argument("out")
    a.add_argument("--cell", default="K562"); a.add_argument("--test-fold", type=int, default=1, help="1..10, the paper's folds")
    a.add_argument("--mode", choices=("frozen", "decoder", "full"), default="full")
    a.add_argument("--random-init", action="store_true")
    a.add_argument("--form", choices=("genomic", "insert"), default="genomic"); a.add_argument("--window", type=int, default=2048)
    a.add_argument("--share", type=float, default=1.0); a.add_argument("--seed", type=int, default=0)
    a.add_argument("--epochs", type=int, default=3); a.add_argument("--batch", type=int, default=32)
    a.add_argument("--lr-head", type=float, default=1e-3); a.add_argument("--lr-trunk", type=float, default=2e-5)
    a.add_argument("--mem-gib", type=float, default=0.0, help="cap this process's card memory (0: no cap)")
    a.add_argument("--max-rows", type=int, default=0, help="smoke runs: this many rows per split")
    a.add_argument("--rc", action="store_true", help="train on both strands (random per row) and average both at test")
    a = a.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev = torch.device("cuda"); torch.cuda.init()
    if a.mem_gib:
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.mem_gib * 2 ** 30 / torch.cuda.get_device_properties(dev).total_memory))
    torch.cuda.reset_peak_memory_stats(dev)

    model, state = G.load_checkpoint(a.ckpt)
    if a.random_init:
        model = G.KronGenome(model.config)
    model = model.to(dev); model.checkpoint_core = True
    unit = 2 ** model.config.n_down
    if a.form == "genomic" and a.window % unit:
        raise SystemExit(f"--window must be a multiple of {unit}")
    feats = {}
    run_x = model.run_xheads

    def keep(x):
        feats["x"] = x
        return run_x(x)
    model.run_xheads = keep
    model.x_active = set()                                                # no track head runs

    t = pl.read_parquet(interim_dir("eval", "lentimpra-v1") / "elements.parquet").filter(
        (pl.col("cell") == a.cell) & pl.col("activity").is_not_null())
    if "alt" in t.columns:                                                # reference elements only
        t = t.filter(pl.col("alt").is_null() | (pl.col("alt") == ""))
    if not 1 <= a.test_fold <= 10:
        raise SystemExit("--test-fold is 1..10")
    val_fold = a.test_fold % 10 + 1
    parts = {"test": t.filter(pl.col("fold") == a.test_fold), "val": t.filter(pl.col("fold") == val_fold),
             "train": t.filter(~pl.col("fold").is_in([a.test_fold, val_fold]))}
    if a.share < 1.0:
        parts["train"] = parts["train"].sample(fraction=a.share, seed=a.seed)
    if a.max_rows:
        parts = {k: v.head(a.max_rows) for k, v in parts.items()}
    codes = hg38_reader()
    data = {k: (*encode_elements(v, a.form, a.window, unit, codes), v["activity"].to_numpy().astype(np.float32))
            for k, v in parts.items()}

    with torch.no_grad():                                                 # the decoder's width, read once
        x0 = torch.from_numpy(data["val"][0][:1]).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(x0, None, torch.zeros(1, dtype=torch.long, device=dev))
    head = Head(feats["x"].shape[1]).to(dev)
    for p in model.parameters():
        p.requires_grad_(False)
    if a.mode == "full":
        trunk = [p for p in model.parameters()]
    elif a.mode == "decoder":
        trunk = [p for n, p in model.named_parameters() if n.split(".")[0] in ("up", "up2", "skip_proj", "skip_scale", "local", "inter")]
    else:
        trunk = []
    for p in trunk:
        p.requires_grad_(True)
    groups = [{"params": head.parameters(), "lr": a.lr_head}] + ([{"params": trunk, "lr": a.lr_trunk}] if trunk else [])
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    n_train = sum(p.numel() for p in head.parameters()) + sum(p.numel() for p in trunk)

    def revcomp(xb: torch.Tensor, sb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The reverse strand: complemented bases in reverse order (N and others kept), the span mirrored."""
        xr = torch.where(xb < 4, 3 - xb, xb).flip(-1)
        n = xb.shape[-1]
        return xr, torch.stack([n - sb[:, 1], n - sb[:, 0]], 1)

    def forward_head(xb: torch.Tensor, sb: torch.Tensor) -> torch.Tensor:
        model(xb, None, torch.zeros(len(xb), dtype=torch.long, device=dev))
        return head(feats["x"], sb)

    def predict(split: str) -> np.ndarray:
        model.eval(); head.eval(); x, span, _ = data[split]; out = []
        with torch.no_grad():
            for i in range(0, len(x), a.batch):
                xb = torch.from_numpy(x[i:i + a.batch]).to(dev); sb = torch.from_numpy(span[i:i + a.batch]).to(dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    p = forward_head(xb, sb).float()
                    if a.rc:                                              # average both strands (MPRALegNet's test rule)
                        p = 0.5 * (p + forward_head(*revcomp(xb, sb)).float())
                    out.append(p.cpu().numpy())
        return np.concatenate(out)

    t0 = time.time(); best = (-2.0, None, -1); best_trunk = {}
    x, span, y = data["train"]; y_t = torch.from_numpy(y)
    steps_per_epoch = math.ceil(len(x) / a.batch)
    for ep in range(a.epochs):
        model.train(a.mode != "frozen"); head.train(); order = rng.permutation(len(x))
        for i in range(steps_per_epoch):
            idx = order[i * a.batch:(i + 1) * a.batch]
            xb = torch.from_numpy(x[idx]).to(dev); sb = torch.from_numpy(span[idx]).to(dev)
            if a.rc:                                                      # each row on a random strand
                flip = torch.from_numpy(rng.random(len(idx)) < 0.5).to(dev)
                xr, sr = revcomp(xb, sb)
                xb = torch.where(flip[:, None], xr, xb); sb = torch.where(flip[:, None], sr, sb)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if a.mode == "frozen":
                    with torch.no_grad():
                        model(xb, None, torch.zeros(len(xb), dtype=torch.long, device=dev))
                else:
                    model(xb, None, torch.zeros(len(xb), dtype=torch.long, device=dev))
                pred = head(feats["x"], sb)
            loss = torch.nn.functional.mse_loss(pred.float(), y_t[idx].to(dev))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        pv = predict("val"); r = float(pearsonr(pv, data["val"][2])[0])
        print(f"epoch {ep + 1}: train mse {loss.item():.4f}, val Pearson {r:.4f}, {time.time() - t0:.0f} s", flush=True)
        if r > best[0]:
            best = (r, {k: v.detach().clone() for k, v in head.state_dict().items()},
                    ep + 1)
            best_trunk = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    head.load_state_dict(best[1])
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in best_trunk:
                p.copy_(best_trunk[n])
    pt, pv = predict("test"), predict("val")
    row = dict(ckpt=str(a.ckpt), step=state.get("step"), random_init=a.random_init, cell=a.cell, test_fold=a.test_fold,
               mode=a.mode, rc=a.rc, form=a.form, window=a.window if a.form == "genomic" else None, share=a.share, seed=a.seed,
               n_train=len(data["train"][2]), n_val=len(data["val"][2]), n_test=len(data["test"][2]),
               best_epoch=best[2], epochs=a.epochs, batch=a.batch, lr_head=a.lr_head, lr_trunk=a.lr_trunk,
               trainable_params=n_train,
               test_pearson=float(pearsonr(pt, data["test"][2])[0]), test_spearman=float(spearmanr(pt, data["test"][2])[0]),
               val_pearson=float(pearsonr(pv, data["val"][2])[0]),
               wall_s=round(time.time() - t0, 1), peak_gib=round(torch.cuda.max_memory_allocated(dev) / 2 ** 30, 2),
               mem_cap_gib=a.mem_gib or None, card=torch.cuda.get_device_name(dev), host=socket.gethostname())
    print(json.dumps(row), flush=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
