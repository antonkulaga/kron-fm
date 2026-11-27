"""Enhancer-gene links read from input gradients (AlphaGenome 2's readout), on the CRISPRi element-gene pairs.

    python scripts/kron_dna/diag/egrad.py CKPT PAIRS.parquet OUT.parquet [--max-genes N] [--reach 250000] [--context 524288]

For each gene (its TSS = read_pos, its cell type's RNA channels = channel_ids, head = the pairs' head), one 1 Mb window
centred on the TSS: the head's predicted RNA summed over the TSS bin and its two neighbours, and its gradient with
respect to the one-hot input. Each candidate element's score is |sum over the element of gradient x input|. Elements
farther than --reach from the TSS are left out (the window reaches +-context/2). Writes one row per pair with its score;
AUPRC by distance bin, beside a distance-only baseline, is printed at the end.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import polars as pl
import torch

from kron_core.paths import interim_dir
from kron_dna import genome as G
from kron_dna import xheads as X

CTX, BIN = 1048576, 128


def ap(score: np.ndarray, label: np.ndarray) -> float:
    o = np.argsort(-score); l = label[o].astype(float); tp = np.cumsum(l); k = np.arange(1, len(l) + 1)
    return float((tp[l > 0] / k[l > 0]).mean()) if l.sum() else float("nan")


def main():
    a = argparse.ArgumentParser(); a.add_argument("ckpt"); a.add_argument("pairs"); a.add_argument("out")
    a.add_argument("--max-genes", type=int, default=0); a.add_argument("--reach", type=int, default=250000)
    a.add_argument("--context", type=int, default=524288, help="window length (1 Mb needs more than a shared card)")
    a = a.parse_args()
    global CTX
    CTX = a.context
    t = pl.read_parquet(a.pairs).filter(pl.col("distance").abs() < a.reach)
    genes = t.select("gene", "read_pos", "chrom", "cell_type", "head").unique(maintain_order=True)
    if a.max_genes:
        genes = genes.head(a.max_genes)
    src = interim_dir("genome", "hg38")
    bases = np.memmap(src / "bases.u8", dtype=np.uint8, mode="r"); contigs = pl.read_parquet(src / "contigs.parquet")

    def codes_of(chrom, lo, hi):
        out = np.full(hi - lo, 4, dtype=np.int64)
        for r in contigs.filter((pl.col("chrom") == chrom) & (pl.col("start") < hi) & (pl.col("end") > lo)).iter_rows(named=True):
            s, e = max(lo, r["start"]), min(hi, r["end"]); out[s - lo:e - lo] = bases[r["offset"] + s - r["start"]:r["offset"] + e - r["start"]]
        return out

    dev = torch.device("cuda"); torch.cuda.init()
    model, state = G.load_checkpoint(a.ckpt); model = model.to(dev).eval(); model.checkpoint_core = True
    G.LEAN_CONV = True                                    # recompute norm+act in backward: memory
    stores = ((state.get("run_record") or {}).get("xheads") or {}).get("stores") or {}
    unit = 2 ** model.config.n_down; rows, t0 = [], time.time()
    for gi, g in enumerate(genes.iter_rows(named=True)):
        tss = int(g["read_pos"]); lo = (tss - CTX // 2) // BIN * BIN; hi = lo + CTX
        codes = torch.from_numpy(codes_of(g["chrom"], lo, hi)).to(dev)
        x = G.base_channels(codes[None]).float().requires_grad_(True)       # [1, 6, L] one-hot
        ids = list(stores[g["head"]]["ids"])
        sub = t.filter((pl.col("gene") == g["gene"]) & (pl.col("read_pos") == tss) & (pl.col("cell_type") == g["cell_type"]))
        ch = [ids.index(c) for c in sub["channel_ids"][0] if c in ids]
        if not ch:
            continue
        b0 = (tss - lo) // BIN * BIN
        model.x_active = {g["head"]}; model.x_span = (max(0, b0 - 4 * BIN), min(CTX, b0 + 5 * BIN))
        seg = torch.zeros(1, CTX // unit, dtype=torch.long, device=dev); ctx = torch.zeros(1, dtype=torch.long, device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(x, seg, ctx)
            out = model.last_x[g["head"]][0].float()                         # [span bins or bases, channels]
        a0, a1 = model.last_x_at
        rel = (tss - lo) - a0
        step = (a1 - a0) // out.shape[0]
        k = rel // step
        y = out[max(0, k - 1):k + 2, ch].sum()
        model.zero_grad(set_to_none=True)
        y.backward()
        gx = (x.grad[0] * x[0].detach()).sum(0)                              # [L] gradient x input at the true base
        for r in sub.iter_rows(named=True):
            s, e = r["del_start"] - lo, max(r["del_end"], r["del_start"] + 1) - lo
            if 0 <= s < CTX and 0 < e <= CTX:
                rows.append(dict(gene=r["gene"], cell_type=r["cell_type"], distance=r["distance"], regulated=r["regulated"],
                                 bench_set=r["bench_set"], score=float(gx[s:e].sum().abs())))
        if gi % 50 == 0:
            print(f"gene {gi + 1}/{genes.height}, {len(rows)} pairs, {time.time() - t0:.0f} s, peak "
                  f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    df = pl.DataFrame(rows); df.write_parquet(a.out)
    bins = [(0, 10_000), (10_000, 100_000), (100_000, a.reach)]
    for lo_, hi_ in bins:
        d = df.filter((pl.col("distance").abs() >= lo_) & (pl.col("distance").abs() < hi_))
        if d.height == 0:
            continue
        lab = d["regulated"].to_numpy()
        print(f"{lo_:>7}-{hi_:<7} pairs {d.height:5d} positives {int(lab.sum()):4d}  AUPRC gradient {ap(d['score'].to_numpy(), lab):.3f}"
              f"  distance-only {ap(-d['distance'].abs().to_numpy().astype(float), lab):.3f}  base rate {lab.mean():.3f}")


if __name__ == "__main__":
    main()
