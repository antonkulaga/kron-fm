"""How much a block-JEPA predictor's output can be decoded, with the decoders the model already has.

Anton (23 Nov): "does it learn enough to also make a decoder for it?" On held-out windows (chr8/chr9 of the primary
genome) blocks of bins are hidden (MASK input, as block JEPA trains them); then the core states at the hidden bins
are set four ways and the model's own decoder, masked-base head and NTv3 track head run on each:

- ``real``: the core states of the clean (unmasked) window: the upper bound;
- ``jepa_raw`` / ``jepa`` / ``jepa_oracle_norm``: the JEPA predictor on the masked window's core, as it comes, rescaled
  to the masked core's norm per bin, and rescaled to the clean state's norm (the loss is cosine, so norm is free);
- ``online``: the masked window's core as it is (the trunk's own guess, no predictor; the training log's "copy
  baseline" is this state against the target);
- ``nearest_visible``: each hidden bin takes the nearest visible bin's clean state.
``real`` keeps the masked skips, so it is the honest ceiling: the decoder never sees the hidden bases.

The decoder's skip connections come from the masked input in every case, so inside a block the only information
about the hidden bases is the core state. Scored inside the blocks: masked-base accuracy against the true bases,
and per-track Pearson of the NTv3 predictions against the clean window's own predictions (how much of the model's
output a variant recovers). Usage:

    python scripts/kron_dna/jepa_decode.py CKPT [CKPT ...] --windows 32 --window 131072 --out OUT.csv
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import polars as pl
import torch
import typer

from kron_dna import genome as G

app = typer.Typer(add_completion=False)


def core_of(model: G.KronGenome, seq: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """The core states ``[B, bins, d]`` and the encoder skips of ``seq`` (nuclear context, one segment per row)."""
    x, skips = model.encode(seq, None)
    h = x.transpose(1, 2)
    b, bins = h.shape[0], h.shape[1]
    if model.context is not None:
        h = h + model.context_rows(torch.zeros(b, dtype=torch.long, device=seq.device), b, seq.device).unsqueeze(1).to(h.dtype)
    cu = torch.arange(0, b * bins + 1, bins, device=seq.device, dtype=torch.int32)
    return model.run_core(h.reshape(1, b * bins, -1), cu).reshape(b, bins, -1), skips


def heads_of(model: G.KronGenome, core: torch.Tensor, skips: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None]:
    x = model.add_local(model.decode(core.transpose(1, 2), skips), skips)
    logits = model.head(x).transpose(1, 2)                                   # [B, L, 5 or 6]
    nt = model.ntv3(x).transpose(1, 2) if model.ntv3 is not None else None   # [B, L, T]
    return logits, nt


def nearest_visible(hidden: np.ndarray) -> np.ndarray:
    """For each bin, the index of the nearest visible bin (itself when visible)."""
    idx = np.arange(hidden.shape[0])
    vis = idx[~hidden]
    pos = np.searchsorted(vis, idx).clip(1, len(vis) - 1)
    left, right = vis[pos - 1], vis[pos]
    return np.where(np.abs(idx - left) <= np.abs(right - idx), left, right)


def pearson_tracks(a: np.ndarray, b: np.ndarray) -> float:
    """Mean over tracks of Pearson(a[:, t], b[:, t]), tracks with variance on both sides only."""
    a = a - a.mean(0)
    b = b - b.mean(0)
    den = np.sqrt((a * a).sum(0) * (b * b).sum(0))
    ok = den > 1e-12
    return float(((a * b).sum(0)[ok] / den[ok]).mean()) if ok.any() else float("nan")


@app.command()
def main(checkpoints: list[Path] = typer.Argument(..., help="checkpoint files with a JEPA predictor"),
         genome: Path = typer.Option(None, help="primary genome dir (default: KRON_FM_DATA/interim/genome/hg38)"),
         chroms: str = typer.Option("chr8,chr9", help="held-out chromosomes to draw windows from"),
         windows: int = typer.Option(32), window: int = typer.Option(131072),
         share: float = typer.Option(0.2, help="share of bins hidden, as --jepa-block-share"),
         lo: int = typer.Option(1024), hi: int = typer.Option(8192), seed: int = typer.Option(0),
         out: Path = typer.Option(Path("jepa_decode.csv"))) -> None:
    import os
    genome = genome or Path(os.environ["KRON_FM_DATA"]) / "interim" / "genome" / "hg38"
    contigs = pl.read_parquet(genome / "contigs.parquet").filter(pl.col("chrom").is_in(chroms.split(",")))
    if "len" not in contigs.columns:                              # a prepared genome's contigs: start/end/offset
        contigs = contigs.with_columns((pl.col("end") - pl.col("start")).alias("len"))
    contigs = contigs.filter(pl.col("len") > window)
    bases = np.memmap(genome / "bases.u8", dtype=np.uint8, mode="r")
    rng = np.random.default_rng(seed)
    picks = []
    while len(picks) < windows:                                   # the same windows for every checkpoint
        r = contigs.row(int(rng.integers(contigs.height)), named=True)
        at = int(r["offset"]) + int(rng.integers(0, int(r["len"]) - window + 1))
        s = np.asarray(bases[at:at + window]).astype(np.int64)
        if (s < 4).mean() >= 0.95:
            picks.append((at, s))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for ck in checkpoints:
        model, state = G.load_checkpoint(ck)
        model = model.to(dev).eval()
        if model.jepa_pred is None:
            print(f"{ck}: no JEPA predictor, skipped")
            continue
        unit = 2 ** model.config.n_down
        acc = {k: [] for k in ("real", "jepa_raw", "jepa", "jepa_oracle_norm", "online", "nearest_visible")}
        cor = {k: [] for k in ("real", "jepa_raw", "jepa", "jepa_oracle_norm", "online", "nearest_visible")}
        brng = np.random.default_rng(seed + 1)
        for at, s in picks:
            hid = G.block_mask(np.zeros((1, window // unit), dtype=np.int64), s[None], unit, share, lo, hi, brng)[0]
            if not hid.any():
                continue
            base_hidden = np.repeat(hid, unit)
            seq = torch.tensor(s, device=dev).unsqueeze(0)
            masked = seq.clone()
            masked[0, torch.tensor(base_hidden, device=dev)] = 5
            with torch.no_grad():
                core_c, skips_c = core_of(model, seq)
                core_m, skips_m = core_of(model, masked)
                _, nt_clean = heads_of(model, core_c, skips_c)
                hb = torch.tensor(hid, device=dev)
                near = torch.tensor(nearest_visible(hid), device=dev)
                pred = model.jepa_pred(core_m)
                # the JEPA loss is 1 - cosine, so the predictor's norm is free: rescale each predicted state to the norm
                # of the masked core's state at that bin (available at inference), and to the clean state's norm
                # (an oracle norm, the ceiling of the direction alone)
                pn = pred.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                variants = {"real": core_c, "jepa_raw": pred, "jepa": pred / pn * core_m.norm(dim=-1, keepdim=True),
                            "jepa_oracle_norm": pred / pn * core_c.norm(dim=-1, keepdim=True),
                            "online": core_m, "nearest_visible": core_c[:, near]}
                for k, cv in variants.items():
                    mix = core_m.clone()
                    mix[0, hb] = cv[0, hb].to(mix.dtype)
                    logits, nt = heads_of(model, mix, skips_m)
                    m = torch.tensor(base_hidden & (s < 4), device=dev)
                    acc[k].append(float((logits[0, m, :4].argmax(-1) == seq[0, m]).float().mean()))
                    if nt is not None and nt_clean is not None:
                        cor[k].append(pearson_tracks(nt[0, m].float().cpu().numpy(), nt_clean[0, m].float().cpu().numpy()))
        row = dict(checkpoint=str(ck), step=int(state.get("step", -1)), windows=len(acc["real"]),
                   **{f"base_acc_{k}": float(np.mean(v)) for k, v in acc.items()},
                   **{f"ntv3_r_vs_clean_{k}": float(np.nanmean(v)) if v else float("nan") for k, v in cor.items()})
        rows.append(row)
        print(json.dumps(row))
    pl.DataFrame(rows).write_csv(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    app()
