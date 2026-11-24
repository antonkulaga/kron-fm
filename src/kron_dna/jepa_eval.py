"""What a block-JEPA predictor learned, read on held-out windows with the model's own decoder and heads.

Blocks of bins are hidden (MASK input, as block JEPA trains them). The core states at the hidden bins are then set
several ways, and each is scored two ways: decoded, and as a distance to the clean state.

Decoded (the decoder's skips always come from the masked input, so inside a block the only information about the
hidden bases is the core state):

- ``real``: the clean window's core: the ceiling;
- ``jepa_raw`` / ``jepa`` / ``jepa_oracle_norm``: the predictor's states as they come, rescaled per bin to the masked
  core's norm (known at inference), and to the clean state's norm (an oracle: the ceiling of the direction alone);
- ``online``: the masked core as it is, no predictor (the training log's copy baseline);
- ``nearest_visible``: each hidden bin takes the nearest visible bin's clean state;
- ``interp``: inside each block, the line between the clean states at its two edges.

Scored inside the blocks: masked-base accuracy against the true bases, and per-track Pearson of the NTv3 predictions
against the clean window's own (``ntv3_r_vs_clean_*``).

Distance (``raw_*`` and ``centred_*``): ``1 - cosine`` to the clean state at hidden bins, for the predictor, the masked
core and the interpolation. ``centred`` first centres and scales every dimension by the clean states' mean and std
over all windows: a raw cosine loss is easy to win by predicting the states' common mean (``cos_state_vs_mean``
says how much it dominates), and the centred gain of the predictor over the masked core (``centred_pred_gain``) is
the one that counts.

Spread of the clean states: per-dimension std (``dim_std_min``/``_mean``) and the share of each dimension's variance
that lies inside windows rather than between them (``within_share_*``; ``within_share_weakest_dim`` for the dimension
with the lowest std): a dimension near 0 there is a per-window constant.

    python -m kron_dna.jepa_eval CKPT [CKPT ...] --windows 32 --window 131072 --out jepa_eval.csv
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
import typer

from kron_dna import genome as G

DECODE_VARIANTS = ("real", "jepa_raw", "jepa", "jepa_oracle_norm", "online", "nearest_visible", "interp")


def nearest_visible(hidden: np.ndarray) -> np.ndarray:
    """For each bin, the index of the nearest visible bin (itself when visible; the left one on a tie)."""
    idx = np.arange(hidden.shape[0])
    vis = idx[~hidden]
    if vis.size == 0:
        raise ValueError("every bin is hidden")
    if vis.size == 1:
        return np.full_like(idx, vis[0])
    pos = np.searchsorted(vis, idx).clip(1, len(vis) - 1)
    left, right = vis[pos - 1], vis[pos]
    return np.where(np.abs(idx - left) <= np.abs(right - idx), left, right)


def interpolate(core: torch.Tensor, hidden: np.ndarray) -> torch.Tensor:
    """``core`` [1, bins, d]; each hidden run replaced by the line between its visible neighbours (the one neighbour,
    held flat, at a window edge)."""
    out = core.clone()
    h = np.concatenate([[False], hidden, [False]]).astype(np.int8)
    d = np.diff(h)
    n = hidden.shape[0]
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):       # hidden run [a, b)
        left = core[0, a - 1] if a > 0 else None
        right = core[0, b] if b < n else None
        if left is None and right is None:
            raise ValueError("every bin is hidden")
        left = right if left is None else left
        right = left if right is None else right
        t = torch.linspace(0, 1, b - a + 2, device=core.device, dtype=core.dtype)[1:-1, None]
        out[0, a:b] = (1 - t) * left + t * right
    return out


def pearson_tracks(a: np.ndarray, b: np.ndarray) -> float:
    """Mean over tracks (columns) of Pearson(a[:, t], b[:, t]), over tracks with variance on both sides only."""
    a = a - a.mean(0)
    b = b - b.mean(0)
    den = np.sqrt((a * a).sum(0) * (b * b).sum(0))
    ok = den > 1e-12
    return float(((a * b).sum(0)[ok] / den[ok]).mean()) if ok.any() else float("nan")


def cosine_distance(a: torch.Tensor, b: torch.Tensor, mu: torch.Tensor | None = None,
                    sd: torch.Tensor | None = None) -> torch.Tensor:
    """``1 - cosine`` per row, after centring and scaling each dimension by ``mu``/``sd`` when given."""
    if mu is not None:
        a, b = (a - mu) / sd, (b - mu) / sd
    return 1 - F.cosine_similarity(a, b, dim=-1)


def within_window_share(cores: list[torch.Tensor]) -> torch.Tensor:
    """Per dimension, the share of the clean states' variance that lies inside windows (``[d]``, 0 to 1), from
    ``cores``, one ``[bins, d]`` per window. Near 0: the dimension is a per-window constant, the slow-feature collapse
    a regulariser on the overall spread (SIGReg) cannot see (H-JEPA, arXiv 2610.06805)."""
    allc = torch.cat(cores)
    total = allc.var(0, unbiased=False).clamp_min(1e-12)
    within = sum(c.var(0, unbiased=False) * c.shape[0] for c in cores) / allc.shape[0]
    return within / total


def core_of(model: G.KronGenome, seq: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """The core states ``[B, bins, d]`` and the encoder skips of ``seq`` (nuclear context, one segment per row)."""
    x, skips = model.encode(seq, None)
    h = x.transpose(1, 2)
    b, bins = h.shape[0], h.shape[1]
    if model.context is not None:
        rows = model.context_rows(torch.zeros(b, dtype=torch.long, device=seq.device), b, seq.device)
        h = h + rows.unsqueeze(1).to(h.dtype)
    cu = torch.arange(0, b * bins + 1, bins, device=seq.device, dtype=torch.int32)
    return model.run_core(h.reshape(1, b * bins, -1), cu).reshape(b, bins, -1), skips


def heads_of(model: G.KronGenome, core: torch.Tensor, skips: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Masked-base logits ``[B, L, 5 or 6]`` and NTv3 tracks ``[B, L, T]`` (None without the head) from core states."""
    x = model.add_local(model.decode(core.transpose(1, 2), skips), skips)
    logits = model.head(x).transpose(1, 2)
    nt = model.ntv3(x).transpose(1, 2) if model.ntv3 is not None else None
    return logits, nt


def predict(model: G.KronGenome, core_masked: torch.Tensor, hidden: np.ndarray) -> torch.Tensor:
    """The JEPA predictor on the masked core; the context predictor is told which bins are hidden."""
    model.jepa_hidden = torch.tensor(hidden, device=core_masked.device)[None]
    try:
        return model.predict_jepa(core_masked)
    finally:
        model.jepa_hidden = None


def sample_windows(bases: np.ndarray, contigs: pl.DataFrame, n: int, window: int, seed: int = 0,
                   min_acgt: float = 0.95) -> list[tuple[int, np.ndarray]]:
    """``n`` windows ``(offset, bases)`` drawn from ``contigs`` (``offset``/``len``) with at least ``min_acgt`` real
    bases; the same seed gives the same windows."""
    contigs = contigs.filter(pl.col("len") > window)
    if contigs.height == 0:
        raise ValueError(f"no contig longer than the window ({window})")
    rng = np.random.default_rng(seed)
    picks, tries = [], 0
    while len(picks) < n:
        tries += 1
        if tries > 100 * n:
            raise ValueError(f"found {len(picks)} of {n} windows with >= {min_acgt:.0%} ACGT")
        r = contigs.row(int(rng.integers(contigs.height)), named=True)
        at = int(r["offset"]) + int(rng.integers(0, int(r["len"]) - window + 1))
        s = np.asarray(bases[at:at + window]).astype(np.int64)
        if (s < 4).mean() >= min_acgt:
            picks.append((at, s))
    return picks


@torch.no_grad()
def evaluate(model: G.KronGenome, picks: list[tuple[int, np.ndarray]], share: float = 0.2, lo: int = 1024,
             hi: int = 8192, seed: int = 0) -> dict:
    """One row of scores for ``model`` (with a JEPA predictor, in eval mode) on the windows ``picks``."""
    if model.jepa_pred is None:
        raise ValueError("the model has no JEPA predictor")
    dev = next(model.parameters()).device
    unit = 2 ** model.config.n_down
    brng = np.random.default_rng(seed + 1)
    acc = {k: [] for k in DECODE_VARIANTS}
    cor = {k: [] for k in DECODE_VARIANTS}
    keep = []
    for _, s in picks:
        hid = G.block_mask(np.zeros((1, s.shape[0] // unit), dtype=np.int64), s[None], unit, share, lo, hi, brng)[0]
        if not hid.any() or hid.all():
            continue
        base_hidden = np.repeat(hid, unit)
        seq = torch.tensor(s, device=dev).unsqueeze(0)
        masked = seq.clone()
        masked[0, torch.tensor(base_hidden, device=dev)] = 5
        core_c, skips_c = core_of(model, seq)
        core_m, skips_m = core_of(model, masked)
        _, nt_clean = heads_of(model, core_c, skips_c)
        pred = predict(model, core_m, hid)
        hb = torch.tensor(hid, device=dev)
        pn = pred.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        variants = {"real": core_c, "jepa_raw": pred, "jepa": pred / pn * core_m.norm(dim=-1, keepdim=True),
                    "jepa_oracle_norm": pred / pn * core_c.norm(dim=-1, keepdim=True), "online": core_m,
                    "nearest_visible": core_c[:, torch.tensor(nearest_visible(hid), device=dev)],
                    "interp": interpolate(core_c, hid)}
        m = torch.tensor(base_hidden & (s < 4), device=dev)
        for k, cv in variants.items():
            mix = core_m.clone()
            mix[0, hb] = cv[0, hb].to(mix.dtype)
            logits, nt = heads_of(model, mix, skips_m)
            if m.any():
                acc[k].append(float((logits[0, m, :4].argmax(-1) == seq[0, m]).float().mean()))
                if nt is not None:
                    cor[k].append(pearson_tracks(nt[0, m].float().cpu().numpy(), nt_clean[0, m].float().cpu().numpy()))
        keep.append((core_c[0].float().cpu(), core_m[0].float().cpu(), pred[0].float().cpu(), hid))
    if not keep:
        raise ValueError("no window had hidden bins")
    allc = torch.cat([k[0] for k in keep])
    mu, sd = allc.mean(0), allc.std(0).clamp_min(1e-6)
    share = within_window_share([k[0] for k in keep])
    weakest = int(sd.argmin())
    row = dict(windows=len(keep), cos_state_vs_mean=float(F.cosine_similarity(allc, mu[None], dim=-1).mean()),
               dim_std_min=float(sd.min()), dim_std_mean=float(sd.mean()), within_share_min=float(share.min()),
               within_share_mean=float(share.mean()), within_share_weakest_dim=float(share[weakest]),
               **{f"base_acc_{k}": float(np.mean(v)) if v else float("nan") for k, v in acc.items()},
               **{f"ntv3_r_vs_clean_{k}": float(np.nanmean(v)) if v else float("nan") for k, v in cor.items()})
    for tag, (m_, s_) in (("raw", (None, None)), ("centred", (mu, sd))):
        dp = torch.cat([cosine_distance(k[2][k[3]], k[0][k[3]], m_, s_) for k in keep]).mean()
        dm = torch.cat([cosine_distance(k[1][k[3]], k[0][k[3]], m_, s_) for k in keep]).mean()
        di = torch.cat([cosine_distance(interpolate(k[0][None], k[3])[0][k[3]], k[0][k[3]], m_, s_) for k in keep]).mean()
        row.update({f"{tag}_pred_hidden": float(dp), f"{tag}_masked_hidden": float(dm),
                    f"{tag}_interp_hidden": float(di), f"{tag}_pred_gain": float(1 - dp / dm)})
    return row


app = typer.Typer(add_completion=False)


@app.command()
def main(checkpoints: list[Path] = typer.Argument(..., help="checkpoint files with a JEPA predictor"),
         genome: Path = typer.Option(None, help="prepared genome dir (default: KRON_FM_DATA/interim/genome/hg38)"),
         chroms: str = typer.Option("chr8,chr9", help="held-out chromosomes to draw windows from"),
         windows: int = typer.Option(32), window: int = typer.Option(131072),
         share: float = typer.Option(0.2, help="share of bins hidden, as --jepa-block-share"),
         lo: int = typer.Option(1024), hi: int = typer.Option(8192), seed: int = typer.Option(0),
         out: Path = typer.Option(Path("jepa_eval.csv"))) -> None:
    """Score each checkpoint's JEPA predictor on the same windows and blocks; one CSV row per checkpoint."""
    genome = genome or Path(os.environ["KRON_FM_DATA"]) / "interim" / "genome" / "hg38"
    contigs = pl.read_parquet(genome / "contigs.parquet").filter(pl.col("chrom").is_in(chroms.split(",")))
    if "len" not in contigs.columns:
        contigs = contigs.with_columns((pl.col("end") - pl.col("start")).alias("len"))
    picks = sample_windows(np.memmap(genome / "bases.u8", dtype=np.uint8, mode="r"), contigs, windows, window, seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for ck in checkpoints:
        model, state = G.load_checkpoint(ck)
        model = model.to(dev).eval()
        if model.jepa_pred is None:
            print(f"{ck}: no JEPA predictor, skipped")
            continue
        row = dict(checkpoint=str(ck), step=int(state.get("step", -1)), **evaluate(model, picks, share, lo, hi, seed))
        rows.append(row)
        print(json.dumps(row))
    pl.DataFrame(rows).write_csv(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    app()
