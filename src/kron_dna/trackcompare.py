"""Track predictions of several models scored against one target at 128 bp, the same windows and tracks for all.

Each prediction is an array ``[windows, bins, tracks]`` in the target's units (or rescaled to them: ``scale``
multiplies each track by target total / prediction total on the scored windows, for a model whose outputs are in
other units, such as AlphaGenome's). Per track: Spearman, and Pearson of ``log1p`` (predictions clipped at 0). Then
the means by assay, over all scored tracks, and over the tracks shared by every model.

The manifest is a JSON list, one entry per prediction::

    [{"model": "KronGenome F 240k", "context": "1 Mb", "path": "pred_128bp_F.npy"},
     {"model": "AlphaGenome", "context": "1 Mb", "path": "ag.npy", "scale": true},
     {"model": "AlphaGenome folds", "context": "1 Mb", "path": "ag_f1.npy", "windows": [0, 3, 7], "scale": true}]

An entry's ``windows`` (indices into the target) says which windows its array holds, in order; without it the array
holds every window. Window sets (``--sets``, JSON ``{"name": [indices]}``; default every window as ``all``) are
scored for every entry that holds them.

    python -m kron_dna.trackcompare MANIFEST TARGET.npy TRACKS.csv --sets sets.json --out compare.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import typer
from scipy.stats import spearmanr


def track_metrics(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Spearman and Pearson of log1p over every bin of one track (any shape; predictions clipped at 0)."""
    p = np.clip(pred.reshape(-1).astype(np.float64), 0, None)
    y = target.reshape(-1).astype(np.float64)
    return float(spearmanr(p, y).statistic), float(np.corrcoef(np.log1p(p), np.log1p(y))[0, 1])


def total_scale(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per track, target total / prediction total (``[..., tracks]``); NaN where the prediction is not finite or sums
    to 0."""
    axes = tuple(range(pred.ndim - 1))
    ps = pred.sum(axes)
    out = np.full(pred.shape[-1], np.nan)
    ok = np.isfinite(ps) & (ps != 0)
    out[ok] = target.sum(axes)[ok] / ps[ok]
    return out


def score(model: str, context: str, window_set: str, pred: np.ndarray, target: np.ndarray,
          scale: bool = False) -> list[dict]:
    """One row per track with a finite prediction: ``pred`` and ``target`` are ``[windows, bins, tracks]`` over the
    same windows."""
    if pred.shape != target.shape:
        raise ValueError(f"{model}: prediction {pred.shape} and target {target.shape} differ")
    if scale:
        pred = pred * total_scale(pred, target)
    rows = []
    for ch in range(pred.shape[-1]):
        p = pred[..., ch]
        if not np.isfinite(p).all():
            continue
        sp, pe = track_metrics(p, target[..., ch])
        rows.append(dict(model=model, context=context, window_set=window_set, n_windows=pred.shape[0], channel=ch,
                         spearman_128bp=sp, pearson_log1p_128bp=pe))
    return rows


def compare(manifest: list[dict], target: np.ndarray, sets: dict[str, list[int]], root: Path = Path(".")) -> pl.DataFrame:
    """Every manifest entry scored on every window set it holds (see the module's text)."""
    rows = []
    for e in manifest:
        pred = np.load(root / e["path"], mmap_mode="r")
        held = list(e.get("windows", range(target.shape[0])))
        if pred.shape[0] != len(held):
            raise ValueError(f"{e['path']}: {pred.shape[0]} windows, the manifest says {len(held)}")
        at = {w: i for i, w in enumerate(held)}
        for name, wins in sets.items():
            if not all(w in at for w in wins):
                continue
            rows += score(e["model"], e.get("context", ""), name, np.asarray(pred[[at[w] for w in wins]]),
                          target[wins], bool(e.get("scale", False)))
    if not rows:
        raise ValueError("nothing scored: no entry holds any window set")
    return pl.DataFrame(rows)


def aggregate(scores: pl.DataFrame, tracks: pl.DataFrame, exclude: list[int] = ()) -> pl.DataFrame:
    """Mean Spearman and Pearson by model, context, window set and assay; plus ``all tracks`` and ``shared tracks``
    (the tracks every model of that window set scored). ``tracks`` has ``channel`` and ``assay``; ``exclude`` drops
    channels first."""
    d = scores.join(tracks.select(pl.col("channel").cast(pl.Int64), "assay"), on="channel").filter(
        ~pl.col("channel").is_in(list(exclude)))
    keys = ["model", "context", "window_set"]
    agg = [pl.col("pearson_log1p_128bp").mean().alias("pearson"), pl.col("spearman_128bp").mean().alias("spearman"),
           pl.len().alias("n_tracks")]
    n_models = d.group_by("window_set").agg(pl.struct("model", "context").n_unique().alias("n"))
    shared = (d.group_by("window_set", "channel").agg(pl.struct("model", "context").n_unique().alias("k"))
              .join(n_models, on="window_set").filter(pl.col("k") == pl.col("n")).select("window_set", "channel"))
    parts = [d.group_by(keys + ["assay"]).agg(agg),
             d.group_by(keys).agg(agg).with_columns(pl.lit("all tracks").alias("assay")),
             d.join(shared, on=["window_set", "channel"]).group_by(keys).agg(agg)
              .with_columns(pl.lit("shared tracks").alias("assay"))]
    cols = keys + ["assay", "pearson", "spearman", "n_tracks"]
    return pl.concat([p.select(cols) for p in parts]).sort("window_set", "model", "context", "assay")


app = typer.Typer(add_completion=False)


@app.command()
def main(manifest: Path, target: Path, tracks: Path = typer.Argument(..., help="CSV with channel and assay columns"),
         sets: Path = typer.Option(None, help='JSON {"name": [window indices]}; default every window as "all"'),
         exclude: str = typer.Option("", help="channels to leave out of the means, e.g. 19-28,33"),
         out: Path = typer.Option(Path("trackcompare.csv"))) -> None:
    """Score the manifest's predictions; write per-track rows to OUT and the means to OUT with ``_by_assay``."""
    tgt = np.load(target, mmap_mode="r")
    entries = json.loads(manifest.read_text())
    wsets = json.loads(sets.read_text()) if sets else {"all": list(range(tgt.shape[0]))}
    ex = []
    for part in filter(None, exclude.split(",")):
        a, _, b = part.partition("-")
        ex += list(range(int(a), int(b or a) + 1))
    tr = pl.read_csv(tracks)
    per = compare(entries, np.asarray(tgt), wsets, manifest.parent)
    per.join(tr.with_columns(pl.col("channel").cast(pl.Int64)), on="channel", how="left").write_csv(out)
    means = aggregate(per, tr, ex)
    means.write_csv(out.with_name(out.stem + "_by_assay" + out.suffix))
    print(means.filter(pl.col("assay").is_in(["all tracks", "shared tracks"])))


if __name__ == "__main__":
    app()
