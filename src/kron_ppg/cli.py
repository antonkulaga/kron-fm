"""``kron-ppg``: download, prepare, pretrain, fit the age clock, evaluate and export to ONNX.

    uv run kron-ppg download bidmc                       # or: uv run kron-data download bidmc
    uv run kron-ppg download vitaldb --cases 1000        # VitalDB tracks, one case per person
    uv run kron-ppg prepare bidmc vitaldb                # -> data/interim/ppg/<source>/fs25-w30s
    uv run kron-ppg pretrain vitaldb bidmc --steps 5000  # -> data/output/runs/kron-ppg/pretrain-...
    uv run kron-ppg clock vitaldb --encoder <run>/checkpoint.pt
    uv run kron-ppg evaluate <clock run> bidmc           # a trained clock on another source
    uv run kron-ppg export <clock run>                   # stream.onnx for a phone, checked with ONNX Runtime
    uv run kron-ppg info --d-model 256 --n-layers 6
    uv run kron-ppg hub-export vitaldb bidmc galaxyppg --rates 25 --rates 125  # -> data/output/hub/<name>

A source name stands for its prepared variant at the given ``--sample-rate`` and ``--seconds``; a path
to a variant folder works too.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import typer

from kron_core.paths import input_dir, run_name, runs_dir
from kron_ppg.model import KronPPG, KronPPGConfig

app = typer.Typer(add_completion=False, no_args_is_help=True)
MODEL = "kron-ppg"


def _variants(names: list[str], sample_rate: float, seconds: float) -> list[Path]:
    from kron_ppg.data import variant_dir

    out = []
    for name in names:
        path = Path(name)
        folder = path if (path / "windows.parquet").is_file() else variant_dir(name, sample_rate, seconds)
        if not (folder / "windows.parquet").is_file():
            raise typer.BadParameter(f"{folder} is not prepared: run `uv run kron-ppg prepare {name}`")
        out.append(folder)
    return out


def _config(sample_rate: float, patch: int, d_model: int, n_heads: int, n_layers: int, ff_units: int,
            dropout: float, horizon: int) -> KronPPGConfig:
    return KronPPGConfig(sample_rate=sample_rate, patch=patch, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                         ff_units=ff_units, dropout=dropout, horizon=horizon)


@app.command()
def download(names: list[str] = typer.Argument(..., help="bidmc, vitaldb, galaxyppg"),
             cases: int | None = typer.Option(None, help="VitalDB: how many cases (one per person)"),
             force: bool = typer.Option(False, help="download again even when the files are present")) -> None:
    """Download PPG sources into data/input/ppg/<name> with SOURCE.json."""
    from kron_core.download import fetch_source
    from kron_ppg.sources import SOURCES, fetch_vitaldb

    for name in names:
        if name not in SOURCES:
            raise typer.BadParameter(f"unknown source {name!r}; one of {', '.join(SOURCES)}")
        source = SOURCES[name]
        typer.echo(f"{name} -> {source.folder(input_dir())}")
        if name == "vitaldb":
            record = fetch_vitaldb(source, source.folder(input_dir()), force, cases)
        else:
            record = fetch_source(source, input_dir(), force=force)
        typer.echo(f"  {len(record.get('files', []))} files" if "files" in record else f"  {record}")


@app.command()
def prepare(sources: list[str] = typer.Argument(..., help="downloaded sources to prepare"),
            sample_rate: float = typer.Option(25.0, help="the model's rate, Hz"),
            seconds: float = typer.Option(30.0, help="window length, s"),
            hop: float | None = typer.Option(None, help="window step, s (default: no overlap)"),
            max_windows: int = typer.Option(240, help="most windows kept per recording, spread over it"),
            min_periodicity: float = typer.Option(0.35, help="the quality gate's autocorrelation threshold"),
            workers: int | None = typer.Option(None, help="parallel processes (default: available CPUs, at most 16)")
            ) -> None:
    """Band-pass, resample, window, scale and quality-gate into data/interim/ppg/<source>/fs<rate>-w<s>s."""
    from kron_ppg.prepare import prepare_source

    for source in sources:
        info = prepare_source(source, sample_rate, seconds, hop, max_windows, min_periodicity, workers=workers)
        typer.echo(json.dumps({k: info[k] for k in ("source", "windows", "subjects", "subjects_with_windows",
                                                    "recordings_without_windows")}))


@app.command()
def info(sample_rate: float = 25.0, patch: int = 5, d_model: int = 256, n_heads: int = 4, n_layers: int = 6,
         ff_units: int = 768, horizon: int = 5) -> None:
    """Parameter count of a configuration (built on the meta device)."""
    config = _config(sample_rate, patch, d_model, n_heads, n_layers, ff_units, 0.0, horizon)
    with torch.device("meta"):
        n = KronPPG(config).n_parameters()
    typer.echo(f"{n / 1e6:.2f}M parameters, {config.patch_seconds:.2f} s per token: {json.dumps(config.to_dict())}")


@app.command()
def pretrain(sources: list[str] = typer.Argument(..., help="prepared sources (or variant folders)"),
             out: Path | None = typer.Option(None, help="run folder (default data/output/runs/kron-ppg/...)"),
             objective: str = typer.Option("next", help="next, or next+subject"),
             subject_weight: float = 0.1,
             sample_rate: float = 25.0, seconds: float = 30.0, patch: int = 5, d_model: int = 256, n_heads: int = 4,
             n_layers: int = 6, ff_units: int = 768, dropout: float = 0.1, horizon: int = 5,
             batch_size: int = 64, steps: int = 5_000, lr: float = 1e-3, eval_every: int = 250,
             seed: int = 0, device: str = "cuda") -> None:
    """Self-supervised pretraining of the encoder (next-patch prediction, optionally with participant contrast)."""
    from kron_ppg.train import PretrainArgs
    from kron_ppg.train import pretrain as run

    folders = _variants(sources, sample_rate, seconds)
    out = out or runs_dir(MODEL) / run_name("pretrain", objective.replace("+", "_"), *sources)
    config = _config(sample_rate, patch, d_model, n_heads, n_layers, ff_units, dropout, horizon)
    result = run(PretrainArgs(folders, out, config, objective, subject_weight, batch_size, steps, lr,
                              eval_every=eval_every, seed=seed, device=device))
    typer.echo(f"{out}: best val next-patch L1 {result['best_val_next']:.4f}")


@app.command()
def clock(sources: list[str] = typer.Argument(..., help="prepared sources with ages"),
          encoder: Path | None = typer.Option(None, help="a pretraining checkpoint; none trains from scratch"),
          freeze: bool = typer.Option(False, help="train the head only (a probe of the pretrained encoder)"),
          out: Path | None = typer.Option(None, help="run folder (default data/output/runs/kron-ppg/...)"),
          sample_rate: float = 25.0, seconds: float = 30.0, patch: int = 5, d_model: int = 256, n_heads: int = 4,
          n_layers: int = 6, ff_units: int = 768, dropout: float = 0.1,
          batch_size: int = 64, steps: int = 3_000, lr: float = 3e-4, head_lr: float = 1e-3,
          eval_every: int = 250, seed: int = 0, device: str = "cuda") -> None:
    """Fit the age clock: per-person scores, bias-corrected gap, and the gap's outcome associations."""
    from kron_ppg.train import ClockArgs, fit_clock

    folders = _variants(sources, sample_rate, seconds)
    tag = "probe" if freeze else ("finetune" if encoder else "scratch")
    out = out or runs_dir(MODEL) / run_name("clock", tag, *sources)
    config = _config(sample_rate, patch, d_model, n_heads, n_layers, ff_units, dropout, 5)
    result = fit_clock(ClockArgs(folders, out, encoder, config, freeze, batch_size, steps, lr, head_lr,
                                 eval_every=eval_every, seed=seed, device=device))
    from kron_ppg.clock import json_safe

    typer.echo(json.dumps(json_safe({k: result[k] for k in ("best_step", "val", "test", "outcomes")}), indent=2))
    typer.echo(f"written to {out}")


@app.command()
def evaluate(run: Path = typer.Argument(..., help="a clock run folder (or its checkpoint.pt)"),
             sources: list[str] = typer.Argument(..., help="prepared sources to score every subject of"),
             seconds: float = 30.0, batch_size: int = 64, device: str = "cuda") -> None:
    """Score a trained clock on every subject with an age in other sources, with the run's bias correction."""
    from kron_ppg.clock import gap_associations, json_safe, scores
    from kron_ppg.data import Pool
    from kron_ppg.train import load_clock, predict

    ckpt = run / "checkpoint.pt" if run.is_dir() else run
    dev = torch.device(device)
    model = load_clock(ckpt, dev)
    bias = torch.load(ckpt, map_location="cpu", weights_only=True)["bias"]
    pool = Pool.load(_variants(sources, model.encoder.config.sample_rate, seconds))
    idx = np.flatnonzero(np.isfinite(pool.age))
    subj = predict(model, pool, idx, batch_size, dev)
    pred = np.array([p for p, _, _ in subj.values()])
    age = np.array([a for _, a, _ in subj.values()])
    gaps = {s: p - (bias["a"] + bias["b"] * a) for s, (p, a, _) in subj.items()}
    result = {"run": str(ckpt), "sources": sources, **scores(pred, age, (bias["a"], bias["b"])),
              "outcomes": gap_associations(gaps, pool.table, ages={s: a for s, (_, a, _) in subj.items()})}
    typer.echo(json.dumps(json_safe(result), indent=2))


@app.command()
def export(run: Path = typer.Argument(..., help="a clock run folder (or its checkpoint.pt)"),
           out: Path | None = typer.Option(None, help="folder for stream.onnx (default: <run>/onnx)"),
           chunk: int = typer.Option(25, help="patches per graph call (25 x 0.2 s = 5 s)"),
           opset: int = 17) -> None:
    """Export the streaming age graph to ONNX and check it against PyTorch with ONNX Runtime."""
    from kron_ppg.onnx_export import export_stream
    from kron_ppg.train import load_clock

    ckpt = run / "checkpoint.pt" if run.is_dir() else run
    model = load_clock(ckpt, "cpu")
    out = out or ckpt.parent / "onnx"
    record = export_stream(model, out, chunk=chunk, opset=opset)
    typer.echo(json.dumps(record, indent=2))


if __name__ == "__main__":
    app()


@app.command("hub-export")
def hub_export(sources: list[str] = typer.Argument(..., help="prepared sources"),
               repo: str = typer.Option("GlucoseDao/ppg-windows", help="the Hub dataset id the card names"),
               name: str = typer.Option("PPG windows", help="the card's title"),
               out: Path | None = typer.Option(None, help="output folder (default data/output/hub/<repo name>)"),
               rates: list[float] = typer.Option([25.0], help="rates to export (each prepared; a source "
                                                 "gets only rates at or below its own)"),
               seconds: float = typer.Option(30.0)) -> None:
    """Prepared windows as a Hugging Face dataset folder: parquet shards per source, rate and split, and a card."""
    from kron_core.paths import output_dir
    from kron_ppg.hub import export_hub

    out = out or output_dir() / "hub" / repo.split("/")[-1]
    summary = export_hub(sources, out, repo, name, tuple(rates), seconds)
    typer.echo(json.dumps(summary, indent=2))
    typer.echo(f"-> {out}")
