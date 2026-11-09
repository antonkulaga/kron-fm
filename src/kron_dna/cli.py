"""``kron-dna``: prepare coding sequences, pretrain a kron-dna model and export it to ONNX.

    uv run kron-data download ecoli-k12                  # data/input/cds/ecoli-k12
    uv run kron-dna prepare ecoli-k12                    # -> data/interim/cds/ecoli-k12/packed_1024
    uv run kron-dna train ecoli-k12 --steps 2000         # -> data/output/runs/kron-dna/ecoli-k12-mlm-<UTC time>
    uv run kron-dna export-onnx data/output/runs/kron-dna/<run>   # -> <run>/onnx/
    uv run kron-dna info --d-model 512 --n-layers 12

A dataset is named by its source (``kron-data list``) or given as paths; the folders come from
:mod:`kron_core.paths`.

The validation split here is a random split of packed rows, for watching the loss only. It shares
genes with training, so it says nothing about generalisation: the report's evaluation section explains
why results must be gene- or species-held-out.
"""
from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import typer

from kron_core.paths import input_dir, interim_dir, run_name, runs_dir
from kron_dna.data import PackedRows, doc_ids_from_tokens, mask_codons, sequence_files
from kron_dna.data import prepare as prepare_rows
from kron_dna.sources import DOMAIN, SOURCES
from kron_dna.losses import OBJECTIVES, mlm_cross_entropy, objective_loss
from kron_dna.model import KronDNA, KronDNAConfig

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _config(d_model: int, n_heads: int, n_layers: int, ff_units: int, dropout: float, attention_every: int,
            tie_directions: bool) -> KronDNAConfig:
    return KronDNAConfig(d_model=d_model, n_heads=n_heads, n_layers=n_layers, ff_units=ff_units, dropout=dropout,
                         attention_every=attention_every, tie_directions=tie_directions)


def packed_dir(source: str, seq_len: int = 1024, subset: str = "") -> Path:
    """Where ``prepare`` writes a source's packed rows: ``data/interim/cds/<source>/packed_<seq_len>[_<subset>]``."""
    return interim_dir(DOMAIN, source, f"packed_{seq_len}" + (f"_{subset}" if subset else ""))


def resolve_packed(data: str, seq_len: int = 1024) -> Path:
    """A packed-rows directory from a path or a source name."""
    path = Path(data)
    if (path / "tokens.npy").is_file():
        return path
    if data in SOURCES and (packed_dir(data, seq_len) / "tokens.npy").is_file():
        return packed_dir(data, seq_len)
    raise typer.BadParameter(f"{data!r} is neither a packed directory nor a prepared source; "
                             f"run `kron-dna prepare {data}` first")


@app.command()
def prepare(inputs: list[str] = typer.Argument(..., help="a source name (`kron-data list`) or FASTA/CSV paths"),
            out: Path | None = typer.Option(None, help="output directory (default: data/interim/cds/<source>/packed_<seq_len>)"),
            seq_len: int = typer.Option(1024, help="tokens per packed row"),
            alt_starts: bool | None = typer.Option(None, help="also accept GTG and TTG starts (default: the source's setting)"),
            keep_set: str | None = typer.Option(None, help="CSV sources: keep rows whose Set column has this value (train, test)"),
            max_sequences: int | None = typer.Option(None, help="stop after this many accepted sequences")) -> None:
    """Filter, tokenize and pack coding sequences."""
    source = SOURCES.get(inputs[0]) if len(inputs) == 1 else None
    if source is not None:
        folder = source.folder(input_dir())
        paths = sequence_files(folder)
        if not paths:
            raise typer.BadParameter(f"nothing downloaded in {folder}; run `kron-data download {source.name}`")
        out = out or packed_dir(source.name, seq_len, keep_set or "")
        alt = source.notes.get("alt_starts", False) if alt_starts is None else alt_starts
        keep = {source.notes.get("set_column", "Set"): keep_set} if keep_set else None
    else:
        paths = [Path(p) for p in inputs]
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise typer.BadParameter(f"not a source name and not files: {missing}")
        if out is None:
            raise typer.BadParameter("--out is needed when the inputs are paths")
        alt, keep = bool(alt_starts), ({"Set": keep_set} if keep_set else None)
    starts = ("ATG", "GTG", "TTG") if alt else ("ATG",)
    meta = prepare_rows(paths, out, seq_len=seq_len, starts=starts, max_sequences=max_sequences, keep=keep)
    typer.echo(f"wrote {out}")
    typer.echo(json.dumps(meta, indent=2))


@app.command()
def info(d_model: int = 512, n_heads: int = 8, n_layers: int = 12, ff_units: int = 1536, attention_every: int = 6,
         tie_directions: bool = False) -> None:
    """Parameter count of a configuration (built on the meta device, so no GPU or memory is needed)."""
    config = _config(d_model, n_heads, n_layers, ff_units, 0.0, attention_every, tie_directions)
    with torch.device("meta"):
        n = KronDNA(config).n_parameters()
    typer.echo(f"{n / 1e6:.1f}M parameters: {json.dumps(config.to_dict())}")


def _param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        exempt = getattr(p, "_no_weight_decay", False) or p.dim() < 2 or "embed" in name or "norm" in name
        (no_decay if exempt else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def _lr_at(step: int, steps: int, warmup: int, lr: float) -> float:
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, steps - warmup)
    return lr * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


@app.command()
def train(data: str = typer.Argument(..., help="a prepared source name or a directory written by `kron-dna prepare`"),
          out: Path | None = typer.Option(None, help="run directory (default: data/output/runs/kron-dna/<data>-<objective>-<UTC time>)"),
          objective: str = typer.Option("mlm", help=f"one of {', '.join(OBJECTIVES)}"),
          alpha: float = typer.Option(0.2, help="the hierarchical loss's depth discount"),
          d_model: int = 512, n_heads: int = 8, n_layers: int = 12, ff_units: int = 1536, dropout: float = 0.1,
          attention_every: int = 6, tie_directions: bool = False,
          batch_size: int = 32, steps: int = 10_000, lr: float = 2e-4, warmup_fraction: float = 0.1,
          weight_decay: float = 0.01, mask_prob: float = 0.15, grad_clip: float = 1.0,
          val_fraction: float = 0.01, eval_every: int = 500, eval_batches: int = 20,
          seed: int = 0, device: str = "cuda", bf16: bool = True) -> None:
    """Pretrain with AdamW, linear warmup and cosine decay (SynCodonLM's schedule)."""
    if objective not in OBJECTIVES:
        raise typer.BadParameter(f"objective must be one of {OBJECTIVES}")
    torch.manual_seed(seed)
    dev = torch.device(device)
    name = data if data in SOURCES else Path(data).name
    data = resolve_packed(data)
    out = out or runs_dir("kron-dna") / run_name(name, objective)
    out.mkdir(parents=True, exist_ok=True)
    typer.echo(f"run directory: {out}")
    n_rows = len(np.load(data / "tokens.npy", mmap_mode="r"))
    order = np.random.default_rng(seed).permutation(n_rows)
    n_val = max(1, int(round(val_fraction * n_rows)))
    train_set, val_set = PackedRows(data, order[n_val:]), PackedRows(data, order[:n_val])
    loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True,
                                         num_workers=2, persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size)

    config = _config(d_model, n_heads, n_layers, ff_units, dropout, attention_every, tie_directions)
    model = KronDNA(config).to(dev)
    (out / "config.json").write_text(json.dumps({"model": config.to_dict(), "objective": objective, "alpha": alpha,
                                                 "data": str(data), "lr": lr, "batch_size": batch_size,
                                                 "steps": steps, "seed": seed}, indent=2) + "\n")
    typer.echo(f"{model.n_parameters() / 1e6:.1f}M parameters, {len(train_set)} train rows, {n_val} val rows")
    opt = torch.optim.AdamW(_param_groups(model, weight_decay), lr=lr, betas=(0.9, 0.98), eps=1e-8)
    warmup = max(1, int(warmup_fraction * steps))
    autocast = torch.autocast(dev.type, dtype=torch.bfloat16, enabled=bf16 and dev.type == "cuda")
    gen = torch.Generator(device=dev).manual_seed(seed)

    def evaluate() -> dict[str, float]:
        model.eval()
        val_gen = torch.Generator(device=dev).manual_seed(seed + 1)
        totals = {"loss": 0.0, "ce": 0.0}
        n = 0
        with torch.no_grad(), autocast:
            for i, rows in enumerate(val_loader):
                if i >= eval_batches:
                    break
                rows = rows.to(dev)
                inputs, labels = mask_codons(rows, mask_prob, generator=val_gen)
                logits, _ = model(inputs, doc_ids=doc_ids_from_tokens(rows))
                totals["loss"] += float(objective_loss(objective, logits, labels, alpha))
                totals["ce"] += float(mlm_cross_entropy(logits, labels))
                n += 1
        model.train()
        return {"val_loss": totals["loss"] / n, "val_bits_per_codon": totals["ce"] / n / math.log(2)}

    metrics_path = out / "metrics.csv"
    with metrics_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "lr", "train_loss", "val_loss", "val_bits_per_codon", "tokens_per_s"])
    step, t0, seen = 0, time.time(), 0
    model.train()
    while step < steps:
        for rows in loader:
            if step >= steps:
                break
            rows = rows.to(dev, non_blocking=True)
            inputs, labels = mask_codons(rows, mask_prob, generator=gen)
            for group in opt.param_groups:
                group["lr"] = _lr_at(step, steps, warmup, lr)
            with autocast:
                logits, _ = model(inputs, doc_ids=doc_ids_from_tokens(rows))
                loss = objective_loss(objective, logits, labels, alpha)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            seen += rows.numel()
            step += 1
            if step % eval_every == 0 or step == steps:
                m = evaluate()
                rate = seen / (time.time() - t0)
                with metrics_path.open("a", newline="") as fh:
                    csv.writer(fh).writerow([step, opt.param_groups[0]["lr"], loss.item(), m["val_loss"],
                                             m["val_bits_per_codon"], round(rate)])
                torch.save({"model": model.state_dict(), "config": config.to_dict(), "step": step,
                            "objective": objective}, out / "checkpoint.pt")
                typer.echo(f"step {step}: train {loss.item():.4f}, val {m['val_loss']:.4f}, "
                           f"{m['val_bits_per_codon']:.3f} bits/codon, {rate:,.0f} tokens/s")


@app.command("export-onnx")
def export_onnx_cmd(run: Path = typer.Argument(..., help="a `kron-dna train` run directory or checkpoint"),
                    out: Path | None = typer.Option(None, help="output directory (default: <run>/onnx)"),
                    seq_len: int = typer.Option(256, help="codon tokens the graph reads; the recurrence is unrolled, "
                                                          "so the graph and the export time grow with it"),
                    opset: int = typer.Option(17, help="ONNX opset"),
                    check_kernels: bool = typer.Option(True, help="compare with the Triton kernels when CUDA is present")
                    ) -> None:
    """Export a trained model to ONNX and check ONNX Runtime against PyTorch."""
    from kron_dna.onnx_export import export_onnx, load_run

    model, record = load_run(run)
    out = out or (run if run.is_dir() else run.parent) / "onnx"
    meta = export_onnx(model, out, seq_len=seq_len, opset=opset, record=record, check_kernels=check_kernels)
    typer.echo(f"wrote {out / meta['graph']} ({meta['bytes'] / 1e6:,.1f} MB, {meta['export_seconds']} s)")
    typer.echo(json.dumps(meta["verification"], indent=2))


if __name__ == "__main__":
    app()
