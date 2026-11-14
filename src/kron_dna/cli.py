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
from kron_dna.data import PackedRows, doc_ids_from_tokens, is_packed, mask_codons, sequence_files
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
    if is_packed(path):
        return path
    if data in SOURCES and is_packed(packed_dir(data, seq_len)):
        return packed_dir(data, seq_len)
    raise typer.BadParameter(f"{data!r} is neither a packed directory nor a prepared source; "
                             f"run `kron-dna prepare {data}` first")


@app.command()
def prepare(inputs: list[str] = typer.Argument(..., help="a source name (`kron-data list`) or FASTA/CSV paths"),
            out: Path | None = typer.Option(None, help="output directory (default: data/interim/cds/<source>/packed_<seq_len>)"),
            seq_len: int = typer.Option(1024, help="tokens per packed row"),
            alt_starts: bool | None = typer.Option(None, help="also accept GTG and TTG starts (default: the source's setting)"),
            keep_set: str | None = typer.Option(None, help="CSV sources: keep rows whose Set column has this value (train, test)"),
            max_sequences: int | None = typer.Option(None, help="stop after this many accepted sequences"),
            holdout_genes: float = typer.Option(0.0, help="this fraction of genes (all isoforms) to <out>_holdout, "
                                                          "for gene-held-out validation"),
            seed: int = typer.Option(0, help="seed of the gene hash")) -> None:
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
    meta = prepare_rows(paths, out, seq_len=seq_len, starts=starts, max_sequences=max_sequences, keep=keep,
                        holdout_genes=holdout_genes, seed=seed)
    typer.echo(f"wrote {out}")
    typer.echo(json.dumps(meta, indent=2))


@app.command("prepare-stream")
def prepare_stream_cmd(source: str = typer.Argument(..., help="a CSV source name (syncodonlm) or a CSV path"),
                       out: Path | None = typer.Option(None, help="output root (default: data/interim/cds/<source>/packed_<seq_len>); "
                                                                  "one prepared directory per split below it"),
                       seq_len: int = typer.Option(1024, help="tokens per packed row"),
                       alt_starts: bool | None = typer.Option(None, help="also accept GTG and TTG starts (default: the source's setting)"),
                       shard_rows: int = typer.Option(262_144, help="rows per shard (256 MB at 1,024 tokens)"),
                       workers: int = typer.Option(1, help="processes that check and encode sequences"),
                       max_rows: int | None = typer.Option(None, help="stop after this many CSV rows (for trials)"),
                       holdout_species: float = typer.Option(0.0, help="this fraction of species (hashed names) to a "
                                                                       "species_holdout split"),
                       seed: int = typer.Option(0, help="seed of the species hash")) -> None:
    """Stream a large CSV corpus into sharded packed rows, keeping its own train/test split."""
    from kron_dna.stream import prepare_stream

    src = SOURCES.get(source)
    if src is not None:
        csvs = [p for p in sequence_files(src.folder(input_dir())) if p.name.endswith((".csv", ".csv.gz"))]
        if len(csvs) != 1:
            raise typer.BadParameter(f"expected one CSV in {src.folder(input_dir())}, found {csvs}")
        path, notes = csvs[0], src.notes
        out = out or packed_dir(src.name, seq_len)
    else:
        path, notes = Path(source), {}
        if out is None:
            raise typer.BadParameter("--out is needed when the input is a path")
    alt = notes.get("alt_starts", False) if alt_starts is None else alt_starts
    summary = prepare_stream(path, out, seq_len=seq_len, starts=("ATG", "GTG", "TTG") if alt else ("ATG",),
                             split_column=notes.get("set_column", "Set"),
                             species_column=notes.get("species_column", "Species"),
                             group_column=notes.get("group_column", "Group"),
                             sequence_column=notes.get("sequence_column", "CDS"),
                             shard_rows=shard_rows, workers=workers, max_rows=max_rows,
                             holdout_species=holdout_species, seed=seed)
    typer.echo(f"wrote {out}")
    typer.echo(json.dumps(summary, indent=2))


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


def _save_atomic(obj: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _fixed_rows(n_rows: int, n: int, seed: int) -> np.ndarray:
    """``n`` row indices drawn once without replacement, sorted, so evaluation always reads the same rows."""
    return np.sort(np.random.default_rng(seed).choice(n_rows, size=min(n, n_rows), replace=False))


@app.command()
def train(data: str = typer.Argument(..., help="a prepared source name or a directory written by `kron-dna prepare`"),
          out: Path | None = typer.Option(None, help="run directory (default: data/output/runs/kron-dna/<data>-<objective>-<UTC time>); "
                                                     "a directory holding a checkpoint is resumed"),
          val_data: str | None = typer.Option(None, help="a prepared directory to validate on (e.g. a corpus's test set) "
                                                         "instead of a random split of DATA's rows"),
          objective: str = typer.Option("mlm", help=f"one of {', '.join(OBJECTIVES)}"),
          alpha: float = typer.Option(0.2, help="the hierarchical loss's depth discount"),
          d_model: int = 512, n_heads: int = 8, n_layers: int = 12, ff_units: int = 1536, dropout: float = 0.1,
          attention_every: int = 6, tie_directions: bool = False,
          batch_size: int = 32, steps: int = 10_000, lr: float = 2e-4, warmup_fraction: float = 0.1,
          weight_decay: float = 0.01, mask_prob: float = 0.15, grad_clip: float = 1.0,
          val_fraction: float = 0.01, eval_every: int = 500, eval_batches: int = 20,
          checkpoint_every: int = typer.Option(0, help="steps between checkpoints (default: every evaluation)"),
          max_hours: float = typer.Option(0.0, help="stop cleanly, with a checkpoint, after this many hours (0: no limit)"),
          snapshot_every: int = typer.Option(0, help="also keep the weights (no optimizer) as snapshots/step_N.pt "
                                                     "every this many steps (0: never)"),
          num_workers: int = 2, seed: int = 0, device: str = "cuda", bf16: bool = True) -> None:
    """Pretrain with AdamW, linear warmup and cosine decay (SynCodonLM's schedule); resumable."""
    if objective not in OBJECTIVES:
        raise typer.BadParameter(f"objective must be one of {OBJECTIVES}")
    started = time.time()
    torch.manual_seed(seed)
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.zeros(1, device=dev)   # hold the card from the start: a shared node's launchers read a card with no process as free
    name = data if data in SOURCES else Path(data).name
    data_dir = resolve_packed(data)
    out = out or runs_dir("kron-dna") / run_name(name, objective)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "checkpoint.pt"
    resume = torch.load(ckpt_path, map_location="cpu", weights_only=False) if ckpt_path.is_file() else None
    if resume is not None and "optimizer" not in resume:
        raise typer.BadParameter(f"{ckpt_path} has no optimizer state, so it cannot be resumed")
    typer.echo(f"run directory: {out}" + (f" (resuming at step {resume['step']})" if resume else ""))

    all_rows = PackedRows(data_dir)
    if val_data is None:
        order = np.random.default_rng(seed).permutation(all_rows.n_total)
        n_val = max(1, int(round(val_fraction * all_rows.n_total)))
        train_set = PackedRows(data_dir, order[n_val:])
        val_rows = PackedRows(data_dir, np.sort(order[:n_val])[:eval_batches * batch_size])
    else:
        val_dir = resolve_packed(val_data)
        train_set = all_rows
        n_val_total = PackedRows(val_dir).n_total
        val_rows = PackedRows(val_dir, _fixed_rows(n_val_total, eval_batches * batch_size, seed + 1))
    loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True,
                                         num_workers=num_workers, persistent_workers=num_workers > 0,
                                         generator=torch.Generator().manual_seed(seed + (resume["step"] if resume else 0)))
    val_loader = torch.utils.data.DataLoader(val_rows, batch_size=batch_size)

    config = KronDNAConfig(**resume["config"]) if resume else \
        _config(d_model, n_heads, n_layers, ff_units, dropout, attention_every, tie_directions)
    model = KronDNA(config).to(dev)
    record = {"model": config.to_dict(), "objective": objective, "alpha": alpha, "data": str(data_dir),
              "val_data": str(resolve_packed(val_data)) if val_data else None, "lr": lr, "batch_size": batch_size,
              "steps": steps, "mask_prob": mask_prob, "seed": seed}
    (out / "config.json").write_text(json.dumps(record, indent=2) + "\n")
    typer.echo(f"{model.n_parameters() / 1e6:.1f}M parameters, {len(train_set)} train rows, {len(val_rows)} val rows")
    opt = torch.optim.AdamW(_param_groups(model, weight_decay), lr=lr, betas=(0.9, 0.98), eps=1e-8)
    step = 0
    if resume:
        model.load_state_dict(resume["model"])
        opt.load_state_dict(resume["optimizer"])
        step = int(resume["step"])
    warmup = max(1, int(warmup_fraction * steps))
    autocast = torch.autocast(dev.type, dtype=torch.bfloat16, enabled=bf16 and dev.type == "cuda")
    gen = torch.Generator(device=dev).manual_seed(seed + step)
    checkpoint_every = checkpoint_every or eval_every

    def evaluate() -> dict[str, float]:
        model.eval()
        val_gen = torch.Generator(device=dev).manual_seed(seed + 1)
        totals = {"loss": 0.0, "ce": 0.0}
        n = 0
        with torch.no_grad(), autocast:
            for rows in val_loader:
                rows = rows.to(dev)
                inputs, labels = mask_codons(rows, mask_prob, generator=val_gen)
                logits, _ = model(inputs, doc_ids=doc_ids_from_tokens(rows))
                totals["loss"] += float(objective_loss(objective, logits, labels, alpha))
                totals["ce"] += float(mlm_cross_entropy(logits, labels))
                n += 1
        model.train()
        return {"val_loss": totals["loss"] / n, "val_bits_per_codon": totals["ce"] / n / math.log(2)}

    metrics_path = out / "metrics.csv"
    if not (resume and metrics_path.is_file()):
        with metrics_path.open("w", newline="") as fh:
            csv.writer(fh).writerow(["step", "lr", "train_loss", "val_loss", "val_bits_per_codon", "tokens_per_s",
                                     "hours"])
    t0, seen, last_loss = time.time(), 0, float("nan")
    hours_before = float(resume.get("hours", 0.0)) if resume else 0.0
    model.train()
    stop = step >= steps
    while not stop:
        for rows in loader:
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
            last_loss = loss.item() if step % 50 == 0 or step % eval_every == 0 or step == steps else last_loss
            out_of_time = max_hours > 0 and time.time() - started > max_hours * 3600
            stop = step >= steps or out_of_time
            if step % eval_every == 0 or stop:
                m = evaluate()
                rate = seen / (time.time() - t0)
                hours = hours_before + (time.time() - t0) / 3600
                with metrics_path.open("a", newline="") as fh:
                    csv.writer(fh).writerow([step, opt.param_groups[0]["lr"], loss.item(), m["val_loss"],
                                             m["val_bits_per_codon"], round(rate), round(hours, 4)])
                peak = f", peak {torch.cuda.max_memory_allocated(dev) / 2**30:.1f} GiB" if dev.type == "cuda" else ""
                typer.echo(f"step {step}: train {loss.item():.4f}, val {m['val_loss']:.4f}, "
                           f"{m['val_bits_per_codon']:.3f} bits/codon, {rate:,.0f} tokens/s, {hours:.2f} h{peak}")
            if step % checkpoint_every == 0 or stop:
                hours = hours_before + (time.time() - t0) / 3600
                _save_atomic({"model": model.state_dict(), "optimizer": opt.state_dict(), "config": config.to_dict(),
                              "step": step, "objective": objective, "steps": steps, "hours": hours}, ckpt_path)
            if snapshot_every and step % snapshot_every == 0:
                (out / "snapshots").mkdir(exist_ok=True)
                _save_atomic({"model": model.state_dict(), "config": config.to_dict(), "step": step,
                              "objective": objective}, out / "snapshots" / f"step_{step}.pt")
            if stop:
                if out_of_time and step < steps:
                    typer.echo(f"stopped at step {step} of {steps} after {max_hours} h; rerun with --out {out} to resume")
                break


def masked_scores(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    """Sums over the masked codons of a batch: nats for the codon, for its amino acid and for the codon given
    its amino acid (the first is the sum of the other two), and correct codon and amino-acid guesses.

    A model trained with the synonymous loss never learns the amino acid, so only its synonymous part is
    comparable with the other objectives; that is why the parts are reported apart.
    """
    from kron_dna.codons import codon_amino_acid_index, synonym_matrix

    keep = labels != -100
    sel, lab = logits[keep].float(), labels[keep]
    log_p = torch.log_softmax(sel, dim=-1)
    allowed = synonym_matrix().to(sel.device)[lab]
    log_p_aa = torch.logsumexp(log_p.masked_fill(~allowed, float("-inf")), dim=-1)
    log_p_codon = log_p.gather(1, lab[:, None])[:, 0]
    aa = codon_amino_acid_index().to(sel.device)
    guess = sel.argmax(-1)
    return {"n": float(lab.numel()), "codon_nats": float(-log_p_codon.sum()), "aa_nats": float(-log_p_aa.sum()),
            "syn_nats": float(-(log_p_codon - log_p_aa).sum()), "codon_correct": float((guess == lab).sum()),
            "aa_correct": float((aa[guess] == aa[lab]).sum())}


@app.command()
def evaluate(run: Path = typer.Argument(..., help="a `kron-dna train` run directory or checkpoint"),
             data: list[str] = typer.Argument(..., help="prepared directories or source names to score"),
             rows: int = typer.Option(4096, help="rows drawn (once, by seed) from each set"),
             batch_size: int = 32, mask_prob: float = 0.15, seed: int = 0, device: str = "cuda",
             bf16: bool = True) -> None:
    """Masked-codon scores of a trained run on prepared sets, written to ``<run>/eval/<set>.json``
    (``<dir>/eval/<stem>/<set>.json`` for a checkpoint file, so snapshots keep their own scores).

    Bits per masked codon = amino-acid bits + synonymous bits (the codon given its amino acid), and the
    codon and amino-acid accuracies, with the same masking for every run (fixed seed).
    """
    from kron_dna.onnx_export import load_run

    dev = torch.device(device)
    model, record = load_run(run, map_location="cpu")
    model = model.to(dev).eval()
    run_dir = run if run.is_dir() else run.parent
    autocast = torch.autocast(dev.type, dtype=torch.bfloat16, enabled=bf16 and dev.type == "cuda")
    for item in data:
        data_dir = resolve_packed(item)
        n_total = PackedRows(data_dir).n_total
        subset = PackedRows(data_dir, _fixed_rows(n_total, rows, seed))
        loader = torch.utils.data.DataLoader(subset, batch_size=batch_size)
        gen = torch.Generator(device=dev).manual_seed(seed)
        totals: dict[str, float] = {}
        with torch.no_grad(), autocast:
            for batch in loader:
                batch = batch.to(dev)
                inputs, labels = mask_codons(batch, mask_prob, generator=gen)
                logits, _ = model(inputs, doc_ids=doc_ids_from_tokens(batch))
                for k, v in masked_scores(logits, labels).items():
                    totals[k] = totals.get(k, 0.0) + v
        n, ln2 = totals["n"], math.log(2)
        result = {"run": str(run_dir), "step": record.get("step"), "objective": record.get("objective"),
                  "data": str(data_dir), "rows": len(subset), "masked_codons": int(n), "mask_prob": mask_prob,
                  "seed": seed, "bits_per_codon": totals["codon_nats"] / n / ln2,
                  "amino_acid_bits": totals["aa_nats"] / n / ln2, "synonymous_bits": totals["syn_nats"] / n / ln2,
                  "codon_accuracy": totals["codon_correct"] / n, "amino_acid_accuracy": totals["aa_correct"] / n}
        name = item if item in SOURCES else "-".join(Path(data_dir).parts[-3:])
        eval_dir = run_dir / "eval" if run.is_dir() else run.parent / "eval" / run.stem   # one folder per snapshot
        eval_dir.mkdir(parents=True, exist_ok=True)
        (eval_dir / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
        typer.echo(json.dumps(result))


@app.command()
def probe(task: Path = typer.Argument(..., help="a CodonBERT-style CSV (Sequence, Value, Split = train/val/test)"),
          runs: list[Path] = typer.Option([], "--run", help="kron-dna run directories to probe (mean hidden state)"),
          random_init: bool = typer.Option(True, help="also probe an untrained twin of the first run's configuration"),
          baselines: bool = typer.Option(True, help="probe gc, codon_usage, cai, kmer and onehot_pos"),
          cai_reference: str = typer.Option("ecoli-k12", help="a CDS source name or FASTA path for CAI's codon weights"),
          out: Path | None = typer.Option(None, help="results JSON (default: data/output/eval/<task>/probe.json)"),
          device: str = "cuda", seed: int = 0) -> None:
    """Frozen features + ridge head (penalty by the val split), Spearman on the test split, for every feature set."""
    from kron_dna import probe as P
    from kron_dna.onnx_export import load_run
    from kron_core.paths import output_dir

    data = P.read_task(task)
    missing = {"train", "val", "test"} - set(data)
    if missing:
        raise typer.BadParameter(f"{task} lacks the split(s) {sorted(missing)}")
    seqs = {k: v[0] for k, v in data.items()}
    ys = {k: v[1] for k, v in data.items()}
    results: dict[str, dict] = {}

    def run_probe(name: str, featurize) -> None:
        x = {k: featurize(seqs[k]) for k in ("train", "val", "test")}
        results[name] = P.ridge_probe(x, ys)
        lo, hi = results[name]["test_spearman_ci"] or (float("nan"), float("nan"))
        typer.echo(f"{name:40s} test rho {results[name]['test_spearman']:+.3f} [{lo:+.3f}, {hi:+.3f}]  val rho "
                   f"{results[name]['val_spearman']:+.3f}  ({results[name]['dims']} dims, alpha {results[name]['alpha']:g})")

    if baselines:
        if cai_reference in SOURCES:
            ref_files = sequence_files(SOURCES[cai_reference].folder(input_dir()))
            if not ref_files:
                raise typer.BadParameter(f"run `kron-data download {cai_reference}` first")
        else:
            ref_files = [Path(cai_reference)]
        from kron_dna.data import read_sequences
        w = P.relative_adaptiveness(seq for f in ref_files for _, seq in read_sequences(f))
        max_codons = max(len(s) // 3 for split in seqs.values() for s in split)
        run_probe("gc", P.gc_features)
        run_probe("codon_usage", P.codon_usage)
        run_probe("cai", lambda s: P.cai(s, w))
        run_probe("gc+cai", lambda s: np.concatenate([P.gc_features(s), P.cai(s, w)], axis=1))
        run_probe("kmer", P.kmer_features)
        run_probe("onehot_pos", lambda s: P.onehot_pos(s, max_codons))
    dev = torch.device(device)
    for i, run in enumerate(runs):
        model, record = load_run(run)
        run_probe(f"model:{run.name}", lambda s, m=model: P.model_embeddings(m, s, dev))
        if i == 0 and random_init:
            torch.manual_seed(seed)
            twin = KronDNA(model.config if hasattr(model, "config") else KronDNAConfig(**record["config"]))
            run_probe("random_init_twin", lambda s, m=twin: P.model_embeddings(m, s, dev))
    out = out or output_dir("eval", task.stem) / "probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"task": str(task), "n": {k: len(v) for k, v in seqs.items()}, "head": "ridge",
                               "metric": "spearman", "results": results}, indent=2) + "\n")
    typer.echo(f"wrote {out}")


@app.command()
def clinvar(vcf: Path = typer.Option(None, help="ClinVar GRCh38 VCF (default: data/input/variants/clinvar/clinvar.vcf.gz)"),
            cds: str = typer.Option("human-grch38", help="a CDS source name or NCBI cds_from_genomic FASTA path"),
            runs: list[Path] = typer.Option([], "--run", help="kron-dna run directories (or checkpoints) to score"),
            window: int = typer.Option(256, help="tokens of context around the masked codon"),
            holdout_genes: float = typer.Option(0.05, help="the gene fraction `prepare --holdout-genes` set aside"),
            max_per_class: int = typer.Option(5000, help="variants kept per (consequence, label), by seed; 0 = all"),
            batch_size: int = 32, device: str = "cuda", seed: int = 0,
            alphamissense: Path | None = typer.Option(None, help="AlphaMissense_hg38.tsv.gz: score its pathogenicity too"),
            alphagenome_key: Path | None = typer.Option(None, help="a file holding an AlphaGenome API key: score "
                                                        "its splicing and RNA-seq effects too (`baselines` group)"),
            alphagenome_version: int = typer.Option(1, help="1, or 2 for AlphaGenome 2 (needs --ag-condition)"),
            ag_conditions: list[str] = typer.Option([], "--ag-condition", help="AlphaGenome 2 conditions (e.g. "
                                                    "UBERON:0001134); the largest effect over them is kept"),
            esm_models: list[str] = typer.Option([], "--esm", help="ESM-2 models to score missense with "
                                                 "(e.g. esm2_t33_650M_UR50D; needs the `baselines` group)"),
            out: Path | None = typer.Option(None, help="results JSON (default: data/output/eval/clinvar/zero_shot.json)"),
            ) -> None:
    """Zero-shot ClinVar: reviewed pathogenic vs benign coding SNVs, AUROC by consequence and gene split.

    Needs the ``genomics`` group (polars-bio): ``uv run --group genomics kron-dna clinvar --run RUN``. The
    mapped table is cached as ``data/interim/cds/clinvar/variants.parquet``.
    """
    import polars as pl

    from kron_core.paths import output_dir
    from kron_dna import probe as P
    from kron_dna import variants as V
    from kron_dna.onnx_export import load_run

    vcf = vcf or input_dir("variants", "clinvar") / "clinvar.vcf.gz"
    fasta = sequence_files(SOURCES[cds].folder(input_dir()))[0] if cds in SOURCES else Path(cds)
    t0 = time.time()
    transcripts = V.read_transcripts(fasta)
    typer.echo(f"{len(transcripts):,} genes with a usable CDS ({time.time() - t0:.0f} s)")
    cache = interim_dir("cds", "clinvar") / "variants.parquet"
    if cache.exists():
        table = pl.read_parquet(cache)
    else:
        variants = V.read_clinvar(vcf)
        typer.echo(f"{variants.height:,} reviewed P/LP or B/LB SNVs ({time.time() - t0:.0f} s)")
        table, dropped = V.map_variants(variants, transcripts, holdout_genes, seed)
        typer.echo(f"{table.height:,} placed on a codon; dropped {dict(dropped)}")
        cache.parent.mkdir(parents=True, exist_ok=True)
        table.write_parquet(cache)
    table = V.sample_per_class(table.filter(pl.col("consequence").is_in(["synonymous", "missense"])),
                               max_per_class, seed)
    typer.echo(f"{table.height:,} variants scored (at most {max_per_class or 'all'} per consequence and label)")
    seqs = [t["seq"] for t in transcripts.values()]
    scores = V.baseline_scores(table, V.codon_frequencies(seqs), P.relative_adaptiveness(seqs))
    positions = (V.read_clinvar(vcf).filter(pl.col("id").is_in(table["id"].implode()))
                 if alphamissense or alphagenome_key else None)
    if alphagenome_key:
        cache = "alphagenome.parquet" if alphagenome_version == 1 else "alphagenome2.parquet"
        scores |= V.alphagenome_scores(table, positions, alphagenome_key.read_text().strip(),
                                       interim_dir("cds", "clinvar") / cache, version=alphagenome_version,
                                       conditions=tuple(ag_conditions))
    if alphamissense:
        scores["alphamissense"] = V.alphamissense_scores(table, positions, alphamissense)
        typer.echo(f"AlphaMissense: {int(np.isfinite(scores['alphamissense']).sum()):,} of {table.height:,} variants scored")
    dev = torch.device(device)
    for name in esm_models:
        t1 = time.time()
        scores[f"esm:{name}"] = V.esm_marginals(table, transcripts, dev, name)
        typer.echo(f"scored {name} ({time.time() - t1:.0f} s)")
    for run in runs:
        model, record = load_run(run)
        if dev.type == "cpu":                       # the Triton kernels need CUDA; the export's plain graph does not
            from kron_dna.onnx_export import KronDNAGraph
            model = KronDNAGraph(model).eval()
        t1 = time.time()
        run_dir = run if run.is_dir() else run.parent.parent if run.parent.name == "snapshots" else run.parent
        scores[f"model:{run_dir.name}@{record.get('step')}"] = V.masked_marginals(
            model, table, transcripts, dev, window=window, batch_size=batch_size)
        typer.echo(f"scored {run} ({time.time() - t1:.0f} s)")
    summary = V.summarize(table, scores)
    for row in summary:
        typer.echo(json.dumps(row))
    out = out or output_dir("eval", "clinvar") / "zero_shot.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"vcf": str(vcf), "cds": str(fasta), "window": window, "max_per_class": max_per_class, "seed": seed, "score": "log p(alt) - log p(ref)",
                               "metric": "AUROC of -score, pathogenic positive", "ci": "95% bootstrap over variants, 1000 resamples",
                               "results": summary}, indent=2) + "\n")
    per_variant = out.with_suffix(".parquet")         # every variant's scores, for later paired comparisons
    table.select("id", "gene", "consequence", "label", "in_training").with_columns(
        [pl.Series(name, s) for name, s in scores.items()]).write_parquet(per_variant)
    typer.echo(f"wrote {out} and {per_variant}")


@app.command()
def ensemble(scores: list[Path] = typer.Argument(..., help="per-variant parquet files from `kron-dna clinvar`"),
             columns: list[str] = typer.Option(..., "--column", help="LONG=SHORT: a score column and its short name"),
             leave_out: list[str] = typer.Option([], "--leave-out", help="a short name: also fit without it, "
                                                 "and report its paired gain"),
             out: Path | None = typer.Option(None, help="results JSON (default: data/output/eval/clinvar/ensemble.json)"),
             ) -> None:
    """Gradient-boosted trees over per-variant scores, 5 folds split by gene; AUROC by consequence with intervals."""
    from kron_core.paths import output_dir
    from kron_dna import ensemble as E

    pairs = dict(c.split("=", 1) for c in columns)
    table = E.join_scores(scores, pairs)
    every = [*pairs.values(), "is_missense"]
    sets = {"all": every} | {f"all but {x}": [f for f in every if f != x] for x in leave_out}
    rows, preds = E.evaluate(table, sets)
    y, cons = table["label"].to_numpy(), table["consequence"].to_numpy()
    gains = {}
    for x in leave_out:
        for consequence in ("missense", "synonymous"):
            m = cons == consequence
            gains[f"{x}:{consequence}"] = E.paired_gain(preds[f"all but {x}"][m], preds["all"][m], y[m])
    for r in rows:
        typer.echo(f"{r['consequence']:10s} {r['inputs']:30s} {r['auroc']:.3f} ({r['ci'][0]:.3f}-{r['ci'][1]:.3f})")
    for k, g in gains.items():
        typer.echo(f"gain of {k}: {g['gain']:+.4f} ({g['ci'][0]:+.4f} to {g['ci'][1]:+.4f})")
    out = out or output_dir("eval", "clinvar") / "ensemble.json"
    out.write_text(json.dumps({"inputs": [str(p) for p in scores], "columns": pairs, "folds": "5, by gene",
                               "model": "HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05)",
                               "results": rows, "gains": gains}, indent=2) + "\n")
    typer.echo(f"wrote {out}")


@app.command("genome-prepare")
def genome_prepare(fasta: Path = typer.Argument(..., help="a genome FASTA (e.g. UCSC hg38.fa.gz)"),
                   min_gap: int = typer.Option(50, help="a run of N at least this long splits a contig"),
                   min_len: int = typer.Option(10_000, help="contigs shorter than this are dropped"),
                   name: str = typer.Option("hg38", help="folder under data/interim/genome/")) -> None:
    """Whole gap-free contigs of a genome, as one uint8 array (A, C, G, T, N = 0..4) and a contig index."""
    import numpy as np
    import polars as pl
    import polars_bio as pb

    from kron_dna import genome as G

    out = interim_dir("genome", name)
    out.mkdir(parents=True, exist_ok=True)
    keep = {f"chr{c}" for c in [*map(str, range(1, 23)), "X", "Y"]}
    rows, offset = [], 0
    with open(out / "bases.u8", "wb") as fh:
        for chrom, seq in pb.scan_fasta(str(fasta)).select("name", "sequence").collect().iter_rows():
            if chrom not in keep:
                continue
            codes = G.encode_bases(seq)
            for start, end in G.contigs_of(codes, min_gap, min_len):
                fh.write(codes[start:end].tobytes())
                rows.append((chrom, start, end, offset))
                offset += end - start
            typer.echo(f"{chrom}: {sum(1 for r in rows if r[0] == chrom)} contigs")
    index = pl.DataFrame(rows, schema=["chrom", "start", "end", "offset"], orient="row")
    index.write_parquet(out / "contigs.parquet")
    typer.echo(f"{index.height} contigs, {offset / 1e9:.2f} Gb, longest {(index['end'] - index['start']).max() / 1e6:.1f} Mb"
               f" -> {out}")


@app.command("genome-labels")
def genome_labels(regulatory: Path = typer.Option(None, help="Ensembl Regulatory Build GFF3 (default: data/input/regulation/"
                                                  "ensembl116/Homo_sapiens.GRCh38.regulatory_features.v116.gff3.gz)"),
                  activity: Path = typer.Option(None, help="its activity matrix TSV (feature_id, feature_type, then one "
                                                "ACTIVE/INACTIVE column per epigenome)"),
                  gtf: Path = typer.Option(None, help="GENCODE basic GTF (default: data/input/regulation/gencode/"
                                           "gencode.basic.gtf.gz)"),
                  name: str = typer.Option("hg38", help="prepared genome under data/interim/genome/")) -> None:
    """Per-base label tracks for stage 2, laid out like ``bases.u8``: ``labels.u8`` (bits of ``LABEL_BITS``: Ensembl
    promoter, enhancer, CTCF, open chromatin; GENCODE exon, splice donor, acceptor, CDS) and ``activity.u8`` (share of
    the Regulatory Build's epigenomes where the covering feature is ACTIVE, 0..255; the largest where features overlap)."""
    import numpy as np
    import polars as pl
    import polars_bio as pb

    from kron_dna import genome as G

    reg_dir = input_dir("regulation")
    regulatory = regulatory or reg_dir / "ensembl116" / "Homo_sapiens.GRCh38.regulatory_features.v116.gff3.gz"
    activity = activity or reg_dir / "ensembl116" / "Homo_sapiens.GRCh38.regulatory_activity.v116.tsv.gz"
    gtf = gtf or reg_dir / "gencode" / "gencode.basic.gtf.gz"
    out = interim_dir("genome", name)
    index = pl.read_parquet(out / "contigs.parquet")
    total = int((index["end"] - index["start"]).sum())
    labels = np.zeros(total, dtype=np.uint8)
    act = np.zeros(total, dtype=np.uint8)

    acts = pl.read_csv(activity, separator="\t")
    epis = acts.columns[2:]
    acts = acts.select(id="feature_id", frac=pl.sum_horizontal(pl.col(c) == "ACTIVE" for c in epis) / len(epis))
    reg = (pb.scan_gff(str(regulatory)).collect()
           .with_columns(id=pl.col("attributes").list.eval(pl.element().struct.field("value")).list.first(),
                         chrom="chr" + pl.col("chrom"), start=pl.col("start").cast(pl.Int64) - 1,
                         end=pl.col("end").cast(pl.Int64))
           .join(acts, on="id", how="left"))
    typer.echo(f"regulatory: {reg.height} features, {len(epis)} epigenomes, "
               f"{reg['frac'].is_null().sum()} without activity")
    gt = (pb.scan_gtf(str(gtf), attr_fields=["transcript_id"]).filter(pl.col("type").is_in(["exon", "CDS"]))
          .select(["chrom", "start", "end", "type", "strand", "transcript_id"]).collect())
    exons = gt.filter(pl.col("type") == "exon")
    donors, acceptors = G.splice_sites(exons)
    tracks = [(reg.filter(pl.col("type") == t), G.LABEL_BITS[t]) for t in
              ("promoter", "enhancer", "CTCF_binding_site", "open_chromatin_region")]
    zero = lambda f: f.with_columns(start=pl.col("start").cast(pl.Int64) - 1, end=pl.col("end").cast(pl.Int64))
    tracks += [(zero(exons), G.LABEL_BITS["exon"]), (zero(gt.filter(pl.col("type") == "CDS")), G.LABEL_BITS["CDS"]),
               (donors, G.LABEL_BITS["donor"]), (acceptors, G.LABEL_BITS["acceptor"])]
    counts = {}
    for frame, bit in tracks:
        n = 0
        for (chrom,), part in frame.group_by("chrom"):
            n += G.paint(index, chrom, part["start"].to_numpy(), part["end"].to_numpy(), labels, 1 << bit)
        counts[bit] = n
    for (chrom,), part in reg.filter(pl.col("frac").is_not_null()).group_by("chrom"):
        G.paint(index, chrom, part["start"].to_numpy(), part["end"].to_numpy(), act,
                np.round(part["frac"].to_numpy() * 255).astype(np.uint8), mode="max")
    labels.tofile(out / "labels.u8")
    act.tofile(out / "activity.u8")
    for label, bit in G.LABEL_BITS.items():
        share = float(((labels >> bit) & 1).mean())
        typer.echo(f"{label:>22}: {counts[bit]:>9} intervals, {100 * share:6.3f} % of bases")
    typer.echo(f"activity > 0 on {100 * float((act > 0).mean()):.2f} % of bases -> {out}")


@app.command("genome-labels-extra")
def genome_labels_extra(regulatory: Path = typer.Option(None, help="Ensembl Regulatory Build GFF3 (as genome-labels)"),
                        activity: Path = typer.Option(None, help="its activity matrix TSV (as genome-labels)"),
                        gtf: Path = typer.Option(None, help="GENCODE basic GTF (as genome-labels)"),
                        name: str = typer.Option("hg38", help="prepared genome under data/interim/genome/")) -> None:
    """The next tracks, beside ``genome-labels``' files: ``labels_extra.u8`` (bits of ``EXTRA_BITS``: 5' and 3' UTR,
    start and stop codons, TSS, lncRNA exons, gene span on each strand) and the Regulatory Build per epigenome:
    ``regfeat.i32`` (the row of the regulatory feature covering each base, -1 for none; the later feature where they
    overlap; features without a row in the activity table, all CTCF sites in v116, are left out), ``regfeat_active.npy`` ([features, epigenomes] bool, ACTIVE) and ``regfeat_epigenomes.json``."""
    import json

    import numpy as np
    import polars as pl
    import polars_bio as pb

    from kron_dna import genome as G

    reg_dir = input_dir("regulation")
    regulatory = regulatory or reg_dir / "ensembl116" / "Homo_sapiens.GRCh38.regulatory_features.v116.gff3.gz"
    activity = activity or reg_dir / "ensembl116" / "Homo_sapiens.GRCh38.regulatory_activity.v116.tsv.gz"
    gtf = gtf or reg_dir / "gencode" / "gencode.basic.gtf.gz"
    out = interim_dir("genome", name)
    index = pl.read_parquet(out / "contigs.parquet")
    total = int((index["end"] - index["start"]).sum())

    gt = (pb.scan_gtf(str(gtf), attr_fields=["transcript_id", "transcript_type"])
          .filter(pl.col("type").is_in(["transcript", "exon", "CDS", "UTR", "start_codon", "stop_codon"]))
          .select(["chrom", "start", "end", "type", "strand", "transcript_id", "transcript_type"]).collect())
    labels = np.zeros(total, dtype=np.uint8)
    for label, frame in G.gene_structure(gt).items():
        n = sum(G.paint(index, chrom, part["start"].to_numpy(), part["end"].to_numpy(), labels, 1 << G.EXTRA_BITS[label])
                for (chrom,), part in frame.group_by("chrom"))
        typer.echo(f"{label:>12}: {n:>9} intervals, {100 * float(((labels >> G.EXTRA_BITS[label]) & 1).mean()):6.3f} % of bases")
    labels.tofile(out / "labels_extra.u8")

    acts = pl.read_csv(activity, separator="\t")
    epis = acts.columns[2:]
    reg = (pb.scan_gff(str(regulatory)).collect()
           .with_columns(id=pl.col("attributes").list.eval(pl.element().struct.field("value")).list.first(),
                         chrom="chr" + pl.col("chrom"), start=pl.col("start").cast(pl.Int64) - 1,
                         end=pl.col("end").cast(pl.Int64))
           .join(acts.select("feature_id", *epis).rename({"feature_id": "id"}), on="id", how="inner")
           .sort("chrom", "start"))
    active = reg.select([(pl.col(e) == "ACTIVE") for e in epis]).to_numpy()
    feat = np.full(total, -1, dtype=np.int32)
    rows = np.arange(reg.height, dtype=np.int32)
    reg = reg.with_columns(row=pl.Series(rows))
    n = sum(G.paint_ids(index, chrom, part["start"].to_numpy(), part["end"].to_numpy(), feat, part["row"].to_numpy())
            for (chrom,), part in reg.group_by("chrom"))
    feat.tofile(out / "regfeat.i32")
    np.save(out / "regfeat_active.npy", active)
    (out / "regfeat_epigenomes.json").write_text(json.dumps(list(epis)))
    typer.echo(f"regulatory: {reg.height} features with activity over {len(epis)} epigenomes, {n} intervals painted, "
               f"{100 * float((feat >= 0).mean()):.2f} % of bases covered; mean share active "
               f"{float(active.mean()):.3f} -> {out}")


@app.command("genome-prepare-mito")
def genome_prepare_mito(gbff: Path = typer.Argument(..., help="RefSeq mitochondrion GenBank flat file (.gbff.gz)"),
                        species: Path = typer.Option(None, help="MitoAge species table, for the held-out orders "
                                                     "(default: data/input/mitoage/species.parquet)"),
                        name: str = typer.Option("mito", help="output under data/interim/genome/"),
                        held_out_mod: int = typer.Option(10, help="an order is held out when md5(order) % this == 0"),
                        min_len: int = typer.Option(1000, help="drop records shorter than this")) -> None:
    """Organelle genomes in genome-prepare's layout: ``bases.u8`` and ``contigs.parquet`` (chrom = accession, start,
    end, offset, code = genetic code table, circular, organism, held_out). Whole orders of MitoAge's species table
    are held out (Cetacea counted as Artiodactyla), and so is every record whose lineage names a held-out order;
    ``held_out_orders.json`` lists them."""
    import gzip
    import hashlib
    import json

    import numpy as np
    import polars as pl

    from kron_dna import genome as G

    sp = pl.read_parquet(species or input_dir("mitoage") / "species.parquet").filter(pl.col("record_status") != "retired")
    orders = sorted({("Artiodactyla" if o == "Cetacea" else o) for o in sp["order"].drop_nulls().to_list()})
    held = sorted(o for o in orders if int(hashlib.md5(o.encode()).hexdigest(), 16) % held_out_mod == 0)
    held_set = set(held) | ({"Cetacea"} if "Artiodactyla" in held else set())
    out = interim_dir("genome", name)
    out.mkdir(parents=True, exist_ok=True)
    rows, chunks, offset, skipped = [], [], 0, 0
    with gzip.open(gbff, "rt") as fh:
        for r in G.read_genbank(fh):
            if len(r["sequence"]) < min_len:
                skipped += 1
                continue
            codes = G.encode_bases(r["sequence"])
            chunks.append(codes)
            rows.append({"chrom": r["accession"], "start": 0, "end": len(codes), "offset": offset,
                         "code": int(r["transl_table"]), "circular": bool(r["circular"]), "organism": r["organism"],
                         "kingdom": r["lineage"][1] if len(r["lineage"]) > 1 else "",
                         "held_out": bool(held_set & set(r["lineage"]))})
            offset += len(codes)
    np.concatenate(chunks).tofile(out / "bases.u8")
    table = pl.DataFrame(rows)
    table.write_parquet(out / "contigs.parquet")
    (out / "held_out_orders.json").write_text(json.dumps(held, indent=1))
    by = table.group_by("held_out").agg(pl.len(), pl.col("end").sum())
    typer.echo(f"{table.height:,} genomes, {offset:,} bases ({skipped} shorter than {min_len} or without sequence "
               f"skipped); held-out orders ({len(held)} of {len(orders)}): {', '.join(held)}")
    typer.echo(f"{by}")
    typer.echo(f"{table.group_by('code').len().sort('len', descending=True).head(10)}")


@app.command("genome-train")
def genome_train(out: Path = typer.Argument(..., help="run directory (resumes from its checkpoint.pt)"),
                 name: str = typer.Option("hg38", help="prepared genome under data/interim/genome/"),
                 window: int = typer.Option(1 << 20, help="bases per window (a multiple of 128); a shorter contig "
                                            "is taken whole and padded"),
                 batch_size: int = 1, steps: int = 20_000, lr: float = 3e-4, warmup: int = 500,
                 mask_prob: float = 0.15, val_chroms: str = "chr8,chr9", eval_every: int = 500,
                 d_model: int = 384, n_layers: int = 8, stem_channels: int = 96, max_hours: float = 0.0,
                 n_down: int = typer.Option(7, help="halvings before the core: 2**n_down bases per core step"),
                 bench: bool = typer.Option(False, help="time one step and report peak memory, then stop"),
                 labels: bool = typer.Option(False, help="stage 2: also predict the genome-labels tracks (TRACKS)"),
                 init: Path | None = typer.Option(None, help="start a new run from this checkpoint's weights (stage 1)"),
                 track_weight: float = typer.Option(1.0, help="weight of the track loss beside the masked-base loss"),
                 tracks: str = typer.Option("all", help="with --labels: tracks the loss trains, a group (all, "
                                            "regulatory, gene) or names from TRACKS; all are still scored"),
                 whole_contigs: bool = typer.Option(False, help="one whole contig per step in one recurrence; the "
                                                    "encoder runs in --window chunks, the loss on --train-chunks of them"),
                 train_chunks: int = typer.Option(4, help="with --whole-contigs: chunks decoded and scored per step"),
                 max_contig: int = typer.Option(0, help="with --whole-contigs: a longer contig gives a random stretch "
                                                "of this many bases (0: none)"),
                 arch: str = typer.Option("v1", help="v1, or v2: per-position RMS norm, two convs per block, scaled "
                                          "projected decoder skips, MLP heads"),
                 head_hidden: int = typer.Option(0, help="v2: hidden width of the MLP heads (0: 2 x stem channels)"),
                 bert_mask: bool = typer.Option(False, help="of the masked bases 80 % MASK, 10 % random, 10 % kept"),
                 rc: bool = typer.Option(False, help="train on the reverse complement of half the windows"),
                 curriculum: str = typer.Option("", help="window:steps,... trained in order (e.g. 8192:6000,"
                                                "131072:2000,1048576:2000); overrides --window and --steps"),
                 tokens_per_step: int = typer.Option(0, help="with --curriculum: bases per optimizer step over all "
                                                     "cards and micro-batches (0: --batch-size windows)"),
                 grad_accum: int = typer.Option(1, help="micro-batches summed per optimizer step"),
                 compile_convs: bool = typer.Option(False, help="torch.compile each conv block"),
                 eval_window: int = typer.Option(0, help="held-out windows of this length (0: the curriculum's last)"),
                 jepa_weight: float = typer.Option(0.0, help="JEPA-DNA latent loss: predict an EMA target encoder's "
                                                   "core states of the unmasked window (cosine), at this weight"),
                 jepa_ema: float = typer.Option(0.996, help="with --jepa-weight: EMA decay of the target encoder"),
                 mito: str = typer.Option("", help="also train on this prepared organelle genome (genome-prepare-mito), "
                                          "each sequence tagged with its genetic code table as context (nuclear: 0)"),
                 mito_share: float = typer.Option(0.03, help="with --mito: share of training windows from it"),
                 seed: int = 0) -> None:
    """The multi-track model (KronGenome) over whole gap-free contigs: stage 1 masked-base prediction; with
    ``--labels``, stage 2 adds the Ensembl and GENCODE tracks of ``genome-labels`` and writes ``tracks.csv``.
    v2 (``--arch v2``) trains with a short-to-long ``--curriculum`` at a fixed ``--tokens-per-step``; held-out bits
    are always read on windows of the curriculum's last length, from windows and masks drawn by ``seed`` and the
    step alone, so two runs with one seed are scored on the same bases at the same step (paired evals)."""
    import numpy as np
    import polars as pl

    import torch.nn.functional as F

    from kron_dna import genome as G

    src = interim_dir("genome", name)
    bases = np.memmap(src / "bases.u8", dtype=np.uint8, mode="r")
    if labels:
        lab_mm = np.memmap(src / "labels.u8", dtype=np.uint8, mode="r")
        act_mm = np.memmap(src / "activity.u8", dtype=np.uint8, mode="r")
    index = pl.read_parquet(src / "contigs.parquet").with_columns((pl.col("end") - pl.col("start")).alias("len"))
    held = set(val_chroms.split(","))
    train_idx = index.filter(~pl.col("chrom").is_in(list(held)))
    val_idx = index.filter(pl.col("chrom").is_in(list(held)))
    import os

    import torch.distributed as dist
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world > 1:                                     # torchrun: one process per card, data-parallel
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    main = rank == 0
    say = typer.echo if main else (lambda *a, **k: None)
    rng = np.random.default_rng([seed, rank])        # each card its own windows; rank 0 keeps the eval stream
    unit = 2 ** n_down
    phases = [(int(w), int(s)) for w, s in (x.split(":") for x in curriculum.split(","))] if curriculum else \
        [(window, steps)]
    steps = sum(s for _, s in phases)
    eval_window = eval_window or phases[-1][0]
    if whole_contigs and (curriculum or rc or mito or bert_mask or grad_accum > 1):
        raise typer.BadParameter("--whole-contigs does not take --curriculum, --rc, --mito, --bert-mask or --grad-accum")
    if any(w % unit for w, _ in phases):
        raise typer.BadParameter(f"curriculum windows must be multiples of {unit}")
    if mito:
        mito_src = interim_dir("genome", mito)
        mito_bases = np.memmap(mito_src / "bases.u8", dtype=np.uint8, mode="r")
        mito_idx = pl.read_parquet(mito_src / "contigs.parquet").with_columns((pl.col("end") - pl.col("start")).alias("len"))
        mito_train, mito_val = mito_idx.filter(~pl.col("held_out")), mito_idx.filter(pl.col("held_out"))
        say(f"mito: {mito_train.height} train / {mito_val.height} held-out genomes "
                   f"({int(mito_train['len'].sum()):,} / {int(mito_val['len'].sum()):,} bases), share {mito_share}")

    def phase_at(s: int) -> tuple[int, int]:
        """(window, batch) at optimizer step s."""
        acc = 0
        for w, n in phases:
            acc += n
            if s < acc:
                break
        if tokens_per_step and tokens_per_step < w * world * grad_accum:
            raise typer.BadParameter(f"--tokens-per-step {tokens_per_step} is less than one window of {w} per card "
                                     f"and micro-batch")
        return w, (tokens_per_step // w // world // grad_accum if tokens_per_step else batch_size)

    def sample_contig(table, longest: bool = False) -> tuple[torch.Tensor, ...]:
        """One whole contig (by length, so every base equally), padded with N to a multiple of 2**n_down."""
        p = table["len"].to_numpy() / table["len"].sum()
        i = int(np.argmax(p)) if longest else int(rng.choice(table.height, p=p))
        r = dict(table.row(i, named=True))
        at = r["offset"]
        if max_contig and r["len"] > max_contig:
            at += 0 if longest else int(rng.integers(0, r["len"] - max_contig + 1))
            r["len"] = max_contig
        n = -(-r["len"] // unit) * unit
        codes = np.full(n, 4, dtype=np.int64)
        codes[:r["len"]] = bases[at:at + r["len"]]
        out = [torch.from_numpy(codes)]
        if labels:
            lab, act = np.zeros(n, dtype=np.uint8), np.zeros(n, dtype=np.uint8)
            lab[:r["len"]], act[:r["len"]] = lab_mm[at:at + r["len"]], act_mm[at:at + r["len"]]
            out += [torch.from_numpy(lab), torch.from_numpy(act), torch.from_numpy(np.arange(n) < r["len"])]
        return tuple(out)

    def sample(table, window=window, batch=None, gen=None, flip=False, mito_table=None, share=0.0
               ) -> tuple[torch.Tensor, ...]:
        """``batch`` windows: (seqs, segments, contexts[, labels, activity, valid]). With ``mito_table``, each row comes
        from it with probability ``share``; a circular genome is read across its origin when the window is shorter.
        Context 0 is nuclear; an organelle sequence's context is its genetic code table (1..33)."""
        gen = gen or rng
        batch = batch or batch_size
        p = table["len"].to_numpy() / table["len"].sum()                 # contigs by length: every base equally
        seqs, segs, ctxs, labs, acts, valid = [], [], [], [], [], []
        for _ in range(batch):
            from_mito = mito_table is not None and gen.random() < share
            if from_mito:
                r = mito_table.row(int(gen.integers(0, mito_table.height)), named=True)   # each genome alike
                take = min(window, r["len"])
                if r["circular"]:                                         # read across the origin
                    pos = (int(gen.integers(0, r["len"])) + np.arange(take)) % r["len"]
                else:
                    pos = int(gen.integers(0, r["len"] - take + 1)) + np.arange(take)
                codes = np.full(window, 4, dtype=np.int64)
                codes[:take] = mito_bases[r["offset"] + pos]
                ctxs.append(int(r["code"]))                               # organelle: its genetic code table
            else:
                r = table.row(int(gen.choice(table.height, p=p)), named=True)
                take = min(window, r["len"])
                at = r["offset"] + int(gen.integers(0, r["len"] - take + 1))
                codes = np.full(window, 4, dtype=np.int64)
                codes[:take] = bases[at:at + take]
                ctxs.append(0)                                            # nuclear (context 0)
            flip_row = flip and gen.random() < 0.5
            if flip_row:
                codes[:take] = G.reverse_complement(codes[:take].astype(np.uint8))
            seg = np.zeros(window // unit, dtype=np.int64)
            seg[(take + unit - 1) // unit:] = 1                           # padding is its own segment
            seqs.append(codes)
            segs.append(seg)
            if labels:
                lab, act = np.zeros(window, dtype=np.uint8), np.zeros(window, dtype=np.uint8)
                if not from_mito:
                    lab[:take], act[:take] = lab_mm[at:at + take], act_mm[at:at + take]
                    if flip_row:                                          # the current tracks are not strand-specific
                        lab[:take], act[:take] = lab[:take][::-1].copy(), act[:take][::-1].copy()
                labs.append(lab)
                acts.append(act)
                valid.append((np.arange(window) < take) & (not from_mito))
        out = [torch.from_numpy(np.stack(seqs)), torch.from_numpy(np.stack(segs)), torch.tensor(ctxs)]
        if labels:
            out += [torch.from_numpy(np.stack(x)) for x in (labs, acts, valid)]
        return tuple(out)

    dev = torch.device("cuda", torch.cuda.current_device())
    config = G.KronGenomeConfig(d_model=d_model, n_layers=n_layers, stem_channels=stem_channels, n_down=n_down,
                                n_tracks=len(G.TRACKS) if labels else 0, arch=arch, head_hidden=head_hidden,
                                n_contexts=34 if mito else 0, jepa=jepa_weight > 0)
    model = G.KronGenome(config).to(dev)
    if compile_convs:
        for blocks in [model.down, model.up] + ([model.up2] if arch == "v2" else []):
            for i in range(len(blocks)):
                blocks[i].compile(dynamic=True)                # in place: state-dict keys unchanged
    pos_weight = None
    train_mask = torch.tensor(G.track_mask(tracks), dtype=torch.float32, device=dev)
    if labels:                       # rare tracks (splice sites) weighted up by sqrt(neg/pos), from a sample of bases
        probe = np.asarray(lab_mm[:: max(1, len(lab_mm) // 50_000_000)])
        share = np.array([((probe >> b) & 1).mean() for b in range(len(G.LABEL_BITS))] + [0.5])
        pos_weight = torch.tensor(np.clip(np.sqrt((1 - share) / np.maximum(share, 1e-9)), 1, 100),
                                  dtype=torch.float32, device=dev)
        say("track positive weights: " + ", ".join(f"{t} {w:.1f}" for t, w in zip(G.TRACKS, pos_weight)))
        say("tracks trained: " + ", ".join(t for t, m in zip(G.TRACKS, train_mask.tolist()) if m))
    if arch == "v2":   # no weight decay on norms, biases, scales and embeddings
        decay = [p_ for n_, p_ in model.named_parameters() if p_.ndim >= 2 and "context" not in n_]
        rest = [p_ for n_, p_ in model.named_parameters() if not (p_.ndim >= 2 and "context" not in n_)]
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": rest, "weight_decay": 0.0}],
                                lr=lr, betas=(0.9, 0.95))
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "checkpoint.pt"
    step, hours = 0, 0.0
    if ckpt.exists() and not bench:
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        step, hours = state["step"], state.get("hours", 0.0)
        say(f"resumed at step {step}")
    elif init is not None and not bench:
        missing, unexpected = model.load_state_dict(torch.load(init, map_location="cpu")["model"], strict=False)
        say(f"initialised from {init}; new: {missing}; dropped: {unexpected}")
    say(f"KronGenome {arch} {model.n_parameters() / 1e6:.1f}M parameters, curriculum "
               f"{', '.join(f'{w:,} x {n:,} steps' for w, n in phases)}, eval window {eval_window:,}, "
               f"{train_idx.height} train / {val_idx.height} held-out contigs")
    target = None
    if jepa_weight > 0:                               # the EMA target: a frozen copy, updated after every step
        import copy
        target = copy.deepcopy(model).eval()
        for p_ in target.parameters():
            p_.requires_grad_(False)
    train_model = model
    if world > 1:
        train_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dev.index])
        say(f"data-parallel over {world} cards: {tokens_per_step // world if tokens_per_step else batch_size} "
            f"bases per card per micro-batch")
    autocast = torch.autocast("cuda", dtype=torch.bfloat16)

    def contig_loss(seq, lab=None, act=None, valid=None, keep=None):
        inputs, targets = G.mask_bases(seq.to(dev), mask_prob, bert=bert_mask)
        n_chunks = -(-seq.numel() // window)
        picks = sorted(rng.choice(n_chunks, size=min(train_chunks, n_chunks), replace=False).tolist())
        with autocast:
            outs = model.forward_contig(inputs, [k * window for k in picks], chunk=window)
        mlm, track, count = 0.0, 0.0, 0
        for k, (logits, tracks) in zip(picks, outs):
            sl = slice(k * window, k * window + logits.shape[1])
            mlm = mlm + F.cross_entropy(logits.float()[0], targets[sl], ignore_index=-100, reduction="sum")
            count += int((targets[sl] != -100).sum())
            if lab is not None:
                want = G.track_targets(lab[sl].to(dev), act[sl].to(dev))
                ok = valid[sl].to(dev)
                per = F.binary_cross_entropy_with_logits(tracks.float()[0], want, pos_weight=pos_weight,
                                                         reduction="none")
                track = track + (per * ok.unsqueeze(-1) * train_mask).sum() / (ok.sum() * train_mask.sum()) / len(picks)
                if keep is not None:
                    keep.append((tracks.float()[0][ok], want[ok]))
        return mlm / max(count, 1), (track if lab is not None else None)

    def loss_on(seq, seg, ctx, lab=None, act=None, valid=None, keep=None, mask_gen=None):
        # held-out masks are always plain MASK (bert only in training), so runs with and without it compare
        inputs, targets = G.mask_bases(seq.to(dev), mask_prob, generator=mask_gen, bert=bert_mask and model.training)
        with autocast:
            logits, _, tracks = (train_model if model.training else model)(inputs, seg.to(dev),
                                                                            ctx.to(dev) if mito else None)
        mlm = F.cross_entropy(logits.float().reshape(-1, 4), targets.reshape(-1), ignore_index=-100)
        if target is not None and model.training and model.last_jepa is not None:
            with torch.no_grad(), autocast:
                _, want_core, _ = target(seq.to(dev), seg.to(dev), ctx.to(dev) if mito else None)
            keep_bins = seg.to(dev) == 0                                # not padding
            cos = F.cosine_similarity(model.last_jepa.float(), want_core.float(), dim=-1)
            jepa_losses.append(float((1 - cos)[keep_bins].mean().detach()))
            mlm = mlm + jepa_weight * (1 - cos)[keep_bins].mean()
        if lab is None:
            return mlm, None
        want = G.track_targets(lab.to(dev), act.to(dev))
        ok = valid.to(dev)
        per = F.binary_cross_entropy_with_logits(tracks.float(), want, pos_weight=pos_weight, reduction="none")
        track = (per * ok.unsqueeze(-1) * train_mask).sum() / (ok.sum().clamp_min(1) * train_mask.sum())
        if keep is not None:                                      # held-out scores for tracks.csv
            keep.append((tracks.float()[ok], want[ok]))
        return mlm, track

    jepa_losses: list[float] = []
    t0, started = time.time(), time.time()
    metrics = out / "metrics.csv"
    if main and not metrics.exists():
        metrics.write_text("step,lr,train_loss,val_bits_per_base,hours" + (",train_track_loss,val_track_loss"
                                                                           if labels else "")
                           + (",val_mito_bits_per_base" if mito else "") + "\n")
    track_csv = out / "tracks.csv"
    if main and labels and not track_csv.exists():
        track_csv.write_text("step,track,auroc,average_precision,prevalence\n")
    model.train()
    while step < steps:
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, (step + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(step / steps, 1.0)))
        torch.cuda.reset_peak_memory_stats(dev)
        opt.zero_grad(set_to_none=True)
        win, bat = phase_at(step)
        for _ in range(grad_accum):
            if whole_contigs:
                loss, track = contig_loss(*sample_contig(train_idx, longest=bench))
            else:
                loss, track = loss_on(*sample(train_idx, window=win, batch=bat, flip=rc,
                                              mito_table=mito_train if mito else None, share=mito_share))
            ((loss + track_weight * track if track is not None else loss) / grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if target is not None:
            with torch.no_grad():
                for pt, pm in zip(target.parameters(), model.parameters()):
                    pt.lerp_(pm.detach(), 1 - jepa_ema)
        step += 1
        if bench and step == 1:                                  # the first step compiles kernels: time the second
            torch.cuda.synchronize()
            t0 = time.time()
            continue
        if bench:
            torch.cuda.synchronize()
            what = f"longest contig {min(train_idx['len'].max(), max_contig or 1 << 62):,} bases in {window:,} chunks, {train_chunks} scored" \
                if whole_contigs else f"window {win:,}, batch {bat} x {grad_accum} accumulated"
            say(f"bench: {what}: {time.time() - t0:.1f} s for one step, "
                       f"peak {torch.cuda.max_memory_allocated(dev) / 2**30:.1f} GiB, loss {loss.item():.3f}")
            return
        out_of_time = max_hours > 0 and time.time() - started > max_hours * 3600
        if world > 1:                                   # every card stops at the same step
            flag = torch.tensor([float(out_of_time)], device=dev)
            dist.broadcast(flag, 0)
            out_of_time = bool(flag.item())
        if (step % eval_every == 0 or step == steps or out_of_time) and not main:
            dist.barrier()                              # rank 0 evaluates and saves
        elif step % eval_every == 0 or step == steps or out_of_time:
            model.eval()
            keep: list = []
            eval_rng = np.random.default_rng([seed, step])               # the same windows for every run with this seed
            mask_gen = torch.Generator(device=dev).manual_seed(seed * 1_000_003 + step)
            with torch.no_grad():
                pairs = ([contig_loss(*sample_contig(val_idx), keep=keep if labels else None) for _ in range(2)]
                         if whole_contigs else
                         [loss_on(*sample(val_idx, window=eval_window, batch=1 if arch == "v2" else batch_size,
                                          gen=eval_rng),
                                  keep=keep if labels else None, mask_gen=mask_gen) for _ in range(8)])
                mito_bits = None
                if mito:
                    mw = -(-int(mito_val["len"].max()) // unit) * unit
                    mito_bits = float(np.mean([loss_on(*sample(mito_val, window=mw, batch=4, gen=eval_rng,
                                                               mito_table=mito_val, share=1.0), mask_gen=mask_gen)[0].item()
                                               for _ in range(8)])) / math.log(2)
            model.train()
            h = hours + (time.time() - t0) / 3600
            bits = float(np.mean([m.item() for m, _ in pairs])) / math.log(2)
            extra = ""
            if labels:
                val_track = float(np.mean([t.item() for _, t in pairs]))
                extra = f",{track.item():.4f},{val_track:.4f}"
                scores = torch.cat([k[0] for k in keep])
                truth = torch.cat([k[1] for k in keep])
                with track_csv.open("a") as fh:
                    for j, name_ in enumerate(G.TRACKS):
                        y = truth[:, j] >= 0.5
                        auroc, ap = G.ranking_scores(scores[:, j], y)
                        fh.write(f"{step},{name_},{auroc:.4f},{ap:.4f},{float(y.float().mean()):.6f}\n")
                del keep, scores, truth
            tracks_note = extra
            if mito_bits is not None:
                extra += f",{mito_bits:.4f}"
            with metrics.open("a") as fh:
                fh.write(f"{step},{opt.param_groups[0]['lr']:.3e},{loss.item():.4f},{bits:.4f},{h:.3f}{extra}\n")
            say(f"step {step} (window {win:,} x {bat}): train {loss.item():.4f}, held-out {bits:.3f} bits/base"
                       + (f", mito {mito_bits:.3f}" if mito_bits is not None else "")
                       + (f", jepa {np.mean(jepa_losses):.4f}" if jepa_losses else "") + f", {h:.2f} h, "
                       f"peak {torch.cuda.max_memory_allocated(dev) / 2**30:.1f} GiB" + (f", tracks {tracks_note}" if tracks_note else ""))
            jepa_losses.clear()
            _save_atomic({"model": model.state_dict(), "optimizer": opt.state_dict(), "config": config.to_dict(),
                          "step": step, "hours": h}, ckpt)
            if world > 1:
                dist.barrier()
        if out_of_time:
            say("stopping at --max-hours; resubmit the same command to continue")
            if world > 1:
                dist.destroy_process_group()
            return


@app.command("genome-clinvar")
def genome_clinvar(runs: list[Path] = typer.Option([], "--run", help="genome-train run directories (or checkpoints)"),
                   vcf: Path = typer.Option(None, help="ClinVar GRCh38 VCF (default: data/input/variants/clinvar/"
                                            "clinvar.vcf.gz)"),
                   name: str = typer.Option("hg38", help="prepared genome (with genome-labels) under data/interim/genome/"),
                   window: int = typer.Option(1 << 20, help="bases of genome per pass; a variant has at least a "
                                              "quarter of it on each side"),
                   per_class: int = typer.Option(2000, help="variants kept per (region, label, held-out or not), by "
                                                 "seed; 0 = all"),
                   val_chroms: str = typer.Option("chr8,chr9", help="the chromosomes genome-train held out"),
                   near: int = typer.Option(10, help="bases from a splice site that count as near it"),
                   delta_window: int = typer.Option(1 << 17, help="with a track head: also score the largest "
                                                    "change of the track predictions within 50 bp, ref against alt, "
                                                    "in a window this long centred on the variant (0: skip)"),
                   delta_batch: int = 8,
                   batch_size: int = 2, device: str = "cuda", seed: int = 0, n_boot: int = 1000,
                   out: Path | None = typer.Option(None, help="JSON summary (default: data/output/eval/clinvar/"
                                                   "genome.json; the scores go beside it as .parquet)")) -> None:
    """Zero-shot ClinVar for KronGenome by region (coding, splice site, near one, UTR exon, regulatory element,
    intron or intergenic): AUROC of the masked marginal ``log p(alt) - log p(ref)``, pathogenic against benign."""
    import json

    import numpy as np
    import polars as pl

    from kron_core.paths import output_dir
    from kron_dna import genome as G
    from kron_dna import genome_variants as GV
    from kron_dna import variants as V

    vcf = vcf or input_dir("variants", "clinvar") / "clinvar.vcf.gz"
    src = interim_dir("genome", name)
    bases = np.memmap(src / "bases.u8", dtype=np.uint8, mode="r")
    labels = np.memmap(src / "labels.u8", dtype=np.uint8, mode="r")
    index = pl.read_parquet(src / "contigs.parquet")
    table, dropped = GV.place(V.read_clinvar(vcf), index, bases, labels, set(val_chroms.split(",")), near)
    typer.echo(f"{table.height} variants placed; dropped {dropped}")
    if per_class:
        table = table.filter(pl.int_range(pl.len()).shuffle(seed).over("region", "label", "held_out") < per_class)
    table = table.sort("g")
    with pl.Config(tbl_rows=-1):
        typer.echo(str(table.group_by("region", "held_out", "label").len().sort("region", "held_out", "label")))
    dev = torch.device(device)
    g, c_off, c_len = (table[c].to_numpy() for c in ("g", "c_off", "c_len"))
    t_start, t_len = GV.tile_of(g, c_off, c_len, window)
    scores = {"distance to splice site": GV.splice_distance(labels, g)}   # lower = closer = more damaging
    for run in runs:
        ckpt = run / "checkpoint.pt" if run.is_dir() else run
        state = torch.load(ckpt, map_location="cpu")
        model = G.KronGenome(G.KronGenomeConfig(**state["config"])).to(dev)
        model.load_state_dict(state["model"])
        model.eval()
        t0 = time.time()
        key = run.name if run.is_dir() else run.parent.name
        scores[key] = GV.masked_marginals(model, bases, g, table["ref_code"].to_numpy(), table["alt_code"].to_numpy(),
                                          t_start, t_len, window, dev, batch_size)
        typer.echo(f"{key} (step {state.get('step')}): {len(np.unique(t_start))} tiles in {time.time() - t0:.0f} s")
        if model.tracks is not None and delta_window:
            t0 = time.time()
            d = GV.track_deltas(model, bases, g, table["ref_code"].to_numpy(), table["alt_code"].to_numpy(),
                                c_off, c_len, delta_window, dev, delta_batch)
            splice = [G.TRACKS.index("donor"), G.TRACKS.index("acceptor")]
            scores[f"{key}:splice-delta"] = -d[:, splice].max(1)        # lower = more damaging, as the marginal
            scores[f"{key}:track-delta"] = -d.max(1)
            typer.echo(f"{key}: track deltas in {delta_window:,}-base windows, {time.time() - t0:.0f} s")
        del model
    rows = GV.summarize(table, scores, n_boot)
    out = out or output_dir("eval", "clinvar") / "genome.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"dropped": dropped, "window": window, "delta_window": delta_window,
                               "runs": [str(r) for r in runs], "rows": rows},
                              indent=2) + "\n")
    table.select("id", "chrom", "pos", "ref", "alt", "region", "held_out", "label").with_columns(
        **{k: pl.Series(v) for k, v in scores.items()}).write_parquet(out.with_suffix(".parquet"))
    for r in rows:
        typer.echo(f"{r['region']:>21} | {r['chromosomes']:>20} | P {r['pathogenic']:>5} B {r['benign']:>5} | "
                   + "  ".join(f"{k} {r[k]:.3f}" if r[k] is not None else f"{k} -" for k in scores))
    typer.echo(f"-> {out}")


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
