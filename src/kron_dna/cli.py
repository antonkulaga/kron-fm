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
    dev = torch.device(device)
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
