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

#: tests: set to a list and genome-train appends every training micro-batch's loss terms when jepa.csv is on
#: (``jepa_log``), plus ``jepa_term`` (the JEPA loss in the total) and ``reg`` (the --jepa-reg term before its weight)
JEPA_TRACE: list | None = None


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


def genome_lr(step: int, lr: float, warmup: int, steps: int, lr_total: int = 0, cooldown_from: int = 0) -> float:
    """genome-train's learning rate at ``step``: linear warmup, then a cosine over ``lr_total`` steps (0: ``steps``,
    the curriculum's total). With ``cooldown_from`` S, from step S on the rate starts at the cosine's own value at S
    and follows a second cosine from there to zero at ``steps``: a run resumed with a longer curriculum keeps its rate
    continuous and gets a longer cooldown instead of a rewarm (raising ``steps`` alone moves the whole cosine)."""
    total = lr_total or steps
    def base(t: int) -> float:
        return lr * min(1.0, (t + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(t / total, 1.0)))
    if not cooldown_from or step < cooldown_from:
        return base(step)
    span = max(1, steps - cooldown_from)
    return base(cooldown_from) * 0.5 * (1 + math.cos(math.pi * min((step - cooldown_from) / span, 1.0)))


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


def clinvar_cache_key(vcf: Path, cds: Path, holdout_genes: float, seed: int, version: int = 1) -> str:
    """Identity of a mapped ClinVar table: the VCF and CDS files (path, size, modification time), the held-out gene
    fraction, the seed and the mapping version. A changed input or split gets its own cache file instead of the
    rows (and in_training flags) mapped for another (audit 2026-11-18, EV-01)."""
    import hashlib

    def ident(p: Path) -> list:
        p = Path(p)
        st = p.stat() if p.exists() else None
        return [str(p.resolve()), st.st_size if st else None, st.st_mtime_ns if st else None]
    blob = json.dumps(dict(vcf=ident(vcf), cds=ident(cds), holdout_genes=float(holdout_genes), seed=int(seed),
                           version=version), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def run_keys(runs: list[Path]) -> list[str]:
    """One score name per checkpoint of a comparison: the run directory's name, or for a checkpoint file its parent's
    name, extended by the file's stem (or the whole path) where two would collide. Duplicates are refused before
    any scoring (audit 2026-11-18, EV-03: two files of one snapshots/ folder overwrote each other's scores)."""
    base = [r.name if r.is_dir() else r.parent.name for r in runs]
    keys = []
    for r, b in zip(runs, base):
        if base.count(b) == 1:
            keys.append(b)
        else:
            keys.append(f"{r.parent.name}/{r.stem}" if not r.is_dir() else str(r))
    if len(set(keys)) < len(keys):
        keys = [str(r) for r in runs]
    dup = sorted({k for k in keys if keys.count(k) > 1})
    if dup:
        raise typer.BadParameter(f"--run lists the same checkpoint more than once: {dup}")
    return keys


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
    cache = interim_dir("cds", "clinvar") / f"variants-{clinvar_cache_key(vcf, fasta, holdout_genes, seed)}.parquet"
    if cache.exists():                          # audit 2026-11-18, EV-01: keyed by the inputs and the gene split
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
                   name: str = typer.Option("hg38", help="folder under data/interim/genome/"),
                   overwrite: bool = typer.Option(False, help="replace an existing bases.u8 + contigs.parquet (the old "
                                                  "pair is kept as *.prev)")) -> None:
    """Whole gap-free contigs of a genome, as one uint8 array (A, C, G, T, N = 0..4) and a contig index.

    The pair is built in a staging folder inside the output and published only after the FASTA parsed and the array
    length matched the index, so a failed or mistaken run never touches a prepared genome other runs read."""
    import os
    import shutil

    import numpy as np
    import polars as pl
    import polars_bio as pb

    from kron_dna import genome as G

    out = interim_dir("genome", name)
    out.mkdir(parents=True, exist_ok=True)
    finals = (out / "bases.u8", out / "contigs.parquet")
    existing = [f for f in finals if f.exists()]
    if existing and not overwrite:
        raise typer.BadParameter(f"{out} already holds {', '.join(f.name for f in existing)}; pass --overwrite to "
                                 "replace them (the old pair is kept as *.prev) or choose another --name")
    stage = out / f".prepare-{os.getpid()}"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir()
    try:
        keep = {f"chr{c}" for c in [*map(str, range(1, 23)), "X", "Y"]}
        records = pb.scan_fasta(str(fasta)).select("name", "sequence").collect()   # parse before writing anything
        rows, offset = [], 0
        with open(stage / "bases.u8", "wb") as fh:
            for chrom, seq in records.iter_rows():
                if chrom not in keep:
                    continue
                codes = G.encode_bases(seq)
                for start, end in G.contigs_of(codes, min_gap, min_len):
                    fh.write(codes[start:end].tobytes())
                    rows.append((chrom, start, end, offset))
                    offset += end - start
                typer.echo(f"{chrom}: {sum(1 for r in rows if r[0] == chrom)} contigs")
        if not rows:
            raise typer.BadParameter(f"{fasta}: no contig of chr1-22, X or Y passed the filters")
        index = pl.DataFrame(rows, schema=["chrom", "start", "end", "offset"], orient="row")
        index.write_parquet(stage / "contigs.parquet")
        size = (stage / "bases.u8").stat().st_size
        extent = int((index["offset"] + index["end"] - index["start"]).max())
        if size != offset or extent != offset:
            raise RuntimeError(f"staged genome is inconsistent: bases.u8 {size} bytes, index extent {extent}, "
                               f"written {offset}")
        for f in finals:                     # publish: the old pair moves aside, then the staged pair takes its place
            if f.exists():
                os.replace(f, f.with_name(f.name + ".prev"))
        for f in finals:
            os.replace(stage / f.name, f)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
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
                 xhead: list[str] = typer.Option([], help="a further per-base head, its own module beside the "
                                                 "inherited ones (repeatable, kept in this order): NAME:STORE, a "
                                                 "measured-track store on the primary genome's layout "
                                                 "(prep_bigwig_tracks.py), or NAME:N to register N outputs ahead "
                                                 "of their data (no forward, no update). Needs --ntv3-dir (its "
                                                 "training split) and arch v2 or v2l; with --init the heads are "
                                                 "the only new parameters"),
                 xhead_split: Path | None = typer.Option(None, help="with --xhead and no --ntv3-dir: the primary "
                                                         "genome's split.u8 (1 = train, 4 = the held-out "
                                                         "chromosomes), read for the heads' training mask without "
                                                         "an NTv3 head; default: --ntv3-dir's own"),
                 xhead_weight: float = typer.Option(0.5, help="weight of each --xhead loss beside the other terms"),
                 xhead_hidden: int = typer.Option(0, help="hidden width of the --xhead heads (0: as the other heads, "
                                                  "2 x stem channels); lower it when a 1 Mb window runs out of "
                                                  "memory"),
                 xhead_bin: list[str] = typer.Option([], help="NAME=B: --xhead NAME predicts at bins of B bases "
                                                     "(a power of two up to the core step; targets are the per-base "
                                                     "targets averaged per bin): wide track sets cost B-fold less in "
                                                     "the head and its loss. Track heads only; fixed for a head's life"),
                 xhead_kind: str = typer.Option("mlp", help="the --xhead heads' kind: mlp (per position) or kron "
                                                "(a per-base readout plus bidirectional KronLSTM layers over "
                                                "--xhead-step-bases means; context is the supervised span)"),
                 xhead_layers: int = typer.Option(2, help="with --xhead-kind kron: KronLSTM layers per head"),
                 xhead_step_bases: int = typer.Option(32, help="with --xhead-kind kron: bases per KronLSTM step"),
                 xhead_width: int = typer.Option(128, help="with --xhead-kind kron: the heads' KronLSTM width"),
                 xhead_cycle: str = typer.Option("", help="train one group of --xhead heads per optimizer step, in "
                                                 "turn: groups split by ';', heads by ',' (e.g. 'dnase,splice;rna;chip'"
                                                 "); every head in some group; a head's loss is scaled by groups / the "
                                                 "groups that hold it; held-out scoring still runs every head. Peak "
                                                 "memory is the largest group's, not the sum"),
                 xhead_recompute: bool = typer.Option(False, help="recompute each --xhead head's inside in backward: "
                                                      "only its input and output are kept"),
                 xhead_genome_split: list[str] = typer.Option([], help="GENOME=SPLIT: a further genome's per-base split "
                                                              "(uint8, 1 training, 4 held out; prep_syntenic_split.py): "
                                                              "its rows are supervised where it is 1, scored where 4, "
                                                              "and its training rows read N at every held-out base (no "
                                                              "input, no masked-base target); without one, a window's "
                                                              "code is its contig's"),
                 xhead_species: list[str] = typer.Option([], help="NAME=GENOME:STORE: head NAME also trains on windows "
                                                         "of --genome GENOME from that genome's store; a channel that "
                                                         "matches one of the primary store's (assay, target, biosample, "
                                                         "strand; or a label) is shared, the rest are appended"),
                 xhead_only: int = typer.Option(200, help="with --xhead: labelled updates of the first head before "
                                                "the inherited weights train too (their rate is 0 until then)"),
                 xhead_lr: float = typer.Option(4e-4, help="peak rate of each --xhead, reached over --xhead-warmup "
                                                "of its own labelled updates, under the run's schedule"),
                 xhead_warmup: int = 200,
                 inherit_lr: float = typer.Option(4e-5, help="with --xhead: peak rate of the inherited weights in "
                                                  "joint training, reached over --inherit-warmup joint updates, "
                                                  "under the run's schedule (it scales genome_lr by "
                                                  "--inherit-lr / --lr)"),
                 inherit_warmup: int = 200,
                 track_weight: float = typer.Option(1.0, help="weight of the track loss beside the masked-base loss"),
                 tracks: str = typer.Option("all", help="with --labels: tracks the loss trains, a group (all, "
                                            "regulatory, gene) or names from TRACKS; all are still scored"),
                 whole_contigs: bool = typer.Option(False, help="one whole contig per step in one recurrence; the "
                                                    "encoder runs in --window chunks, the loss on --train-chunks of them"),
                 train_chunks: int = typer.Option(4, help="with --whole-contigs: chunks decoded and scored per step"),
                 max_contig: int = typer.Option(0, help="with --whole-contigs: a longer contig gives a random stretch "
                                                "of this many bases (0: none)"),
                 arch: str = typer.Option("v1", help="v1, or v2: per-position RMS norm, two convs per block, scaled "
                                          "projected decoder skips, MLP heads; v2l: v2 with NTv3's lighter conv "
                                          "layout (pool right after the stem, one conv per decoder step)"),
                 head_hidden: int = typer.Option(0, help="v2: hidden width of the MLP heads (0: 2 x stem channels)"),
                 n_in_extra: int = typer.Option(0, help="reserved per-base input channels beside the 6 base channels, "
                                                "zero-initialised and fed zeros until a later run supplies them"),
                 n_contexts: int = typer.Option(-1, help="context embedding slots (genome.CONTEXT_RANGES; 256 "
                                                "provisions species, assemblies and cell states); -1: 34 with "
                                                "--mito, else 0, as before"),
                 new_contexts: str = typer.Option("zero", help="with --init, when the model has more context rows than "
                                                  "the checkpoint: 'zero' (new rows add nothing, as before: the "
                                                  "checkpoint WITHOUT a context vector, which is not the trained "
                                                  "nuclear model, whose row 0 is non-zero) or 'nuclear' (new rows "
                                                  "copy row 0, so a new id starts exactly as the nuclear model)"),
                 bert_mask: bool = typer.Option(False, help="of the masked bases 80 % MASK, 10 % random, 10 % kept"),
                 rc: bool = typer.Option(False, help="train on the reverse complement of half the windows"),
                 curriculum: str = typer.Option("", help="window:steps,... trained in order (e.g. 8192:6000,"
                                                "131072:2000,1048576:2000); overrides --window and --steps"),
                 tokens_per_step: int = typer.Option(0, help="with --curriculum: bases per optimizer step over all "
                                                     "cards and micro-batches (0: --batch-size windows)"),
                 grad_accum: int = typer.Option(1, help="micro-batches summed per optimizer step"),
                 compile_convs: bool = typer.Option(False, help="torch.compile each conv block"),
                 local_track: str = typer.Option("", help="two-pathway: width,layers[,span] of a light 1-bp KronLSTM "
                                                 "beside the long-range core, e.g. 128,2,8192"),
                 checkpoint_core: bool = typer.Option(False, help="recompute each core layer in backward: ~8x less "
                                                      "core activation memory for ~30 % more compute"),
                 eval_window: int = typer.Option(0, help="held-out windows of this length (0: the curriculum's last)"),
                 jepa_weight: float = typer.Option(0.0, help="JEPA-DNA latent loss: predict an EMA target encoder's "
                                                   "core states of the unmasked window (cosine), at this weight"),
                 jepa_ema: float = typer.Option(0.996, help="with --jepa-weight: EMA decay of the target encoder"),
                 jepa_reg: str = typer.Option("none", help="with --jepa-weight: anti-collapse term on the context's core "
                                              "states: none, vicreg (variance hinge + covariance) or sigreg (LeJEPA: "
                                              "random projections towards an isotropic Gaussian, Epps-Pulley)"),
                 jepa_reg_weight: float = typer.Option(1.0, help="weight of --jepa-reg"),
                 jepa_block_share: float = typer.Option(0.0, help="with --jepa-weight, block JEPA: hide this fraction "
                                                        "of each training row's bins in contiguous blocks (MASK in "
                                                        "the online input, the repeat channel zeroed, no soft bases "
                                                        "there; the target reads the clean row) and train the "
                                                        "predictor on the hidden bins. 0: off, as before"),
                 jepa_block_len: str = typer.Option("1024:8192", help="with --jepa-block-share: LO:HI block lengths "
                                                    "in bases (multiples of 2**n_down), drawn log-uniformly"),
                 jepa_on: str = typer.Option("hidden", help="with --jepa-block-share: the bins the JEPA loss is a mean "
                                             "over: hidden (the blocks) or all (every non-padding bin)"),
                 jepa_block_heads: str = typer.Option("exclude", help="with --jepa-block-share: exclude (default) "
                                                      "keeps positions inside blocks out of the masked-base, NTv3 "
                                                      "track, epigenome and --labels track losses; include scores "
                                                      "them there as elsewhere"),
                 jepa_pred: str = typer.Option("mlp", help="with --jepa-weight: the predictor, mlp (per bin, as "
                                               "before) or context (a mask embedding at hidden bins and "
                                               "--jepa-pred-layers bidirectional KronLSTM layers over the core "
                                               "states; it starts as the masked core). A change needs --init"),
                 jepa_pred_layers: int = typer.Option(2, help="with --jepa-pred context: its KronLSTM layers"),
                 jepa_loss: str = typer.Option("cos", help="with --jepa-weight: cos (1 - cosine on raw core states, as "
                                               "before), zcos (1 - cosine after centring and scaling each dimension "
                                               "by running statistics of the target states: the states share a "
                                               "large common direction, which raw cosine rewards), or zcos-mse "
                                               "(zcos plus the mean squared error of the standardised states, so "
                                               "the norm is fixed too)"),
                 jepa_decode_pred: bool = typer.Option(False, help="with --jepa-block-share and --jepa-block-heads "
                                                       "include: the decoder reads the predictor's states at the "
                                                       "hidden bins, so the heads' losses there train the predictor "
                                                       "(the predictor's output must decode)"),
                 cf_pairs: list[Path] = typer.Option([], help="measured counterfactual pair tables (cf-pairs-v1 "
                                                     "parquet; repeatable): their train rows add a sign loss on the "
                                                     "head's predicted alt-minus-ref change (kron_dna.cfpairs). Rows "
                                                     "on the held-out chromosomes are refused"),
                 cf_weight: float = typer.Option(0.0, help="with --cf-pairs: weight of the pair loss (0: off)"),
                 cf_batch: int = typer.Option(4, help="with --cf-pairs: pairs per step (each two windows)"),
                 cf_every: int = typer.Option(1, help="with --cf-pairs: the pair loss on every Nth step"),
                 cf_window: int = typer.Option(131072, help="with --cf-pairs: bases read around each pair"),
                 cf_temp: float = typer.Option(0.1, help="with --cf-pairs: the change (log1p units) per unit of the "
                                                "logistic sign loss"),
                 jepa_edits: float = typer.Option(0.0, help="with --jepa-block-share, self-made counterfactuals "
                                                  "(JEPA step 2b), at this weight: on a window of its own an edit "
                                                  "(a shuffled, reverse-complemented or blanked segment) is made "
                                                  "outside the hidden blocks; the predictor on the edited, hidden "
                                                  "input must match the target's states of the edited window, and "
                                                  "its change from the clean input the target's change (0: off)"),
                 jepa_edit_every: int = typer.Option(1, help="with --jepa-edits: on every Nth step"),
                 jepa_edit_window: int = typer.Option(131072, help="with --jepa-edits: bases of the edit window"),
                 jepa_edit_len: str = typer.Option("128:1024", help="with --jepa-edits: LO:HI edit lengths in bases"),
                 freeze_trunk_until: int = typer.Option(0, help="until this step only the JEPA predictor trains (the "
                                                        "other parameters' gradients are dropped before the "
                                                        "update): a warm-up of a new predictor on a frozen model"),
                 jepa_log_every: int = typer.Option(0, help="with --jepa-weight: also write a jepa.csv row (and print "
                                                    "it) every this many steps; 0: only at evaluations, and only "
                                                    "with --jepa-block-share"),
                 jepa_guard: str = typer.Option("off", help="with --jepa-weight, JEPA collapse guard: off (default, "
                                                "as before), monitor (at every evaluation, collapse measures of the "
                                                "core states on fixed held-out windows in jepa.csv and the log) or on "
                                                "(monitor, and when the entropy rank falls below --jepa-guard-rank x "
                                                "its running maximum or the smallest per-dimension std below "
                                                "--jepa-guard-std x its running median at two evaluations in a row, "
                                                "double the --jepa-reg weight at each such evaluation up to "
                                                "--jepa-guard-cap x; triggered again at the cap: save and stop with "
                                                "exit code 42)"),
                 jepa_guard_windows: int = typer.Option(8, help="with --jepa-guard: number of fixed monitoring windows"),
                 jepa_guard_from: str = typer.Option("heldout", help="with --jepa-guard: where its fixed windows come "
                                                     "from: heldout (default, as runs before 23 Nov did, so their "
                                                     "comparisons hold) or train (review 23 Nov: the guard steers "
                                                     "training, so the production launchers pass train)"),
                 jepa_guard_window: int = typer.Option(131072, help="with --jepa-guard: bases per monitoring window"),
                 jepa_guard_rank: float = typer.Option(0.5, help="with --jepa-guard: entropy-rank trigger, a share "
                                                       "of its running maximum"),
                 jepa_guard_std: float = typer.Option(0.05, help="with --jepa-guard: per-dimension std trigger, a "
                                                      "share of the running median of the minimum"),
                 jepa_guard_cap: float = typer.Option(8.0, help="with --jepa-guard on: the largest factor on the "
                                                      "--jepa-reg weight"),
                 start_step: int = typer.Option(0, help="with --init: continue the schedule (curriculum, LR, paired "
                                                "evals) from this step, so the init run's own history is the control"),
                 stop_step: int = typer.Option(0, help="stop after this step (0: the end of the curriculum)"),
                 lr_total: int = typer.Option(0, help="the length of the LR cosine (0: the curriculum's total); with "
                                              "--cooldown-from, the run's original total"),
                 cooldown_from: int = typer.Option(0, help="a longer cooldown: from this step the LR decays by a "
                                                   "second cosine from its value here to zero at the curriculum's end "
                                                   "(resume with a longer last phase, --lr-total = the old total)"),
                 chunk_ratio: float = typer.Option(0.0, help="learned chunking (H-Net): the router keeps ~1 in this many "
                                                   "2**n_down-bp positions as chunk starts (use with --n-down 4)"),
                 ratio_weight: float = typer.Option(0.03, help="with --chunk-ratio: weight of the ratio loss"),
                 chunk_length_feature: bool = typer.Option(False, help="with --chunk-ratio: give the core each chunk's "
                                                           "log-length, so its gates can scale with the span"),
                 epigenome_weight: float = typer.Option(0.0, help="cell-type conditioning: also predict Regulatory Build "
                                                        "activity per 128-bp bin in each epigenome "
                                                        "(genome-labels-extra), at this loss weight"),
                 epigenome_focus: float = typer.Option(1.0, help="with --epigenome-weight: the weight of bins that no "
                                                       "Regulatory Build element covers in the cell-type loss. Only "
                                                       "~6% of bins carry an element, and the rest are all-off for "
                                                       "every epigenome whatever the cell type, so at 1.0 (all bins "
                                                       "alike) ~94% of the loss asks 'is there an element here' "
                                                       "rather than 'which cell types'; ~0.05 balances the two"),
                 annot_oversample: float = typer.Option(0.0, help="share of training windows drawn so they contain an "
                                                        "annotated block (--annot-bits) rather than uniformly over "
                                                        "the training contigs by length (0: uniform, as before)"),
                 annot_bits: str = typer.Option("CDS,exon,promoter,enhancer", help="with --annot-oversample: the "
                                                "genome-labels bits that make a base count as annotated"),
                 ntv3_dir: str = typer.Option("", help="also predict quantitative per-base assay tracks prepared from "
                                              "NTv3's bigwigs: this directory holds tracks/<file_id>.u8, split.u8 and "
                                              "tracks.json laid out like bases.u8"),
                 ntv3_weight: float = typer.Option(1.0, help="with --ntv3-dir: weight of the track loss"),
                 ntv3_assays: str = typer.Option("ATAC-seq,Histone ChIP-seq,total RNA-seq,polyA plus RNA-seq",
                                                 help="with --ntv3-dir: assays to train on, or 'all'. PRO-cap and "
                                                 "eCLIP are ~0.1%% non-zero, so they are left out by default"),
                 ntv3_focus: float = typer.Option(1.0, help="with --ntv3-dir: weight of positions where every selected "
                                                  "track reads zero (1.0: all positions alike, as for --epigenome-focus)"),
                 ntv3_target: str = typer.Option("log1p", help="with --ntv3-dir: the units the track head is trained in. "
                                                 "'log1p' is the metric's own units; 'alphagenome' is AlphaGenome's and "
                                                 "NTv3's scaling (divide by the track's non-zero mean, x**0.75 for "
                                                 "RNA-seq, soft sqrt clip above 10), which the metric inverts before "
                                                 "scoring so the two are comparable"),
                 ntv3_loss: str = typer.Option("mse", help="with --ntv3-dir: 'mse' on the transformed targets, or "
                                               "'poisson': a Poisson term on each track's total plus the multinomial "
                                               "cross-entropy of its profile, on a softplus output (NTv3's and Borzoi's "
                                               "objective; needs --ntv3-target alphagenome and --ntv3-focus 1)"),
                 ntv3_center: float = typer.Option(1.0, help="with --ntv3-dir: train the tracks only on this central "
                                                   "fraction of each window, so every supervised base has context on "
                                                   "both sides (NTv3: 0.375; 1.0: the whole window, as before)"),
                 ntv3_extra: str = typer.Option("", help="with --ntv3-dir: prepared genomes of further species "
                                                "(comma-separated names under data/interim/genome/, each with bases.u8, "
                                                "contigs.parquet and ntv3/ laid out like the human one). Each adds its "
                                                "own track channels, and a window from it trains only those"),
                 ntv3_extra_share: float = typer.Option(0.5, help="with --ntv3-extra: share of training windows drawn "
                                                        "from the further species (among them by the square root of "
                                                        "their bases, so small genomes are not swamped or drowned)"),
                 mlm_weight: float = typer.Option(1.0, help="weight of the masked-base loss in training. 0 makes a "
                                                  "track stage: the training input is left unmasked and only the "
                                                  "other heads train; held-out bits/base are still read on masked "
                                                  "input, and the track scores on unmasked input"),
                 mito: str = typer.Option("", help="also train on this prepared organelle genome (genome-prepare-mito), "
                                          "each sequence tagged with its genetic code table as context (nuclear: 0)"),
                 mito_share: float = typer.Option(0.03, help="with --mito: share of training windows from it"),
                 genome: list[str] = typer.Option([], "--genome", help="repeatable: a further prepared genome for the "
                                                  "masked-base loss, NAME:SHARE[:CONTEXT[:HELD]] - SHARE of all "
                                                  "training windows, CONTEXT id (0 nuclear), HELD its held-out "
                                                  "contigs joined by '+' or 'none' (default: its contigs.parquet "
                                                  "'held_out' column). Its windows train masked bases only; "
                                                  "held-out bits/base are reported per genome (see multigenome.py)"),
                 soft_base_rate: float = typer.Option(0.0, help="training only: at this fraction of the positions "
                                                      "that show a base and are not masked-LM targets, feed 0.5 of "
                                                      "the base and 0.5 of one other base (drawn uniformly) through "
                                                      "the fractional-input path, so the model has seen "
                                                      "heterozygous-like input; targets, loss and evaluation (hard "
                                                      "input) are unchanged. 0: off, no random number drawn"),
                 repeat_channel: bool = typer.Option(False, help="feed repeat.u8 (1 = lower-case, soft-masked base "
                                                     "in the source FASTA; laid out like bases.u8) of each genome "
                                                     "as reserved input channel 0 (needs --n-in-extra >= 1); a "
                                                     "genome without the file feeds zeros. Training only: held-out "
                                                     "numbers are read with the channel zeroed, comparable with "
                                                     "runs that lack it"),
                 repeat_channel_drop: float = typer.Option(0.5, help="with --repeat-channel: fraction of training "
                                                           "windows whose repeat channel is zeroed, so the model "
                                                           "also works without it"),
                 short_sources: str = typer.Option("pad", help="a training window whose contig is shorter than "
                                                   "the window: 'pad' takes it whole and fills the window with N (as "
                                                   "before), 'pack' turns it into several shorter rows of the same "
                                                   "source that together hold about one window of real bases (the "
                                                   "row lengths: packing.py). Evaluation is unchanged either way"),
                 mito_slice: int = typer.Option(0, help="with --mito and --short-sources pack: every training step "
                                                "also trains on a slice of this many bases of mitochondrial "
                                                "genomes, drawn from the --mito training table as one packed "
                                                "window of this length (packing.py rows: a ~16.5 kb animal "
                                                "mitogenome gives 4 rows of 16,384, a mitogenome at least this "
                                                "long one row of it), each sequence with its genetic-code "
                                                "context; --mito then leaves the main window draw (its share "
                                                "there is 0, the other sources keep their relative shares). The "
                                                "slice's loss joins the step's with weight slice / (slice + the "
                                                "step's main-draw bases), the main draw's with the rest; it runs "
                                                "after the main draw's backward (gradients accumulate, one "
                                                "optimizer step) and, with --split-cards, unsplit on the first "
                                                "card. Its draws come from a stream of their own. 0: off (as "
                                                "before)"),
                 split_cards: int = typer.Option(0, help="split each training window over this many cards in ONE "
                                                 "process (CUDA_VISIBLE_DEVICES lists them; 0 or 1: off): the conv "
                                                 "encoder, decoder and per-base heads run on contiguous stretches "
                                                 "of the window (with a halo covering their reach), or on whole "
                                                 "windows when the batch divides evenly, one per card; the core, "
                                                 "cell-type head and JEPA predictor run on the first card over the "
                                                 "whole window. Same outputs up to round-off, no new parameters; "
                                                 "held-out evaluation stays on the first card, unsplit. Single "
                                                 "process: --tokens-per-step >= window x --grad-accum"),
                 split_halo: int = typer.Option(0, help="with --split-cards: bases of context each side of a piece "
                                                "(0: the conv path's reach, genome.split_halo)"),
                 bf16: bool = typer.Option(True, help="bfloat16 autocast (--no-bf16: float32 throughout, for checks)"),
                 bench_grads: Path | None = typer.Option(None, help="with --bench: save the first step's loss and "
                                                         "every parameter's gradient (before clipping) here"),
                 seed: int = 0) -> None:
    """The multi-track model (KronGenome) over whole gap-free contigs: stage 1 masked-base prediction; with
    ``--labels``, stage 2 adds the Ensembl and GENCODE tracks of ``genome-labels`` and writes ``tracks.csv``.
    v2 (``--arch v2``) trains with a short-to-long ``--curriculum`` at a fixed ``--tokens-per-step``; held-out bits
    are always read on windows of the curriculum's last length, from windows and masks drawn by ``seed`` and the
    step alone, so two runs with one seed are scored on the same bases at the same step (paired evals)."""
    run_options = dict(locals())                      # every option as given (the run record; nothing else reads it)
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
    if cooldown_from and not 0 < cooldown_from < steps:
        raise typer.BadParameter(f"--cooldown-from {cooldown_from} must lie inside the curriculum (1..{steps - 1})")
    if lr_total < 0 or (cooldown_from and lr_total and lr_total < cooldown_from):
        raise typer.BadParameter("--lr-total must be 0 or at least --cooldown-from")
    eval_window = eval_window or phases[-1][0]
    if tokens_per_step:          # fail now, not hours in at the phase that cannot run (see phase_at)
        bad = sorted({w for w, _ in phases if tokens_per_step < w * world * grad_accum})
        if bad:
            raise typer.BadParameter(
                f"--tokens-per-step {tokens_per_step} is less than one window of {', '.join(f'{w:,}' for w in bad)} "
                f"over {world} card(s) x --grad-accum {grad_accum}, so those curriculum phases can never run")
    if whole_contigs and (curriculum or rc or mito or bert_mask or grad_accum > 1):
        raise typer.BadParameter("--whole-contigs does not take --curriculum, --rc, --mito, --bert-mask or --grad-accum")
    if any(w % unit for w, _ in phases):
        raise typer.BadParameter(f"curriculum windows must be multiples of {unit}")
    if ntv3_extra and not ntv3_dir:
        raise typer.BadParameter("--ntv3-extra needs --ntv3-dir (the human tracks come first)")
    if mlm_weight < 0 or not 0.0 <= ntv3_extra_share < 1.0:
        raise typer.BadParameter("--mlm-weight must be >= 0 and --ntv3-extra-share in [0, 1)")
    if mlm_weight == 0 and (whole_contigs or jepa_weight > 0 or labels):
        raise typer.BadParameter("--mlm-weight 0 (a track stage on unmasked input) does not take --whole-contigs, "
                                 "--jepa-weight or --labels")
    if not 0.0 <= jepa_block_share <= 0.9:
        raise typer.BadParameter("--jepa-block-share must lie in [0, 0.9]")
    if jepa_block_share > 0 and (jepa_weight <= 0 or whole_contigs or arch == "v3"):
        raise typer.BadParameter("--jepa-block-share needs --jepa-weight > 0 and does not take --whole-contigs or "
                                 "--arch v3")
    if jepa_on not in ("hidden", "all") or jepa_block_heads not in ("exclude", "include"):
        raise typer.BadParameter("--jepa-on must be hidden or all, --jepa-block-heads exclude or include")
    if jepa_pred not in ("mlp", "context") or jepa_loss not in ("cos", "zcos", "zcos-mse") or jepa_pred_layers < 1:
        raise typer.BadParameter("--jepa-pred must be mlp or context, --jepa-loss cos, zcos or zcos-mse, "
                                 "--jepa-pred-layers >= 1")
    if jepa_weight <= 0 and (jepa_pred != "mlp" or jepa_loss != "cos" or jepa_decode_pred or freeze_trunk_until):
        raise typer.BadParameter("--jepa-pred, --jepa-loss, --jepa-decode-pred and --freeze-trunk-until need "
                                 "--jepa-weight > 0")
    if jepa_edits < 0 or jepa_edit_every < 1 or (jepa_edits > 0 and (jepa_block_share <= 0 or jepa_edit_window % unit
                                                                      or jepa_edit_window <= 0)):
        raise typer.BadParameter(f"--jepa-edits needs --jepa-block-share > 0 and --jepa-edit-window a positive "
                                 f"multiple of {unit}; --jepa-edit-every >= 1")
    if cf_weight < 0 or cf_batch < 1 or cf_every < 1 or cf_temp <= 0 or (cf_weight > 0 and not cf_pairs):
        raise typer.BadParameter("--cf-weight >= 0 (and > 0 needs --cf-pairs), --cf-batch and --cf-every >= 1, "
                                 "--cf-temp > 0")
    if cf_weight > 0 and (not xhead or cf_window % unit or cf_window <= 0):
        raise typer.BadParameter(f"--cf-pairs needs --xhead heads and --cf-window a positive multiple of {unit}")
    if jepa_decode_pred and (jepa_block_share <= 0 or jepa_block_heads != "include"):
        raise typer.BadParameter("--jepa-decode-pred needs --jepa-block-share > 0 and --jepa-block-heads include "
                                 "(the heads must be scored inside the blocks for the decoded states to train)")
    if jepa_log_every < 0 or (jepa_log_every > 0 and jepa_weight <= 0):
        raise typer.BadParameter("--jepa-log-every must be >= 0 and needs --jepa-weight > 0")
    if jepa_block_share > 0:
        try:
            block_lo, block_hi = G.parse_block_len(jepa_block_len, unit)
        except ValueError as err:
            raise typer.BadParameter(f"--jepa-block-len: {err}") from None
    if jepa_guard_from not in ("train", "heldout"):
        raise typer.BadParameter("--jepa-guard-from must be train or heldout")
    if jepa_guard not in ("off", "monitor", "on"):
        raise typer.BadParameter("--jepa-guard must be off, monitor or on")
    if jepa_guard != "off":
        if jepa_weight <= 0 or whole_contigs or arch == "v3":
            raise typer.BadParameter("--jepa-guard needs --jepa-weight > 0 and does not take --whole-contigs or "
                                     "--arch v3")
        if jepa_guard == "on" and jepa_reg == "none":
            raise typer.BadParameter("--jepa-guard on raises the --jepa-reg weight: it needs --jepa-reg sigreg or vicreg")
        if (jepa_guard_windows < 1 or jepa_guard_window <= 0 or jepa_guard_window % unit
                or not (0 < jepa_guard_rank < 1 and 0 < jepa_guard_std < 1) or jepa_guard_cap < 2):
            raise typer.BadParameter(f"--jepa-guard-windows must be >= 1, --jepa-guard-window a positive multiple of "
                                     f"{unit}, --jepa-guard-rank and --jepa-guard-std in (0, 1), --jepa-guard-cap >= 2")
        try:
            guard_lo, guard_hi = G.parse_block_len(jepa_block_len, unit)
        except ValueError as err:
            raise typer.BadParameter(f"--jepa-block-len: {err}") from None
    guard = G.jepa_guard_new()          # --jepa-guard state; with off its factor stays 1 and the loss is as before
    guard_saved = None                  # the state a resumed checkpoint carries
    jnorm_saved = None                  # --jepa-loss zcos: the running target statistics a resumed checkpoint carries
    teacher_saved, resumed = None, False
    xstage_saved = None
    jepa_log = main and jepa_weight > 0 and (jepa_block_share > 0 or jepa_log_every > 0 or jepa_guard != "off")
    if not (0.0 <= soft_base_rate <= 1.0 and 0.0 <= repeat_channel_drop <= 1.0):
        raise typer.BadParameter("--soft-base-rate and --repeat-channel-drop must lie in [0, 1]")
    if whole_contigs and (soft_base_rate > 0 or repeat_channel):
        raise typer.BadParameter("--whole-contigs does not take --soft-base-rate or --repeat-channel")
    if repeat_channel and n_in_extra < 1:
        raise typer.BadParameter("--repeat-channel feeds reserved input channel 0: it needs --n-in-extra >= 1")
    if short_sources not in ("pad", "pack"):
        raise typer.BadParameter("--short-sources must be 'pad' or 'pack'")
    if short_sources == "pack" and whole_contigs:
        raise typer.BadParameter("--whole-contigs does not take --short-sources pack (it reads whole contigs already)")
    if mito_slice < 0:
        raise typer.BadParameter("--mito-slice must be >= 0")
    if mito_slice > 0:
        if not mito or short_sources != "pack" or whole_contigs:
            raise typer.BadParameter("--mito-slice needs --mito and --short-sources pack, and does not take "
                                     "--whole-contigs")
        if world > 1:
            raise typer.BadParameter("--mito-slice is single-process only (one process, or --split-cards): with "
                                     "torchrun each card would run a second backward per step")
        if not 0.0 < mito_share < 1.0:
            raise typer.BadParameter("--mito-slice takes --mito out of the main draw by its --mito-share: that "
                                     "share must lie in (0, 1)")
        slice_unit_ = math.lcm(unit, 128 if arch == "v3" else unit)
        if mito_slice % slice_unit_ or mito_slice < 8 * slice_unit_:
            raise typer.BadParameter(f"--mito-slice must be a multiple of {slice_unit_} and at least "
                                     f"{8 * slice_unit_:,} bases")
    if split_cards > 1:
        bad_ = [n_ for n_, on_ in (("--arch v1/v3 (v1's GroupNorm takes whole-window statistics)",
                                    arch not in ("v2", "v2l")),
                                   ("--whole-contigs", whole_contigs), ("--chunk-ratio", chunk_ratio > 0),
                                   ("--labels", labels),
                                   ("--compile-convs", compile_convs), ("torchrun (WORLD_SIZE > 1)", world > 1))
                if on_]
        if bad_:
            raise typer.BadParameter("--split-cards does not take " + ", ".join(bad_))
        if split_halo % unit or split_halo < 0:
            raise typer.BadParameter(f"--split-halo must be a non-negative multiple of {unit}")
    if bench_grads is not None and not bench:
        raise typer.BadParameter("--bench-grads needs --bench")
    if new_contexts not in ("zero", "nuclear"):
        raise typer.BadParameter("--new-contexts must be 'zero' or 'nuclear'")
    if new_contexts != "zero" and init is None:
        raise typer.BadParameter("--new-contexts applies to --init only: from scratch every context row starts at "
                                 "zero, as row 0 (nuclear) does")
    if mito:
        mito_src = interim_dir("genome", mito)
        mito_bases = np.memmap(mito_src / "bases.u8", dtype=np.uint8, mode="r")
        mito_idx = pl.read_parquet(mito_src / "contigs.parquet").with_columns((pl.col("end") - pl.col("start")).alias("len"))
        mito_train, mito_val = mito_idx.filter(~pl.col("held_out")), mito_idx.filter(pl.col("held_out"))
        say(f"mito: {mito_train.height} train / {mito_val.height} held-out genomes "
                   f"({int(mito_train['len'].sum()):,} / {int(mito_val['len'].sum()):,} bases), share {mito_share}")

    # --genome: further genomes for the masked-base loss, each with its share, context id and held-out contigs.
    # Their shares are fractions of ALL training windows, and so is --mito's once they are given (it is then drawn
    # with mito_share / (1 - their sum) among the rest); without --genome no extra draw is made, so the default
    # window stream is the one before this option existed.
    from kron_dna import multigenome as MG
    try:
        if genome and whole_contigs:
            raise MG.GenomeSpecError("--whole-contigs does not take --genome")
        genome_specs = MG.parse_genome_specs(genome, primary=name, other_share=mito_share if mito else 0.0)
        genomes = MG.load_further_genomes(genome_specs, lambda n_: interim_dir("genome", n_))
    except MG.GenomeSpecError as err:
        raise typer.BadParameter(str(err)) from None
    genome_total = float(sum(g_.share for g_ in genomes))
    genome_cum = MG.cumulative_shares(genomes)
    mito_codes = {int(c_) for c_ in mito_idx["code"].unique().to_list()} if mito else set()
    for g_ in genomes:
        longest_ = int(g_.train["len"].max())
        say(f"genome {g_.name}: share {g_.share:g}, context {g_.context}, {g_.train.height} train / "
            f"{g_.val.height} held-out contigs ({int(g_.train['len'].sum()):,} / {int(g_.val['len'].sum()):,} bases)"
            + (f", held out: {'+'.join(g_.val['chrom'].unique(maintain_order=True).to_list())}" if g_.val.height
               else ", no held-out set (no held-out bits/base)")
            + (f"; longest training contig {longest_:,} < window {max(w for w, _ in phases):,}: "
               + ("read in shorter rows (--short-sources pack)" if short_sources == "pack" else "taken whole and "
                  "padded") if longest_ < max(w for w, _ in phases) else ""))

    if epigenome_weight > 0:
        regfeat = np.memmap(src / "regfeat.i32", dtype=np.int32, mode="r")
        reg_active = np.load(src / "regfeat_active.npy")                  # [features, epigenomes] bool
        n_epi = reg_active.shape[1]
        say(f"cell-type conditioning: {n_epi} epigenomes, {reg_active.shape[0]:,} regulatory elements")
    epi_batch: list = []                                                  # the last sample()'s bin targets
    x_read: dict = {"heads": None}    # --xhead-cycle: the heads whose stores sample() reads now (None: every head)
    epi_unit = 128 if arch == "v3" else unit                               # v3 pools its 1-bp states per 128 bp

    annot_pos = annot_coff = annot_clen = None
    if annot_oversample > 0:                   # oversample annotated sequence (Enformer/Borzoi sample gene-rich loci)
        want_bits = [b.strip() for b in annot_bits.split(",") if b.strip()]
        unknown = sorted(set(want_bits) - set(G.LABEL_BITS))
        if unknown or not want_bits:
            raise typer.BadParameter(f"unknown label bits {unknown or annot_bits!r}; have {list(G.LABEL_BITS)}")
        bit_mask = sum(1 << G.LABEL_BITS[b] for b in want_bits)
        if not (src / "labels.u8").exists():
            raise typer.BadParameter(f"--annot-oversample needs {src / 'labels.u8'} (run genome-labels first)")
        block = 1024                           # a block counts as annotated when >= 1/8 of its bases carry a bit
        cache = src / f"annot_blocks_m{bit_mask}_b{block}.npy"
        if cache.exists():
            blocks = np.load(cache)
        else:
            lab_all = np.memmap(src / "labels.u8", dtype=np.uint8, mode="r")
            found, piece_n = [], (1 << 26) // block * block
            for a in range(0, len(lab_all) // block * block, piece_n):
                piece = np.asarray(lab_all[a:a + piece_n])
                piece = piece[: len(piece) // block * block].reshape(-1, block)
                found.append(a + np.flatnonzero(((piece & bit_mask) != 0).sum(1) >= block // 8) * block)
            blocks = np.concatenate(found).astype(np.int64)
            np.save(cache, blocks)
        t_off = train_idx["offset"].to_numpy()
        t_len = train_idx["len"].to_numpy()
        order = np.argsort(t_off)                                          # searchsorted needs sorted offsets
        t_off, t_len = t_off[order], t_len[order]
        j = np.searchsorted(t_off, blocks, side="right") - 1
        inside = (j >= 0) & (blocks < t_off[np.maximum(j, 0)] + t_len[np.maximum(j, 0)])
        annot_pos, j = blocks[inside], j[inside]                           # training contigs only
        annot_coff, annot_clen = t_off[j], t_len[j]
        say(f"annotation oversampling: {annot_bits} marks {len(annot_pos):,} blocks of {block} bp on training contigs "
            f"({100 * len(annot_pos) * block / max(float(t_len.sum()), 1):.1f}% of their bases), "
            f"share {annot_oversample}")

    ntv3 = None
    ntv3_batch: list = []
    # --xhead: further heads, each on its own measured-track store over the primary genome (kron_dna.xheads)
    from kron_dna import xheads as XH
    xstores: dict = {}                                # head -> its store (a head registered without data has none)
    xhead_sizes: dict[str, int] = {}                  # head -> outputs, in registration order
    for spec_x in xhead:
        name_x, _, val_x = spec_x.partition(":")
        if not name_x.isidentifier() or not val_x or name_x in xhead_sizes:
            raise typer.BadParameter(f"--xhead {spec_x!r}: give NAME:STORE or NAME:N, each name once")
        if val_x.isdigit():
            xhead_sizes[name_x] = int(val_x)
            continue
        try:
            xstores[name_x] = XH.open_store(val_x)
        except (OSError, ValueError, KeyError) as err:
            raise typer.BadParameter(f"--xhead {name_x}: {err}") from None
        xhead_sizes[name_x] = len(xstores[name_x].ids)
        covers_ = (xstores[name_x].total_bases if xstores[name_x].kind == "tracks_binned"
                   else xstores[name_x].dense.shape[0])
        if covers_ != len(bases):
            raise typer.BadParameter(f"--xhead {name_x}: the store covers {covers_:,} bases, "
                                     f"the primary genome {name} has {len(bases):,}")
    x_species: dict[str, set] = {}                    # --xhead-species: genome -> the heads that read it
    if xhead_species:
        by_head: dict = {}
        g_len = {g_.name: len(g_.bases) for g_ in genomes}
        for spec_s in xhead_species:
            name_s, _, rest_s = spec_s.partition("=")
            g_s, _, path_s = rest_s.partition(":")
            if name_s not in xstores or not g_s or not path_s:
                raise typer.BadParameter(f"--xhead-species {spec_s!r}: give NAME=GENOME:STORE for a NAME of --xhead "
                                         "with a store")
            if g_s not in g_len:
                raise typer.BadParameter(f"--xhead-species {spec_s!r}: genome {g_s} is not a --genome of this run "
                                         f"({sorted(g_len)})")
            try:
                st_s = XH.open_store(path_s)
            except (OSError, ValueError, KeyError) as err:
                raise typer.BadParameter(f"--xhead-species {name_s}={g_s}: {err}") from None
            cov_s = st_s.total_bases if st_s.kind == "tracks_binned" else st_s.dense.shape[0]
            if cov_s != g_len[g_s]:
                raise typer.BadParameter(f"--xhead-species {name_s}={g_s}: the store covers {cov_s:,} bases, the "
                                         f"genome has {g_len[g_s]:,}")
            if g_s in by_head.setdefault(name_s, {}):
                raise typer.BadParameter(f"--xhead-species {name_s}={g_s}: given twice")
            by_head[name_s][g_s] = st_s
            x_species.setdefault(g_s, set()).add(name_s)
        for name_s, extra_s in by_head.items():
            try:
                xstores[name_s] = XH.SpeciesStores(xstores[name_s], extra_s)
            except ValueError as err:
                raise typer.BadParameter(f"--xhead-species {name_s}: {err}") from None
            xhead_sizes[name_s] = len(xstores[name_s].ids)
            say(f"--xhead-species {name_s}: {len(xstores[name_s].primary.ids)} {name} channels + "
                + ", ".join(f"{g_}: {len(st_.ids)} ({xstores[name_s].shared[g_]} shared)" for g_, st_ in extra_s.items())
                + f" = {len(xstores[name_s].ids)} outputs")

    x_gsplit: dict = {}                               # --xhead-genome-split: genome -> its per-base split
    for spec_g in xhead_genome_split:
        g_s, _, path_s = spec_g.partition("=")
        if g_s not in x_species or not path_s:
            raise typer.BadParameter(f"--xhead-genome-split {spec_g!r}: give GENOME=SPLIT for a genome of "
                                     f"--xhead-species ({sorted(x_species)})")
        x_gsplit[g_s] = np.memmap(path_s, dtype=np.uint8, mode="r")
        n_g = len(next(g_ for g_ in genomes if g_.name == g_s).bases)
        if len(x_gsplit[g_s]) != n_g:
            raise typer.BadParameter(f"--xhead-genome-split {g_s}: {len(x_gsplit[g_s]):,} bases, the genome has {n_g:,}")
        held_ = float((np.asarray(x_gsplit[g_s][::1000]) == 4).mean())       # every 1,000th base: an estimate
        say(f"--xhead-genome-split {g_s}: {held_:.1%} of its bases held out")

    def x_mask(st_, g_name):
        """The channels of head store ``st_`` a row of genome ``g_name`` supervises (None: the primary genome)."""
        if isinstance(st_, XH.SpeciesStores):
            return st_.mask(g_name) if g_name in st_.cols else np.zeros(len(st_.ids), dtype=bool)
        return np.full(len(st_.ids), g_name is None, dtype=bool)

    try:                                              # --xhead-cycle: the head groups trained in turn
        x_groups = XH.parse_cycle(xhead_cycle, list(xhead_sizes)) if xhead_cycle else []
    except ValueError as e_:
        raise typer.BadParameter(str(e_))
    x_scale = {n_: len(x_groups) / sum(n_ in g_ for g_ in x_groups) for n_ in xhead_sizes} if x_groups else {}
    x_bin: dict[str, int] = {}                        # head -> bases per output (1: per base)
    for spec_b in xhead_bin:
        name_b, _, val_b = spec_b.partition("=")
        if name_b not in xhead_sizes or not val_b.isdigit():
            raise typer.BadParameter(f"--xhead-bin {spec_b!r}: give NAME=B for a NAME of --xhead")
        if name_b in xstores and xstores[name_b].kind == "binary":
            raise typer.BadParameter(f"--xhead-bin {name_b}: a binary head predicts per base")
        x_bin[name_b] = int(val_b)
    for n_b, st_b in xstores.items():               # a pre-binned store sets its head's bin
        if st_b.kind == "tracks_binned":
            if x_bin.get(n_b, st_b.bin) != st_b.bin:
                raise typer.BadParameter(f"--xhead-bin {n_b}={x_bin[n_b]}: its store is binned at {st_b.bin} bases")
            x_bin[n_b] = st_b.bin
    # primary-genome windows start on the coarsest pre-binned store's grid (cells are global, a window must match)
    x_grid = max([st_b.bin for st_b in xstores.values() if st_b.kind == "tracks_binned"], default=1)

    def on_grid(at_: int, take_: int, off_: int, len_: int) -> tuple[int, int]:
        """``at_``/``take_`` moved onto the ``x_grid`` grid inside the contig ``[off_, off_ + len_)``: the start down
        (or up, if that leaves the contig) to a multiple of the grid, the length cut to whole cells. A contig too
        short for one cell keeps its window; its cells are then all padding, never supervised."""
        if x_grid == 1:
            return at_, take_
        a_ = at_ - at_ % x_grid
        if a_ < off_:
            a_ += x_grid
        t_ = min(take_, off_ + len_ - a_)
        t_ -= t_ % x_grid
        return (a_, t_) if t_ > 0 else (at_, take_)
    if xhead:
        if not (ntv3_dir or xhead_split) or whole_contigs or world > 1 or arch not in ("v2", "v2l"):
            raise typer.BadParameter("--xhead needs --ntv3-dir or --xhead-split (the training split), arch v2 or v2l, "
                                     "windows (not --whole-contigs) and one process (--split-cards, not "
                                     "data-parallel cards)")
        xsplit = np.memmap(xhead_split or Path(ntv3_dir) / "split.u8", dtype=np.uint8, mode="r")
        if len(xsplit) != len(bases):
            raise typer.BadParameter(f"--xhead: the split covers {len(xsplit):,} bases, the primary genome {name} has "
                                     f"{len(bases):,}")
        for n_, s_ in xstores.items():
            lone_ = [i_ for k_, i_ in enumerate(s_.ids) if i_.endswith(("_P", "_M")) and s_.swap[k_] == k_]
            if rc and lone_:
                raise typer.BadParameter(f"--rc: stranded tracks of --xhead {n_} have no partner: {lone_[:6]}")
    if ntv3_dir:
        import json as _json
        from pathlib import Path as _Path
        nd = _Path(ntv3_dir)
        want_assays = None if ntv3_assays.strip() == "all" else {a.strip() for a in ntv3_assays.split(",")}
        sel = [t for t in _json.load(open(nd / "tracks.json"))["tracks"]
               if want_assays is None or t["assay"] in want_assays]
        if not sel:
            raise typer.BadParameter(f"--ntv3-assays {ntv3_assays!r} selected no track")
        ids = [t["file_id"] for t in sel]
        at_pos = {i: k for k, i in enumerate(ids)}
        # reverse complement must SWAP each _P track with its _M partner, not merely reverse it: these assays are
        # strand-specific, unlike the Regulatory Build bits, so reversing alone teaches the wrong strand.
        swap = np.arange(len(ids))
        for k, i in enumerate(ids):
            if i.endswith(("_P", "_M")):
                other = i[:-1] + ("M" if i.endswith("_P") else "P")
                if other in at_pos:
                    swap[k] = at_pos[other]
        paired = int((swap != np.arange(len(ids))).sum())
        orphans = [i for i in ids if i.endswith(("_P", "_M"))
                   and (i[:-1] + ("M" if i.endswith("_P") else "P")) not in at_pos]
        if rc and orphans:      # reversing a stranded track without its partner puts minus signal on the plus strand
            raise typer.BadParameter(
                f"--rc needs both strands of every stranded NTv3 track, but {len(orphans)} have no selected "
                f"partner: {orphans[:6]}{'...' if len(orphans) > 6 else ''}. Widen --ntv3-assays or drop --rc.")
        # One base-major [TOTAL, T] array makes a training window a single contiguous read; per-track files
        # cost 14 random 8 kB reads per window and measured 7.9 s/step against 0.7 s without tracks.
        dense = dense_cols = None
        if (nd / "dense.json").exists():
            dj = _json.load(open(nd / "dense.json"))
            if set(ids) <= set(dj["ids"]):          # dense may hold a superset: take the columns we want
                dense = np.memmap(nd / "dense.u8", dtype=np.uint8, mode="r",
                                  shape=(int(dj["total_bases"]), int(dj["n_tracks"])))
                where = {i: k for k, i in enumerate(dj["ids"])}
                dense_cols = np.array([where[i] for i in ids])
                if "nonzero_mean" in dj:            # keep the stats aligned with the chosen columns
                    dj = dict(dj, nonzero_mean=[dj["nonzero_mean"][k] for k in dense_cols])
        if ntv3_target not in ("log1p", "alphagenome"):
            raise typer.BadParameter("--ntv3-target must be 'log1p' or 'alphagenome'")
        if ntv3_loss not in ("mse", "poisson"):
            raise typer.BadParameter("--ntv3-loss must be 'mse' or 'poisson'")
        if ntv3_loss == "poisson" and (ntv3_target != "alphagenome" or ntv3_focus != 1.0):
            raise typer.BadParameter("--ntv3-loss poisson needs --ntv3-target alphagenome (non-negative targets) and "
                                     "--ntv3-focus 1 (its mask is 0/1; positions are not re-weighted)")
        if not 0.0 < ntv3_center <= 1.0:
            raise typer.BadParameter("--ntv3-center must be in (0, 1]")
        nzmean = None
        if ntv3_target == "alphagenome":
            if dense is None or "nonzero_mean" not in dj:
                raise typer.BadParameter("--ntv3-target alphagenome needs dense.json with 'nonzero_mean'")
            nzmean = np.array(dj["nonzero_mean"], dtype=np.float32)
        ntv3 = dict(ids=ids, swap=swap, dense=dense, cols=dense_cols, nzmean=nzmean,
                    is_rna=np.array(["RNA-seq" in t["assay"] for t in sel], dtype=bool),
                    arrs=None if dense is not None else
                         [np.memmap(nd / "tracks" / f"{i}.u8", dtype=np.uint8, mode="r") for i in ids],
                    split=np.memmap(nd / "split.u8", dtype=np.uint8, mode="r"),
                    scales=np.array([t["log1p_scale"] for t in sel], dtype=np.float32))
        ntv3["n_h"] = len(ids)                                  # the human channels come first
        ntv3["extra"] = []
        for xname in [x.strip() for x in ntv3_extra.split(",") if x.strip()]:
            xsrc = interim_dir("genome", xname)
            xj = _json.load(open(xsrc / "ntv3" / "dense.json"))
            if ntv3_target == "alphagenome" and "nonzero_mean" not in xj:
                raise typer.BadParameter(f"{xsrc / 'ntv3' / 'dense.json'} has no 'nonzero_mean'")
            xids = list(xj["ids"])
            xat = {i_: k_ for k_, i_ in enumerate(xids)}
            xswap = np.arange(len(xids))
            for k_, i_ in enumerate(xids):
                if i_.endswith(("_P", "_M")):
                    other = i_[:-1] + ("M" if i_.endswith("_P") else "P")
                    if other not in xat and rc:
                        raise typer.BadParameter(f"--rc: stranded track {i_} of {xname} has no partner")
                    xswap[k_] = xat.get(other, k_)
            xt = pl.read_parquet(xsrc / "contigs.parquet").with_columns((pl.col("end") - pl.col("start")).alias("len"))
            xsplit = np.memmap(xsrc / "ntv3" / "split.u8", dtype=np.uint8, mode="r")
            # held-out test positions, for evals that land on them: every 1,024th base of the test split, with the
            # contig each lies in (windows are placed inside one contig, as everywhere else)
            tpos = np.flatnonzero(np.asarray(xsplit[::1024]) == 3).astype(np.int64) * 1024
            x_off, x_len = xt["offset"].to_numpy(), xt["len"].to_numpy()
            order = np.argsort(x_off)
            x_off, x_len = x_off[order], x_len[order]
            jx = np.searchsorted(x_off, tpos, side="right") - 1
            inside = (jx >= 0) & (tpos < x_off[np.maximum(jx, 0)] + x_len[np.maximum(jx, 0)])
            lo_ = len(ntv3["ids"])
            ntv3["extra"].append(dict(
                name=xname, lo=lo_, hi=lo_ + len(xids), table=xt, p=x_len[np.argsort(order)] / x_len.sum(),
                bases=np.memmap(xsrc / "bases.u8", dtype=np.uint8, mode="r"), split=xsplit,
                dense=np.memmap(xsrc / "ntv3" / "dense.u8", dtype=np.uint8, mode="r",
                                shape=(int(xj["total_bases"]), int(xj["n_tracks"]))),
                test_pos=tpos[inside], test_off=x_off[jx[inside]], test_len=x_len[jx[inside]],
                train_bases=float(x_len.sum())))
            ntv3["ids"] = ntv3["ids"] + [f"{xname}:{i_}" for i_ in xids]
            ntv3["swap"] = np.concatenate([ntv3["swap"], lo_ + xswap])
            ntv3["scales"] = np.concatenate([ntv3["scales"], np.array(xj["scales"], dtype=np.float32)])
            ntv3["is_rna"] = np.concatenate([ntv3["is_rna"], np.array(["RNA-seq" in a for a in xj["assays"]])])
            if ntv3["nzmean"] is not None:
                ntv3["nzmean"] = np.concatenate([ntv3["nzmean"], np.array(xj["nonzero_mean"], dtype=np.float32)])
        if ntv3["extra"]:
            w_ = np.sqrt(np.array([e["train_bases"] for e in ntv3["extra"]]))
            ntv3["extra_p"] = w_ / w_.sum()
            say("NTv3 tracks, further species: " + ", ".join(
                f"{e['name']} {e['hi'] - e['lo']} tracks (share of all training windows "
                f"{(1.0 - genome_total - (mito_share if mito else 0.0)) * ntv3_extra_share * p_:.4f}, "
                f"{len(e['test_pos']):,} test anchors)" for e, p_ in zip(ntv3["extra"], ntv3["extra_p"]))
                + f"; {len(ntv3['ids'])} channels in all")
        by_assay = {}
        for t in sel:
            by_assay[t["assay"]] = by_assay.get(t["assay"], 0) + 1
        say(f"NTv3 tracks: {len(ids)} selected ({by_assay}), {paired // 2} stranded pairs swapped under --rc, "
            f"loss {ntv3_loss} on split==1 only" + (f", central {ntv3_center:g} of each window" if ntv3_center < 1 else "")
            + f", target {ntv3_target}"
            + (", base-major layout" if dense is not None else ", per-track files"))
        for k_x, e_x in enumerate(ntv3["extra"]):       # also a --genome: its context, and its held-out contigs kept
            g_x = next((g_ for g_ in genomes if g_.name == e_x["name"]), None)   # out of its track windows too
            if g_x is not None:
                e_x["table"], e_x["p"], e_x["context"] = g_x.train, g_x.train_p, g_x.context
                rest_ = 1.0 - genome_total - (mito_share if mito else 0.0)
                say(f"genome {g_x.name} is also an --ntv3-extra species: window share {g_x.share:g} (--genome, "
                    f"masked bases only) + {rest_ * ntv3_extra_share * ntv3['extra_p'][k_x]:.4f} (--ntv3-extra, "
                    f"masked bases and its tracks), context {g_x.context} for both, its held-out contigs in neither")

    # --repeat-channel: per-base repeat.u8 (1 = soft-masked) beside each genome's bases.u8, read only for training
    # windows; a genome without the file feeds zeros. The per-window drop draws from a stream of its own, so the
    # windows (and everything else drawn from rng) are the ones a run without --repeat-channel draws.
    rep_main = rep_mito = None
    rep_genome: list = [None] * len(genomes)
    rep_batch: list = []
    rep_rng = np.random.default_rng([seed, rank, 2027])
    if repeat_channel:
        def open_repeat(d, n_bases: int, label: str):
            f_ = d / "repeat.u8"
            if not f_.is_file():
                say(f"repeat channel: {label} has no {f_.name}: its windows feed zeros")
                return None
            mm_ = np.memmap(f_, dtype=np.uint8, mode="r")
            if len(mm_) != n_bases:
                raise typer.BadParameter(f"{f_} holds {len(mm_):,} values but bases.u8 {n_bases:,}")
            probe_ = np.asarray(mm_[:: max(1, len(mm_) // 2_000_000)])
            say(f"repeat channel: {label} repeat.u8, ~{100 * float((probe_ > 0).mean()):.1f}% soft-masked")
            return mm_
        rep_main = open_repeat(src, len(bases), name)
        if mito:
            rep_mito = open_repeat(mito_src, len(mito_bases), mito)
        rep_genome = [open_repeat(interim_dir("genome", g_.name), len(g_.bases), g_.name) for g_ in genomes]
        for e_x in (ntv3["extra"] if ntv3 else []):
            g_k = next((k_ for k_, g_ in enumerate(genomes) if g_.name == e_x["name"]), None)
            e_x["rep"] = rep_genome[g_k] if g_k is not None else open_repeat(
                interim_dir("genome", e_x["name"]), len(e_x["bases"]), e_x["name"])
        say(f"repeat channel: extra input channel 0, zeroed in {repeat_channel_drop:g} of training windows; "
            f"held-out evaluation with the channel zeroed")

    def epi_targets(at: int, take: int, window: int, flip: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``[bins, E]`` bool: an epigenome is active in a 128-bp bin when an element active there covers a base of
        it; ``[bins]`` valid (inside the window's bases); and ``[bins]`` bool: a Regulatory Build element covers a
        base of the bin at all, which is where the cell-type question is actually asked (--epigenome-focus)."""
        ids = np.full(window, -1, dtype=np.int64)
        ids[:take] = regfeat[at:at + take]
        if flip:
            ids[:take] = ids[:take][::-1]
        ids = ids.reshape(-1, epi_unit)
        hit = np.zeros((ids.shape[0], n_epi), dtype=bool)
        covered = ids >= 0
        rows, cols = np.nonzero(covered)
        if len(rows):
            np.logical_or.at(hit, rows, reg_active[ids[rows, cols]])
        return hit, np.arange(window // epi_unit) < (take + epi_unit - 1) // epi_unit, covered.any(1)

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

    # --short-sources pack (packing.py): a training window whose contig is shorter than the window becomes a pack of
    # shorter rows. pack_cells: each source's contigs by the row length they select, per window; tally: what the
    # training sampler drew since the last report (kept for 'pad' too: it draws no random number).
    from kron_dna import packing as PK
    pack_short = short_sources == "pack"
    pack_unit = math.lcm(unit, epi_unit)
    pack_cells: dict = {}
    tally = PK.Tally()
    if pack_short:
        w_ = max(w for w, _ in phases)
        lad_ = PK.row_ladder(w_, pack_unit)
        say(f"short sources: a training window whose contig is shorter than the window becomes a pack of shorter rows "
            f"of that source (same length bucket) holding about one window of real bases; row lengths x rows at "
            f"window {w_:,}: " + ", ".join(f"{r_}x{n_}" for r_, n_ in lad_[:10])
            + (f", ... {lad_[-1][0]}x{lad_[-1][1]} ({len(lad_)} lengths)" if len(lad_) > 10 else ""))

    def short(key, lengths, p_, length: int):
        """(row length, further rows to draw, their bucket) of the pack that a contig of ``length`` (shorter than
        the window) leads: ``key`` names the source, ``lengths`` and ``p_`` are its table's lengths and draw
        probabilities (None: drawn uniformly)."""
        window_ = int(key[-1])
        if key not in pack_cells:
            pack_cells[key] = PK.Buckets(lengths.to_numpy(), p_, PK.row_ladder(window_, pack_unit))
        cell_ = pack_cells[key].cell(length)
        if not len(cell_.members):    # a bucket with no contig to draw further rows from (D crashed on one, 23 Nov,
            return window_, 0, None   # in gen.choice(0)): this row is not packed - one padded row, as without packing
        return cell_.row, cell_.n - 1, cell_

    def sample(table, window=window, batch=None, gen=None, flip=False, mito_table=None, share=0.0, annot=0.0,
               extra=0.0, extra_force=None, genomes=(), force_genome=None, rep=False, pack=None, cum=None,
               tally_to=None):
        """``batch`` windows: (seqs, segments, contexts[, labels, activity, valid]). With ``mito_table``, each row comes
        from it with probability ``share``; a circular genome is read across its origin when the window is shorter.
        Context 0 is nuclear; an organelle sequence's context is its genetic code table (1..33). With ``genomes``
        (--genome), one uniform draw first sends a row to further genome k with probability its share (its training
        contigs, by length, masked bases only, its context id); ``force_genome`` k draws every row from ``table``
        (that genome's held-out contigs) instead. ``rep`` (--repeat-channel, training only) also reads each row's
        repeat.u8 (zeros where a genome has none, or for a dropped window) into ``rep_batch``. ``cum``
        (--mito-slice): the further genomes' cumulative shares to draw with instead of ``genome_cum``; ``tally_to``:
        the Tally the rows are counted in instead of ``tally``.

        ``pack`` is given by the training loop only (None: evaluation, nothing below applies). True
        (--short-sources pack): a window whose contig is shorter than ``window`` becomes a pack - rows of the longest
        ladder length that contig fills, as many as fit one window, the further ones drawn from the same source's
        contigs of the same length bucket (packing.py) - and the call returns a list of parts, one per row length
        drawn: ``(weight, tensors, (epi, ntv3, repeat targets))``, the weight being the part's share of the
        ``batch`` windows. Every window is still drawn exactly as before (same draws, same order), so the source
        shares are per window as they were; a pack only replaces the padding by more rows of that source."""
        gen = gen or rng
        batch = batch or batch_size
        p = table["len"].to_numpy() / table["len"].sum()                 # contigs by length: every base equally
        seqs, segs, ctxs, labs, acts, valid = [], [], [], [], [], []
        epi_t, epi_v, epi_e = [], [], []
        nt_t, nt_s, nt_c, nt_x = [], [], [], []
        xh_t = {n_: [] for n_ in xstores}            # --xhead: each head's stored targets per row
        xh_m = {n_: [] for n_ in xstores}            # and the channels each row supervises
        rep_rows = []
        row_len, row_wt = [], []                     # per row: its length, and its share of one window
        left, more, cell, wrow = batch, 0, None, window   # windows to draw; rows left in the pack being filled, their
        while left or more:                               # bucket and length (cell None, wrow == window: a full row)
            rsrc = None                              # --repeat-channel: this row's repeat.u8, read at rsel
            if more:                                 # a further row of the pack: the source its first row drew
                more -= 1
            else:
                left -= 1
                cell, wrow = None, window
                fg = force_genome                    # a further genome's index (--genome), else None
                if fg is None and genomes:           # no draw at all without --genome: the default stream is unchanged
                    fg = MG.pick_genome(gen.random(), genome_cum if cum is None else cum)
                from_mito = fg is None and mito_table is not None and gen.random() < share
                xs = None                            # a further species' index (--ntv3-extra), else None
                if extra_force is not None:
                    xs = extra_force
                elif fg is None and not from_mito and extra > 0 and gen.random() < extra:
                    xs = int(gen.choice(len(ntv3["extra"]), p=ntv3["extra_p"]))
            if fg is not None:
                g_ = genomes[fg]
                tab_, p_ = (table, p) if force_genome is not None else (g_.train, g_.train_p)
                r = tab_.row(cell.draw(gen) if cell else int(gen.choice(tab_.height, p=p_)), named=True)
                if pack and cell is None and r["len"] < window:
                    wrow, more, cell = short(("genome", fg, window), tab_["len"], p_, r["len"])
                take = min(wrow, r["len"])
                at = r["offset"] + int(gen.integers(0, r["len"] - take + 1))
                if g_.name in x_species:             # its windows start on the head stores' grid, as the primary's
                    at, take = on_grid(at, take, r["offset"], r["len"])
                codes = np.full(wrow, 4, dtype=np.int64)
                codes[:take] = XH.read_rows(g_.bases, at, at + take)          # pread, not the mapping
                if force_genome is None and g_.name in x_gsplit:   # training rows: its held-out bases (orthologs of
                    codes[:take][XH.read_rows(x_gsplit[g_.name], at, at + take) == 4] = 4   # the primary's held-out) read as N: no
                    # input, no masked-base target, no label - as if cut out; its held-out draws keep them for scoring
                ctxs.append(int(g_.context))
                rsrc, rsel = rep_genome[fg], slice(at, at + take)
            elif xs is not None:
                e_ = ntv3["extra"][xs]
                if extra_force is not None and len(e_["test_pos"]):       # an eval window around a test position
                    k = int(gen.integers(0, len(e_["test_pos"])))
                    c_off, c_len = int(e_["test_off"][k]), int(e_["test_len"][k])
                    take = min(window, c_len)
                    at = min(max(c_off, int(e_["test_pos"][k]) - take // 2), c_off + c_len - take)
                else:
                    r = e_["table"].row(cell.draw(gen) if cell else int(gen.choice(e_["table"].height, p=e_["p"])),
                                        named=True)
                    if pack and cell is None and r["len"] < window:
                        wrow, more, cell = short(("ntv3_extra", xs, window), e_["table"]["len"], e_["p"], r["len"])
                    take = min(wrow, r["len"])
                    at = r["offset"] + int(gen.integers(0, r["len"] - take + 1))
                codes = np.full(wrow, 4, dtype=np.int64)
                codes[:take] = e_["bases"][at:at + take]
                ctxs.append(int(e_.get("context", 0)))                    # nuclear (0) unless --genome gave one
                rsrc, rsel = e_.get("rep"), slice(at, at + take)
            elif from_mito:
                r = mito_table.row(cell.draw(gen) if cell else int(gen.integers(0, mito_table.height)),
                                   named=True)                            # each genome alike
                if pack and cell is None and r["len"] < window:
                    wrow, more, cell = short(("mito", window), mito_table["len"], None, r["len"])
                take = min(wrow, r["len"])
                if r["circular"]:                                         # read across the origin
                    pos = (int(gen.integers(0, r["len"])) + np.arange(take)) % r["len"]
                else:
                    pos = int(gen.integers(0, r["len"] - take + 1)) + np.arange(take)
                codes = np.full(wrow, 4, dtype=np.int64)
                codes[:take] = mito_bases[r["offset"] + pos]
                ctxs.append(int(r["code"]))                               # organelle: its genetic code table
                rsrc, rsel = rep_mito, r["offset"] + pos
            elif cell is None and annot and annot_pos is not None and gen.random() < annot:
                k = int(gen.integers(0, len(annot_pos)))                  # a window that contains an annotated block
                c_off, c_len = int(annot_coff[k]), int(annot_clen[k])
                if pack and c_len < window:                               # its pack's further rows: primary contigs
                    wrow, more, cell = short(("primary", window), table["len"], p, c_len)   # of its bucket, by length
                take = min(wrow, c_len)
                lo = max(c_off, int(annot_pos[k]) - take + 1)             # the block stays inside the window,
                hi = min(c_off + c_len - take, int(annot_pos[k]))         # the window inside its contig
                at = int(gen.integers(lo, max(lo, hi) + 1))
                at, take = on_grid(at, take, c_off, c_len)
                codes = np.full(wrow, 4, dtype=np.int64)
                codes[:take] = XH.read_rows(bases, at, at + take)
                ctxs.append(0)                                            # nuclear (context 0)
                rsrc, rsel = rep_main, slice(at, at + take)
            else:
                r = table.row(cell.draw(gen) if cell else int(gen.choice(table.height, p=p)), named=True)
                if pack and cell is None and r["len"] < window:
                    wrow, more, cell = short(("primary", window), table["len"], p, r["len"])
                take = min(wrow, r["len"])
                at = r["offset"] + int(gen.integers(0, r["len"] - take + 1))
                at, take = on_grid(at, take, r["offset"], r["len"])
                codes = np.full(wrow, 4, dtype=np.int64)
                codes[:take] = XH.read_rows(bases, at, at + take)
                ctxs.append(0)                                            # nuclear (context 0)
                rsrc, rsel = rep_main, slice(at, at + take)
            flip_row = flip and gen.random() < 0.5
            if flip_row:
                codes[:take] = G.reverse_complement(codes[:take].astype(np.uint8))
            if rep:                                  # a repeat is a repeat on either strand: reversed, not swapped
                r_ = np.zeros(wrow, dtype=np.uint8)
                if rep_rng.random() >= repeat_channel_drop and rsrc is not None:
                    r_[:take] = (XH.read_rows(rsrc, rsel.start, rsel.stop) if isinstance(rsel, slice) else rsrc[rsel])
                    if flip_row:
                        r_[:take] = r_[:take][::-1].copy()
                rep_rows.append(r_)
            if epigenome_weight > 0:
                if from_mito or xs is not None or fg is not None:         # no human Regulatory Build there
                    epi_t.append(np.zeros((wrow // epi_unit, n_epi), dtype=bool))
                    epi_v.append(np.zeros(wrow // epi_unit, dtype=bool))
                    epi_e.append(np.zeros(wrow // epi_unit, dtype=bool))
                else:
                    h_, v_, e_ = epi_targets(at, take, wrow, flip_row)
                    epi_t.append(h_)
                    epi_v.append(v_)
                    epi_e.append(e_)
            if ntv3 is not None:
                nt = np.zeros((wrow, len(ntv3["ids"])), dtype=np.uint8)
                sp_ = np.zeros(wrow, dtype=np.uint8)
                cm_ = np.zeros(len(ntv3["ids"]), dtype=bool)              # the channels this window can train
                if xs is not None:
                    nt[:take, e_["lo"]:e_["hi"]] = XH.read_rows(e_["dense"], at, at + take)   # pread, not the mapping
                    sp_[:take] = e_["split"][at:at + take]
                    cm_[e_["lo"]:e_["hi"]] = True
                elif not from_mito and fg is None:                        # organelle windows have no nuclear tracks
                    n_h = ntv3["n_h"]
                    if ntv3["dense"] is not None:
                        nt[:take, :n_h] = XH.read_rows(ntv3["dense"], at, at + take)[:, ntv3["cols"]]   # pread
                    else:
                        for k_, arr_ in enumerate(ntv3["arrs"]):
                            nt[:take, k_] = arr_[at:at + take]
                    sp_[:take] = XH.read_rows(ntv3["split"], at, at + take)
                    cm_[:n_h] = True
                elif fg is not None and genomes[fg].name in x_species:   # a further genome with head stores: its
                    if genomes[fg].name in x_gsplit:                     # per-base split (orthologs held out), else
                        sp_[:take] = XH.read_rows(x_gsplit[genomes[fg].name], at, at + take)
                    else:                                                # its held-out draw / training contigs
                        sp_[:take] = 4 if force_genome is not None else 1
                if flip_row and (xs is not None or not from_mito):        # reverse AND swap the strand partners
                    nt[:take] = nt[:take][::-1][:, ntv3["swap"]]
                    sp_[:take] = sp_[:take][::-1]
                g_row = genomes[fg].name if fg is not None else None
                for n_, st_ in xstores.items():           # primary rows, and further-genome rows of a species store
                    read_ = (xs is None and not from_mito and (x_read["heads"] is None or n_ in x_read["heads"])
                             and (fg is None or n_ in x_species.get(g_row, ())))
                    xh_t[n_].append((st_.window(at, take, wrow, flip_row, genome=g_row) if fg is not None else
                                     st_.window(at, take, wrow, flip_row)) if read_ else st_.empty(wrow))
                    xh_m[n_].append(x_mask(st_, g_row) if read_ else np.zeros(len(st_.ids), dtype=bool))
                nt_t.append(nt)
                nt_s.append(sp_)
                nt_c.append(cm_)
                nt_x.append(-1 if xs is None else xs)
            elif xstores:                            # --xhead without an NTv3 head (--xhead-split): the same side
                primary_ = xs is None and not from_mito and fg is None    # tuple, with no NTv3 channel in it
                g_row = genomes[fg].name if fg is not None else None
                sp_ = np.zeros(wrow, dtype=np.uint8)
                if primary_:
                    sp_[:take] = XH.read_rows(xsplit, at, at + take)
                    if flip_row:
                        sp_[:take] = sp_[:take][::-1]
                elif fg is not None and g_row in x_species:
                    if g_row in x_gsplit:
                        sp_[:take] = XH.read_rows(x_gsplit[g_row], at, at + take)
                        if flip_row:
                            sp_[:take] = sp_[:take][::-1]
                    else:
                        sp_[:take] = 4 if force_genome is not None else 1
                for n_, st_ in xstores.items():
                    read_ = ((primary_ or (fg is not None and n_ in x_species.get(g_row, ())))
                             and (x_read["heads"] is None or n_ in x_read["heads"]))
                    xh_t[n_].append((st_.window(at, take, wrow, flip_row, genome=g_row) if fg is not None else
                                     st_.window(at, take, wrow, flip_row)) if read_ else st_.empty(wrow))
                    xh_m[n_].append(x_mask(st_, g_row) if read_ else np.zeros(len(st_.ids), dtype=bool))
                nt_t.append(np.zeros((wrow, 0), dtype=np.uint8))
                nt_s.append(sp_)
                nt_c.append(np.zeros(0, dtype=bool))
                nt_x.append(-1)
            seg = np.zeros(wrow // unit, dtype=np.int64)
            seg[(take + unit - 1) // unit:] = 1                           # padding is its own segment
            seqs.append(codes)
            segs.append(seg)
            row_len.append(wrow)
            row_wt.append(1.0 / cell.n if cell else 1.0)
            if pack is not None:                                          # the training loop's calls only
                (tally if tally_to is None else tally_to).add(f"genome:{genomes[fg].name}" if fg is not None else
                          f"ntv3_extra:{ntv3['extra'][xs]['name']}" if xs is not None else
                          "mito" if from_mito else "primary", window, wrow, take, cell.n if cell else 1,
                          cell is None or more == cell.n - 1)
            if labels:
                lab, act = np.zeros(wrow, dtype=np.uint8), np.zeros(wrow, dtype=np.uint8)
                if not from_mito and xs is None and fg is None:
                    lab[:take], act[:take] = lab_mm[at:at + take], act_mm[at:at + take]
                    if flip_row:                                          # the current tracks are not strand-specific
                        lab[:take], act[:take] = lab[:take][::-1].copy(), act[:take][::-1].copy()
                labs.append(lab)
                acts.append(act)
                valid.append((np.arange(wrow) < take) & (not from_mito) & (xs is None) & (fg is None))

        def stacked(ix):
            """Rows ``ix`` as tensors, and the heads' targets for them (what epi_batch, ntv3_batch, rep_batch hold)."""
            def rows(xs_):
                return np.stack([xs_[i] for i in ix])
            epi_ = [(torch.from_numpy(rows(epi_t)), torch.from_numpy(rows(epi_v)),
                     torch.from_numpy(rows(epi_e)))] if epi_t else []
            nt_ = [(torch.from_numpy(rows(nt_t)), torch.from_numpy(rows(nt_s)),
                    torch.from_numpy(rows(nt_c)), [nt_x[i] for i in ix],
                    {n_: torch.from_numpy(rows(v_)) for n_, v_ in xh_t.items()},
                    {n_: torch.from_numpy(rows(v_)) for n_, v_ in xh_m.items()})] if nt_t else []
            rep_ = [torch.from_numpy(rows(rep_rows))] if rep_rows else []
            out = [torch.from_numpy(rows(seqs)), torch.from_numpy(rows(segs)), torch.tensor([ctxs[i] for i in ix])]
            if labels:
                out += [torch.from_numpy(rows(x)) for x in (labs, acts, valid)]
            return tuple(out), (epi_, nt_, rep_)

        if pack:                                     # one part per row length, the full window first
            parts = []
            for w_ in sorted(set(row_len), reverse=True):
                ix = [i for i, l_ in enumerate(row_len) if l_ == w_]
                parts.append((sum(row_wt[i] for i in ix) / batch, *stacked(ix)))
            return parts
        out, (epi_batch[:], ntv3_batch[:], rep_batch[:]) = stacked(range(len(seqs)))
        return out

    dev = torch.device("cuda", torch.cuda.current_device())
    if n_contexts < 0:
        n_contexts = G.N_CONTEXTS_ORGANELLE if mito else 0
    elif mito and n_contexts < G.N_CONTEXTS_ORGANELLE:
        raise typer.BadParameter(f"--mito tags windows with genetic-code ids up to 33: --n-contexts must be at least "
                                 f"{G.N_CONTEXTS_ORGANELLE}")
    config = G.KronGenomeConfig(d_model=d_model, n_layers=n_layers, stem_channels=stem_channels, n_down=n_down,
                                xheads=[[n_x, k_x] + ([x_bin[n_x]] if x_bin.get(n_x, 1) > 1 else [])
                                        for n_x, k_x in xhead_sizes.items()], xhead_hidden=xhead_hidden,
                                xhead_kind=xhead_kind, xhead_layers=xhead_layers, xhead_bin=xhead_step_bases,
                                xhead_width=xhead_width,
                                n_tracks=len(G.TRACKS) if labels else 0, arch=arch, head_hidden=head_hidden,
                                n_contexts=n_contexts, n_in_extra=n_in_extra, jepa=jepa_weight > 0,
                                jepa_pred_kind=jepa_pred, jepa_pred_layers=jepa_pred_layers,
                                n_epigenomes=n_epi if epigenome_weight > 0 else 0, chunk_ratio=chunk_ratio,
                                n_ntv3=len(ntv3["ids"]) if ntv3 else 0,
                                chunk_length_feature=chunk_length_feature,
                                **(dict(zip(("local_width", "local_layers", "local_span"),
                                            map(int, local_track.split(",")))) if local_track else {}))
    try:                                              # --genome context ids against the table this model has
        MG.check_contexts(genomes, config.n_contexts, mito_codes)
    except MG.GenomeSpecError as err:
        raise typer.BadParameter(str(err)) from None
    ctx_on = config.n_contexts > 0                    # the model reads context ids whenever it has a table
    torch.manual_seed(seed)                       # initial weights from --seed (audit 2026-11-18, TR03)
    model = G.KronGenome(config).to(dev)
    model.checkpoint_core = checkpoint_core
    if xhead:
        model.checkpoint_xheads = xhead_recompute
    split_devs = None                                 # --split-cards: the cards one window is divided over
    split_by = "balanced" if pack_short else None     # packs may hold an odd number of short rows (genome.split_plan)
    if split_cards > 1:
        if dev.type == "cuda" and torch.cuda.device_count() < (dev.index or 0) + split_cards:
            raise typer.BadParameter(f"--split-cards {split_cards} needs {split_cards} visible cards from "
                                     f"{dev}; CUDA_VISIBLE_DEVICES shows {torch.cuda.device_count()}")
        split_devs = [dev] + [torch.device("cuda", (dev.index or 0) + k) for k in range(1, split_cards)]
        say(f"split windows over {split_cards} cards ({', '.join(str(d_) for d_ in split_devs)}): conv path per "
            f"piece with a halo of {split_halo or G.split_halo(config):,} bases, core on {dev}; "
            f"held-out evaluation unsplit on {dev}")
    if compile_convs:
        for blocks in [model.down, model.up] + ([model.up2] if arch in ("v2", "v2l") else []):
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
    if xhead:                         # the same split, per role (inherited, each head), so each takes its own rate
        opt = torch.optim.AdamW(XH.param_groups(model, 0.1), lr=lr, betas=(0.9, 0.95))
    elif arch in ("v2", "v2l", "v3"):   # no weight decay on norms, biases, scales and embeddings
        decay = [p_ for n_, p_ in model.named_parameters() if p_.ndim >= 2 and "context" not in n_]
        rest = [p_ for n_, p_ in model.named_parameters() if not (p_.ndim >= 2 and "context" not in n_)]
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": rest, "weight_decay": 0.0}],
                                lr=lr, betas=(0.9, 0.95))
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "checkpoint.pt"
    # The run record (runrecord.py): written at startup, stored in every checkpoint, checked on resume.
    from kron_dna import runrecord as RR
    x_list = ntv3["extra"] if ntv3 else []
    shares_ = MG.window_shares([g_.share for g_ in genomes], mito_share if mito else 0.0,
                               ntv3_extra_share if x_list else 0.0, [float(p_) for p_ in ntv3["extra_p"]] if x_list else None)
    slice_rec = None
    if mito_slice:      # --mito-slice: the main draw without mito (the others rescaled by 1 / (1 - --mito-share))
        shares_ = {k_: (0.0 if k_ == "mito" else v_ / (1.0 - mito_share)) for k_, v_ in shares_.items()}
        per_phase_ = []
        for w_, n_ in phases:
            b_ = tokens_per_step // w_ // world // grad_accum if tokens_per_step else batch_size
            main_ = w_ * b_ * grad_accum
            per_phase_.append(dict(window=w_, steps=n_, main_bases=main_, slice_bases=mito_slice,
                                   mito_base_share=mito_slice / (main_ + mito_slice),
                                   base_shares={k_: (mito_slice / (main_ + mito_slice) if k_ == "mito" else
                                                     v_ * main_ / (main_ + mito_slice)) for k_, v_ in shares_.items()}))
        slice_rec = dict(bases=mito_slice, configured_mito_share=mito_share, main_window_shares=dict(shares_),
                         per_phase=per_phase_, placement="unsplit on the first card" if split_cards > 1 else "",
                         stream="np.random.default_rng([seed, rank, 2031, start step])")
        say(f"mito slice: every step also {mito_slice:,} bases of packed mitogenomes (unsplit"
            + (f" on {dev}" if split_cards > 1 else "") + "); main draw without mito, its window shares "
            + ", ".join(f"{k_} {v_:.4f}" for k_, v_ in shares_.items()) + "; mito share of step bases "
            + ", ".join(f"{q_['mito_base_share']:.4f} at window {q_['window']:,}" for q_ in per_phase_))
    rec_genomes = [RR.genome_entry("primary", name, shares_["primary"], 0,
                                   val_idx["chrom"].unique(maintain_order=True).to_list(), src / "contigs.parquet",
                                   annot_oversample=annot_oversample)]
    rec_genomes += [RR.genome_entry("genome", g_.name, g_.share / (1.0 - mito_share) if mito_slice else g_.share,
                                    g_.context,
                                    g_.val["chrom"].unique(maintain_order=True).to_list(),
                                    interim_dir("genome", g_.name) / "contigs.parquet") for g_ in genomes]
    if mito:
        rec_genomes.append(RR.genome_entry("mito", mito, mito_share, "per sequence: its genetic code table",
                                           mito_val["chrom"].to_list(), mito_src / "contigs.parquet",
                                           codes=sorted(mito_codes),
                                           **({"main_draw_share": 0.0, "mito_slice": mito_slice,
                                               "share_note": "--mito-slice: not in the main window draw; "
                                                             "resolved.mito_slice has its share of step bases"}
                                              if mito_slice else {})))
    for k_x, e_x in enumerate(x_list):
        g_x = next((g_ for g_ in genomes if g_.name == e_x["name"]), None)
        rec_genomes.append(RR.genome_entry("ntv3_extra", e_x["name"], shares_.get(f"extra{k_x}"),
                                           int(e_x.get("context", 0)),
                                           g_x.val["chrom"].unique(maintain_order=True).to_list() if g_x else [],
                                           interim_dir("genome", e_x["name"]) / "contigs.parquet"))
    ctx_rows = {}
    if ctx_on:
        ctx_rows["nuclear"] = 0
        for c_ in sorted(mito_codes):
            ctx_rows[f"organelle:genetic_code_{c_}"] = int(c_)
        for e_ in rec_genomes:
            if e_["role"] in ("primary", "genome", "ntv3_extra") and isinstance(e_["context"], int):
                ctx_rows[f"{e_['role']}:{e_['name']}"] = e_["context"]
    record = RR.make_record(
        options=run_options, source=RR.source_info(Path(G.__file__).parent), genomes=rec_genomes,
        input_channels=G.input_channel_registry(config.n_in_extra, ("repeat",) if repeat_channel else ()),
        context_rows=ctx_rows,
        resolved=dict(n_contexts=config.n_contexts, n_in_extra=config.n_in_extra, ctx_on=ctx_on, arch=arch,
                      curriculum=phases, steps=steps, eval_window=eval_window, world_size=world,
                      held_out_chroms_requested=sorted(held), window_shares=shares_, config=config.to_dict(),
                      n_parameters=model.n_parameters(), torch=torch.__version__,
                      **({"mito_slice": slice_rec} if slice_rec else {})))
    step, hours = 0, 0.0
    if ckpt.exists() and not bench:
        state = torch.load(ckpt, map_location="cpu")
        kind_saved = (state.get("config") or {}).get("xhead_kind", "mlp")
        rows_saved = {str(h[0]): (int(h[1]), int(h[2]) if len(h) > 2 else 1)
                      for h in (state.get("config") or {}).get("xheads", [])}
        moved = [n_ for n_, (k_, b_) in rows_saved.items()
                 if n_ in xhead_sizes and (xhead_sizes[n_], x_bin.get(n_, 1)) != (k_, b_)]
        if xhead and moved:
            raise typer.BadParameter(f"refusing to resume {ckpt}: heads {moved} were trained with other outputs or "
                                     f"bins ({ {n_: rows_saved[n_] for n_ in moved} }); add a new head instead")
        if xhead and kind_saved != xhead_kind:
            raise typer.BadParameter(f"refusing to resume {ckpt}: its --xhead heads are of kind {kind_saved}, this "
                                     f"run asks {xhead_kind}; start a new run directory with --init (the heads then "
                                     "start untrained, every other weight is the checkpoint's)")
        if state.get("run_record") is not None:
            diffs = RR.identity_differences(state["run_record"], record)
        else:                         # older code: the multi-genome branch stored only the further genomes
            diffs = RR.legacy_differences(state.get("genomes", []), record)
            say("note: the checkpoint has no run record (older code); only its --genome list can be checked")
        if diffs:
            raise typer.BadParameter(f"refusing to resume {ckpt}: this run's genomes, held-out sets or context ids "
                                     f"differ from the checkpoint's record: " + "; ".join(diffs)
                                     + ". Start a new run directory instead (with --init to keep the weights).")
        # TR01 (audit 2026-11-18): the NTv3 head and its optimizer rows load by position, so the ordered channel ids
        # and the target transform must be the checkpoint's own; a reordered or different same-width selection
        # would otherwise silently train each head row on another track
        saved_nt = state.get("ntv3")
        if ntv3 and saved_nt is not None:
            nt_diffs = []
            if list(saved_nt.get("ids", [])) != list(ntv3["ids"]):
                nt_diffs.append(f"channel ids (in order): checkpoint {list(saved_nt.get('ids', []))}, now {list(ntv3['ids'])}")
            for k_, v_ in (("target", ntv3_target), ("center", ntv3_center), ("loss", ntv3_loss)):
                if k_ in saved_nt and saved_nt[k_] != v_:
                    nt_diffs.append(f"{k_}: checkpoint {saved_nt[k_]!r}, now {v_!r}")
            if nt_diffs:
                raise typer.BadParameter(f"refusing to resume {ckpt}: its NTv3 head was trained on other targets: "
                                         + "; ".join(nt_diffs) + ". Keep the checkpoint's --ntv3-dir/--ntv3-assays and "
                                         "target options, or start a new run directory (with --init to keep the weights).")
        elif ntv3 and saved_nt is None and any(k_.startswith("ntv3") for k_ in state["model"]):
            say("note: the checkpoint records no NTv3 channel ids (older code); their order cannot be checked")
        elif saved_nt is not None and not ntv3:
            raise typer.BadParameter(f"refusing to resume {ckpt}: it has an NTv3 head over {len(saved_nt.get('ids', []))} "
                                     "tracks and this run selects none (pass the checkpoint's --ntv3-dir)")
        # --xhead heads registered beyond the checkpoint's: a continuation that gains heads (their weights start
        # untrained, their optimizer groups fresh; everything the checkpoint holds, its optimizer state too, stands)
        x_added: list[str] = []
        if xhead and state.get("xstage") is not None:
            try:
                x_added = XH.added_heads(state["xstage"]["plan"], XH.StagePlan(
                    heads=list(xhead_sizes), head_only=xhead_only, lr_inherited=inherit_lr, lr_head=xhead_lr,
                    warmup_inherited=inherit_warmup, warmup_head=xhead_warmup))
            except ValueError as err:
                raise typer.BadParameter(f"refusing to resume {ckpt}: {err}") from None
        if x_added:
            missing_, unexpected_ = model.load_state_dict(state["model"], strict=False)
            stray_ = [k_ for k_ in missing_ if k_.split(".")[:2] not in [["xheads", n_] for n_ in x_added]]
            if stray_ or unexpected_:
                raise typer.BadParameter(f"refusing to resume {ckpt}: beside the added heads {x_added}, the model "
                                         f"lacks {stray_[:6]} and the checkpoint has {list(unexpected_)[:6]}")
            try:
                opt.load_state_dict(XH.merge_optimizer_state(state["optimizer"], opt.state_dict(),
                                                             2 * (1 + len(state["xstage"]["plan"]["heads"]))))
            except ValueError as err:
                raise typer.BadParameter(f"refusing to resume {ckpt}: {err}") from None
            say(f"--xhead: added on resume: {x_added} (new parameters and optimizer groups; every other weight, "
                f"moment and count is the checkpoint's)")
        else:
            model.load_state_dict(state["model"])
            opt.load_state_dict(state["optimizer"])
        step, hours = state["step"], state.get("hours", 0.0)
        say(f"resumed at step {step}")
        guard_saved = state.get("jepa_guard")
        jnorm_saved = state.get("jepa_norm")
        teacher_saved, resumed = state.get("jepa_target"), True
        xstage_saved = state.get("xstage")
    elif init is not None and not bench:
        # reserved room (stem input channels, context rows) is zero-padded explicitly, function-preserving; new
        # heads stay missing (strict=False) as before, and any other shape mismatch still raises
        init_state = torch.load(init, map_location="cpu")
        kind_init = (init_state.get("config") or {}).get("xhead_kind", "mlp")
        if kind_init != config.xhead_kind:            # another kind of head: none of its weights is this run's
            dropped_x = [k_ for k_ in init_state["model"] if k_.startswith("xheads.")]
            init_state["model"] = {k_: v_ for k_, v_ in init_state["model"].items() if not k_.startswith("xheads.")}
            if dropped_x:
                say(f"--xhead: the checkpoint's {kind_init} heads are not loaded ({len(dropped_x)} tensors); "
                    f"the {config.xhead_kind} heads start untrained")
        # --xhead-species: a head with more outputs than the checkpoint's (a further genome's channels appended): its
        # readout rows of the old channels are copied, the new rows keep this model's fresh initialisation
        own_ = model.state_dict()
        widened_, moved_ = {}, {}
        # each head's channel ids in the checkpoint (its stage records the store it trained on): readout rows are
        # matched by id, not by position (review 23 Nov: a reordered or re-selected store would otherwise put an old
        # channel's weights on another channel); a checkpoint without ids keeps the positional copy
        old_ids_ = {h_: list(st_.get("ids") or []) for h_, st_ in
                    (((init_state.get("xstage") or {}).get("state") or {}).get("stores") or {}).items()}
        for k_, v_ in list(init_state["model"].items()):
            w_ = own_.get(k_)
            if not (k_.startswith("xheads.") and w_ is not None and v_.dim() >= 1 and w_.shape[1:] == v_.shape[1:]):
                continue
            h_ = k_.split(".")[1]
            ids_old, ids_new = old_ids_.get(h_), (list(xstores[h_].ids) if h_ in xstores else None)
            if ids_old and ids_new and len(ids_old) == v_.shape[0] and len(ids_new) == w_.shape[0] \
                    and ids_new[:len(ids_old)] != ids_old:
                at_ = {i_: k2_ for k2_, i_ in enumerate(ids_old)}
                new_ = w_.clone()
                hit_ = [(j_, at_[i_]) for j_, i_ in enumerate(ids_new) if i_ in at_]
                if hit_:
                    jj_, ii_ = zip(*hit_)
                    new_[list(jj_)] = v_[list(ii_)]
                init_state["model"][k_] = new_
                moved_[h_] = (len(hit_), len(ids_new))
            elif w_.shape[0] > v_.shape[0]:
                new_ = w_.clone()
                new_[:v_.shape[0]] = v_
                init_state["model"][k_] = new_
                widened_[h_] = (int(v_.shape[0]), int(w_.shape[0]))
        if widened_:
            say("--xhead: widened readouts (old channels kept, new rows fresh): "
                + ", ".join(f"{h_} {a_} -> {b_}" for h_, (a_, b_) in widened_.items()))
        if moved_:
            say("--xhead: readout rows matched to the checkpoint's channels by id (the order or selection changed): "
                + ", ".join(f"{h_} {a_} of {b_} kept" for h_, (a_, b_) in moved_.items()))
        grown, notes = G.expand_state_dict(init_state["model"], model, new_contexts=new_contexts)
        missing, unexpected = model.load_state_dict(grown, strict=False)
        say(f"initialised from {init}; new: {missing}; dropped: {unexpected}; padded: {notes or 'nothing'}")
        if xhead:
            other_ = [k_ for k_ in missing if not k_.startswith("xheads.")]
            say("--xhead: " + ("the new parameters are the heads' only" if not other_ else
                               f"WARNING: beside the heads, {len(other_)} parameters start untrained: {other_[:8]}"))
        old_rows = int(init_state["model"]["context.weight"].shape[0]) if "context.weight" in init_state["model"] else 0
        fresh = sorted({g_.context for g_ in genomes if g_.context >= old_rows})
        if fresh:
            say(f"context ids {fresh} are new rows of this model: they start as "
                + ("copies of the checkpoint's nuclear row 0" if new_contexts == "nuclear" else
                   "zero (the checkpoint without a context vector, not its nuclear row; --new-contexts nuclear "
                   "copies row 0)"))
        record["init_from"] = dict(path=str(init), step=init_state.get("step"), new_contexts=new_contexts,
                                   padded=notes, commit=(init_state.get("run_record") or {}).get("source", {})
                                   .get("commit"))
        del init_state
        step = start_step
    # --xhead: the stage (head-only, then joint; a rate per group), continued from the checkpoint on a resume
    xstage = None
    x_step: set = set()                               # heads with supervised positions in the current step
    x_rates: dict = {}                                # the last step's rate per role
    x_losses: dict = {n_: [] for n_ in xhead_sizes}
    x_pearson_g: dict = {}            # (head, further genome): its held-out score, kept apart from hg38 chr8/chr9
    x_where = {"genome": None}         # which genome's held-out rows loss_on is scoring (None: the primary's)
    x_pearson: dict = {n_: (xstores[n_].scorer(x_bin.get(n_, 1)) if n_ in xstores else XH.Pearson(k_))
                       for n_, k_ in xhead_sizes.items()}
    x_consts: dict = {}

    cf = None                           # --cf-pairs: the training rows, their stream, their log
    if cf_weight > 0:
        from kron_dna import cfpairs as CF
        cf = CF.CfPairs(list(cf_pairs), {n_: list(st_.ids) for n_, st_ in xstores.items()},
                        set(val_chroms.split(",")), cf_window)
        if not cf.pairs:
            raise typer.BadParameter(f"--cf-pairs: no usable train row for the heads {sorted(xstores)}")
        cf_rng = np.random.default_rng([seed, rank, 2041, step])
        cf_consts = {n_: (None if st_.kind == "binary" else tuple(st_.consts(dev))[1:]) for n_, st_ in xstores.items()}
        cf_contigs = index.select("chrom", "start", "end", "offset")
        record["cf_pairs"] = dict(paths=[str(p_) for p_ in cf_pairs], weight=cf_weight, batch=cf_batch,
                                  every=cf_every, window=cf_window, temp=cf_temp, **cf.counts)
        say(f"--cf-pairs: {cf.counts['pairs']:,} train pairs ({cf.counts['by_table']}), weight {cf_weight:g}, "
            f"{cf_batch} per {'step' if cf_every == 1 else f'{cf_every} steps'}, window {cf_window:,}")

    def x_consts_on(name_):
        """Head ``name_``'s constants (scales, non-zero means, RNA flags) per card, moved once."""
        def on(d_):
            if (name_, d_) not in x_consts:
                x_consts[(name_, d_)] = xstores[name_].consts(d_)
            return x_consts[(name_, d_)]
        return on

    if xhead:
        plan_x = XH.StagePlan(heads=list(xhead_sizes), head_only=xhead_only, lr_inherited=inherit_lr, lr_head=xhead_lr,
                              warmup_inherited=inherit_warmup, warmup_head=xhead_warmup)
        if resumed and xstage_saved is None:
            raise typer.BadParameter(f"refusing to resume {ckpt}: it has no --xhead stage; start a new run directory "
                                     "with --init")
        try:
            if xstage_saved is not None:              # the same stage, or the same with heads added at its end
                xstage_saved = dict(xstage_saved, state=XH.extend_state(
                    xstage_saved["state"], XH.added_heads(xstage_saved["plan"], plan_x)))
            xstage = XH.Stage(plan_x, xstage_saved["state"] if xstage_saved is not None else None)
            for n_, st_ in xstores.items():           # adopted here: a start or a restart is a checkpoint boundary
                xstage.activate(n_, st_.identity)
        except ValueError as err:
            raise typer.BadParameter(f"--xhead: {err} (resuming {ckpt})" if resumed else f"--xhead: {err}") from None
        record["xheads"] = dict(plan=plan_x.to_dict(), weight=xhead_weight,
                                stores={n_: dict(path=str(st_.path), **st_.identity) for n_, st_ in xstores.items()})
        say(f"--xhead: {', '.join(f'{n_} ({k_} outputs' + ('' if n_ in xstores else ', no data yet') + ')' for n_, k_ in xhead_sizes.items())}; "
            f"phase {xstage.state['phase']}, stage step {xstage.state['stage_step']}, updates {xstage.state['updates']}")
    elif resumed and xstage_saved is not None:
        raise typer.BadParameter(f"refusing to resume {ckpt}: it carries an --xhead stage "
                                 f"({xstage_saved['plan']['heads']}); pass its --xhead options")
    # TR03/TR04 (audit 2026-11-18): the training-mask/dropout stream gets its own seed per card and start step, and a
    # run starting past step 0 draws its windows (and repeat-channel drops) from step-keyed streams instead of
    # replaying the first windows of a fresh run. A fresh run (step 0) keeps the windows of earlier code.
    torch.manual_seed(int(np.random.SeedSequence([seed, rank, 2035, step]).generate_state(1)[0]))
    if step > 0:
        rng = np.random.default_rng([seed, rank, step])
        rep_rng = np.random.default_rng([seed, rank, 2027, step])
    record["rng"] = dict(init="torch.manual_seed(seed) before the model is built",
                         masks="torch.manual_seed(SeedSequence([seed, rank, 2035, start step]))",
                         windows="np.random.default_rng([seed, rank]) from step 0, else [seed, rank, start step]",
                         repeat_drop="np.random.default_rng([seed, rank, 2027]) from step 0, else [seed, rank, 2027, start step]")
    record["lr_schedule"] = dict(lr=lr, warmup=warmup, steps=steps, lr_total=lr_total or steps,
                                 cooldown_from=cooldown_from)
    if main and not bench:
        record["started_at_step"] = step
        rec_path = out / "run_record.json"
        if rec_path.exists():                                    # a resumption keeps the original record
            rec_path = out / f"run_record.step{step}.json"
        RR.write_json(rec_path, record)
        say(f"run record: {rec_path} (commit {record['source'].get('commit')}, "
            f"{len(record['genomes'])} genomes, input channels {record['input_channels']})")
    say(f"KronGenome {arch} {model.n_parameters() / 1e6:.1f}M parameters, curriculum "
               f"{', '.join(f'{w:,} x {n:,} steps' for w, n in phases)}, eval window {eval_window:,}, "
               f"{train_idx.height} train / {val_idx.height} held-out contigs")
    target = None
    if jepa_weight > 0:                               # the EMA target: a frozen copy, updated after every step
        import copy
        target = copy.deepcopy(model).eval()
        target.x_active = set()                       # the target's further heads are never read
        for p_ in target.parameters():
            p_.requires_grad_(False)
        if jepa_ema > 0 and teacher_saved is not None:            # audit 2026-11-18, TR05
            target.load_state_dict(teacher_saved)
            say("jepa: the EMA teacher continues from the checkpoint")
        elif jepa_ema > 0 and resumed:
            say("note: the checkpoint carries no EMA teacher (older code): the teacher restarts as a copy of the "
                "resumed weights")
    # --jepa-loss zcos: running per-dimension mean and variance of the target's core states (momentum 0.99), the
    # frame the loss is taken in; carried in the checkpoint so a resumption keeps its frame
    jnorm = {k_: v_.to(dev) for k_, v_ in jnorm_saved.items()} if jnorm_saved and jepa_loss != "cos" else {}
    if jepa_weight > 0:
        model.jepa_decode_pred = jepa_decode_pred
        if jepa_pred != "mlp" or jepa_loss != "cos" or jepa_decode_pred or freeze_trunk_until:
            say(f"jepa step 2: predictor {jepa_pred}" + (f" ({jepa_pred_layers} KronLSTM layers)" if jepa_pred == "context"
                else "") + f", loss {jepa_loss}" + (", decoder reads the predictor at hidden bins" if jepa_decode_pred
                else "") + (f", only the predictor trains until step {freeze_trunk_until}" if freeze_trunk_until else "")
                + ("; target statistics restored" if jnorm else ""))

    def jnorm_frame(want: torch.Tensor, keep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Updates the running statistics from this batch's target states at ``keep`` bins (training) and returns
        (mean, std) [d]."""
        with torch.no_grad():
            z_ = want.float()[keep]
            if z_.shape[0] > 1:
                m_, v_ = z_.mean(0), z_.var(0)
                if not jnorm:
                    jnorm.update(mu=m_, var=v_)
                else:
                    jnorm["mu"].lerp_(m_, 0.01)
                    jnorm["var"].lerp_(v_, 0.01)
        return jnorm["mu"], jnorm["var"].clamp_min(1e-6).sqrt()

    edit_lo, edit_hi = (int(v_) for v_ in jepa_edit_len.split(":"))
    edit_rng = np.random.default_rng([seed, rank, 2043, step]) if jepa_edits > 0 else None
    edit_blk_rng = np.random.default_rng([seed, rank, 2047, step]) if jepa_edits > 0 else None
    edit_tab = train_idx.filter(pl.col("len") >= jepa_edit_window) if jepa_edits > 0 else None
    if jepa_edits > 0:
        if edit_tab.height == 0:
            raise typer.BadParameter(f"--jepa-edits: no training contig of {jepa_edit_window:,} bases")
        say(f"jepa step 2b: self-made edits at weight {jepa_edits:g} every {jepa_edit_every} step(s), "
            f"{jepa_edit_window:,}-base windows, edits {edit_lo}-{edit_hi} bases (shuffle, reverse complement, N)")

    def edit_loss() -> tuple[torch.Tensor | None, dict]:
        """--jepa-edits: one window, clean and edited, both hidden in the same blocks; the predictor's states at the
        hidden bins against the target's for the edited window, and its change (edited - clean) against the
        target's change, in the --jepa-loss frame."""
        p_ = edit_tab["len"].to_numpy() / edit_tab["len"].sum()
        for _ in range(20):
            r_ = edit_tab.row(int(edit_rng.choice(edit_tab.height, p=p_)), named=True)
            at_ = int(r_["offset"]) + int(edit_rng.integers(0, int(r_["len"]) - jepa_edit_window + 1))
            s_ = XH.read_rows(bases, at_, at_ + jepa_edit_window).astype(np.int64)
            if (s_ < 4).mean() >= 0.95:
                break
        else:
            return None, {}
        n_bins = jepa_edit_window // unit
        hid_ = G.block_mask(np.zeros((1, n_bins), dtype=np.int64), s_[None], unit, jepa_block_share, block_lo,
                            block_hi, edit_blk_rng)[0]
        vis_ = np.flatnonzero(~np.repeat(hid_, unit))
        if not hid_.any() or len(vis_) < edit_hi * 2:
            return None, {}
        e_len = int(edit_rng.integers(edit_lo, edit_hi + 1))
        e_at = int(vis_[edit_rng.integers(0, len(vis_))])
        e_at = min(e_at, jepa_edit_window - e_len)
        if np.repeat(hid_, unit)[e_at:e_at + e_len].any():     # the edit stays outside every block
            return None, {}
        ed_ = s_.copy()
        kind_ = int(edit_rng.integers(0, 3))
        seg_ = ed_[e_at:e_at + e_len]
        if kind_ == 0:
            ed_[e_at:e_at + e_len] = seg_[edit_rng.permutation(e_len)]
        elif kind_ == 1:
            ed_[e_at:e_at + e_len] = np.where(seg_ < 4, 3 - seg_, seg_)[::-1]
        else:
            ed_[e_at:e_at + e_len] = 4
        seqs_ = torch.from_numpy(np.stack([s_, ed_])).to(dev)
        segm_ = torch.zeros(2, n_bins, dtype=torch.long, device=dev)
        ctx_ = torch.zeros(2, dtype=torch.long, device=dev) if ctx_on else None
        hb_ = torch.from_numpy(np.stack([hid_, hid_])).to(dev)
        inp_, _, _, _ = G.hide_blocks(seqs_.clone(), None, hb_, unit)
        old_ = (model.x_active, getattr(model, "jepa_hidden", None), getattr(model, "jepa_decode_pred", False))
        try:
            model.x_active, model.jepa_hidden, model.jepa_decode_pred = set(), hb_, False
            with torch.no_grad(), autocast:
                _, want_, _ = target(seqs_, segm_, ctx_)
            with autocast:
                train_model(inp_, segm_, ctx_)
            pred_ = model.last_jepa.float()
        finally:
            model.x_active, model.jepa_hidden, model.jepa_decode_pred = old_
        want_ = want_.float()
        if jepa_loss == "cos":
            mu_, sd_ = torch.zeros_like(want_[0, 0]), torch.ones_like(want_[0, 0])
        else:
            mu_, sd_ = jnorm_frame(want_, torch.ones_like(hb_)) if not jnorm else (jnorm["mu"], jnorm["var"].clamp_min(1e-6).sqrt())
        zp_, zt_ = (pred_ - mu_) / sd_, (want_ - mu_) / sd_
        h1_ = hb_[1]
        d_e = (1 - F.cosine_similarity(zp_[1], zt_[1], dim=-1))[h1_].mean()
        dp_, dt_ = (zp_[1] - zp_[0])[h1_], (zt_[1] - zt_[0])[h1_]
        d_delta = (dp_ - dt_).pow(2).mean()
        st_ = dict(kind=("shuffle", "revcomp", "N")[kind_], length=e_len, edited=float(d_e.detach()),
                   delta_mse=float(d_delta.detach()), target_change=float(dt_.pow(2).mean().detach()),
                   delta_cos=float(F.cosine_similarity(dp_.flatten(), dt_.flatten(), dim=0).detach()))
        return d_e + d_delta, st_

    train_model = model
    if world > 1:
        train_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dev.index])
        say(f"data-parallel over {world} cards: {tokens_per_step // world if tokens_per_step else batch_size} "
            f"bases per card per micro-batch")
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16)

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

    def split_mlm(pieces, targets, clean):
        """--split-cards: loss_on's masked-base term from the pieces of a split window, each piece's cross-entropy
        summed on its own card; the mean over the window's targets (or a zero tied to every piece's logits, as
        unsplit, when the input is clean). Unweighted: loss_on applies --mlm-weight as it does unsplit.
        With block JEPA (--jepa-block-heads exclude) the targets inside blocks are -100 already (hide_blocks)."""
        if clean:
            return sum((p_.logits.float().sum() * 0.0).to(dev) for p_ in pieces)
        count_ = (targets != -100).sum()
        tot_ = sum(F.cross_entropy(p_.logits.float().reshape(-1, 4),
                                   targets[p_.rows, p_.start:p_.end].to(p_.device).reshape(-1),
                                   ignore_index=-100, reduction="sum").to(dev) for p_ in pieces)
        return tot_ / count_                                              # loss_on applies --mlm-weight

    split_consts: dict = {}

    def split_ntv3_loss(pieces, batch_, skip_=None):
        """--split-cards: loss_on's training track loss (every --ntv3-loss form) from the pieces of a split window:
        targets go from the host straight to each piece's card, per-position terms are summed there, and only
        per-(row, track) sums and scalars cross to the first card, where the window's loss is assembled exactly as
        loss_on assembles it from the whole window. ``skip_`` [B, L] bool (block JEPA, --jepa-block-heads exclude):
        bases inside hidden blocks, left out of the numerator and the denominator as loss_on leaves them out."""
        want_c, sp_c, col_c = batch_[:3]                                  # host tensors [B,L,T] u8, [B,L], [B,T]
        ok_full = (sp_c == 1).float()                                     # as loss_on: NTv3 train split only,
        if ntv3_center < 1.0:                                             # the central fraction of each window
            edge = int(round(ok_full.shape[1] * (1 - ntv3_center) / 2))
            if edge:
                ok_full[:, :edge] = 0
                ok_full[:, -edge:] = 0
        if skip_ is not None:                                             # block JEPA: visible positions only
            ok_full = ok_full * (~skip_).float().cpu()
        parts_, num_ = [], 0.0
        for p_ in pieces:
            d_ = p_.device
            if d_ not in split_consts:
                split_consts[d_] = (ntv3_scale.to(d_), None if ntv3_nz is None else ntv3_nz.to(d_), ntv3_rna.to(d_))
            sc_, nz_, rna_ = split_consts[d_]
            want_k = want_c[p_.rows, p_.start:p_.end].to(d_)
            col_k = col_c[p_.rows].to(d_)
            ok_k = ok_full[p_.rows, p_.start:p_.end].to(d_)
            log1p_k = want_k.float() / 255.0 * sc_
            pred_k = p_.ntv3.float()
            if ntv3_loss == "poisson":
                pred_k = F.softplus(pred_k)
            if nz_ is None:
                tgt_k = log1p_k
            else:                                                         # ntv3_ag on this card's constants
                x_ = torch.expm1(log1p_k).clamp_min(0) / nz_
                x_ = torch.where(rna_, x_.clamp_min(1e-12).pow(0.75), x_)
                tgt_k = torch.where(x_ > 10, 2 * torch.sqrt(10 * x_) - 10, x_)
            if ntv3_focus < 1.0:
                sig_ = (want_k > 0).any(-1).float()
                ok_k = ok_k * (sig_ + (1 - sig_) * ntv3_focus)
            if ntv3_loss == "poisson":
                parts_.append((p_.rows, G.poisson_multinomial_parts(pred_k, tgt_k,
                                                                    ok_k.unsqueeze(-1) * col_k.unsqueeze(1).float())))
            elif ntv3["extra"]:
                sq_ = (pred_k - tgt_k).pow(2) * col_k.unsqueeze(1).float()
                per_ = sq_.sum(-1) / col_k.float().sum(-1, keepdim=True).clamp_min(1)
                num_ = num_ + (per_ * ok_k).sum().to(dev)
            else:
                num_ = num_ + ((pred_k - tgt_k).pow(2).mean(-1) * ok_k).sum().to(dev)
        if ntv3_loss == "poisson":
            return G.poisson_multinomial_combine(parts_, want_c.shape[0], dev)
        den_ = ok_full if ntv3_focus >= 1.0 else None
        if den_ is None:                                                  # the focus weights need the targets
            sig_ = (want_c > 0).any(-1).float()
            den_ = ok_full * (sig_ + (1 - sig_) * ntv3_focus)
        return num_ / den_.sum().clamp_min(1).to(dev)

    def loss_on(seq, seg, ctx, lab=None, act=None, valid=None, keep=None, mask_gen=None,
                track_only=False, clean=False, score_ntv3=True, unsplit=False):
        clean = clean or (mlm_weight == 0 and model.training)        # a track stage reads the sequence unmasked
        if clean:
            inputs, targets = seq.to(dev), None
        else:   # held-out masks are always plain MASK (bert only in training), so runs with and without it compare
            inputs, targets = G.mask_bases(seq.to(dev), mask_prob, generator=mask_gen,
                                           bert=bert_mask and model.training)
        fwd_kw = {}                     # training-only inputs; evaluation always reads hard bases, no repeat channel
        # --jepa-block-share: hidden bins [B, bins] and bases [B, L] of this training batch (None: no blocks). The
        # blocks come from a stream of their own, so the masks and everything else drawn are those of a run without
        # them; inside a block the online input is MASK only, drawn before the soft bases so none lands there.
        blk_bins = blk_at = None
        tgt_kw = fwd_kw                 # what the JEPA target reads beside the clean sequence
        if jepa_block_share > 0 and model.training and not clean:
            blk_bins = torch.from_numpy(G.block_mask(seg.numpy(), seq.numpy(), unit, jepa_block_share, block_lo,
                                                     block_hi, block_rng)).to(dev)
            inputs, targets, _, blk_at = G.hide_blocks(inputs, targets, blk_bins, unit,
                                                       exclude=jepa_block_heads == "exclude")
        if model.training and soft_base_rate > 0:                       # [B, 6, L] fractional bases (heterozygous-like)
            inputs = G.soft_base_channels(inputs, targets, soft_base_rate)
        if model.training and repeat_channel and rep_batch:
            extra_in = torch.zeros(seq.shape[0], n_in_extra, seq.shape[-1], device=dev)
            extra_in[:, G.EXTRA_INPUT_CHANNELS["repeat"]] = rep_batch[0].to(dev).float()
            fwd_kw["extra"] = extra_in
            if blk_at is not None:      # the target keeps the repeat channel; the online input loses it in blocks
                tgt_kw = {"extra": extra_in}
                fwd_kw = {"extra": extra_in.masked_fill(blk_at.unsqueeze(1), 0.0)}
        skip_at = blk_at if jepa_block_heads == "exclude" else None    # [B, L] bases the other heads do not score
        x_now, x_ok = [], None          # --xhead: the heads this pass runs, and where they are supervised [B, L]
        if xstage is not None:
            if ntv3_batch and len(ntv3_batch[0]) > 4 and xstage.active:
                if model.training:      # the training split, the window's centre, visible bases
                    x_ok = XH.supervised(ntv3_batch[0][1], ntv3_center, skip_at)
                else:                   # scored on the held-out chromosomes only (split code 4)
                    x_ok = (ntv3_batch[0][1] == 4).float()
                if float(x_ok.sum()) > 0:
                    x_now = [h_ for h_ in xstage.active      # --xhead-cycle: this step's group in training
                             if not model.training or x_read["heads"] is None or h_ in x_read["heads"]]
            model.x_active = set(x_now)
            # training: the heads run on the supervised stretch only (the window's centre): the same loss and
            # gradients, without their activations at every other base. Evaluation scores whole windows.
            model.x_span = XH.span_of(x_ok, unit) if model.training and x_now else None
        parts_log = {} if jepa_log and model.training else None        # this micro-batch's loss terms, for jepa.csv
        model.jepa_hidden = blk_bins if model.training else None       # the context predictor and --jepa-decode-pred
        # --split-cards: training windows only (``unsplit``: the --mito-slice part, whole on the first card)
        split_now = split_devs is not None and model.training and not unsplit
        with autocast:
            if split_now:
                pieces, ctx_core = model.forward_split(inputs, seg.to(dev), ctx.to(dev) if ctx_on else None,
                                                       devices=split_devs, halo=split_halo or None, by=split_by,
                                                       **fwd_kw)
                logits = tracks = None
            else:
                logits, ctx_core, tracks = (train_model if model.training else model)(inputs, seg.to(dev),
                                                                                       ctx.to(dev) if ctx_on else None,
                                                                                       **fwd_kw)
        if split_now:                   # each piece's loss on its own card; sums (not outputs) cross to the first
            mlm = split_mlm(pieces, targets, clean)
            if not clean:
                if parts_log is not None:
                    parts_log["mlm"] = float(mlm.detach())
                if mlm_weight != 1.0:
                    mlm = mlm * mlm_weight
        elif clean:
            mlm = logits.float().sum() * 0.0                             # no masked base to score
        else:
            mlm = F.cross_entropy(logits.float().reshape(-1, 4), targets.reshape(-1), ignore_index=-100)
            if parts_log is not None:
                parts_log["mlm"] = float(mlm.detach())
            if model.training and mlm_weight != 1.0:
                mlm = mlm * mlm_weight
        if chunk_ratio and model.training and model.last_ratio_loss is not None:
            mlm = mlm + ratio_weight * model.last_ratio_loss
            chunk_rates.append(model.last_chunks)
        if epigenome_weight > 0 and model.last_epi is not None and epi_batch:
            want_e, ok_e, has_e = (x.to(dev) for x in epi_batch[0])
            per_e = F.binary_cross_entropy_with_logits(model.last_epi.float(), want_e.float(), reduction="none")
            # --epigenome-focus: a bin no element covers is all-off for every epigenome whatever the cell type, and
            # those are ~94% of bins, so down-weighting them puts the loss on the bins that carry the question.
            w_e = ok_e.float() if epigenome_focus >= 1.0 else \
                ok_e.float() * (has_e.float() + (~has_e).float() * epigenome_focus)
            if skip_at is not None:                                      # a weighted mean over the visible bins
                w_e = w_e * (~blk_bins).float()
            loss_e = (per_e * w_e.unsqueeze(-1)).sum() / (w_e.sum().clamp_min(1) * per_e.shape[-1])
            if parts_log is not None:
                parts_log["epi"] = float(loss_e.detach())
            if model.training:
                mlm = mlm + epigenome_weight * loss_e
            elif not track_only:
                epi_keep.append((model.last_epi.float()[ok_e].cpu(), want_e[ok_e].cpu()))
        if split_now and ntv3 is not None and model.ntv3 is not None and ntv3_batch:
            loss_n = split_ntv3_loss(pieces, ntv3_batch[0], skip_at)
            mlm = mlm + ntv3_weight * loss_n
            ntv3_losses.append(float(loss_n.detach()))
            if parts_log is not None:
                parts_log["ntv3"] = ntv3_losses[-1]
        elif ntv3 is not None and model.last_ntv3 is not None and ntv3_batch:
            want_n, sp_n, col_n = (x.to(dev) for x in ntv3_batch[0][:3])  # [B,L,T] uint8, [B,L] split, [B,T] bool
            spc_n = ntv3_batch[0][3]                                      # per row: -1 human, else the species index
            # split.u8: 1 NTv3 train, 2 NTv3 val, 3 NTv3 test, 4 chr8/chr9 (KronGenome's held-out chromosomes)
            log1p_t = want_n.float() / 255.0 * ntv3_scale                 # the units the metric is defined in
            pred = model.last_ntv3.float()
            if ntv3_loss == "poisson":
                pred = F.softplus(pred)                                   # a non-negative rate, as the loss assumes
            tgt = log1p_t if ntv3_nz is None else ntv3_ag(torch.expm1(log1p_t))
            if model.training:                                            # NTv3's own train regions only, which is
                ok_n = (sp_n == 1).float()                                # also what keeps chr8/chr9 out
                if ntv3_center < 1.0:                                     # supervise the middle of the window only
                    edge = int(round(ok_n.shape[1] * (1 - ntv3_center) / 2))
                    if edge:
                        ok_n[:, :edge] = 0
                        ok_n[:, -edge:] = 0
                if ntv3_focus < 1.0:                                      # all-zero positions are most of the genome
                    sig = (want_n > 0).any(-1).float()
                    ok_n = ok_n * (sig + (1 - sig) * ntv3_focus)
                if skip_at is not None:                                   # supervise visible positions only
                    ok_n = ok_n * (~skip_at).float()
                if ntv3_loss == "poisson":
                    loss_n = G.poisson_multinomial_loss(pred, tgt, ok_n.unsqueeze(-1) * col_n.unsqueeze(1).float())
                elif ntv3["extra"]:                                       # each window: its own species' channels
                    sq = (pred - tgt).pow(2) * col_n.unsqueeze(1).float()
                    per_pos = sq.sum(-1) / col_n.float().sum(-1, keepdim=True).clamp_min(1)
                    loss_n = (per_pos * ok_n).sum() / ok_n.sum().clamp_min(1)
                else:
                    loss_n = ((pred - tgt).pow(2).mean(-1) * ok_n).sum() / ok_n.sum().clamp_min(1)
                mlm = mlm + ntv3_weight * loss_n
                ntv3_losses.append(float(loss_n.detach()))
                if parts_log is not None:
                    parts_log["ntv3"] = ntv3_losses[-1]
            elif score_ntv3:
                # score in log1p units whatever the head was trained in, so arms stay comparable and the number
                # means the same thing the paper's does. 4 = chr8/chr9 (never seen at all), 3 = NTv3's own test
                # split (on chromosomes the masked-LM trains on, as it is for NTv3 itself).
                p_log = pred if ntv3_nz is None else torch.log1p(ntv3_ag_inv(pred).clamp_min(0))
                for code_ in (3, 4):
                    for b_ in range(sp_n.shape[0]):
                        m_ = sp_n[b_] == code_
                        st_ = ntv3_stats.get((code_, spc_n[b_]))
                        if st_ is None or not m_.any():
                            continue
                        cols_ = col_n[b_]
                        x_, y_ = p_log[b_][m_][:, cols_].double(), log1p_t[b_][m_][:, cols_].double()
                        st_[0] += x_.shape[0]
                        st_[1] += x_.sum(0).cpu(); st_[2] += y_.sum(0).cpu()
                        st_[3] += (x_ * y_).sum(0).cpu()
                        st_[4] += (x_ * x_).sum(0).cpu(); st_[5] += (y_ * y_).sum(0).cpu()
        for n_ in x_now:                # --xhead: each active head on its own store
            want_x = ntv3_batch[0][4][n_]
            cm_x = ntv3_batch[0][5][n_]     # [B, T] the channels each row supervises
            if model.training:
                kind_x = xstores[n_].kind
                lo_x, hi_x = (0, 0) if split_now else model.last_x_at
                bin_x = x_bin.get(n_, 1)
                cell_x = bin_x if kind_x == "tracks_binned" else 1   # a pre-binned store's targets are per cell
                loss_x = (XH.split_head_loss(pieces, n_, want_x, x_ok, x_consts_on(n_), dev, kind_x, bin_x, cm_x)
                          if split_now else
                          XH.head_loss(model.last_x[n_], want_x[:, lo_x // cell_x:hi_x // cell_x].to(dev),
                                       x_ok[:, lo_x:hi_x].to(dev), x_consts_on(n_)(dev), kind_x, bin_x,
                                       cm_x.to(dev)))
                # a head's loss reaches the shared trunk ramped over its own warmup (review 23 Nov: a head added
                # in place, or on --init with --xhead-only 0, otherwise pulls the trunk at full weight while its
                # readout is still random); a head past its warmup counts in full, as before
                ramp_x = min(1.0, (xstage.state["updates"][n_] + 1) / max(xstage.plan.warmup_head, 1))
                mlm = mlm + xhead_weight * x_scale.get(n_, 1.0) * ramp_x * loss_x
                x_losses[n_].append(float(loss_x.detach()))
                x_step.add(n_)
            else:                       # held-out rows whole: log1p signal at 128 bp, or logits against labels
                rows_x = (x_ok.sum(1) > 0).to(dev)
                p_x, w_x = XH.eval_pair(xstores[n_], model.last_x[n_], want_x.to(dev), x_consts_on(n_)(dev),
                                        x_bin.get(n_, 1))
                pm_x = XH.positions_at(x_ok.to(dev), p_x.shape[1])          # held-out positions only, no padding
                acc_x = (x_pearson[n_] if x_where["genome"] is None else       # a further genome scores apart
                         x_pearson_g.setdefault((n_, x_where["genome"]), xstores[n_].scorer(x_bin.get(n_, 1))))
                acc_x.add(p_x[rows_x], w_x[rows_x], cm_x.to(dev)[rows_x], pm_x[rows_x])
        reg_t = None                    # the anti-collapse term of this micro-batch (JEPA_TRACE)
        if target is not None and model.training and model.last_jepa is not None:
            with torch.no_grad(), autocast:
                if split_now:           # the target's core states only: encoder pieces + core, no decoder; it reads
                    # the clean row and, with block JEPA, the repeat channel without the blocks zeroed (tgt_kw)
                    _, want_core = target.forward_split(seq.to(dev), seg.to(dev), ctx.to(dev) if ctx_on else None,
                                                        devices=split_devs, halo=split_halo or None, by=split_by,
                                                        core_only=True, **tgt_kw)
                else:
                    _, want_core, _ = target(seq.to(dev), seg.to(dev), ctx.to(dev) if ctx_on else None, **tgt_kw)
            keep_bins = seg.to(dev) == 0                                # not padding
            if jepa_loss == "cos":
                frame = None
                cos = F.cosine_similarity(model.last_jepa.float(), want_core.float(), dim=-1)
                dist = 1 - cos
            else:                       # --jepa-loss zcos: both sides in the target's standardised frame
                frame = jnorm_frame(want_core, keep_bins)
                zp_ = (model.last_jepa.float() - frame[0]) / frame[1]
                zt_ = (want_core.float() - frame[0]) / frame[1]
                cos = F.cosine_similarity(zp_, zt_, dim=-1)
                dist = 1 - cos + ((zp_ - zt_).pow(2).mean(-1) if jepa_loss == "zcos-mse" else 0.0)
            if blk_bins is None:
                jepa_losses.append(float(dist[keep_bins].mean().detach()))
                mlm = mlm + jepa_weight * dist[keep_bins].mean()
                z = ctx_core.float()[keep_bins]                         # [N, d] context core states, no padding
            else:
                # block JEPA: the loss is a mean over the hidden bins (--jepa-on all: over every non-padding bin).
                # The anti-collapse term below reads the online states of the VISIBLE bins only: those are the
                # states of real sequence (what the target produces), while a hidden bin's online state is the
                # model's summary of a blanked stretch and need not follow the same distribution.
                vis_bins = keep_bins & ~blk_bins
                on_bins = blk_bins if jepa_on == "hidden" else keep_bins
                jepa_term = dist[on_bins].mean() if bool(on_bins.any()) else cos.sum() * 0.0
                jepa_losses.append(float(jepa_term.detach()))
                mlm = mlm + jepa_weight * jepa_term
                z = ctx_core.float()[vis_bins]
            if parts_log is not None:                                   # jepa.csv: no gradient, no random numbers
                with torch.no_grad():
                    hid_ = blk_bins if blk_bins is not None else torch.zeros_like(keep_bins)
                    vis_ = keep_bins & ~hid_
                    d_pred = 1 - cos.detach()
                    if frame is None:
                        d_copy = 1 - F.cosine_similarity(ctx_core.detach().float(), want_core.float(), dim=-1)
                    else:               # zcos: the copy baseline in the same frame, so the two compare
                        d_copy = 1 - F.cosine_similarity((ctx_core.detach().float() - frame[0]) / frame[1],
                                                         (want_core.float() - frame[0]) / frame[1], dim=-1)
                    nan_ = float("nan")

                    def mean_on(x, m):
                        return float(x[m].mean()) if bool(m.any()) else nan_

                    def std_on(m):
                        zz = ctx_core.detach().float()[m]
                        return float(torch.sqrt(zz.var(0) + 1e-4).mean()) if zz.shape[0] > 1 else nan_
                    parts_log.update(jepa_hidden=mean_on(d_pred, hid_), jepa_visible=mean_on(d_pred, vis_),
                                     copy_hidden=mean_on(d_copy, hid_), copy_visible=mean_on(d_copy, vis_),
                                     hidden_share=float(hid_.sum()) / max(float(keep_bins.sum()), 1.0),
                                     core_std_hidden=std_on(hid_), core_std_visible=std_on(vis_))
            if z.shape[0] > 4096:                                       # a random subset keeps the terms cheap
                z = z[torch.randperm(z.shape[0], device=z.device)[:4096]]
            z = z - z.mean(0)
            std = torch.sqrt(z.var(0) + 1e-4)
            jepa_std.append(float(std.mean().detach()))
            # --jepa-guard on raises the regulariser's weight; at factor 1 it is the configured weight, untouched
            reg_w = jepa_reg_weight if guard["mult"] == 1.0 else jepa_reg_weight * guard["mult"]
            if jepa_reg == "vicreg":
                cov = (z.T @ z) / (z.shape[0] - 1)
                off = cov - torch.diag(torch.diag(cov))
                reg_t = torch.relu(1 - std).mean() + off.pow(2).sum() / z.shape[1]
                mlm = mlm + reg_w * reg_t
            elif jepa_reg == "sigreg":
                dirs = torch.randn(z.shape[1], 64, device=z.device)
                proj = (z / std.mean()) @ (dirs / dirs.norm(dim=0))     # [N, 64] 1-D projections
                tg = torch.linspace(-3, 3, 17, device=z.device)
                arg = proj.unsqueeze(-1) * tg                           # [N, 64, 17]
                ecf_re, ecf_im = torch.cos(arg).mean(0), torch.sin(arg).mean(0)
                gauss = torch.exp(-tg.pow(2) / 2)                       # N(0, 1)'s characteristic function
                ep = ((ecf_re - gauss).pow(2) + ecf_im.pow(2)) * gauss  # Epps-Pulley, Gaussian-weighted
                reg_t = ep.mean()
                mlm = mlm + reg_w * reg_t
        if parts_log is not None:
            parts_log["core_std"] = jepa_std[-1] if target is not None and model.last_jepa is not None else float("nan")
            parts_log["total"] = float(mlm.detach())
            jepa_rows.append(parts_log)
            if JEPA_TRACE is not None:  # tests: this micro-batch's terms, with the JEPA term and the unweighted
                # regulariser (not written to jepa.csv)
                JEPA_TRACE.append(dict(parts_log, jepa_term=jepa_losses[-1] if jepa_losses and target is not None
                                       and model.last_jepa is not None else float("nan"),
                                       reg=float(reg_t.detach()) if reg_t is not None else float("nan")))
        if lab is None:
            return mlm, None
        want = G.track_targets(lab.to(dev), act.to(dev))
        ok = valid.to(dev)
        if skip_at is not None:                                   # a mean over the visible positions
            ok = ok & ~skip_at
        per = F.binary_cross_entropy_with_logits(tracks.float(), want, pos_weight=pos_weight, reduction="none")
        track = (per * ok.unsqueeze(-1) * train_mask).sum() / (ok.sum().clamp_min(1) * train_mask.sum())
        if keep is not None:                                      # held-out scores for tracks.csv
            keep.append((tracks.float()[ok], want[ok]))
        return mlm, track

    def loss_parts(parts, **kw):
        """A training micro-batch from sample(): the tensors of one rectangular batch (--short-sources pad), or with
        ``pack`` its parts, one per row length - each a forward pass with its own targets, their losses added with
        the parts' shares of the micro-batch's windows as weights. The caller's single backward covers every part,
        so data-parallel cards (which sync gradients once per backward, and whose micro-batches now differ in shape
        and in the number of parts) stay in step; the parts together hold at most the micro-batch's bases.
        ``kw`` goes to loss_on (``unsplit``: --mito-slice)."""
        if not pack_short:
            return loss_on(*parts, **kw)
        loss = track = None
        for w_, rows_, sides_ in parts:
            epi_batch[:], ntv3_batch[:], rep_batch[:] = sides_
            l_, t_ = loss_on(*rows_, **kw)
            loss = w_ * l_ if loss is None else loss + w_ * l_
            if t_ is not None:
                track = w_ * t_ if track is None else track + w_ * t_
        return loss, track

    ntv3_losses: list[float] = []
    ntv3_scale = ntv3_nz = ntv3_rna = None
    ntv3_stats = None
    if ntv3:
        ntv3_scale = torch.from_numpy(ntv3["scales"]).to(dev)
        ntv3_rna = torch.from_numpy(ntv3["is_rna"]).to(dev)
        if ntv3["nzmean"] is not None:
            ntv3_nz = torch.from_numpy(ntv3["nzmean"]).to(dev)
        # running sums for Pearson per (split code, species): -1 is human (codes 3 and 4), then each further
        # species' test split, each over its own channels
        ntv3_stats = {(k, -1): [0] + [torch.zeros(ntv3["n_h"], dtype=torch.float64) for _ in range(5)] for k in (3, 4)}
        for x_i, e_x in enumerate(ntv3["extra"]):
            ntv3_stats[(3, x_i)] = [0] + [torch.zeros(e_x["hi"] - e_x["lo"], dtype=torch.float64) for _ in range(5)]

    def ntv3_ag(raw):
        """AlphaGenome/NTv3 scaling: /non-zero mean, **0.75 for RNA-seq, soft sqrt clip above 10."""
        x = raw.clamp_min(0) / ntv3_nz
        x = torch.where(ntv3_rna, x.clamp_min(1e-12).pow(0.75), x)
        return torch.where(x > 10, 2 * torch.sqrt(10 * x) - 10, x)

    def ntv3_ag_inv(y):
        x = torch.where(y > 10, ((y + 10) / 2).pow(2) / 10, y)
        x = torch.where(ntv3_rna, x.clamp_min(0).pow(1 / 0.75), x)
        return x * ntv3_nz

    jepa_losses: list[float] = []
    bench_losses: list[float] = []
    jepa_std: list[float] = []
    epi_keep: list = []
    chunk_rates: list[float] = []
    t0, started = time.time(), time.time()
    metrics = out / "metrics.csv"
    g_eval = [k for k, g_ in enumerate(genomes) if g_.val.height]     # further genomes with a held-out set
    header = ("step,lr,train_loss,val_bits_per_base,hours" + (",train_track_loss,val_track_loss" if labels else "")
              + (",val_mito_bits_per_base" if mito else "")
              + "".join(f",val_bits_per_base_{genomes[k].name}" for k in g_eval))   # new columns go last
    if main and not metrics.exists():
        metrics.write_text(header + "\n")
    elif main and metrics.read_text().split("\n", 1)[0] != header:
        say(f"warning: {metrics} has other columns than this run writes ({header}); rows are appended anyway")
    track_csv = out / "tracks.csv"
    if main and labels and not track_csv.exists():
        track_csv.write_text("step,track,auroc,average_precision,prevalence\n")
    # block JEPA (--jepa-block-share) and jepa.csv. The blocks have a random stream of their own (per card, and per
    # start step so a resumed run does not replay its first blocks); metrics.csv is not touched.
    block_rng = np.random.default_rng([seed, rank, 2029, step]) if jepa_block_share > 0 else None
    jepa_rows: list[dict] = []                        # each training micro-batch's terms since the last jepa.csv row
    jepa_secs, jepa_peak, jepa_steps = 0.0, 0.0, 0
    jepa_csv = out / "jepa.csv"
    jepa_cols = ("jepa_hidden", "jepa_visible", "copy_hidden", "copy_visible", "hidden_share", "core_std",
                 "core_std_visible", "core_std_hidden", "train_mlm", "train_epigenome", "train_ntv3", "train_total")
    jepa_keys = ("jepa_hidden", "jepa_visible", "copy_hidden", "copy_visible", "hidden_share", "core_std",
                 "core_std_visible", "core_std_hidden", "mlm", "epi", "ntv3", "total")
    if jepa_block_share > 0:
        say(f"block JEPA: {jepa_block_share:g} of each row's bins hidden in blocks of {block_lo * unit:,}-"
            f"{block_hi * unit:,} bases (log-uniform, {unit}-base bins); JEPA loss on {jepa_on} bins; other heads "
            f"{jepa_block_heads} block positions; {jepa_reg} on visible bins")
    jepa_head = ["step", *jepa_cols, "sec_per_step", "peak_gib", "window", "steps"]
    if jepa_guard != "off":             # the guard's rows (one per evaluation) share the file: row = train | guard
        jepa_head += ["row", "g_entropy_rank", "g_participation_ratio", "g_std_min", "g_std_mean", "g_std_median",
                      "g_jepa_hidden", "g_jepa_visible", "g_copy_hidden", "g_copy_visible", "g_top1_hidden",
                      "g_top1_chance", "g_reg_weight", "g_mult", "g_hits", "g_trigger_rank", "g_trigger_std",
                      "g_ref_max_rank", "g_ref_median_std_min", "g_action", "g_windows"]
    if jepa_log and not jepa_csv.exists():
        jepa_csv.write_text(",".join(jepa_head) + "\n")
    elif jepa_log:                      # a resumed run: keep its rows, widened (or kept wider) to one header
        lines_ = jepa_csv.read_text().rstrip("\n").split("\n")
        have_ = lines_[0].split(",")
        if have_ != jepa_head:
            if have_ == jepa_head[:len(have_)]:
                pad_ = ["train" if c_ == "row" else "" for c_ in jepa_head[len(have_):]]
                jepa_csv.write_text("\n".join([",".join(jepa_head)] + [",".join([ln_] + pad_) for ln_ in lines_[1:]
                                                                         if ln_]) + "\n")
            elif jepa_head == have_[:len(jepa_head)]:
                jepa_head = have_
            else:
                jepa_csv.rename(out / f"jepa.csv.before-step{step}")
                jepa_csv.write_text(",".join(jepa_head) + "\n")

    def jepa_write(vals: dict) -> None:
        """Append one jepa.csv row; ``vals`` maps column names to formatted text (missing: empty)."""
        with jepa_csv.open("a") as fh:
            fh.write(",".join(vals.get(c_, "") for c_ in jepa_head) + "\n")

    def jepa_row(step_: int, win_: int) -> None:
        """One jepa.csv row and log line: means over the training micro-batches since the last row (nan: no bin of
        that kind, or the term is off)."""
        nonlocal jepa_secs, jepa_peak, jepa_steps
        if not jepa_rows:
            return
        avg = []
        for k_ in jepa_keys:
            v_ = [r_[k_] for r_ in jepa_rows if k_ in r_ and not math.isnan(r_[k_])]
            avg.append(float(np.mean(v_)) if v_ else float("nan"))
        a_ = dict(zip(jepa_keys, avg))
        sec_ = jepa_secs / max(jepa_steps, 1)
        jepa_write({"step": str(step_), **{c_: f"{v_:.5f}" for c_, v_ in zip(jepa_cols, avg)},
                    "sec_per_step": f"{sec_:.3f}", "peak_gib": f"{jepa_peak:.2f}", "window": str(win_),
                    "steps": str(jepa_steps), "row": "train"})
        say(f"jepa step {step_}: hidden bins {a_['jepa_hidden']:.4f}, visible bins {a_['jepa_visible']:.4f}; copy "
            f"baseline hidden {a_['copy_hidden']:.4f}, visible {a_['copy_visible']:.4f}; hidden share "
            f"{a_['hidden_share']:.4f}; core std {a_['core_std']:.3f} (hidden bins {a_['core_std_hidden']:.3f}); "
            f"masked-base {a_['mlm']:.4f}, epigenome {a_['epi']:.4f}, NTv3 {a_['ntv3']:.4f}, total {a_['total']:.4f}; "
            f"{sec_:.2f} s/step, peak {jepa_peak:.1f} GiB, mean of {jepa_steps} steps")
        jepa_rows.clear()
        jepa_secs, jepa_peak, jepa_steps = 0.0, 0.0, 0

    # --jepa-guard: restore its state on resume (with off a saved state is ignored and the loss is the configured
    # one); fixed held-out monitoring windows (rank 0 measures, whatever the seed: the same windows in every run)
    if guard_saved is not None and guard_saved.get("from", "heldout") != jepa_guard_from and jepa_guard != "off":
        # the reference statistics were measured on other windows: keep the regulariser factor already applied
        # (the loss does not jump), restart the references on the new windows
        say(f"jepa guard: the checkpoint's guard watched {guard_saved.get('from', 'heldout')} windows, this run "
            f"watches {jepa_guard_from} windows: running references restart (factor x{guard_saved.get('mult', 1.0):g} kept)")
        guard_saved = {k_: v_ for k_, v_ in guard_saved.items() if k_ not in ("max_rank", "std_mins", "evals", "hits", "from")}
    guard["from"] = jepa_guard_from
    if guard_saved is not None:
        if jepa_guard == "on":
            guard.update({k_: v_ for k_, v_ in guard_saved.items() if k_ in guard})
            guard["stopped"] = False                # a resumed run gets its evaluations again
            say(f"jepa guard: state restored from the checkpoint: --jepa-reg weight x{guard['mult']:g}, "
                f"{guard['hits']} triggered evaluation(s) in a row, running max entropy rank "
                f"{guard['max_rank'] if guard['max_rank'] is None else round(guard['max_rank'], 2)}, "
                f"{len(guard['std_mins'])} reference minima")
        else:
            say(f"jepa guard: the checkpoint carries a guard state (--jepa-reg weight x{guard_saved.get('mult', 1.0):g}); "
                f"not applied with --jepa-guard {jepa_guard}")
            if jepa_guard == "monitor":
                guard.update({k_: v_ for k_, v_ in guard_saved.items() if k_ in ("max_rank", "std_mins", "evals")})
    guard_win = []
    if jepa_guard != "off" and main:
        g_rng = np.random.default_rng(20261117)
        g_src = train_idx if jepa_guard_from == "train" else val_idx
        g_word = "training" if jepa_guard_from == "train" else "held-out"
        g_rows = [r_ for r_ in g_src.iter_rows(named=True) if r_["len"] > jepa_guard_window]
        if not g_rows:
            raise typer.BadParameter(f"--jepa-guard-window {jepa_guard_window:,}: no {g_word} contig is that long")
        g_share = jepa_block_share if jepa_block_share > 0 else 0.2
        for _ in range(200 * jepa_guard_windows):
            if len(guard_win) == jepa_guard_windows:
                break
            r_ = g_rows[int(g_rng.integers(len(g_rows)))]
            at_ = int(r_["offset"]) + int(g_rng.integers(0, int(r_["len"]) - jepa_guard_window + 1))
            s_ = np.asarray(bases[at_:at_ + jepa_guard_window]).astype(np.int64)
            if (s_ < 4).mean() < 0.9:               # assembly gaps: no core states worth watching
                continue
            pick_ = (g_rng.random(jepa_guard_window) < mask_prob) & (s_ < 4)
            hid_ = G.block_mask(np.zeros((1, jepa_guard_window // unit), dtype=np.int64), s_[None], unit, g_share,
                                guard_lo, guard_hi, g_rng)[0]
            guard_win.append((s_, pick_, hid_))
        if not guard_win:
            raise typer.BadParameter(f"--jepa-guard: no {g_word} window with at least 90% known bases")
        say(f"jepa guard ({jepa_guard}): {len(guard_win)} fixed {g_word} windows of {jepa_guard_window:,} bases "
            f"({', '.join(sorted(set(g_src['chrom'].to_list())))[:200]}), blocks {g_share:g} of bins + {mask_prob:g} "
            f"scattered MASK; trigger: entropy rank < {jepa_guard_rank:g} x running max or min dim std < "
            f"{jepa_guard_std:g} x running median, two evaluations in a row"
            + (f"; then --jepa-reg weight x2 per evaluation up to x{jepa_guard_cap:g}, then stop (exit 42)"
               if jepa_guard == "on" else "; monitor only"))

    def guard_measure() -> dict:
        """Collapse measures on the fixed windows (inference; the online model, its predictor and the target)."""
        tgt_ = target if target is not None and jepa_ema > 0 else None     # EMA 0: the target is the model
        cl_, tg_, on_, pr_ = [], [], [], []
        with torch.no_grad(), autocast:
            for s_, pick_, hid_ in guard_win:
                x_ = torch.from_numpy(s_)[None].to(dev)
                seg_ = torch.zeros(1, s_.shape[0] // unit, dtype=torch.long, device=dev)
                _, c_, _ = model(x_, seg_)
                t_ = c_ if tgt_ is None else tgt_(x_, seg_)[1]
                xo_ = x_.masked_fill(torch.from_numpy(pick_)[None].to(dev), G.MASK_CHANNEL)
                xo_, _, _, _ = G.hide_blocks(xo_, None, torch.from_numpy(hid_)[None].to(dev), unit)
                _, o_, _ = model(xo_, seg_)
                if hasattr(model, "predict_jepa"):     # the context predictor reads the hidden bins
                    model.jepa_hidden = torch.from_numpy(hid_)[None].to(dev)
                    p_ = model.predict_jepa(o_)
                    model.jepa_hidden = None
                else:
                    p_ = model.jepa_pred(o_)
                for l_, v_ in ((cl_, c_), (tg_, t_), (on_, o_), (pr_, p_)):
                    l_.append(v_.float()[0])
        hid_all = torch.from_numpy(np.stack([h_ for _, _, h_ in guard_win])).to(dev)
        return G.jepa_guard_metrics(torch.stack(cl_), torch.stack(tg_), torch.stack(on_), torch.stack(pr_), hid_all)

    # --mito-slice: a stream of its own (per card, and per start step so a resumed run does not replay its first
    # slices), the further genomes' cumulative shares for the main draw without mito, and its own row tally
    slice_rng = np.random.default_rng([seed, rank, 2031, step]) if mito_slice else None
    cum_main = genome_cum / (1.0 - mito_share) if mito_slice else None
    slice_tally = PK.Tally()
    slice_secs, slice_steps, slice_peak = 0.0, 0, {}
    slice_rows: list[dict] = []                       # the slice's loss terms (jepa.csv keys) since the last report
    slice_losses: list[float] = []                    # its unweighted training loss per step
    model.train()
    end_step = stop_step or steps                     # TR02: evaluate and save at the step the run stops at
    while step < end_step:
        t_step = time.time()
        for g in opt.param_groups:
            g["lr"] = genome_lr(step, lr, warmup, steps, lr_total, cooldown_from)
        for d_ in (split_devs or [dev]):
            torch.cuda.reset_peak_memory_stats(d_)
        opt.zero_grad(set_to_none=True)
        win, bat = phase_at(step)
        if mito_slice:          # base-proportional weights: the main draw's bases and the slice's
            main_bases = win * bat * grad_accum
            w_main, w_slice = main_bases / (main_bases + mito_slice), mito_slice / (main_bases + mito_slice)
        x_read["heads"] = x_groups[step % len(x_groups)] if x_groups else None    # --xhead-cycle
        for _ in range(grad_accum):
            if whole_contigs:
                loss, track = contig_loss(*sample_contig(train_idx, longest=bench))
            else:
                loss, track = loss_parts(sample(train_idx, window=win, batch=bat, flip=rc,
                                                mito_table=mito_train if mito and not mito_slice else None,
                                                share=mito_share / (1.0 - genome_total),   # absolute (x / 1.0 == x)
                                                annot=annot_oversample,
                                                extra=ntv3_extra_share if ntv3 and ntv3["extra"] else 0.0,
                                                genomes=genomes, rep=repeat_channel, pack=pack_short, cum=cum_main))
            if mito_slice:
                loss = loss * w_main
                track = None if track is None else track * w_main
            ((loss + track_weight * track if track is not None else loss) / grad_accum).backward()
            if bench_grads is not None and step == 0:
                bench_losses.append(float(loss.detach()))
        if jepa_edits > 0 and step % jepa_edit_every == 0:   # --jepa-edits: its own window and backward
            e_loss, e_st = edit_loss()
            if e_loss is not None:
                (jepa_edits * e_loss).backward()
                if main:
                    with open(out / "edits.csv", "a") as fh_:
                        if fh_.tell() == 0:
                            fh_.write("step,kind,length,edited,delta_mse,target_change,delta_cos\n")
                        fh_.write(f"{step + 1},{e_st['kind']},{e_st['length']},{e_st['edited']:.5f},"
                                  f"{e_st['delta_mse']:.5f},{e_st['target_change']:.5f},{e_st['delta_cos']:.4f}\n")
        if cf is not None and step % cf_every == 0:   # --cf-pairs: the sign loss, its own backward
            cf_loss, cf_st = CF.pair_loss(model, cf.sample(cf_rng, cf_batch), cf_contigs,
                                          lambda a_, b_: XH.read_rows(bases, a_, b_), cf_consts, XH.signal_of,
                                          cf_window, cf_temp, dev, autocast)
            if cf_loss is not None:
                (cf_weight * cf_loss).backward()
                if main:
                    with open(out / "cf.csv", "a") as fh_:
                        if fh_.tell() == 0:
                            fh_.write("step,pairs,signed,agree,loss\n")
                        fh_.write(f"{step + 1},{cf_st['pairs']},{cf_st['signed']},{cf_st['agree']:.4f},"
                                  f"{cf_st['loss']:.5f}\n")
        x_read["heads"] = None          # every other draw (the --mito-slice part, evaluation) reads every store
        if mito_slice:          # after the main draw's backward: its activations are freed; one optimizer step
            if dev.type == "cuda":                          # time the slice alone (the main backward has finished)
                torch.cuda.synchronize(dev)
            t_slice = time.time()
            # the slice's logged terms are kept apart, so jepa.csv's train rows and the evaluation line's JEPA and
            # NTv3 train means stay those of the main draw (comparable across the switch)
            n_logged = (len(jepa_rows), len(jepa_losses), len(jepa_std), len(ntv3_losses))
            s_loss, s_track = loss_parts(sample(mito_train, window=mito_slice, batch=1, gen=slice_rng, flip=rc,
                                                mito_table=mito_train, share=1.0, rep=repeat_channel, pack=True,
                                                tally_to=slice_tally), unsplit=True)
            slice_rows.extend(jepa_rows[n_logged[0]:])
            del jepa_rows[n_logged[0]:], jepa_losses[n_logged[1]:], jepa_std[n_logged[2]:], ntv3_losses[n_logged[3]:]
            slice_losses.append(float(s_loss.detach()))
            s_loss = s_loss * w_slice
            s_track = None if s_track is None else s_track * w_slice
            (s_loss + track_weight * s_track if s_track is not None else s_loss).backward()
            loss = loss.detach() + s_loss.detach()          # the step's training loss: both parts, base-weighted
            if bench_grads is not None and step == 0:
                bench_losses.append(float(s_loss.detach()))
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            slice_secs += time.time() - t_slice
            slice_steps += 1
            for d_ in (split_devs or [dev]):
                slice_peak[str(d_)] = max(slice_peak.get(str(d_), 0.0), torch.cuda.max_memory_allocated(d_) / 2**30)
        if bench_grads is not None and step == 0:              # the first step's gradients, before clipping
            torch.save(dict(losses=bench_losses, grads={n_: (None if p_.grad is None else p_.grad.detach().cpu())
                                                        for n_, p_ in model.named_parameters()}), bench_grads)
        if xstage is not None and not xstage.inherited_trains:   # head-only: the inherited weights' gradients go,
            for n_, p_ in model.named_parameters():                # so AdamW leaves their moments and step count as
                if not n_.startswith("xheads."):                   # the checkpoint had them and they do not enter the
                    p_.grad = None                                 # clip norm (review 23 Nov)
        if step < freeze_trunk_until:   # --freeze-trunk-until: the predictor's warm-up on the frozen model
            for n_, p_ in model.named_parameters():
                if not n_.startswith("jepa_pred."):
                    p_.grad = None
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if xstage is not None:          # each group's rate: the run's schedule (the line above) as a share of --lr,
            x_rates = XH.set_rates(opt, xstage, sorted(x_step),               # times the group's own peak and warmup
                                   genome_lr(step, lr, warmup, steps, lr_total, cooldown_from) / lr)
        opt.step()
        if xstage is not None:
            if xstage.stepped(sorted(x_step)) == "joint":
                say(f"--xhead: head-only phase done after step {step + 1} ({xstage.state['updates']} labelled "
                    f"updates); the inherited weights train from here, up to {inherit_lr:g} over {inherit_warmup} "
                    "updates")
                if main:                # the head-only milestone: weights and stage, beside the run's checkpoint
                    _save_atomic({"model": model.state_dict(), "config": config.to_dict(), "step": step + 1,
                                  "xstage": {"plan": xstage.plan.to_dict(), "state": xstage.state},
                                  "run_record": record}, out / f"head-only-step{step + 1}.pt")
            x_step.clear()
        if target is not None:
            with torch.no_grad():
                for pt, pm in zip(target.parameters(), model.parameters()):
                    pt.lerp_(pm.detach(), 1 - jepa_ema)
        step += 1
        if bench and step == 1:                                  # the first step compiles kernels: time the second
            for d_ in (split_devs or [None]):
                torch.cuda.synchronize(d_)
            t0 = time.time()
            continue
        if bench:
            for d_ in (split_devs or [None]):
                torch.cuda.synchronize(d_)
            what = f"longest contig {min(train_idx['len'].max(), max_contig or 1 << 62):,} bases in {window:,} chunks, {train_chunks} scored" \
                if whole_contigs else f"window {win:,}, batch {bat} x {grad_accum} accumulated"
            if split_devs is not None:
                what += f", split over {len(split_devs)} cards"
                peak_ = ", ".join(f"{d_} {torch.cuda.max_memory_allocated(d_) / 2**30:.2f}" for d_ in split_devs)
                say(f"bench: {what}: {time.time() - t0:.2f} s for one step, peak {peak_} GiB, loss {loss.item():.3f}")
                return
            say(f"bench: {what}: {time.time() - t0:.1f} s for one step, "
                       f"peak {torch.cuda.max_memory_allocated(dev) / 2**30:.1f} GiB, loss {loss.item():.3f}")
            return
        out_of_time = max_hours > 0 and time.time() - started > max_hours * 3600
        if world > 1:                                   # every card stops at the same step
            flag = torch.tensor([float(out_of_time)], device=dev)
            dist.broadcast(flag, 0)
            out_of_time = bool(flag.item())
        if jepa_log:
            jepa_secs += time.time() - t_step
            jepa_steps += 1
            jepa_peak = max(jepa_peak, max(torch.cuda.max_memory_allocated(d_) / 2**30  # split: the busiest card
                                           for d_ in (split_devs or [dev])))
            if (step % eval_every == 0 or step == steps or step == end_step or out_of_time
                    or (jepa_log_every > 0 and step % jepa_log_every == 0)):
                jepa_row(step, win)
        if (step % eval_every == 0 or step == steps or step == end_step or out_of_time) and not main:
            dist.barrier()                              # rank 0 evaluates and saves
        elif step % eval_every == 0 or step == steps or step == end_step or out_of_time:
            model.eval()
            keep: list = []
            eval_rng = np.random.default_rng([seed, step])               # the same windows for every run with this seed
            mask_gen = torch.Generator(device=dev).manual_seed(seed * 1_000_003 + step)
            with torch.no_grad():
                pairs = ([contig_loss(*sample_contig(val_idx), keep=keep if labels else None) for _ in range(2)]
                         if whole_contigs else
                         [loss_on(*sample(val_idx, window=eval_window, batch=1 if arch in ("v2", "v2l", "v3") else batch_size,
                                          gen=eval_rng),
                                  keep=keep if labels else None, mask_gen=mask_gen,
                                  score_ntv3=mlm_weight != 0) for _ in range(8)])
                if ntv3 is not None and not whole_contigs:   # NTv3's test split lies on chromosomes the
                    for _ in range(24):                          # masked-LM trains on, so draw from there too;
                        # only ~1.6% of a random training window is split==3, so draw enough to cover regions
                        loss_on(*sample(train_idx, window=eval_window, batch=1, gen=eval_rng),
                                mask_gen=mask_gen, track_only=True,   # scored only via the split==3 accumulator
                                clean=mlm_weight == 0)
                    if mlm_weight == 0:      # a track stage is scored as it trains, on unmasked input: the same
                        again = np.random.default_rng([seed, step])   # eight held-out windows once more, unmasked
                        for _ in range(8):
                            loss_on(*sample(val_idx, window=eval_window, batch=1, gen=again), track_only=True,
                                    clean=True)
                    x_rng = np.random.default_rng([seed, step, 7])    # its own stream: the paired draws above and
                    x_gen = torch.Generator(device=dev).manual_seed(seed * 1_000_003 + step + 7)   # below stay put
                    for x_i in range(len(ntv3["extra"])):             # each further species: windows on its test split
                        for _ in range(8):
                            loss_on(*sample(train_idx, window=min(eval_window, 262144), batch=1, gen=x_rng,
                                            extra_force=x_i), mask_gen=x_gen, track_only=True, clean=mlm_weight == 0)
                mito_bits = None
                if mito:
                    mw = -(-int(mito_val["len"].max()) // max(unit, 128)) * max(unit, 128)
                    mito_bits = float(np.mean([loss_on(*sample(mito_val, window=mw, batch=4, gen=eval_rng,
                                                               mito_table=mito_val, share=1.0), mask_gen=mask_gen)[0].item()
                                               for _ in range(8)])) / math.log(2)
                genome_bits = []
                for k in g_eval:   # each further genome on its own held-out contigs, from streams of its own so the
                    g_rng = np.random.default_rng([seed, step, 1009, k])     # paired draws above stay bit-for-bit
                    g_mask = torch.Generator(device=dev).manual_seed(int(g_rng.integers(1 << 62)))
                    x_where["genome"] = genomes[k].name                      # its head scores go to their own rows
                    genome_bits.append(float(np.mean([loss_on(*sample(
                        genomes[k].val, window=eval_window, batch=batch_size if arch == "v1" else 1,  # as above
                        gen=g_rng, genomes=genomes, force_genome=k), mask_gen=g_mask, track_only=True,
                        score_ntv3=False)[0].item() for _ in range(8)])) / math.log(2))
                    x_where["genome"] = None
            guard_note = None
            if jepa_guard != "off":         # collapse measures on the fixed windows, then the guard's decision
                t_guard = time.time()
                gm_ = guard_measure()
                t_guard = time.time() - t_guard
                gi_ = G.jepa_guard_update(guard, gm_["entropy_rank"], gm_["std_min"], act=jepa_guard == "on",
                                          rank_frac=jepa_guard_rank, std_frac=jepa_guard_std, cap=jepa_guard_cap)
                reg_now = jepa_reg_weight * guard["mult"] if jepa_reg != "none" else 0.0
                fmt_ = lambda v_: "" if v_ is None else (f"{v_:.6g}" if isinstance(v_, float) else str(v_))  # noqa: E731
                if jepa_log:
                    jepa_write({"step": str(step), "row": "guard", "window": str(jepa_guard_window),
                                **{f"g_{k_}": fmt_(gm_[k_]) for k_ in ("entropy_rank", "participation_ratio", "std_min",
                                                                       "std_mean", "std_median", "jepa_hidden",
                                                                       "jepa_visible", "copy_hidden", "copy_visible",
                                                                       "top1_hidden", "top1_chance", "windows")},
                                "g_reg_weight": fmt_(float(reg_now)), "g_mult": fmt_(float(guard["mult"])),
                                "g_hits": str(gi_["hits"]), "g_trigger_rank": str(int(gi_["trigger_rank"])),
                                "g_trigger_std": str(int(gi_["trigger_std"])), "g_ref_max_rank": fmt_(gi_["ref_max_rank"]),
                                "g_ref_median_std_min": fmt_(gi_["ref_median_std_min"]), "g_action": gi_["action"]})
                guard_note = (f"jepa guard step {step}: entropy rank {gm_['entropy_rank']:.2f}, participation ratio "
                              f"{gm_['participation_ratio']:.2f}, dim std min {gm_['std_min']:.4f} / mean "
                              f"{gm_['std_mean']:.4f}; JEPA hidden {gm_['jepa_hidden']:.4f}, visible "
                              f"{gm_['jepa_visible']:.4f}, copy hidden {gm_['copy_hidden']:.4f}; top-1 hidden "
                              f"{gm_['top1_hidden']:.4f} (chance {gm_['top1_chance']:.4f}); --jepa-reg weight "
                              f"{reg_now:g} (x{guard['mult']:g}); {gi_['action']}; {t_guard:.1f} s")
                if gi_["action"] != "ok":
                    why_ = " and ".join(w_ for w_, t_ in (
                        (f"entropy rank {gm_['entropy_rank']:.2f} < {jepa_guard_rank:g} x running max "
                         f"{gi_['ref_max_rank']}", gi_["trigger_rank"]),
                        (f"min dim std {gm_['std_min']:.4g} < {jepa_guard_std:g} x running median "
                         f"{gi_['ref_median_std_min']}", gi_["trigger_std"])) if t_)
                    what_ = {"triggered": "first triggered evaluation, no action yet",
                             "raise": f"--jepa-reg weight raised to x{guard['mult']:g} ({reg_now:g})",
                             "would-raise": "would raise the --jepa-reg weight (monitor only)",
                             "stop": f"still triggered at the cap x{jepa_guard_cap:g}: saving and stopping"}[gi_["action"]]
                    guard_note += f"\nWARNING: JEPA collapse guard at step {step}: {why_} ({gi_['hits']} in a row): {what_}"
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
            if ntv3 is not None:
                names = {(3, -1): "NTv3 test split", (4, -1): "chr8/chr9 (never trained on)"}
                for x_i, e_x in enumerate(ntv3["extra"]):
                    names[(3, x_i)] = f"{e_x['name']} test split"
                how = "unmasked input" if mlm_weight == 0 else "masked input"
                cols = []
                for key_ in names:
                    n_, sx, sy, sxy, sxx, syy = ntv3_stats[key_]
                    if n_ == 0:
                        continue
                    cov = sxy / n_ - (sx / n_) * (sy / n_)
                    sd = ((sxx / n_ - (sx / n_) ** 2).clamp_min(0).sqrt()
                          * (syy / n_ - (sy / n_) ** 2).clamp_min(0).sqrt())
                    r_ = torch.where(sd > 0, cov / sd.clamp_min(1e-12), torch.zeros_like(cov))
                    say(f"NTv3 tracks [{names[key_]}]: Pearson(log1p) mean {float(r_.mean()):.4f} "
                        f"(min {float(r_.min()):.3f}, max {float(r_.max()):.3f}, {len(r_)} tracks, {n_:,} bases, {how})")
                    cols.append((str(key_[0]) if key_[1] < 0 else f"3:{ntv3['extra'][key_[1]]['name']}", r_))
                    ntv3_stats[key_][0] = 0
                    for t_ in ntv3_stats[key_][1:]:
                        t_.zero_()
                if ntv3["extra"] and len(cols) > 2:
                    say(f"NTv3 tracks [further species]: mean of {len(cols) - 2} species means "
                        f"{float(np.mean([float(r_.mean()) for _, r_ in cols[2:]])):.4f}")
                if ntv3_losses:
                    say(f"NTv3 tracks: train {'MSE' if ntv3_loss == 'mse' else 'Poisson-multinomial'} "
                        f"{np.mean(ntv3_losses):.4f} ({ntv3_target} units)")
                if main and cols:
                    with (out / "ntv3_tracks.csv").open("a") as fh:
                        for code_, r_ in cols:
                            fh.write(f"{step},{code_},{float(r_.mean()):.5f},{float(r_.min()):.5f},"
                                     f"{float(r_.max()):.5f}," + ",".join(f"{v:.5f}" for v in r_.tolist()) + "\n")
                ntv3_losses.clear()
            if xstage is not None:
                x_csv = out / "xheads.csv"
                if main and not x_csv.exists():
                    x_csv.write_text("step,head,phase,stage_step,updates,lr,lr_inherited,train_loss,heldout_metric,"
                                     "heldout_mean,heldout_n,heldout_per_channel\n")
                for n_ in xstage.plan.heads:
                    r_x = x_pearson[n_].value()
                    ok_x = r_x[~np.isnan(r_x)]
                    mean_x = float(ok_x.mean()) if len(ok_x) else float("nan")
                    tl_x = float(np.mean(x_losses[n_])) if x_losses[n_] else float("nan")
                    met_x = ("average precision" if n_ in xstores and xstores[n_].kind == "binary" else
                             "Pearson(log1p, 128 bp)")
                    say(f"--xhead {n_} [{xstage.state['phase']}]: {xstage.state['updates'][n_]} labelled updates, train "
                        f"{tl_x:.4f}, chr8/chr9 {met_x} mean {mean_x:.4f} over {len(ok_x)} of {len(r_x)} "
                        f"channels, n {XH.scored_n(x_pearson[n_]):,}")
                    if main:
                        with x_csv.open("a") as fh:
                            fh.write(f"{step},{n_},{xstage.state['phase']},{xstage.state['stage_step']},"
                                     f"{xstage.state['updates'][n_]},{x_rates.get(n_, 0.0):.3e},"
                                     f"{x_rates.get('inherited', 0.0):.3e},{tl_x:.5f},{met_x.split('(')[0].strip().replace(' ', '_')},"
                                     f"{mean_x:.5f},{XH.scored_n(x_pearson[n_])},"
                                     + " ".join(f"{v_:.5f}" for v_ in r_x) + "\n")
                    x_pearson[n_] = (xstores[n_].scorer(x_bin.get(n_, 1)) if n_ in xstores
                                     else XH.Pearson(xhead_sizes[n_]))
                    x_losses[n_].clear()
                if x_pearson_g:     # further genomes' held-out head scores, each on its own contigs
                    g_csv = out / "xheads_genomes.csv"
                    if main and not g_csv.exists():
                        g_csv.write_text("step,head,genome,heldout_metric,heldout_mean,heldout_n,heldout_per_channel\n")
                    for (n_, g_), sc_ in sorted(x_pearson_g.items()):
                        r_g = sc_.value()
                        ok_g = r_g[~np.isnan(r_g)]
                        mean_g = float(ok_g.mean()) if len(ok_g) else float("nan")
                        met_g = "average_precision" if xstores[n_].kind == "binary" else "Pearson"
                        say(f"--xhead {n_} on {g_} held-out: {met_g} mean {mean_g:.4f} over {len(ok_g)} of "
                            f"{len(r_g)} channels, n {XH.scored_n(sc_):,}")
                        if main:
                            with g_csv.open("a") as fh:
                                fh.write(f"{step},{n_},{g_},{met_g},{mean_g:.5f},{XH.scored_n(sc_)},"
                                         + " ".join(f"{v_:.5f}" for v_ in r_g) + "\n")
                    x_pearson_g.clear()
            if epi_keep:
                sc = torch.cat([k[0] for k in epi_keep])
                tr = torch.cat([k[1] for k in epi_keep])
                aucs = [G.ranking_scores(sc[:, j], tr[:, j])[0] for j in range(sc.shape[1]) if 0 < tr[:, j].sum() < len(tr)]
                epi_note = f"epigenome AUROC mean {np.mean(aucs):.4f} (min {np.min(aucs):.3f}, {len(aucs)} epigenomes)"
                if main:
                    with (out / "epigenomes.csv").open("a") as fh:
                        fh.write(f"{step},{np.mean(aucs):.4f},{np.median(aucs):.4f},{np.min(aucs):.4f},"
                                 f"{float(tr.float().mean()):.5f}\n")
                say(epi_note)
                epi_keep.clear()
            tracks_note = extra
            if mito_bits is not None:
                extra += f",{mito_bits:.4f}"
            extra += "".join(f",{b_:.4f}" for b_ in genome_bits)
            with metrics.open("a") as fh:
                fh.write(f"{step},{opt.param_groups[0]['lr']:.3e},{loss.item():.4f},{bits:.4f},{h:.3f}{extra}\n")
            if guard_note:
                say(guard_note)
            say(f"step {step} (window {win:,} x {bat}): train {loss.item():.4f}, held-out {bits:.3f} bits/base"
                       + (f", mito {mito_bits:.3f}" if mito_bits is not None else "")
                       + "".join(f", {genomes[k].name} {b_:.3f}" for k, b_ in zip(g_eval, genome_bits))
                       + (f", jepa {np.mean(jepa_losses):.4f} (core std {np.mean(jepa_std):.3f})" if jepa_losses else "")
                       + (f", chunk every {unit / np.mean(chunk_rates):.0f} bp" if chunk_rates else "") + f", {h:.2f} h, "
                       + (f"peak {torch.cuda.max_memory_allocated(dev) / 2**30:.1f} GiB" if split_devs is None else
                          "peak " + ", ".join(f"{d_} {torch.cuda.max_memory_allocated(d_) / 2**30:.1f}"
                                              for d_ in split_devs) + " GiB")
                       + (f", tracks {tracks_note}" if tracks_note else ""))
            if tally.by:
                say(f"training rows since the last report (--short-sources {short_sources}): {tally.line()}")
            if slice_tally.by:
                terms_ = {k_: [r_[k_] for r_ in slice_rows if k_ in r_ and not math.isnan(r_[k_])]
                          for k_ in ("mlm", "jepa_hidden", "total")}
                say(f"mito slice rows since the last report (--mito-slice {mito_slice:,}, one packed window per "
                    f"step): {slice_tally.line()}; slice loss {np.mean(slice_losses):.4f} (unweighted"
                    + "".join(f", {k_} {np.mean(v_):.4f}" for k_, v_ in terms_.items() if v_)
                    + f"), {slice_secs / max(slice_steps, 1):.3f} s/step; peak per card over the {slice_steps} "
                    f"steps: " + ", ".join(f"{k_} {v_:.2f}" for k_, v_ in slice_peak.items()) + " GiB")
                slice_secs, slice_steps, slice_peak = 0.0, 0, {}
                slice_rows.clear()
                slice_losses.clear()
            jepa_losses.clear()
            jepa_std.clear()
            chunk_rates.clear()
            _save_atomic({"model": model.state_dict(), "optimizer": opt.state_dict(), "config": config.to_dict(),
                          "step": step, "hours": h,
                          **({"ntv3": {"ids": list(ntv3["ids"]), "loss": ntv3_loss, "target": ntv3_target,
                                       "center": ntv3_center, "extra": [e_x["name"] for e_x in ntv3["extra"]],
                                       "mlm_weight": mlm_weight}} if ntv3 else {}),
                          "run_record": record,
                          **({"xstage": {"plan": xstage.plan.to_dict(), "state": xstage.state}}
                             if xstage is not None else {}),
                          **({"jepa_target": target.state_dict()} if target is not None and jepa_ema > 0 else {}),
                          **({"jepa_guard": dict(guard, std_mins=list(guard["std_mins"]))} if jepa_guard != "off" else {}),
                          **({"jepa_norm": {k_: v_.cpu() for k_, v_ in jnorm.items()}} if jnorm else {}),
                          **({"genomes": [dict(name=g_.name, share=g_.share, context=g_.context,
                                               held_out=g_.val["chrom"].unique(maintain_order=True).to_list())
                                          for g_ in genomes]} if genomes else {})}, ckpt)
            if world > 1:
                dist.barrier()
        if jepa_guard == "on" and (step % eval_every == 0 or step == steps or step == end_step or out_of_time):
            if world > 1:                               # rank 0 decided; every card takes its factor and verdict
                flag_ = torch.tensor([guard["mult"], float(guard["stopped"])], device=dev, dtype=torch.float64)
                dist.broadcast(flag_, 0)
                guard["mult"], guard["stopped"] = float(flag_[0].item()), bool(flag_[1].item())
            if guard["stopped"]:
                msg_ = (f"JEPA collapse guard: stopped at step {step} (exit code 42): collapse still detected with the "
                        f"--jepa-reg weight at x{guard['mult']:g}; checkpoint {ckpt} saved at this step")
                say(msg_)
                if main:
                    (out / "JEPA_GUARD_STOP").write_text(msg_ + "\n")
                if world > 1:
                    dist.destroy_process_group()
                raise typer.Exit(42)
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
                   rc_average: bool = typer.Option(False, help="also score the reverse-complement strand and add "
                                                   "strand-averaged scores (<run>@rc-avg...) and the strand agreement"),
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
    keys = run_keys(runs)                       # checked before any scoring (audit 2026-11-18, EV-03)
    for run, key in zip(runs, keys):
        ckpt = run / "checkpoint.pt" if run.is_dir() else run
        state = torch.load(ckpt, map_location="cpu")
        model = G.KronGenome(G.KronGenomeConfig(**state["config"])).to(dev)
        model.load_state_dict(state["model"])
        model.eval()
        t0 = time.time()
        scores[key] = GV.masked_marginals(model, bases, g, table["ref_code"].to_numpy(), table["alt_code"].to_numpy(),
                                          t_start, t_len, window, dev, batch_size)
        typer.echo(f"{key} (step {state.get('step')}): {len(np.unique(t_start))} tiles in {time.time() - t0:.0f} s")
        if rc_average:
            back = GV.masked_marginals(model, bases, g, table["ref_code"].to_numpy(), table["alt_code"].to_numpy(),
                                       t_start, t_len, window, dev, batch_size, rc=True)
            scores[f"{key}@rc-avg"] = (scores[key] + back) / 2
            typer.echo(f"{key}: masked marginal, strand agreement (Pearson) {np.corrcoef(scores[key], back)[0, 1]:.3f}")
        if model.tracks is not None and delta_window:
            t0 = time.time()
            d = GV.track_deltas(model, bases, g, table["ref_code"].to_numpy(), table["alt_code"].to_numpy(),
                                c_off, c_len, delta_window, dev, delta_batch)
            splice = [G.TRACKS.index("donor"), G.TRACKS.index("acceptor")]
            scores[f"{key}:splice-delta"] = -d[:, splice].max(1)        # lower = more damaging, as the marginal
            scores[f"{key}:track-delta"] = -d.max(1)
            if rc_average:
                d2 = GV.track_deltas(model, bases, g, table["ref_code"].to_numpy(), table["alt_code"].to_numpy(),
                                     c_off, c_len, delta_window, dev, delta_batch, rc=True)
                scores[f"{key}:splice-delta@rc-avg"] = -(d[:, splice] + d2[:, splice]).max(1) / 2
                scores[f"{key}:track-delta@rc-avg"] = -((d + d2) / 2).max(1)
                typer.echo(f"{key}: splice delta, strand agreement (Pearson) "
                           f"{np.corrcoef(d[:, splice].max(1), d2[:, splice].max(1))[0, 1]:.3f}")
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


@app.command("export-genome-onnx")
def export_genome_onnx_cmd(run: Path = typer.Argument(..., help="a `genome-train` run directory or checkpoint"),
                           out: Path | None = typer.Option(None, help="output directory (default: <run>/onnx-<window>)"),
                           window: int = typer.Option(131072, help="bases the graph reads per row (a multiple of the "
                                                                   "core step); the recurrence is unrolled, so the "
                                                                   "graph and the export time grow with it"),
                           opset: int = typer.Option(18, help="ONNX opset"),
                           ntv3: bool = typer.Option(True, help="include the NTv3 tracks output"),
                           check_kernels: bool = typer.Option(True, help="compare with the Triton kernels when CUDA "
                                                                         "is present"),
                           optimize: bool = typer.Option(False, help="run the exporter's ONNX optimisation pass "
                                                                     "(slow at full windows; ONNX Runtime optimises "
                                                                     "on load anyway)")) -> None:
    """Export a KronGenome checkpoint to ONNX (base logits, NTv3 tracks, further heads) and check ONNX Runtime
    against PyTorch; see kron_dna.genome_onnx."""
    from kron_dna.genome_onnx import export_genome_onnx

    out = out or (run if run.is_dir() else run.parent) / f"onnx-{window}"
    meta = export_genome_onnx(run, out, window=window, opset=opset, check_kernels=check_kernels, with_ntv3=ntv3,
                              optimize=optimize)
    typer.echo(f"wrote {out / meta['graph']} ({meta['bytes'] / 1e6:,.1f} MB, {meta['export_seconds']} s)")
    typer.echo(json.dumps(meta["verification"], indent=2))


if __name__ == "__main__":
    app()
