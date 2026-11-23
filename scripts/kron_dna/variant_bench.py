"""Reference against alternative allele through a track head: the variant-effect scorer behind the allele-specific
binding benchmark (ADASTRA, ``prep_asb.py``) and any table of the same shape (MPRA saturation mutagenesis, eQTLs).

    uv run python scripts/kron_dna/variant_bench.py score OUT LABEL=CHECKPOINT [...] --variants T.parquet --head chiptf
        [--where "match = 'tf_cell'"] [--max-variants N] [--read-col COL] [--flank 192] [--context 131072]
        [--batch 1] [--dtype bf16|fp16|fp32] [--shard i/n] [--chunk 131072]
    uv run python scripts/kron_dna/variant_bench.py merge OUT LABEL --shards n       (after every --shard i/n)
    uv run python scripts/kron_dna/variant_bench.py report-asb OUT [--variants T.parquet]
    uv run python scripts/kron_dna/variant_bench.py report-mpra OUT | report-eqtl OUT | report-lentimpra OUT

``score``: the table's distinct sites (``chrom``, ``pos`` 0-based, ``ref``, ``alt``, and the read position: ``pos``,
or ``--read-col``, e.g. a gene's TSS). One window of ``--context`` bases per site, centred between the variant and
the read position (128-bp grid), read twice: the assembly's sequence and the same with ``ref`` replaced by ``alt``
(an indel shifts what follows; the window keeps its length, cut or N-padded at its end). The head runs on the bins
around the read position; a site's readout is the predicted signal (``expm1`` of the head's log1p) summed over the
bins within ``--flank`` of it, for **every** channel of the head's store: ``ref.npy`` and ``alt.npy``,
``[sites, channels]``, with ``sites.parquet`` and the channel names in ``index.json``, so a report picks channels
afterwards. Sites whose ``ref`` is not the assembly's, or longer than 64 bases, are left out and counted.

``report-asb``: per row of the ASB table, the prediction is ``log2((alt + 1) / (ref + 1))`` averaged over the row's
channels (its TF in its cell type). Scored against the favoured allele: AUROC of the prediction for "alt binds
more", sign accuracy, and Spearman against the measured effect size; beside two baselines on the same rows, the
motif fold change (HOCOMOCO, in the table) and the same prediction averaged over every channel of the head (is the
TF's own channel better than a generic one). Per match tier, hg38 chr8/chr9 (held out) apart, with 95 % bootstrap
intervals.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import typer

from kron_core.paths import interim_dir
from kron_dna import evalkit as EK
from kron_dna import genome as G
from kron_dna import xheads as X

app = typer.Typer(add_completion=False)
BIN = 128
EPS = 1.0
CODE = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
KEY = ["chrom", "pos", "ref", "alt", "read"]
STORES = {"chiptf": "chip-tf-v1-bin128", "chiphr": "chip-histone-rest-v1-bin128", "chip492b": "chip-histone-core-v1-bin128",
          "dnase196b": "dnase-biosamples-v1-bin128", "rna462b": "rnaseq-stranded-v1-bin128"}


def codes_of(contigs: pl.DataFrame, bases: np.ndarray, chrom: str, lo: int, hi: int) -> np.ndarray:
    out = np.full(hi - lo, 4, dtype=np.int64)
    for r in contigs.filter((pl.col("chrom") == chrom) & (pl.col("start") < hi) & (pl.col("end") > lo)).iter_rows(named=True):
        a, b = max(lo, r["start"]), min(hi, r["end"])
        out[a - lo:b - lo] = bases[r["offset"] + a - r["start"]:r["offset"] + b - r["start"]]
    return out


def with_alt(seq: np.ndarray, at: int, ref: str, alt: str) -> np.ndarray | None:
    """``seq`` with ``ref`` at ``at`` replaced by ``alt``, at ``seq``'s length; None when ``seq`` does not hold ``ref``
    there."""
    r = np.array([CODE.get(c, 4) for c in ref.upper()], dtype=seq.dtype)
    if at < 0 or at + len(r) > len(seq) or not np.array_equal(seq[at:at + len(r)], r):
        return None
    a = np.array([CODE.get(c, 4) for c in alt.upper()], dtype=seq.dtype)
    new = np.concatenate([seq[:at], a, seq[at + len(r):]])
    if len(new) < len(seq):
        new = np.concatenate([new, np.full(len(seq) - len(new), 4, dtype=seq.dtype)])
    return new[:len(seq)]


def sites_of(variants: Path, where: str, read_col: str, max_variants: int, seed: int) -> pl.DataFrame:
    t = pl.read_parquet(variants)
    if "pos" not in t.columns and "position" in t.columns:      # prep_eqtl_finemap.py's name for it
        t = t.rename({"position": "pos"})
    if where:
        t = t.filter(pl.sql_expr(where))
    t = t.with_columns(read=pl.col(read_col).cast(pl.Int64) if read_col else pl.col("pos"))
    s = t.select(KEY).unique().sort(KEY)
    if max_variants and s.height > max_variants:
        s = s[np.sort(np.random.default_rng(seed).choice(s.height, max_variants, replace=False))]
    return s.with_row_index("site")


def save(d: Path, sites: pl.DataFrame, ref: np.ndarray, alt: np.ndarray, info: dict) -> None:
    d.mkdir(parents=True, exist_ok=True)
    sites.write_parquet(d / "sites.parquet")
    np.save(d / "ref.npy", ref)
    np.save(d / "alt.npy", alt)
    (d / "index.json").write_text(json.dumps(info, indent=1))


@app.command()
def score(out: Path, checkpoints: list[str] = typer.Argument(..., help="LABEL=CHECKPOINT"),
          variants: Path = typer.Option(..., help="a table with chrom, pos (0-based), ref, alt"),
          head: str = "chiptf", store_path: Path = typer.Option(None, "--store", help="the head's track store "
                                                                 "(default: the checkpoint's record, else the usual)"),
          where: str = typer.Option("", help="an SQL condition on the table's columns"),
          read_col: str = typer.Option("", help="the column of the read position (default: the variant's own)"),
          max_variants: int = 0, name: str = "hg38", context: int = 131072, edge: int = 4096, flank: int = 192,
          batch: int = typer.Option(1, help="sites per forward (each is two sequences)"), seed: int = 0,
          ref_only: bool = typer.Option(False, help="read the reference only (a table of elements, as lentiMPRA's: "
                                                    "ref and alt empty); alt.npy is then ref.npy"),
          device: str = "cuda", dtype: str = typer.Option("bf16", help="bf16 | fp16 | fp32"),
          chunk: int = typer.Option(0, help="bases per encoder chunk (memory cap; 0: the whole window at once)"),
          shard: str = typer.Option("", help="i/n: sites k % n == i only, written under OUT/shards/; then merge")) -> None:
    sh = EK.parse_shard(shard)
    src = interim_dir("genome", name)
    root = src.parent.parent
    sites = sites_of(variants, where, read_col, max_variants, seed)
    sites = sites.with_columns(w_lo=((pl.col("pos") + pl.col("read")) // 2 // BIN) * BIN - context // 2)
    inside = lambda c: (pl.col(c) >= pl.col("w_lo") + edge) & (pl.col(c) < pl.col("w_lo") + context - edge)
    n_all = sites.height
    sites = sites.filter(inside("pos") & inside("read") & (pl.col("ref").str.len_chars() <= 64)
                         & (pl.col("alt").str.len_chars() <= 64))
    mine = sites.filter(pl.col("site") % sh[1] == sh[0])
    bases = np.memmap(src / "bases.u8", dtype=np.uint8, mode="r")
    contigs = pl.read_parquet(src / "contigs.parquet")
    dev = torch.device(device)
    for item in checkpoints:
        label, ck = item.split("=", 1)
        dest = out / "pred" / label if sh[1] == 1 else out / "shards" / label / EK.tag(sh)
        if (dest / "index.json").exists():
            raise typer.BadParameter(f"{dest} exists; choose another label")
        model, state = G.load_checkpoint(ck)
        heads = [h[0] for h in model.config.xheads]
        if head not in heads:
            raise typer.BadParameter(f"{label}: no head {head!r} ({heads})")
        rec = (((state.get("run_record") or {}).get("xheads") or {}).get("stores") or {}).get(head, {})
        where_store = store_path or rec.get("path") or (root / "tracks" / name / STORES[head] if head in STORES else None)
        if where_store is None:
            raise typer.BadParameter(f"{label}: give --store for head {head!r}")
        store = X.open_store(str(where_store))
        if store.kind == "binary":
            raise typer.BadParameter(f"{head}: a label head; this scorer reads track heads")
        model = model.to(dev).eval()
        model.x_active = {head}
        if chunk and not EK.capped_ok(model):
            raise typer.BadParameter(f"{label}: --chunk needs a v2/v2l core without learned chunking")
        unit = 2 ** model.config.n_down
        consts = store.consts(dev)
        n_ch = len(store.ids)

        def signal(seqs: torch.Tensor, span: tuple[int, int]) -> np.ndarray:
            """[B, L] codes -> [B, bins of span, channels] predicted signal."""
            model.x_span = span
            ctx = torch.zeros(len(seqs), dtype=torch.long, device=dev) if model.config.n_contexts > 0 else None
            with torch.no_grad(), EK.autocast(dev, dtype):
                if chunk:
                    raw = torch.cat([EK.forward_capped(model, s, None if ctx is None else ctx[:1], span, chunk)[head]
                                     for s in seqs])
                else:
                    model(seqs, torch.zeros(len(seqs), seqs.shape[1] // unit, dtype=torch.long, device=dev), ctx)
                    raw = model.last_x[head]
            raw = raw.float()[..., :n_ch]
            p_log = X.signal_of(torch.nn.functional.softplus(raw), consts[1], consts[2])   # as xheads.eval_pair
            return torch.expm1(p_log.float()).clamp_min(0).cpu().numpy()

        ref_out = np.zeros((mine.height, n_ch), np.float32)
        alt_out = np.zeros((mine.height, n_ch), np.float32)
        done = np.zeros(mine.height, bool)
        pend: list = []
        t0 = time.time()

        def flush() -> None:
            if not pend:
                return
            span = pend[0][3]
            n_seq = 1 if ref_only else 2
            seqs = torch.from_numpy(np.stack([s for p in pend for s in (p[1], p[2])[:n_seq]])).to(dev)
            sig = signal(seqs, span)
            res = (span[1] - span[0]) // sig.shape[1]         # bases per row of the head's output
            for j, (k, _, _, _, rel) in enumerate(pend):
                a, b = max((rel - flank - span[0]) // res, 0), (rel + flank - span[0]) // res + 1
                ref_out[k], alt_out[k] = sig[n_seq * j, a:b].sum(0), sig[n_seq * j + n_seq - 1, a:b].sum(0)
                done[k] = True
            pend.clear()

        for k, r in enumerate(mine.iter_rows(named=True)):
            lo = r["w_lo"]
            ref = codes_of(contigs, bases, r["chrom"], lo, lo + context)
            alt = with_alt(ref, r["pos"] - lo, r["ref"], r["alt"])
            if alt is None:
                continue
            rel = r["read"] - lo
            s_lo = max(((rel - flank - BIN) // BIN) * BIN, 0)
            s_hi = min(((rel + flank + 2 * BIN) // BIN) * BIN, context)
            if pend and (pend[0][3] != (s_lo, s_hi) or len(pend) >= batch):
                flush()
            pend.append((k, ref, alt, (s_lo, s_hi), rel))
            if k % 500 == 0:
                typer.echo(f"{label}: site {k + 1}/{mine.height}, {time.time() - t0:.0f} s")
        flush()
        kept = mine.filter(pl.Series(done)).drop("w_lo")
        info = dict(label=label, source=str(ck), step=int(state.get("step", -1)), head=head, store=str(store.path),
                    ids=list(store.ids), context=context, edge=edge, flank=flank, where=where, read_col=read_col,
                    max_variants=max_variants, seed=seed, ref_only=ref_only, variants=str(variants), dtype=dtype, chunk=chunk,
                    sites_in_table=n_all, sites_in_window=sites.height, sites_scored=kept.height,
                    ref_mismatch=int((~done).sum()), seconds=round(time.time() - t0, 1),
                    commit=((state.get("run_record") or {}).get("source") or {}).get("commit"))
        if sh[1] > 1:
            info["shard"] = list(sh)
        save(dest, kept, ref_out[done], alt_out[done], info)
        typer.echo(f"{label}: {kept.height} sites ({info['ref_mismatch']} with another reference base) in "
                   f"{time.time() - t0:.0f} s -> {dest}")
        del model
        torch.cuda.empty_cache() if dev.type == "cuda" else None


@app.command()
def merge(out: Path, label: str, shards: int = typer.Option(..., help="n, as in --shard i/n")) -> None:
    """Joins the n shards of LABEL into pred/LABEL (refuses while one is missing or they disagree on settings)."""
    parts = [out / "shards" / label / f"{i}-of-{shards}" for i in range(shards)]
    missing = [p.name for p in parts if not (p / "index.json").exists()]
    if missing:
        raise typer.BadParameter(f"{label}: {len(missing)} of {shards} shards missing: {missing[:5]}")
    infos = [json.loads((p / "index.json").read_text()) for p in parts]
    for k in ("source", "step", "head", "ids", "context", "edge", "flank", "where", "read_col", "max_variants", "seed",
              "variants", "dtype", "sites_in_window"):
        if len({json.dumps(i.get(k)) for i in infos}) > 1:
            raise typer.BadParameter(f"{label}: shards differ in {k}")
    dest = out / "pred" / label
    if (dest / "index.json").exists():
        raise typer.BadParameter(f"{dest} exists")
    sites = pl.concat([pl.read_parquet(p / "sites.parquet") for p in parts])
    if sites["site"].n_unique() != sites.height:
        raise typer.BadParameter(f"{label}: a site is in two shards")
    order = np.argsort(sites["site"].to_numpy(), kind="stable")
    arr = lambda nm: np.concatenate([np.load(p / nm) for p in parts])[order]
    info = dict(infos[0], shards=shards, **{k: sum(i[k] for i in infos) for k in ("sites_scored", "ref_mismatch", "seconds")})
    info.pop("shard", None)
    save(dest, sites[order], arr("ref.npy"), arr("alt.npy"), info)
    typer.echo(f"{label}: {shards} shards, {sites.height} sites -> {dest}")


# ------------------------------------------------------------------------------------------------ the ASB report

def auroc(score: np.ndarray, y: np.ndarray) -> float:
    """Area under the ROC curve, ties at half (the Mann-Whitney statistic)."""
    y = y.astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if not n1 or not n0:
        return float("nan")
    order = np.argsort(score, kind="stable")
    s = score[order]
    rank = np.empty(len(s))
    edges = np.flatnonzero(np.r_[True, s[1:] != s[:-1], True])
    for a, b in zip(edges[:-1], edges[1:]):
        rank[order[a:b]] = (a + b + 1) / 2
    return float((rank[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def ci(fn, n: int, n_boot: int, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    v = [fn(rng.integers(0, n, n)) for _ in range(n_boot)]
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


def asb_rows(d: Path, table: pl.DataFrame) -> pl.DataFrame:
    """The ASB table's rows a prediction covers, with ``pred`` (the row's own channels) and ``pred_any`` (every
    channel of the head)."""
    info = json.loads((d / "index.json").read_text())
    at = {c: i for i, c in enumerate(info["ids"])}
    sites = pl.read_parquet(d / "sites.parquet").with_row_index("at")
    delta = np.log2((np.load(d / "alt.npy") + EPS) / (np.load(d / "ref.npy") + EPS))
    t = table.join(sites.filter(pl.col("read") == pl.col("pos")).select("chrom", "pos", "ref", "alt", "at"),
                   on=["chrom", "pos", "ref", "alt"])
    rows, chans = t["at"].to_numpy(), t["channels"].to_list()
    pred = np.array([delta[r, [at[c] for c in ch if c in at]].mean() if any(c in at for c in ch) else np.nan
                     for r, ch in zip(rows, chans)])
    t = t.with_columns(pred=pl.Series(pred), pred_any=pl.Series(delta[rows].mean(1).astype(np.float64)))
    return t.filter(pl.col("pred").is_not_nan())


@app.command("report-asb")
def report_asb(out: Path, variants: Path = typer.Option(None, help="the ASB table (default: the one scored)"),
               n_boot: int = 200, min_rows: int = 20) -> None:
    lines, table = [], []
    for d in sorted((out / "pred").iterdir()):
        if not (d / "index.json").exists():
            continue
        src = variants or Path(json.loads((d / "index.json").read_text())["variants"])
        r = asb_rows(d, pl.read_parquet(src))
        subsets = []
        for tier in r["match"].unique().sort():
            s = r.filter(pl.col("match") == tier)
            subsets += [(f"{tier}", s), (f"{tier}, chr8/chr9 (held out)", s.filter("held_out")),
                        (f"{tier}, other chromosomes", s.filter(~pl.col("held_out"))),
                        (f"{tier}, with a motif hit", s.filter(pl.col("motif_conc") != "No Hit")),
                        (f"{tier}, no motif hit", s.filter(pl.col("motif_conc") == "No Hit"))]
        for nm, s in subsets:
            if s.height < min_rows:
                continue
            y = (s["side"] == "alt").to_numpy()
            eff = s["effect"].to_numpy()
            row = dict(label=d.name, subset=nm, rows=s.height, alt_side=float(y.mean()))
            scores = {"model": s["pred"].to_numpy(), "any_channel": s["pred_any"].to_numpy()}
            for k, v in scores.items():
                row[f"{k}_auroc"] = auroc(v, y)
                row[f"{k}_auroc_ci"] = ci(lambda i: auroc(v[i], y[i]), len(y), n_boot)
                row[f"{k}_sign"] = float(((v > 0) == y)[v != 0].mean()) if (v != 0).any() else float("nan")
                row[f"{k}_spearman"] = spearman(v, eff)
            m = s.filter(pl.col("motif_fc").is_not_null())
            if m.height >= min_rows:
                ym, fc = (m["side"] == "alt").to_numpy(), m["motif_fc"].to_numpy()
                row |= dict(motif_rows=m.height, motif_auroc=auroc(fc, ym), motif_spearman=spearman(fc, m["effect"].to_numpy()),
                            model_auroc_on_motif_rows=auroc(m["pred"].to_numpy(), ym))
            table.append(row)
            lines.append(f"{d.name:20s} {nm:36s} n={s.height:6d} alt={row['alt_side']:.2f} "
                         f"model={row['model_auroc']:.3f}[{row['model_auroc_ci'][0]:.3f},{row['model_auroc_ci'][1]:.3f}] "
                         f"rho={row['model_spearman']:.3f} any={row['any_channel_auroc']:.3f} "
                         f"motif={row.get('motif_auroc', float('nan')):.3f}")
    (out / "report_asb.json").write_text(json.dumps(table, indent=1))
    typer.echo("\n".join(lines))


# ------------------------------------------------------------------------------------------------ the MPRA report

def norm(s: str) -> str:
    return "".join(c for c in s.lower() if c.isalnum())


@app.command("report-mpra")
def report_mpra(out: Path, variants: Path = typer.Option(None, help="prep_mpra_satmut.py's table (default: the one "
                                                                    "scored)"), min_rows: int = 50) -> None:
    """Per experiment (an element in a cell line), Pearson and Spearman between the measured effect on expression and
    the head's predicted change at the element, over the variants with >= 10 barcodes: the mean over all channels of
    ``log2((alt + 1) / (ref + 1))``, and the same over the channels of the experiment's own cell line where the
    store has it."""
    lines, table = [], []
    for d in sorted((out / "pred").iterdir()):
        if not (d / "index.json").exists():
            continue
        info = json.loads((d / "index.json").read_text())
        t = pl.read_parquet(variants or Path(info["variants"])).filter("confident")
        sites = pl.read_parquet(d / "sites.parquet").with_row_index("at")
        delta = np.log2((np.load(d / "alt.npy") + EPS) / (np.load(d / "ref.npy") + EPS))
        t = t.join(sites.select("chrom", "pos", "ref", "alt", "at", centre="read"), on=["chrom", "pos", "ref", "alt", "centre"])
        meta = Path(info["store"]) / "binned.json"
        tracks = json.loads(meta.read_text())["tracks"] if meta.exists() else []
        cells: dict = {}
        for k, tr in enumerate(tracks):
            cells.setdefault(norm(str(tr.get("biosample", ""))), []).append(k)
        per = []
        for (exp,), s in t.group_by("experiment", maintain_order=True):
            if s.height < min_rows:
                continue
            y, rows = s["effect"].to_numpy(), s["at"].to_numpy()
            own = cells.get(norm(s["cell"][0]), [])
            row = dict(label=d.name, head=info["head"], experiment=exp, element=s["element"][0], kind=s["kind"][0],
                       cell=s["cell"][0], primary=bool(s["primary"][0]), held_out=bool(s["held_out"][0]), variants=s.height,
                       own_channels=len(own))
            for k, v in (("all", delta[rows].mean(1)), ("own", delta[rows][:, own].mean(1) if own else None)):
                if v is not None:
                    row[f"pearson_{k}"] = float(np.corrcoef(v, y)[0, 1]) if v.std() > 0 else float("nan")
                    row[f"spearman_{k}"] = spearman(v, y)
            per.append(row)
        table += per
        prim = [r for r in per if r["primary"]]
        for nm, rs in (("primary experiments", prim), ("promoters", [r for r in prim if r["kind"] == "promoter"]),
                       ("enhancers", [r for r in prim if r["kind"] == "enhancer"]),
                       ("chr8/chr9 (held out)", [r for r in prim if r["held_out"]]),
                       ("with own cell line", [r for r in prim if r["own_channels"]])):
            if rs:
                mean = lambda k: float(np.mean(v)) if (v := [r[k] for r in rs if np.isfinite(r.get(k, np.nan))]) else float("nan")
                lines.append(f"{d.name:24s} {nm:22s} n={len(rs):2d} pearson={mean('pearson_all'):.3f} "
                             f"spearman={mean('spearman_all'):.3f} own: pearson={mean('pearson_own'):.3f} "
                             f"spearman={mean('spearman_own'):.3f}")
        lines += [f"{d.name:24s}   {r['experiment']:14s} {r['cell']:10s} n={r['variants']:5d} pearson={r['pearson_all']:.3f} "
                  f"spearman={r['spearman_all']:.3f}" + (f" own({r['own_channels']})={r['pearson_own']:.3f}" if r["own_channels"] else "")
                  for r in per]
    (out / "report_mpra.json").write_text(json.dumps(table, indent=1))
    typer.echo("\n".join(lines))


# ------------------------------------------------------------------------------------------------ the eQTL report

@app.command("report-eqtl")
def report_eqtl(out: Path, variants: Path = typer.Option(None, help="prep_eqtl_finemap.py's table (default: the one "
                                                                    "scored)"), n_boot: int = 200) -> None:
    """Fine-mapped eQTLs (``label`` 1) against their matched negatives (0), scored with ``--read-col tss`` on the
    RNA head: a variant's score is the mean over the gene strand's channels of ``|log2((alt + 1) / (ref + 1))|`` at
    the TSS (unsigned: the table has no direction). AUROC, average precision, and the share of pairs whose positive
    outscores its negative; chr8/chr9 apart; distance to the TSS as the baseline (matched, so near chance)."""
    lines, table = [], []
    for d in sorted((out / "pred").iterdir()):
        if not (d / "index.json").exists():
            continue
        info = json.loads((d / "index.json").read_text())
        t = pl.read_parquet(variants or Path(info["variants"]))
        t = t.rename({"position": "pos"}) if "pos" not in t.columns else t
        sites = pl.read_parquet(d / "sites.parquet").with_row_index("at")
        mag = np.abs(np.log2((np.load(d / "alt.npy") + EPS) / (np.load(d / "ref.npy") + EPS)))
        ids = info["ids"]
        by = {"+": [k for k, c in enumerate(ids) if c.endswith("_P")], "-": [k for k, c in enumerate(ids) if c.endswith("_M")]}
        t = t.join(sites.select("chrom", "pos", "ref", "alt", "at", tss="read"), on=["chrom", "pos", "ref", "alt", "tss"])
        score = np.array([mag[a, by[s] or slice(None)].mean() for a, s in zip(t["at"].to_numpy(), t["strand"].to_list())])
        t = t.with_columns(score=pl.Series(score))
        for nm, s in (("all", t), ("chr8/chr9 (held out)", t.filter("held_out")), ("other chromosomes", t.filter(~pl.col("held_out"))),
                      ("within 10 kb of the TSS", t.filter(pl.col("distance").abs() <= 10_000)),
                      ("beyond 10 kb", t.filter(pl.col("distance").abs() > 10_000))):
            y = (s["label"] == 1).to_numpy()
            if y.sum() < 10 or (~y).sum() < 10:
                continue
            v, dist = s["score"].to_numpy(), -np.abs(s["distance"].to_numpy()).astype(float)
            pr = s.group_by("pair").agg(p=pl.col("score").filter(pl.col("label") == 1).max(),
                                        q=pl.col("score").filter(pl.col("label") == 0).max()).drop_nulls()
            row = dict(label=d.name, head=info["head"], context=info["context"], subset=nm, variants=s.height,
                       positives=int(y.sum()), auroc=auroc(v, y), auroc_ci=ci(lambda i: auroc(v[i], y[i]), len(y), n_boot),
                       ap=ap(v, y), chance=float(y.mean()), distance_auroc=auroc(dist, y), pairs=pr.height,
                       pair_wins=float(((pr["p"] > pr["q"]).sum() + 0.5 * (pr["p"] == pr["q"]).sum()) / max(pr.height, 1)))
            table.append(row)
            lines.append(f"{d.name:20s} {nm:26s} n={s.height:6d} pos={row['positives']:5d} auroc={row['auroc']:.3f}"
                         f"[{row['auroc_ci'][0]:.3f},{row['auroc_ci'][1]:.3f}] ap={row['ap']:.3f} (chance {row['chance']:.3f}) "
                         f"pair wins={row['pair_wins']:.3f} of {pr.height} distance auroc={row['distance_auroc']:.3f}")
    (out / "report_eqtl.json").write_text(json.dumps(table, indent=1))
    typer.echo("\n".join(lines))


# ------------------------------------------------------------------------------------------------ the lentiMPRA report

def ridge_cv(x: np.ndarray, y: np.ndarray, fold: np.ndarray, lam: float = 100.0) -> np.ndarray:
    """Out-of-fold ridge predictions of ``y`` from ``x`` (standardised on each training part), by the given folds."""
    out = np.full(len(y), np.nan)
    for f in np.unique(fold):
        te = fold == f
        mu, sd = x[~te].mean(0), x[~te].std(0) + 1e-6
        a, b = (x[~te] - mu) / sd, (x[te] - mu) / sd
        w = np.linalg.solve(a.T @ a + lam * np.eye(a.shape[1]), a.T @ (y[~te] - y[~te].mean()))
        out[te] = b @ w + y[~te].mean()
    return out


@app.command("report-lentimpra")
def report_lentimpra(out: Path, variants: Path = typer.Option(None, help="prep_lentimpra.py's table (default: the one "
                                                                         "scored)"), lam: float = 100.0) -> None:
    """lentiMPRA activity of native 200-bp elements (Agarwal et al. 2025) against the head's prediction at their
    place in the genome (scored with ``--ref-only``), per cell line: Pearson of the measured ``log2(RNA/DNA)`` with
    (a) the mean ``log1p`` signal over the cell line's own channels, where the store has them, and (b) a ridge probe
    on every channel's ``log1p`` signal, fitted out-of-fold on the paper's ten folds (their Table S6 gives Enformer,
    Sei and MPRALegNet on the same folds). chr8/chr9 elements apart (probe fitted on the others)."""
    lines, table = [], []
    for d in sorted((out / "pred").iterdir()):
        if not (d / "index.json").exists():
            continue
        info = json.loads((d / "index.json").read_text())
        t = pl.read_parquet(variants or Path(info["variants"])).filter(pl.col("activity").is_not_null())
        sites = pl.read_parquet(d / "sites.parquet").with_row_index("at")
        x_all = np.log1p(np.load(d / "ref.npy")).astype(np.float64)
        t = t.join(sites.select("chrom", "pos", "at"), on=["chrom", "pos"])
        meta = Path(info["store"]) / "binned.json"
        tracks = json.loads(meta.read_text())["tracks"] if meta.exists() else []
        for (cell,), s in t.group_by("cell", maintain_order=True):
            y, x = s["activity"].to_numpy(), x_all[s["at"].to_numpy()]
            own = [k for k, tr in enumerate(tracks) if norm(str(tr.get("biosample", ""))) == norm(cell)]
            held = s["held_out"].to_numpy()
            oof = ridge_cv(x, y, s["fold"].to_numpy(), lam)
            pear = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else float("nan")
            row = dict(label=d.name, head=info["head"], cell=cell, elements=s.height, own_channels=len(own),
                       probe_pearson=pear(oof, y), probe_by_fold=[pear(oof[s["fold"].to_numpy() == f], y[s["fold"].to_numpy() == f])
                                                                  for f in sorted(set(s["fold"].to_list()))],
                       own_pearson=pear(x[:, own].mean(1), y) if own else None, held_out=int(held.sum()))
            if held.sum() > 50:                          # the probe fitted on the other chromosomes
                ho = ridge_cv(x, y, held.astype(int), lam)
                row["probe_pearson_held_out"] = pear(ho[held], y[held])
            table.append(row)
            lines.append(f"{d.name:20s} {cell:6s} n={s.height:6d} probe={row['probe_pearson']:.3f} "
                         f"held-out chr8/chr9={row.get('probe_pearson_held_out', float('nan')):.3f} "
                         + (f"own({len(own)})={row['own_pearson']:.3f}" if own else "own: none"))
    (out / "report_lentimpra.json").write_text(json.dumps(table, indent=1))
    typer.echo("\n".join(lines))


def ap(score: np.ndarray, y: np.ndarray) -> float:
    o = np.argsort(-score, kind="stable")
    y = y[o].astype(np.float64)
    return float((np.cumsum(y) / np.arange(1, len(y) + 1) * y).sum() / y.sum()) if y.sum() else float("nan")


if __name__ == "__main__":
    app()
