"""The further heads' benchmark: KronGenome's DNase and splice-site heads, and AlphaGenome's, on fixed held-out windows.

    uv run python scripts/kron_dna/xhead_bench.py targets BENCH --windows WINDOWS.csv --dnase STORE --splice STORE
    uv run python scripts/kron_dna/xhead_bench.py score BENCH LABEL=CHECKPOINT [...] [--context 1048576]
        [--dtype bf16|fp16|fp32] [--shard i/n]     (a shard writes BENCH/shards/LABEL/i-of-n; then:)
    uv run python scripts/kron_dna/xhead_bench.py merge BENCH LABEL --shards n
    (AlphaGenome: scripts/kron_dna/xhead_bench_ag.py, in AlphaGenome's own environment, writes the same files)
    uv run python scripts/kron_dna/xhead_bench.py report BENCH

One scoring path for every model: ``score`` (and the AlphaGenome script) only write predictions, in one form, under
``BENCH/pred/<label>/``; ``report`` computes every number from those files and the frozen targets.

- ``dnase_128.npy`` float32 ``[windows, 1024, 13]``: per 128-bp bin counted from the window start, the mean over its
  bases of log1p(predicted signal); the target is the same mean of log1p(measured signal), decoded from the store.
  KronGenome's head output goes through ``xheads.eval_pair`` (what its held-out score in training uses); AlphaGenome's
  output is log1p of its own units (Pearson does not see its scale).
- ``splice.npy`` float16 ``[windows, 131072, 4]``: a score per base and channel (donor +, acceptor +, donor -, acceptor -;
  logits for KronGenome, probabilities for AlphaGenome; average precision depends on the order only).
- ``index.json``: the window rows covered (rows of ``windows.csv``; AlphaGenome's fold models cover their test-fold
  windows only), the source, and its settings.

Metrics in ``report``, per channel and pooled over the windows a source covers: Pearson of the 128-bp log1p means for
DNase; exact average precision over every base for splice sites. Intervals: 95 % bootstrap over windows (1,000
resamples), and for each pair of sources on their shared windows, of the paired difference.
"""
from __future__ import annotations

import hashlib
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
WIN, BIN = 131072, 128
SPLICE_IDS = ["donor_plus", "acceptor_plus", "donor_minus", "acceptor_minus"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def windows_of(bench: Path) -> pl.DataFrame:
    return pl.read_csv(bench / "windows.csv")


def log1p_bins(x: np.ndarray) -> np.ndarray:
    """``[L, T]`` log1p values -> ``[L / 128, T]`` bin means (float64 sums, float32 out)."""
    return x.reshape(-1, BIN, x.shape[-1]).mean(1, dtype=np.float64).astype(np.float32)


# ------------------------------------------------------------------------------------------------------- targets

@app.command()
def targets(bench: Path, windows: Path = typer.Option(...), dnase: Path = typer.Option(...),
            splice: Path = typer.Option(...), name: str = "hg38") -> None:
    """Freeze the targets on the windows: DNase from a track store, splice sites from a label store."""
    if (bench / "READY").exists():
        raise typer.BadParameter(f"{bench} is frozen (READY exists)")
    bench.mkdir(parents=True, exist_ok=True)
    win = pl.read_csv(windows)
    assert (win["end"] - win["start"] == WIN).all(), "every window must be 131,072 bases"
    ds, ss = X.open_store(dnase), X.open_store(splice)
    assert ds.kind == "tracks" and ss.kind == "binary" and list(ss.ids) == SPLICE_IDS, (ds.kind, ss.kind, ss.ids)
    bases = np.memmap(interim_dir("genome", name) / "bases.u8", dtype=np.uint8, mode="r")
    n = win.height
    d128 = np.zeros((n, WIN // BIN, len(ds.ids)), dtype=np.float32)
    sp = np.zeros((n, WIN, 4), dtype=np.uint8)
    for k, r in enumerate(win.iter_rows(named=True)):
        a = int(r["bases_offset"])
        assert int(bases[a:a + WIN].max()) < 4, f"{r['window_id']}: N inside the window"
        d128[k] = log1p_bins(np.asarray(ds.dense[a:a + WIN]).astype(np.float32) / 255.0 * ds.scales)
        sp[k] = ss.window(a, WIN, WIN)
    (bench / "windows.csv").write_bytes(windows.read_bytes())
    np.save(bench / "target_dnase_128.npy", d128)
    np.save(bench / "target_splice.npy", sp)
    tracks = json.loads((dnase / "dense.json").read_text())["tracks"]
    meta = dict(written=time.strftime("%Y-%m-%d %H:%M:%S %Z"), windows_source=str(windows), n_windows=n,
                dnase_store=str(dnase), dnase_identity=ds.identity, splice_store=str(splice), splice_identity=ss.identity,
                dnase_channels=[dict(channel=k, id=i, experiment=t.get("experiment"), biosample=t.get("biosample"))
                                for k, (i, t) in enumerate(zip(ds.ids, tracks))],
                splice_channels=SPLICE_IDS, splice_positives=sp.sum((0, 1)).tolist(),
                dnase_target="per 128-bp bin from the window start, mean of log1p(decoded signal)",
                splice_target="0/1 per base: the intron's first base (donor) and last base (acceptor) in the transcript's "
                              "direction, per strand (prep_splice_labels.py)")
    meta["sha256"] = {f: sha256(bench / f) for f in ("windows.csv", "target_dnase_128.npy", "target_splice.npy")}
    (bench / "targets.json").write_text(json.dumps(meta, indent=1))
    (bench / "READY").touch()
    typer.echo(f"{n} windows; DNase {len(ds.ids)} channels; splice positives per channel {meta['splice_positives']}")


# -------------------------------------------------------------------------------------------------------- score

def codes_of(contigs: pl.DataFrame, bases: np.ndarray, chrom: str, lo: int, hi: int) -> np.ndarray:
    """Base codes of chrom[lo:hi); bases outside the gap-free contigs (assembly gaps) are N (code 4)."""
    out = np.full(hi - lo, 4, dtype=np.int64)
    for r in contigs.filter((pl.col("chrom") == chrom) & (pl.col("start") < hi) & (pl.col("end") > lo)).iter_rows(named=True):
        a, b = max(lo, r["start"]), min(hi, r["end"])
        out[a - lo:b - lo] = bases[r["offset"] + a - r["start"]:r["offset"] + b - r["start"]]
    return out


@app.command()
def score(bench: Path, checkpoints: list[str] = typer.Argument(..., help="LABEL=CHECKPOINT (a file or run dir)"),
          context: int = typer.Option(1048576, help="bases given to the model, centred on each window"),
          dnase_head: str = "dnase", splice_head: str = "splice", name: str = "hg38", device: str = "cuda",
          limit: int = typer.Option(0, help="the first N windows only (a smoke run; the label gets a suffix)"),
          dtype: str = typer.Option("bf16", help="bf16 | fp16 | fp32 (bf16 is emulated, slow, on Turing cards)"),
          chunk: int = typer.Option(0, help="bases per encoder chunk (memory cap: e.g. 131072 runs a 1 Mb context "
                                            "on an 11 GB card; 0: the whole window at once)"),
          shard: str = typer.Option("", help="i/n: windows k % n == i only, under BENCH/shards/; then merge")) -> None:
    """KronGenome's further heads on every window: predictions in the benchmark's form, under pred/<label>/."""
    sh = EK.parse_shard(shard)
    if not (bench / "READY").exists():
        raise typer.BadParameter(f"{bench} has no frozen targets (run targets first)")
    if context < WIN or (context - WIN) % (2 * BIN):
        raise typer.BadParameter(f"--context must be {WIN} plus a multiple of {2 * BIN}")
    meta = json.loads((bench / "targets.json").read_text())
    ds = X.open_store(meta["dnase_store"])
    if ds.identity["record_sha256"] != meta["dnase_identity"]["record_sha256"]:
        raise typer.BadParameter("the DNase store changed since the targets were frozen")
    win = windows_of(bench)
    if limit:
        win = win.head(limit)
    src = interim_dir("genome", name)
    bases = np.memmap(src / "bases.u8", dtype=np.uint8, mode="r")
    contigs = pl.read_parquet(src / "contigs.parquet")
    dev = torch.device(device)
    flank = (context - WIN) // 2
    for item in checkpoints:
        label, where = item.split("=", 1)
        label += f"__first{limit}" if limit else ""
        out = bench / "pred" / label if sh[1] == 1 else bench / "shards" / label / EK.tag(sh)
        if (out / "index.json").exists():
            raise typer.BadParameter(f"{out} exists; choose another label")
        model, state = G.load_checkpoint(where)
        heads = [h[0] for h in model.config.xheads]
        if dnase_head not in heads or splice_head not in heads:
            raise typer.BadParameter(f"{label}: heads {heads}, wanted {dnase_head} and {splice_head}")
        rec = ((state.get("run_record") or {}).get("xheads") or {}).get("stores") or {}
        for h, st in ((dnase_head, meta["dnase_identity"]), (splice_head, meta["splice_identity"])):
            got = rec.get(h, {})
            if got.get("ids") is not None and list(got["ids"]) != list(st["ids"]):
                raise typer.BadParameter(f"{label}: head {h} was trained on channels {got['ids']}, the targets are {st['ids']}")
        if chunk and not EK.capped_ok(model):
            raise typer.BadParameter(f"{label}: --chunk needs a v2/v2l core without learned chunking")
        model = model.to(dev).eval()
        model.x_active = {dnase_head, splice_head}
        model.x_span = (flank, flank + WIN)                           # the heads run on the window only, as in training
        consts = ds.consts(dev)
        unit = 2 ** model.config.n_down
        rows = [k for k in range(win.height) if EK.mine(k, sh)]
        d128 = np.zeros((len(rows), WIN // BIN, len(ds.ids)), dtype=np.float32)
        sp = np.zeros((len(rows), WIN, 4), dtype=np.float16)
        t0 = time.time()
        mine = (r for k_, r in enumerate(win.iter_rows(named=True)) if EK.mine(k_, sh))
        for k, r in enumerate(mine):                                  # k: the row in this shard's arrays
            seq = torch.from_numpy(codes_of(contigs, bases, r["chrom"], r["start"] - flank, r["end"] + flank)).to(dev)
            seg = torch.zeros(1, context // unit, dtype=torch.long, device=dev)
            ctx = torch.zeros(1, dtype=torch.long, device=dev) if model.config.n_contexts > 0 else None
            with torch.no_grad(), EK.autocast(dev, dtype):
                if chunk:
                    EK.forward_capped(model, seq, ctx, (flank, flank + WIN), chunk)
                else:
                    model(seq[None], seg, ctx)
            assert model.last_x_at == (flank, flank + WIN), model.last_x_at
            pd = model.last_x[dnase_head][0].float()[:, :len(ds.ids)]  # [WIN, 13]: the primary channels come first
                                                                          # (a head with --xhead-species has more)
            p_log, _ = X.eval_pair(ds, pd, torch.zeros_like(pd), consts)
            d128[k] = log1p_bins(p_log.cpu().numpy())
            sp[k] = model.last_x[splice_head][0][:, :4].float().clamp(-6e4, 6e4).cpu().numpy().astype(np.float16)
            if k % 16 == 0:
                typer.echo(f"{label}: window {k + 1}/{len(rows)}, {time.time() - t0:.0f} s")
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "dnase_128.npy", d128)
        np.save(out / "splice.npy", sp)
        (out / "index.json").write_text(json.dumps(dict(
            label=label, source=str(where), kind="KronGenome", step=int(state.get("step", -1)), context=context,
            windows=rows, dtype=dtype, chunk=chunk, shard=list(sh) if sh[1] > 1 else None, xhead_kind=model.config.xhead_kind,
            xhead_layers=model.config.xhead_layers, n_layers=model.config.n_layers,
            commit=((state.get("run_record") or {}).get("source") or {}).get("commit"),
            seconds=round(time.time() - t0, 1), heads_run_on="the window's 131,072 bases (x_span), trunk on the whole context"),
            indent=1))
        typer.echo(f"{label}: {len(rows)} windows in {time.time() - t0:.0f} s -> {out}")
        del model
        torch.cuda.empty_cache() if dev.type == "cuda" else None


@app.command()
def merge(bench: Path, label: str, shards: int = typer.Option(..., help="n, as in --shard i/n")) -> None:
    """Joins the n shards of LABEL (BENCH/shards/LABEL/i-of-n) into pred/LABEL, as an unsharded score writes it."""
    parts = [bench / "shards" / label / f"{i}-of-{shards}" for i in range(shards)]
    missing = [p.name for p in parts if not (p / "index.json").exists()]
    if missing:
        raise typer.BadParameter(f"{label}: shards {missing} missing")
    infos = [json.loads((p / "index.json").read_text()) for p in parts]
    for k in ("source", "step", "context", "dtype", "commit"):
        if len({json.dumps(i.get(k)) for i in infos}) > 1:
            raise typer.BadParameter(f"{label}: shards differ in {k}")
    rows = np.concatenate([np.asarray(i["windows"], dtype=int) for i in infos])
    o = np.argsort(rows)
    if not np.array_equal(rows[o], np.arange(windows_of(bench).height)):
        raise typer.BadParameter(f"{label}: the shards do not cover every window exactly once")
    out = bench / "pred" / label
    if (out / "index.json").exists():
        raise typer.BadParameter(f"{out} exists")
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "dnase_128.npy", np.concatenate([np.load(p / "dnase_128.npy") for p in parts])[o])
    np.save(out / "splice.npy", np.concatenate([np.load(p / "splice.npy") for p in parts])[o])
    info = dict(infos[0], windows=rows[o].tolist(), shard=None, shards=shards, seconds=sum(i["seconds"] for i in infos))
    (out / "index.json").write_text(json.dumps(info, indent=1))
    typer.echo(f"{label}: {shards} shards, {len(rows)} windows -> {out}")


# ------------------------------------------------------------------------------------------------------- report

def pearson_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-window sufficient statistics ``[n, 6, T]`` of 128-bp bins: n, sx, sy, sxy, sxx, syy."""
    x, y = x.astype(np.float64), y.astype(np.float64)
    return np.stack([np.full(x.shape[::2], x.shape[1], dtype=np.float64), x.sum(1), y.sum(1), (x * y).sum(1),
                     (x * x).sum(1), (y * y).sum(1)], 1)


def pearson_of(s: np.ndarray) -> np.ndarray:
    n, sx, sy, sxy, sxx, syy = (s[k] for k in range(6))
    cov, vx, vy = sxy / n - sx * sy / n ** 2, sxx / n - (sx / n) ** 2, syy / n - (sy / n) ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((vx > 1e-12) & (vy > 1e-12), cov / np.sqrt(vx * vy), np.nan)


class RankedSplice:
    """One source's splice scores for one channel, sorted once: for every base in score order, its window (as an index
    into the source's rows) and its label. Average precision under window weights (a bootstrap resample, or the
    windows two sources share) is then one pass over this order, with no new sort."""

    def __init__(self, score: np.ndarray, label: np.ndarray) -> None:
        n, length = score.shape
        order = np.argsort(-score.reshape(-1).astype(np.float32), kind="stable")
        self.window = (order // length).astype(np.int32)
        self.hit = label.reshape(-1)[order].astype(bool)

    def ap(self, weight: np.ndarray | None = None) -> float:
        """Average precision with each base weighted by its window's weight (None: all 1)."""
        w = np.ones(len(self.hit)) if weight is None else weight[self.window].astype(np.float64)
        tp = np.cumsum(w * self.hit)
        seen = np.cumsum(w)
        pos = w[self.hit]
        if not pos.sum():
            return float("nan")
        # a base of weight w is w tied copies ranked one after another: copy j has precision (tp0 + j) / (seen0 + j)
        tp0, seen0 = tp[self.hit] - pos, seen[self.hit] - pos
        total = np.zeros_like(pos)
        for j in range(1, int(pos.max()) + 1):
            on = pos >= j
            total[on] += (tp0[on] + j) / (seen0[on] + j)
        return float(total.sum() / pos.sum())


@app.command()
def report(bench: Path, n_boot: int = 1000, n_boot_ap: int = 200, seed: int = 0,
           min_shared: int = typer.Option(8, help="fewest shared windows for a paired comparison")) -> None:
    """Every metric from the saved predictions: results.csv (per source and channel), summary.csv, pairs.csv, charts."""
    meta = json.loads((bench / "targets.json").read_text())
    t_d = np.load(bench / "target_dnase_128.npy")
    t_s = np.load(bench / "target_splice.npy", mmap_mode="r")
    win = windows_of(bench)
    preds = {}
    for d in sorted((bench / "pred").glob("*/index.json")):
        info = json.loads(d.read_text())
        preds[d.parent.name] = dict(info=info, rows=np.asarray(info["windows"]), dnase=np.load(d.parent / "dnase_128.npy"),
                                    splice=np.load(d.parent / "splice.npy", mmap_mode="r"),
                                    dnase_channels=info.get("dnase_channels"))
    # a source scored on every window is also shown on each smaller set another source covers (an AlphaGenome fold
    # model's held-out windows), so each chart compares every model on the same windows
    every = [lab for lab, p in preds.items() if len(p["rows"]) == win.height]
    for lab_s, ps in [(lab, p) for lab, p in preds.items() if len(p["rows"]) < win.height]:
        for lab in every:
            if lab.startswith("alphagenome-"):
                continue
            idx = np.searchsorted(preds[lab]["rows"], ps["rows"])
            tag = lab_s.split("__", 1)[-1] if "__" in lab_s else lab_s
            preds[f"{lab} on {tag}"] = dict(info=dict(preds[lab]["info"], windows=ps["rows"].tolist()), rows=ps["rows"],
                                            dnase=preds[lab]["dnase"][idx], splice=np.asarray(preds[lab]["splice"][idx]))
    rng = np.random.default_rng(seed)
    rows_out, pairs = [], []
    dn_ids = [c["id"] for c in meta["dnase_channels"]]
    bios = [c["biosample"] for c in meta["dnase_channels"]]

    def dnase_scores(p: dict) -> tuple[np.ndarray, np.ndarray]:
        """[n_rows, 6, T] statistics and the channel mask (AlphaGenome may lack a channel: NaN columns)."""
        x = p["dnase"]
        have = ~np.isnan(x).all((0, 1))
        return pearson_rows(np.nan_to_num(x), t_d[p["rows"]]), have

    stats = {}
    for lab, p in preds.items():
        s, have = dnase_scores(p)
        stats[lab] = (s, have)
        r = np.where(have, pearson_of(s.sum(0)), np.nan)
        boots = np.stack([pearson_of(s[rng.integers(0, len(s), len(s))].sum(0)) for _ in range(n_boot)])
        p["ranked"] = [RankedSplice(np.asarray(p["splice"][:, :, k]), np.asarray(t_s[p["rows"], :, k])) for k in range(4)]
        aps = np.array([rs.ap() for rs in p["ranked"]])
        for k in range(len(dn_ids)):
            rows_out.append(dict(source=lab, task="dnase", channel=k, id=dn_ids[k], biosample=bios[k], metric="pearson_128bp_log1p",
                                 value=r[k], lo=float(np.nanpercentile(boots[:, k], 2.5)) if have[k] else np.nan,
                                 hi=float(np.nanpercentile(boots[:, k], 97.5)) if have[k] else np.nan, n_windows=len(p["rows"])))
        for k in range(4):
            rows_out.append(dict(source=lab, task="splice", channel=k, id=SPLICE_IDS[k], biosample="", metric="average_precision",
                                 value=aps[k], lo=np.nan, hi=np.nan, n_windows=len(p["rows"])))
        typer.echo(f"{lab}: {len(p['rows'])} windows; DNase r mean {np.nanmean(r):.4f} over {int(have.sum())} channels; "
                   f"splice AP {', '.join(f'{a:.3f}' for a in aps)} (mean {aps.mean():.4f})")
    res = pl.DataFrame(rows_out)
    # written before the paired comparisons, whose bootstraps grow with the square of the sources: a run that
    # is cut short there still leaves every source's own numbers
    res.write_csv(bench / "results.csv")
    summ = (res.group_by("source", "task").agg(pl.col("value").mean().alias("mean"), pl.col("value").is_not_nan().sum().alias("channels"),
                                               pl.col("n_windows").first()).sort("task", "source"))
    summ.write_csv(bench / "summary.csv")

    # paired differences on shared windows (and shared channels): bootstrap over windows
    labs = list(preds)
    for i, a in enumerate(labs):
        for b in labs[i + 1:]:
            if not np.array_equal(preds[a]["rows"], preds[b]["rows"]):     # paired only on the same window set
                continue
            shared = preds[a]["rows"]
            if len(shared) < min_shared:
                continue
            ia = np.searchsorted(preds[a]["rows"], shared)
            ib = np.searchsorted(preds[b]["rows"], shared)
            sa, ha = stats[a]
            sb, hb = stats[b]
            both = ha & hb
            sa, sb = sa[ia], sb[ib]
            diff = lambda idx: np.nanmean((pearson_of(sa[idx].sum(0)) - pearson_of(sb[idx].sum(0)))[both])
            d0 = diff(np.arange(len(shared)))
            bd = np.array([diff(rng.integers(0, len(shared), len(shared))) for _ in range(n_boot)])
            na, nb = len(preds[a]["rows"]), len(preds[b]["rows"])

            def ap_diff(counts: np.ndarray) -> float:
                """Mean over channels of AP(a) - AP(b) with window weights ``counts`` over the shared windows."""
                wa, wb = np.zeros(na), np.zeros(nb)
                wa[ia], wb[ib] = counts, counts
                return float(np.mean([preds[a]["ranked"][k].ap(wa) - preds[b]["ranked"][k].ap(wb) for k in range(4)]))

            s0 = ap_diff(np.ones(len(shared)))
            bs = [ap_diff(np.bincount(rng.integers(0, len(shared), len(shared)), minlength=len(shared)).astype(float))
                  for _ in range(n_boot_ap)]
            pairs.append(dict(a=a, b=b, n_windows=len(shared), dnase_channels=int(both.sum()),
                              dnase_r_mean_diff=d0, dnase_lo=float(np.percentile(bd, 2.5)), dnase_hi=float(np.percentile(bd, 97.5)),
                              splice_ap_mean_diff=s0, splice_lo=float(np.percentile(bs, 2.5)), splice_hi=float(np.percentile(bs, 97.5))))
    pl.DataFrame(pairs).write_csv(bench / "pairs.csv") if pairs else None
    charts(bench, res, preds)
    typer.echo(summ)
    if pairs:
        typer.echo(pl.DataFrame(pairs))


def charts(bench: Path, res: pl.DataFrame, preds: dict) -> None:
    """Two charts: DNase Pearson per biosample and splice AP per channel, one bar per source; whole-set sources
    (all 96 windows) and fold-subset sources are drawn separately, since their windows differ."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = {}
    for lab, p in preds.items():
        groups.setdefault(len(p["rows"]), []).append(lab)
    for n, labs in groups.items():
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw=dict(width_ratios=[3.2, 1]))
        for ax, task in zip(axes, ("dnase", "splice")):
            sub = res.filter((pl.col("task") == task) & pl.col("source").is_in(labs))
            sub_s = {lab: res.filter((pl.col("task") == "splice") & (pl.col("source") == lab))["value"].to_numpy() for lab in labs}
            names = sub.filter(pl.col("source") == labs[0])["biosample" if task == "dnase" else "id"].to_list()
            x = np.arange(len(names))
            w = 0.8 / len(labs)
            for j, lab in enumerate(labs):
                s = sub.filter(pl.col("source") == lab).sort("channel")
                v = s["value"].to_numpy()
                err = None
                if task == "dnase":
                    err = np.abs(np.stack([v - s["lo"].to_numpy(), s["hi"].to_numpy() - v]))
                ax.bar(x + (j - (len(labs) - 1) / 2) * w, v, w, yerr=err, capsize=2,
                       label=f"{lab}" + (f" (DNase {np.nanmean(v):.3f}, splice {np.nanmean(sub_s[lab]):.3f})" if task == "dnase" else ""))
            ax.set_xticks(x, names, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Pearson r, 128-bp log1p" if task == "dnase" else "average precision")
            ax.set_title(f"DNase, {n} held-out chr8/chr9 windows" if task == "dnase" else "splice sites")
            ax.grid(axis="y", alpha=0.3)
            ax.set_ylim(0, 1)
        handles, names_ = axes[0].get_legend_handles_labels()
        fig.legend(handles, names_, fontsize=8, loc="lower center", ncol=min(len(labs), 4), frameon=False)
        fig.tight_layout(rect=(0, 0.06 + 0.035 * ((len(labs) - 1) // 4), 1, 1))
        fig.savefig(bench / f"chart_{n}windows.png", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    app()
