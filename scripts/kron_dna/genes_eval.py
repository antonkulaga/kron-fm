"""The gene-structure head on any prepared genome, against one or more annotations of it (the annotation
validation set: a species in no training genome, an older and a newer annotation of the same assembly).

    uv run python scripts/kron_dna/splice_zeroshot.py windows OUT --name dmel_r6 [--n 96]      (the windows)
    uv run python scripts/kron_dna/genes_eval.py score OUT --name dmel_r6 --labels old=STORE --labels new=STORE \
        LABEL=CHECKPOINT [--dtype bf16|fp16|fp32] [--shard i/n]
    uv run python scripts/kron_dna/genes_eval.py merge OUT LABEL --shards n --labels old=STORE --labels new=STORE

For each window the model reads 1,048,576 bases centred on it; its ``genes`` head runs on the window's 131,072 bases
(as ``splice_zeroshot.py``). Per annotation and channel (tss, polya, utr5, cds, utr3, ncexon, intron; per strand):

- average precision of the head's logits against the labels, exact over every base of every window;
- base-wise F1 of the head's calls (logit > 0) against the labels, and Helixer's summaries: Genic F1 over the
  CDS, UTR (5' and 3') and intron bases of both strands pooled, Subgenic F1 over CDS and intron;
- Tiberius's exon level for CDS exons: a run of CDS calls is right only if an annotated CDS exon (isoforms
  pooled, so close to but not exactly Tiberius's longest-CDS transcript per gene) has exactly its two ends;
- where two annotations disagree on a base and channel, the share of those bases on which the head's call agrees
  with each of them (the newer annotation is the better one; does the head side with it).

Writes ``OUT/<label>.genes.json`` with all of it and ``OUT/<label>.genes.npy`` (the logits, float16
``[windows, 131072, 14]``, so a later metric needs no card), and prints the summary.
"""
from __future__ import annotations

import importlib.util
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

_spec = importlib.util.spec_from_file_location("xhead_bench", Path(__file__).with_name("xhead_bench.py"))
XB = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(XB)
app = typer.Typer(add_completion=False)
WIN = XB.WIN
GENIC = ("cds", "utr5", "utr3", "intron")
SUBGENIC = ("cds", "intron")


def f1(tp: int, fp: int, fn: int) -> float:
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def runs(x: np.ndarray) -> set:
    """``[n, L]`` bool -> {(window, start, end)} of every maximal run of True, end exclusive (a run touching a window
    edge is cut there; such runs are counted on both sides alike)."""
    out = set()
    for w in range(x.shape[0]):
        d = np.diff(np.concatenate([[0], x[w].astype(np.int8), [0]]))
        for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            out.add((w, int(a), int(b)))
    return out


def exon_f1(call: np.ndarray, lab: np.ndarray) -> dict:
    """Tiberius's exon level for CDS exons: a predicted exon (a run of CDS calls on one strand) is right only if an
    annotated one (a run of CDS labels; isoforms pooled) has exactly its two ends."""
    p, t = runs(call), runs(lab)
    tp = len(p & t)
    return dict(f1=f1(tp, len(p) - tp, len(t) - tp), predicted=len(p), annotated=len(t), exact=tp)


@app.command()
def score(out: Path, checkpoints: list[str] = typer.Argument(..., help="LABEL=CHECKPOINT"),
          name: str = typer.Option(...), labels: list[str] = typer.Option(..., help="TAG=STORE, one per annotation"),
          windows_csv: Path = typer.Option(None, "--windows"), head: str = "genes", context: int = 1048576,
          device: str = "cuda",
          dtype: str = typer.Option("bf16", help="bf16 | fp16 | fp32 (bf16 is emulated, slow, on Turing cards)"),
          shard: str = typer.Option("", help="i/n: windows k % n == i only, written under OUT/shards/; then merge")
          ) -> None:
    sh = EK.parse_shard(shard)
    win = pl.read_csv(windows_csv or out / "windows.csv")
    lab, ids = labels_of(labels, win) if sh[1] == 1 else (None, ids_of(labels))
    src = interim_dir("genome", name)
    bases = np.memmap(src / "bases.u8", dtype=np.uint8, mode="r")
    contigs = pl.read_parquet(src / "contigs.parquet")
    dev = torch.device(device)
    flank = (context - WIN) // 2
    for item in checkpoints:
        label, where = item.split("=", 1)
        model, state = G.load_checkpoint(where)
        heads = [h[0] for h in model.config.xheads]
        if head not in heads:
            raise typer.BadParameter(f"{label}: no head {head!r} ({heads})")
        model = model.to(dev).eval()
        model.x_active, model.x_span = {head}, (flank, flank + WIN)
        unit = 2 ** model.config.n_down
        rows = [k for k in range(win.height) if EK.mine(k, sh)]
        pred = np.zeros((len(rows), WIN, len(ids)), dtype=np.float16)
        t0 = time.time()
        for j, (k, r) in enumerate((k, r) for k, r in enumerate(win.iter_rows(named=True)) if EK.mine(k, sh)):
            seq = torch.from_numpy(XB.codes_of(contigs, bases, r["chrom"], r["start"] - flank, r["end"] + flank)).to(dev)
            seg = torch.zeros(1, context // unit, dtype=torch.long, device=dev)
            ctx = torch.zeros(1, dtype=torch.long, device=dev) if model.config.n_contexts > 0 else None
            with torch.no_grad(), EK.autocast(dev, dtype):
                model(seq[None], seg, ctx)
            pred[j] = model.last_x[head][0][:, :len(ids)].float().clamp(-6e4, 6e4).cpu().numpy().astype(np.float16)
        info = dict(label=label, checkpoint=str(where), step=int(state.get("step", -1)), genome=name, dtype=dtype,
                    n_windows=win.height, seconds=round(time.time() - t0, 1))
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        if sh[1] > 1:                                   # a shard: its windows' logits, joined by merge
            d = out / "shards" / label
            d.mkdir(parents=True, exist_ok=True)
            np.save(d / f"{EK.tag(sh)}.npy", pred)
            (d / f"{EK.tag(sh)}.json").write_text(json.dumps(dict(info, rows=rows, shard=list(sh)), indent=1))
            typer.echo(f"{label}: shard {EK.tag(sh)}, {len(rows)} windows in {info['seconds']:.0f} s")
            continue
        finish(out, info, pred, lab, ids)


def ids_of(labels: list[str]) -> list[str]:
    return X.open_store(labels[0].split("=", 1)[1]).ids


def labels_of(labels: list[str], win: pl.DataFrame) -> tuple[dict, list[str]]:
    stores = {t: X.open_store(p) for t, p in (s.split("=", 1) for s in labels)}
    ids = next(iter(stores.values())).ids
    assert all(st.ids == ids for st in stores.values()), "the annotations' channels differ"
    lab = {t: np.stack([st.window(int(r["bases_offset"]), WIN, WIN) for r in win.iter_rows(named=True)]).astype(bool)
           for t, st in stores.items()}                                 # [n, WIN, 14] each
    return lab, ids


@app.command()
def merge(out: Path, label: str, shards: int = typer.Option(..., help="n, as in --shard i/n"),
          labels: list[str] = typer.Option(..., help="TAG=STORE, as given to score"),
          windows_csv: Path = typer.Option(None, "--windows")) -> None:
    """Joins the n shards of LABEL and computes every metric, as an unsharded score would."""
    win = pl.read_csv(windows_csv or out / "windows.csv")
    d = out / "shards" / label
    missing = [i for i in range(shards) if not (d / f"{i}-of-{shards}.npy").exists()]
    if missing:
        raise typer.BadParameter(f"{label}: shards {missing} of {shards} missing")
    infos = [json.loads((d / f"{i}-of-{shards}.json").read_text()) for i in range(shards)]
    for k in ("checkpoint", "step", "genome", "dtype", "n_windows"):
        if len({json.dumps(i[k]) for i in infos}) > 1:
            raise typer.BadParameter(f"{label}: shards differ in {k}")
    if infos[0]["n_windows"] != win.height:
        raise typer.BadParameter(f"{label}: shards were scored on {infos[0]['n_windows']} windows, not {win.height}")
    lab, ids = labels_of(labels, win)
    pred = np.zeros((win.height, WIN, len(ids)), dtype=np.float16)
    seen = np.zeros(win.height, dtype=int)
    for i, inf in enumerate(infos):
        pred[inf["rows"]] = np.load(d / f"{i}-of-{shards}.npy")
        seen[inf["rows"]] += 1
    if not (seen == 1).all():
        raise typer.BadParameter(f"{label}: windows not covered exactly once")
    finish(out, dict(infos[0], seconds=sum(i["seconds"] for i in infos), shards=shards), pred, lab, ids)


def finish(out: Path, info: dict, pred: np.ndarray, lab: dict, ids: list[str]) -> None:
    """Every metric from the logits ``pred`` [windows, WIN, channels] against each annotation's labels."""
    label = info["label"]
    for k in ("rows", "shard"):
        info.pop(k, None)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{label}.genes.npy", pred)         # the logits, so later metrics need no card
    call = pred > 0
    res = dict(info, annotations={})
    for t, y in lab.items():
        per = {}
        for c, i in enumerate(ids):
            yc, pc = y[..., c], call[..., c]
            tp, fp, fn = int((yc & pc).sum()), int((~yc & pc).sum()), int((yc & ~pc).sum())
            ap = XB.RankedSplice(pred[..., c], yc).ap() if yc.any() else float("nan")
            per[i] = dict(ap=ap, f1=f1(tp, fp, fn), positives=int(yc.sum()))
        summ = {}
        for nm, kinds in (("genic_f1", GENIC), ("subgenic_f1", SUBGENIC)):
            cols = [c for c, i in enumerate(ids) if i.rsplit("_", 1)[0] in kinds]
            yc, pc = y[..., cols], call[..., cols]
            summ[nm] = f1(int((yc & pc).sum()), int((~yc & pc).sum()), int((yc & ~pc).sum()))
        ex = {st: exon_f1(call[..., ids.index(f"cds_{st}")], y[..., ids.index(f"cds_{st}")])
              for st in ("plus", "minus")}
        tp_ = sum(v["exact"] for v in ex.values())
        summ["cds_exon_f1"] = f1(tp_, sum(v["predicted"] for v in ex.values()) - tp_,
                                 sum(v["annotated"] for v in ex.values()) - tp_)
        res["annotations"][t] = dict(channels=per, cds_exons=ex, **summ)
    tags = list(lab)
    if len(tags) == 2:                           # where the two annotations disagree: whose side the head takes
        a, b = tags
        dis = {}
        for c, i in enumerate(ids):
            d = lab[a][..., c] != lab[b][..., c]
            n = int(d.sum())
            if n:
                pc = call[..., c][d]
                dis[i] = dict(bases=n, agrees_with={a: float((pc == lab[a][..., c][d]).mean()),
                                                    b: float((pc == lab[b][..., c][d]).mean())})
        res["disagreement"] = dis
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{label}.genes.json").write_text(json.dumps(res, indent=1))
    for t in tags:
        an = res["annotations"][t]
        typer.echo(f"{label} vs {t}: Genic F1 {an['genic_f1']:.3f}, Subgenic F1 {an['subgenic_f1']:.3f}, "
                   f"CDS exon F1 {an['cds_exon_f1']:.3f}; AP "
                   + ", ".join(f"{i} {v['ap']:.2f}" for i, v in an["channels"].items() if i.endswith("_plus")))
    if "disagreement" in res:
        for i, v in res["disagreement"].items():
            typer.echo(f"  {i}: {v['bases']:,} bases differ; head agrees with "
                       + ", ".join(f"{t} {s:.1%}" for t, s in v["agrees_with"].items()))


if __name__ == "__main__":
    app()
