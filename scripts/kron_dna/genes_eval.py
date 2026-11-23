"""The gene-structure head on any prepared genome, against one or more annotations of it (the annotation
validation set: a species in no training genome, an older and a newer annotation of the same assembly).

    uv run python scripts/kron_dna/splice_zeroshot.py windows OUT --name dmel_r6 [--n 96]      (the windows)
    uv run python scripts/kron_dna/genes_eval.py score OUT --name dmel_r6 --labels old=STORE --labels new=STORE \
        LABEL=CHECKPOINT [--dtype bf16|fp16|fp32] [--shard i/n]
    uv run python scripts/kron_dna/genes_eval.py merge OUT LABEL --shards n --labels old=STORE --labels new=STORE
    uv run python scripts/kron_dna/genes_eval.py rescore OUT LABEL --labels old=STORE ...   (from saved logits, no card)

For each window the model reads 1,048,576 bases centred on it; its ``genes`` head runs on the window's 131,072 bases
(as ``splice_zeroshot.py``). Per annotation and channel (tss, polya, utr5, cds, utr3, ncexon, intron; per strand):

- average precision of the head's logits against the labels, exact over every base of every window;
- base-wise F1 of the head's calls (logit > 0) against the labels, and Helixer's summaries: Genic F1 over the
  CDS, UTR (5' and 3') and intron bases of both strands pooled, Subgenic F1 over CDS and intron;
- Tiberius's exon level for CDS exons: a run of CDS calls is right only if an annotated CDS exon (isoforms
  pooled, so close to but not exactly Tiberius's longest-CDS transcript per gene) has exactly its two ends;
- where two annotations disagree on a base and channel, the share of those bases on which the head's call agrees
  with each of them (the newer annotation is the better one; does the head side with it).

- the same exon level after decoding, since per-base calls are fragmented and their ends not base-exact (Tiberius
  and Helixer decode with an HMM): calls smoothed (gaps to 20 bases closed, runs under 30 dropped); ends within 10
  bases; **snapped**: each end of a smoothed exon moved to the splice head's strongest donor or acceptor site
  within 48 bases; and **paired**: an exon between two of the splice head's sites (the low-end and high-end kind
  of its strand, up to 3 kb apart) when the mean CDS probability between them is at least 0.5, internal coding
  exons only. Both scored with exact ends; paired is the number to set beside Tiberius's.

Writes ``OUT/<label>.splice.npy`` (the splice head's logits, float16 ``[windows, 131072, 4]``),
``OUT/<label>.genes.json`` with all of it and ``OUT/<label>.genes.npy`` (the logits, float16
``[windows, 131072, 14]``, so a later metric needs no card), and prints the summary.
"""
from __future__ import annotations

import bisect
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


def exon_f1(call: np.ndarray, lab: np.ndarray, tol: int = 0) -> dict:
    """Tiberius's exon level for CDS exons: a predicted exon (a run of CDS calls on one strand) is right only if an
    annotated one (a run of CDS labels; isoforms pooled) has exactly its two ends. ``tol`` > 0: both ends within
    ``tol`` bases; precision counts the predicted exons with such a partner, recall the annotated ones."""
    p, t = runs(call), runs(lab)
    if not tol:
        tp = len(p & t)
        return dict(f1=f1(tp, len(p) - tp, len(t) - tp), predicted=len(p), annotated=len(t), exact=tp)
    hit_p, hit_t = near(p, t, tol), near(t, p, tol)
    prec, rec = hit_p / max(len(p), 1), hit_t / max(len(t), 1)
    return dict(f1=float(2 * prec * rec / max(prec + rec, 1e-12)), predicted=len(p), annotated=len(t),
                matched_predicted=hit_p, matched_annotated=hit_t, tol=tol)


def near(a: set, b: set, tol: int) -> int:
    """How many runs of ``a`` have a run of ``b`` (same window) with both ends within ``tol`` bases."""
    by = {}
    for w, x, y in b:
        by.setdefault(w, []).append((x, y))
    for w in by:
        by[w].sort()
    n = 0
    for w, x, y in a:
        lst = by.get(w, [])
        i = bisect.bisect_left(lst, (x - tol, -1))
        while i < len(lst) and lst[i][0] <= x + tol:
            if abs(lst[i][1] - y) <= tol:
                n += 1
                break
            i += 1
    return n


SITES = {"plus": (1, 0), "minus": (2, 3)}        # splice channels at an exon's low and high end: acceptor/donor +,
                                                 # donor/acceptor - (channels: donor +, acceptor +, donor -, acceptor -)


def snap(cds: np.ndarray, splice: np.ndarray, strand: str, reach: int = 48, thr: float = 0.0) -> set:
    """Exons ``{(window, start, end)}`` from smoothed CDS calls ``[n, L]`` of one strand, each end moved to the
    splice head's strongest site of the right kind within ``reach`` bases (logit above ``thr``), where there is one:
    an exon ``[a, b)`` begins one base after the site at its low end (the acceptor on +, the donor on -: the
    intron's last or first base) and ends on the site at its high end (the donor on +, the acceptor on -). An end
    with no site near keeps the call's (a first or last coding exon ends at a codon, not a splice site). The least
    decoder that gives exact ends; no gene model, no reading frame."""
    lo_ch, hi_ch = SITES[strand]
    out, L = set(), cds.shape[1]
    for w, a, b in runs(cds):
        s_lo = splice[w, max(a - 1 - reach, 0):min(a - 1 + reach + 1, L), lo_ch].astype(np.float32)
        s_hi = splice[w, max(b - reach, 0):min(b + reach + 1, L), hi_ch].astype(np.float32)
        a2 = max(a - 1 - reach, 0) + int(s_lo.argmax()) + 1 if s_lo.size and s_lo.max() > thr else a
        b2 = max(b - reach, 0) + int(s_hi.argmax()) if s_hi.size and s_hi.max() > thr else b
        if b2 > a2:
            out.add((w, a2, b2))
    return out


def pair_exons(logit: np.ndarray, splice: np.ndarray, strand: str, cover: float = 0.5, longest: int = 3000,
               thr: float = 0.0) -> set:
    """Exons ``{(window, start, end)}`` between the splice head's sites: for every site at an exon's low end (logit
    above ``thr``), the site at a high end within ``longest`` bases whose stretch between them has the highest mean
    CDS probability, if that is at least ``cover``. Internal coding exons only (a first or last one ends at a codon);
    exact ends by construction."""
    lo_ch, hi_ch = SITES[strand]
    prob = 1 / (1 + np.exp(-logit.astype(np.float32)))
    out = set()
    for w in range(prob.shape[0]):
        cs = np.concatenate([[0.0], np.cumsum(prob[w], dtype=np.float64)])
        los = np.flatnonzero(splice[w, :, lo_ch] > thr) + 1
        his = np.flatnonzero(splice[w, :, hi_ch] > thr)
        for a in los:
            j = int(np.searchsorted(his, a + 1))
            k = int(np.searchsorted(his, a + longest, side="right"))
            if k > j:
                b = his[j:k]
                c = (cs[b] - cs[a]) / (b - a)
                i = int(c.argmax())
                if c[i] >= cover:
                    out.add((w, int(a), int(b[i])))
    return out


def smooth(call: np.ndarray, gap: int = 20, short: int = 30) -> np.ndarray:
    """``[n, L]`` calls with gaps of up to ``gap`` bases closed and runs under ``short`` bases dropped: the least
    decoding that turns per-base calls into exons (no gene model; Tiberius and Helixer decode with an HMM)."""
    from scipy.ndimage import binary_closing, binary_opening
    return np.stack([binary_opening(binary_closing(c, np.ones(gap + 1, bool)), np.ones(short, bool)) for c in call])


@app.command()
def score(out: Path, checkpoints: list[str] = typer.Argument(..., help="LABEL=CHECKPOINT"),
          name: str = typer.Option(...), labels: list[str] = typer.Option(..., help="TAG=STORE, one per annotation"),
          windows_csv: Path = typer.Option(None, "--windows"), head: str = "genes", context: int = 1048576,
          splice_head: str = typer.Option("splice", help="its logits are saved too, for the snapped exons ('' = no)"),
          device: str = "cuda",
          dtype: str = typer.Option("bf16", help="bf16 | fp16 | fp32 (bf16 is emulated, slow, on Turing cards)"),
          chunk: int = typer.Option(0, help="bases per encoder chunk (memory cap: e.g. 131072 runs a 1 Mb context "
                                            "on an 11 GB card; 0: the whole window at once)"),
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
        if chunk and not EK.capped_ok(model):
            raise typer.BadParameter(f"{label}: --chunk needs a v2/v2l core without learned chunking")
        model = model.to(dev).eval()
        sp_head = splice_head if splice_head in heads else ""
        model.x_active, model.x_span = {head, sp_head} - {""}, (flank, flank + WIN)
        unit = 2 ** model.config.n_down
        rows = [k for k in range(win.height) if EK.mine(k, sh)]
        pred = np.zeros((len(rows), WIN, len(ids)), dtype=np.float16)
        sp = np.zeros((len(rows), WIN, 4), dtype=np.float16) if sp_head else None
        t0 = time.time()
        for j, (k, r) in enumerate((k, r) for k, r in enumerate(win.iter_rows(named=True)) if EK.mine(k, sh)):
            seq = torch.from_numpy(XB.codes_of(contigs, bases, r["chrom"], r["start"] - flank, r["end"] + flank)).to(dev)
            seg = torch.zeros(1, context // unit, dtype=torch.long, device=dev)
            ctx = torch.zeros(1, dtype=torch.long, device=dev) if model.config.n_contexts > 0 else None
            with torch.no_grad(), EK.autocast(dev, dtype):
                if chunk:
                    EK.forward_capped(model, seq, ctx, (flank, flank + WIN), chunk)
                else:
                    model(seq[None], seg, ctx)
            pred[j] = model.last_x[head][0][:, :len(ids)].float().clamp(-6e4, 6e4).cpu().numpy().astype(np.float16)
            if sp_head:
                sp[j] = model.last_x[sp_head][0][:, :4].float().clamp(-6e4, 6e4).cpu().numpy().astype(np.float16)
        info = dict(label=label, checkpoint=str(where), step=int(state.get("step", -1)), genome=name, dtype=dtype,
                    context=context, chunk=chunk,
                    n_windows=win.height, seconds=round(time.time() - t0, 1))
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        if sh[1] > 1:                                   # a shard: its windows' logits, joined by merge
            d = out / "shards" / label
            d.mkdir(parents=True, exist_ok=True)
            np.save(d / f"{EK.tag(sh)}.npy", pred)
            if sp is not None:
                np.save(d / f"{EK.tag(sh)}.splice.npy", sp)
            (d / f"{EK.tag(sh)}.json").write_text(json.dumps(dict(info, rows=rows, shard=list(sh)), indent=1))
            typer.echo(f"{label}: shard {EK.tag(sh)}, {len(rows)} windows in {info['seconds']:.0f} s")
            continue
        finish(out, info, pred, lab, ids, sp)


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
    has_sp = all((d / f"{i}-of-{shards}.splice.npy").exists() for i in range(shards))
    sp = np.zeros((win.height, WIN, 4), dtype=np.float16) if has_sp else None
    seen = np.zeros(win.height, dtype=int)
    for i, inf in enumerate(infos):
        pred[inf["rows"]] = np.load(d / f"{i}-of-{shards}.npy")
        if has_sp:
            sp[inf["rows"]] = np.load(d / f"{i}-of-{shards}.splice.npy")
        seen[inf["rows"]] += 1
    if not (seen == 1).all():
        raise typer.BadParameter(f"{label}: windows not covered exactly once")
    finish(out, dict(infos[0], seconds=sum(i["seconds"] for i in infos), shards=shards), pred, lab, ids, sp)


@app.command()
def rescore(out: Path, label: str, labels: list[str] = typer.Option(..., help="TAG=STORE, as given to score"),
            windows_csv: Path = typer.Option(None, "--windows")) -> None:
    """Every metric again from the saved logits (OUT/LABEL.genes.npy) and labels: no card."""
    win = pl.read_csv(windows_csv or out / "windows.csv")
    info = json.loads((out / f"{label}.genes.json").read_text())
    info = {k: v for k, v in info.items() if k not in ("annotations", "disagreement")}
    lab, ids = labels_of(labels, win)
    sp = out / f"{label}.splice.npy"
    finish(out, info, np.load(out / f"{label}.genes.npy"), lab, ids, np.load(sp) if sp.exists() else None)


def finish(out: Path, info: dict, pred: np.ndarray, lab: dict, ids: list[str], splice: np.ndarray | None = None
           ) -> None:
    """Every metric from the logits ``pred`` [windows, WIN, channels] against each annotation's labels; with the
    splice head's logits ``splice`` [windows, WIN, 4] also the snapped exons."""
    label = info["label"]
    for k in ("rows", "shard"):
        info.pop(k, None)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{label}.genes.npy", pred)         # the logits, so later metrics need no card
    call = pred > 0
    sm_cds = {st: smooth(call[..., ids.index(f"cds_{st}")]) for st in ("plus", "minus")}
    snapped = None
    if splice is not None:
        np.save(out / f"{label}.splice.npy", splice)
        snapped = {"snapped": {st: snap(sm_cds[st], splice, st) for st in ("plus", "minus")},
                   "paired": {st: pair_exons(pred[..., ids.index(f"cds_{st}")], splice, st) for st in ("plus", "minus")}}
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
        variants = {}                                # the same exons after smoothing, and with ends within 10 bp
        for nm, sm, tol in (("smoothed", True, 0), ("within10", False, 10), ("smoothed_within10", True, 10)):
            per_st = {st: exon_f1(sm_cds[st] if sm else call[..., ids.index(f"cds_{st}")],
                                  y[..., ids.index(f"cds_{st}")], tol) for st in ("plus", "minus")}
            if tol:
                P_, T_ = (sum(v[k] for v in per_st.values()) for k in ("predicted", "annotated"))
                mp, ma = (sum(v[k] for v in per_st.values()) for k in ("matched_predicted", "matched_annotated"))
                pr, rc = mp / max(P_, 1), ma / max(T_, 1)
                variants[nm] = dict(f1=float(2 * pr * rc / max(pr + rc, 1e-12)), per_strand=per_st)
            else:
                tp2 = sum(v["exact"] for v in per_st.values())
                variants[nm] = dict(f1=f1(tp2, sum(v["predicted"] for v in per_st.values()) - tp2,
                                          sum(v["annotated"] for v in per_st.values()) - tp2), per_strand=per_st)
            summ[f"cds_exon_f1_{nm}"] = variants[nm]["f1"]
        for nm, dec in (snapped or {}).items():      # exons with their ends on the splice head's sites: exact ends
            per_st = {}
            for st in ("plus", "minus"):
                tr = runs(y[..., ids.index(f"cds_{st}")])
                tp2 = len(dec[st] & tr)
                per_st[st] = dict(f1=f1(tp2, len(dec[st]) - tp2, len(tr) - tp2), predicted=len(dec[st]),
                                  annotated=len(tr), exact=tp2)
            tp2 = sum(v["exact"] for v in per_st.values())
            variants[nm] = dict(f1=f1(tp2, sum(v["predicted"] for v in per_st.values()) - tp2,
                                      sum(v["annotated"] for v in per_st.values()) - tp2), per_strand=per_st)
            summ[f"cds_exon_f1_{nm}"] = variants[nm]["f1"]
        res["annotations"][t] = dict(channels=per, cds_exons=ex, cds_exon_variants=variants, **summ)
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
                   f"CDS exon F1 {an['cds_exon_f1']:.3f} (smoothed {an['cds_exon_f1_smoothed']:.3f}, ±10 bp "
                   f"{an['cds_exon_f1_within10']:.3f}, both {an['cds_exon_f1_smoothed_within10']:.3f}"
                   + (f", snapped to splice sites {an['cds_exon_f1_snapped']:.3f}, between site pairs "
                      f"{an['cds_exon_f1_paired']:.3f}" if "cds_exon_f1_paired" in an else "")
                   + "); AP "
                   + ", ".join(f"{i} {v['ap']:.2f}" for i, v in an["channels"].items() if i.endswith("_plus")))
    if "disagreement" in res:
        for i, v in res["disagreement"].items():
            typer.echo(f"  {i}: {v['bases']:,} bases differ; head agrees with "
                       + ", ".join(f"{t} {s:.1%}" for t, s in v["agrees_with"].items()))


if __name__ == "__main__":
    app()
