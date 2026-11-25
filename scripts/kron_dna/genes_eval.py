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
  **chained**: the best non-overlapping set of such stretches per window (``chain_exons``); **framed**: the same
  with a reading frame from the sequence (no stop codon inside an exon, the frame kept across introns;
  ``chain_exons_framed``); **terminal**: framed, plus first and last coding exons ending at a start or stop
  codon where the head's UTR channel says so. The last is the decoder to quote.

- every site-based decoder is also scored **against each isoform's exons** (``..._isoforms``): the distinct coding
  exons of every transcript of the annotation's GTF, wholly inside a window, exact ends. That is gffcompare's
  exon level on a CDS-only reference (what Tiberius and Helixer report), except that windows are scored, not whole
  genomes; ``..._isoforms_within3`` allows 3 bases at each end. The label runs above pool the isoforms instead: two
  isoforms' overlapping exons become one run that matches neither.

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


def chain_exons(logit: np.ndarray, splice: np.ndarray, strand: str, calls: np.ndarray | None = None, thr: float = 3.0,
                longest: int = 3000, clip: float = 1.0, site: float = 40.0, floor: float = 160.0) -> set:
    """Exons ``{(window, start, end)}`` as the best chain per window: the set of non-overlapping stretches, each from
    a low-end candidate to a high-end candidate at most ``longest`` bases on, with the highest total score (dynamic
    programming over exon ends). A stretch scores the sum of its bases' CDS logits clipped to ``+-clip``, plus
    ``site`` times each end's splice logit (capped at 8); it must score above ``floor``. Candidates are the splice
    head's sites above ``thr`` and, at ``-2 * site``, the ends of the smoothed calls ``calls`` (a first or last
    coding exon ends at a codon). The sites weigh far more than a base because the genes head is binned and its
    edges are blurred; defaults tuned on beluga (F) and checked on chicken (A3). No reading frame yet."""
    lo_ch, hi_ch = SITES[strand]
    opens: dict = {}
    if calls is not None:
        for w, a, b in runs(calls):
            o = opens.setdefault(w, ([], []))
            o[0].append(a)
            o[1].append(b)
    out = set()
    for w in range(logit.shape[0]):
        cs = np.concatenate([[0.0], np.cumsum(np.clip(logit[w].astype(np.float32), -clip, clip), dtype=np.float64)])
        s_lo, s_hi = splice[w, :, lo_ch].astype(np.float32), splice[w, :, hi_ch].astype(np.float32)
        A = {int(p) + 1: site * min(float(s_lo[p]), 8.0) for p in np.flatnonzero(s_lo > thr)}
        B = {int(p): site * min(float(s_hi[p]), 8.0) for p in np.flatnonzero(s_hi > thr)}
        for a in opens.get(w, ([], []))[0]:
            A.setdefault(int(a), -2 * site)
        for b in opens.get(w, ([], []))[1]:
            B.setdefault(int(b), -2 * site)
        if not A or not B:
            continue
        a_pos = np.array(sorted(A))
        a_sc = np.array([A[a] for a in a_pos])
        ends, best, back = [0], [0.0], [None]            # chains by their last exon's end; best is increasing
        for b in sorted(B):
            j0, j1 = np.searchsorted(a_pos, b - longest), np.searchsorted(a_pos, b)
            if j1 <= j0:
                continue
            aa = a_pos[j0:j1]
            sc = cs[b] - cs[aa] + a_sc[j0:j1] + B[b]
            prev = np.searchsorted(ends, aa, side="right") - 1          # the best chain ending at or before each start
            tot = np.where(sc > floor, sc + np.asarray(best)[prev], -np.inf)
            i = int(tot.argmax())
            if tot[i] > best[-1]:
                ends.append(b)
                best.append(float(tot[i]))
                back.append((int(aa[i]), int(prev[i])))
        k = len(ends) - 1
        while k > 0:
            a, prev_k = back[k]
            out.add((w, a, ends[k]))
            k = prev_k
    return out


STOPS = {"plus": [(3, 0, 0), (3, 0, 2), (3, 2, 0)],          # TAA TAG TGA; on -, read on the forward strand: TTA CTA TCA
         "minus": [(3, 3, 0), (1, 3, 0), (3, 1, 0)]}
START = {"plus": [(0, 3, 2)], "minus": [(1, 0, 3)]}          # ATG; on -, CAT


def codon_hits(seq: np.ndarray, codons: list) -> np.ndarray:
    """Where one of ``codons`` (base codes) starts in ``seq``."""
    n = len(seq)
    hit = np.zeros(n, bool)
    for x, y, z in codons:
        hit[:n - 2] |= (seq[:-2] == x) & (seq[1:-1] == y) & (seq[2:] == z)
    return hit


def last_stops(seq: np.ndarray, strand: str) -> np.ndarray:
    """``[3, L + 1]``: for a frame f and an end b, the largest start p (p % 3 == f, p + 3 <= b) of a stop codon of
    ``strand`` in the base codes ``seq``; -1 where there is none. A stretch [a, b) is open in frame f when that is
    below a."""
    n = len(seq)
    hit = codon_hits(seq, STOPS[strand])
    out = np.full((3, n + 1), -1, np.int64)
    for f in range(3):
        p = np.flatnonzero(hit)
        p = p[p % 3 == f]
        out[f, p + 3] = p
        out[f] = np.maximum.accumulate(out[f])
    return out


def chain_exons_framed(logit: np.ndarray, splice: np.ndarray, strand: str, seqs: list, calls: np.ndarray | None = None,
                       thr: float = 3.0, longest: int = 3000, clip: float = 1.0, site: float = 40.0, floor: float = 160.0,
                       brk: float = 200.0, utr: tuple | None = None, reach: int = 100, gmin: float = 1.0, span: int = 96
                       ) -> set:
    """``chain_exons`` with a reading frame, from the windows' base codes ``seqs``: an exon is read in one of three
    frames (the coordinate of its codons' low base, mod 3) and must have no stop codon in it (a codon across a
    junction is not looked at); the next exon of a chain continues the frame across the intron (frame minus start
    equals the previous frame minus end, mod 3), or the chain pays ``brk`` for a new gene or a wrong exon between.
    Dynamic programming over exon ends, the best chain kept per frame class. Scores and candidates as in
    ``chain_exons``; ``brk`` set on beluga (F), the other species untouched.

    With ``utr`` (two ``[windows, L]`` logit arrays of the genes head: the UTR channel expected below an exon's low
    end and the one above its high end, i.e. utr5 and utr3 on +, utr3 and utr5 on -) also first and last coding
    exons: within ``reach`` bases of a smoothed call's end, a start codon the exon begins with, or a stop codon it
    ends with, is a candidate end when the mean UTR logit over the ``span`` bases outside it (clipped to +-4)
    exceeds ``gmin``; it scores ``site`` times that mean and fixes the exon's frame. ``reach`` and ``gmin`` set on
    beluga (F)."""
    lo_ch, hi_ch = SITES[strand]
    opens: dict = {}
    if calls is not None:
        for w, a, b in runs(calls):
            o = opens.setdefault(w, ([], []))
            o[0].append(a)
            o[1].append(b)
    out = set()
    for w in range(logit.shape[0]):
        cs = np.concatenate([[0.0], np.cumsum(np.clip(logit[w].astype(np.float32), -clip, clip), dtype=np.float64)])
        s_lo, s_hi = splice[w, :, lo_ch].astype(np.float32), splice[w, :, hi_ch].astype(np.float32)
        seq = np.asarray(seqs[w])
        A = {int(p) + 1: (site * min(float(s_lo[p]), 8.0), False) for p in np.flatnonzero(s_lo > thr)}     # (score, the
        B = {int(p): (site * min(float(s_hi[p]), 8.0), False) for p in np.flatnonzero(s_hi > thr)}         # end is a codon)
        if utr is not None:                               # + : a start codon below, a stop above; - : the reverse
            first, last = codon_hits(seq, START[strand]), codon_hits(seq, STOPS[strand])
            lo_hit, hi_hit = (first, last) if strand == "plus" else (last, first)
            for a in opens.get(w, ([], []))[0]:
                for p in np.flatnonzero(lo_hit[max(a - reach, 0):a + reach + 1]) + max(a - reach, 0):
                    g = float(np.clip(utr[0][w, max(p - span, 0):p].astype(np.float32).mean(), -4, 4)) if p > 0 else 0.0
                    if g > gmin:
                        A.setdefault(int(p), (g * site, True))
            for b in opens.get(w, ([], []))[1]:
                for p in np.flatnonzero(hi_hit[max(b - reach - 3, 0):b + reach - 2]) + max(b - reach - 3, 0):
                    g = float(np.clip(utr[1][w, p + 3:p + 3 + span].astype(np.float32).mean(), -4, 4)) if p + 3 < len(seq) else 0.0
                    if g > gmin:
                        B.setdefault(int(p) + 3, (g * site, True))
        for a in opens.get(w, ([], []))[0]:
            A.setdefault(int(a), (-2 * site, False))
        for b in opens.get(w, ([], []))[1]:
            B.setdefault(int(b), (-2 * site, False))
        if not A or not B:
            continue
        stops = last_stops(seq, strand)
        a_pos = np.array(sorted(A))
        a_sc = np.array([A[a][0] for a in a_pos])
        a_fix = np.array([A[a][1] for a in a_pos], bool)
        nodes: list = [None]                              # (start, end, previous node); node 0 is the empty chain
        # chains by their last exon's end, best increasing: one list per frame class (frame - end) % 3, and [3] any
        ends, best, node = [[0] for _ in range(4)], [[0.0] for _ in range(4)], [[0] for _ in range(4)]
        for b in sorted(B):
            j0, j1 = np.searchsorted(a_pos, b - longest), np.searchsorted(a_pos, b)
            if j1 <= j0:
                continue
            aa, afix = a_pos[j0:j1], a_fix[j0:j1]
            b_sc, b_fix = B[b]
            sc = cs[b] - cs[aa] + a_sc[j0:j1] + b_sc
            i_any = np.searchsorted(ends[3], aa, side="right") - 1
            any_node = np.asarray(node[3])[i_any]
            any_best = np.where(any_node == 0, 0.0, np.asarray(best[3])[i_any] - brk)   # a window's first exon is free
            found = []
            for f in range(3):
                if b_fix and b % 3 != f:                  # a codon at the high end sets the frame
                    continue
                ok = (sc > floor) & (~afix | (aa % 3 == f))      # and so does one at the low end
                if strand == "plus":                      # the stop codon that closes the gene is allowed: + at the high end
                    ok &= aa > stops[f, b - 1 if b_fix else b]
                else:                                     # - at the low end
                    ok &= np.where(afix, aa >= stops[f, b], aa > stops[f, b])
                if not ok.any():
                    continue
                key = (f - aa) % 3
                pb, pn = any_best.copy(), any_node.copy()
                for k in range(3):
                    m = np.flatnonzero(key == k)
                    if not len(m):
                        continue
                    ik = np.searchsorted(ends[k], aa[m], side="right") - 1
                    vb, vn = np.asarray(best[k])[ik], np.asarray(node[k])[ik]
                    better = (vb > pb[m]) & (vn != 0)
                    pb[m[better]], pn[m[better]] = vb[better], vn[better]
                tot = np.where(ok, sc + pb, -np.inf)
                i = int(tot.argmax())
                found.append((float(tot[i]), f, int(aa[i]), int(pn[i])))
            for tot, f, a, prev in found:
                k = (f - b) % 3
                if tot > best[k][-1] or tot > best[3][-1]:
                    nodes.append((a, b, prev))
                    for q in (k, 3):
                        if tot > best[q][-1]:
                            ends[q].append(b)
                            best[q].append(tot)
                            node[q].append(len(nodes) - 1)
        n = node[3][-1]
        while n:
            a, b, n = nodes[n]
            out.add((w, a, b))
    return out


def seqs_of(genome: str, win: pl.DataFrame) -> list:
    """The windows' base codes (for the framed decoder)."""
    bases = np.memmap(interim_dir("genome", genome) / "bases.u8", dtype=np.uint8, mode="r")
    return [np.asarray(bases[o:o + WIN]) for o in win["bases_offset"].to_list()]


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
        finish(out, info, pred, lab, ids, sp, seqs_of(name, win), isoforms_of(labels, win))


def ids_of(labels: list[str]) -> list[str]:
    return X.open_store(labels[0].split("=", 1)[1]).ids


def isoform_exons(gtf: Path, win: pl.DataFrame, chrom_prefix: str = "") -> dict:
    """The annotation's distinct coding exons as gffcompare sees a CDS-only reference: per transcript, its CDS rows
    (the stop codon joined on, as in the labels) merged where they touch; the distinct ``(start, end)`` over all
    transcripts, kept when the exon lies wholly inside a window (window coordinates). ``{"plus": {(window, start,
    end)}, "minus": ...}``. Unlike the label runs, isoforms are not pooled: two isoforms' exons with different
    ends stay two exons."""
    from kron_dna.prep.features import read_features
    g = read_features(gtf).filter(pl.col("type").is_in(["CDS", "stop_codon"])).sort("tid", "start0")
    merged: set = set()
    cur = None
    for chrom, strand, tid, a, b in g.select("chrom", "strand", "tid", "start0", "end").iter_rows():
        if cur and cur[2] == tid and a <= cur[4]:
            cur[4] = max(cur[4], b)
            continue
        if cur:
            merged.add((cur[0], cur[1], cur[3], cur[4]))
        cur = [chrom_prefix + chrom, strand, tid, a, b]
    if cur:
        merged.add((cur[0], cur[1], cur[3], cur[4]))
    by: dict = {}
    for w, r in enumerate(win.iter_rows(named=True)):
        by.setdefault(r["chrom"], []).append((int(r["start"]), int(r["end"]), w))
    out = {"plus": set(), "minus": set()}
    for chrom, strand, a, b in merged:
        for lo, hi, w in by.get(chrom, ()):
            if a > lo and b < hi:                      # wholly inside: an exon at a window's edge is not scored
                out["plus" if strand == "+" else "minus"].add((w, a - lo, b - lo))
    return out


def isoforms_of(labels: list[str], win: pl.DataFrame) -> dict:
    """``isoform_exons`` per annotation tag, from the GTF each label store records it was built from (its
    ``labels.json``: ``annotation``); a store without that record, or whose file is gone, is left out."""
    out = {}
    for t, p in (s.split("=", 1) for s in labels):
        meta = Path(p) / "labels.json"
        gtf = Path(json.loads(meta.read_text()).get("annotation", "")) if meta.exists() else None
        if gtf and gtf.is_file():
            out[t] = isoform_exons(gtf, win)
    return out


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
    finish(out, dict(infos[0], seconds=sum(i["seconds"] for i in infos), shards=shards), pred, lab, ids, sp,
           seqs_of(infos[0]["genome"], win), isoforms_of(labels, win))


@app.command()
def rescore(out: Path, label: str, labels: list[str] = typer.Option(..., help="TAG=STORE, as given to score"),
            windows_csv: Path = typer.Option(None, "--windows")) -> None:
    """Every metric again from the saved logits (OUT/LABEL.genes.npy) and labels: no card."""
    win = pl.read_csv(windows_csv or out / "windows.csv")
    info = json.loads((out / f"{label}.genes.json").read_text())
    info = {k: v for k, v in info.items() if k not in ("annotations", "disagreement")}
    lab, ids = labels_of(labels, win)
    sp = out / f"{label}.splice.npy"
    finish(out, info, np.load(out / f"{label}.genes.npy"), lab, ids, np.load(sp) if sp.exists() else None,
           seqs_of(info["genome"], win), isoforms_of(labels, win))


def finish(out: Path, info: dict, pred: np.ndarray, lab: dict, ids: list[str], splice: np.ndarray | None = None,
           seqs: list | None = None, iso: dict | None = None) -> None:
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
                   "paired": {st: pair_exons(pred[..., ids.index(f"cds_{st}")], splice, st) for st in ("plus", "minus")},
                   "chained": {st: chain_exons(pred[..., ids.index(f"cds_{st}")], splice, st, sm_cds[st])
                               for st in ("plus", "minus")}}
        if seqs is not None:
            snapped["framed"] = {st: chain_exons_framed(pred[..., ids.index(f"cds_{st}")], splice, st, seqs, sm_cds[st])
                                 for st in ("plus", "minus")}
            outer = {"plus": ("utr5_plus", "utr3_plus"), "minus": ("utr3_minus", "utr5_minus")}     # below, above an exon
            if all(c in ids for cs_ in outer.values() for c in cs_):
                snapped["terminal"] = {st: chain_exons_framed(
                    pred[..., ids.index(f"cds_{st}")], splice, st, seqs, sm_cds[st],
                    utr=tuple(pred[..., ids.index(c)] for c in outer[st])) for st in ("plus", "minus")}
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
            if iso and t in iso:                     # against each isoform's exons (gffcompare's CDS-exon level)
                edge = pred.shape[1]
                P = {st: {e for e in dec[st] if e[1] > 0 and e[2] < edge} for st in ("plus", "minus")}
                n_p, n_t = sum(map(len, P.values())), sum(len(iso[t][st]) for st in P)
                tp3 = sum(len(P[st] & iso[t][st]) for st in P)
                hp = sum(near(P[st], iso[t][st], 3) for st in P)
                ht = sum(near(iso[t][st], P[st], 3) for st in P)
                pr, rc = hp / max(n_p, 1), ht / max(n_t, 1)
                variants[nm]["isoforms"] = dict(f1=f1(tp3, n_p - tp3, n_t - tp3), predicted=n_p, annotated=n_t, exact=tp3,
                                                precision=tp3 / max(n_p, 1), recall=tp3 / max(n_t, 1),
                                                f1_within3=float(2 * pr * rc / max(pr + rc, 1e-12)))
                summ[f"cds_exon_f1_{nm}_isoforms"] = variants[nm]["isoforms"]["f1"]
                summ[f"cds_exon_f1_{nm}_isoforms_within3"] = variants[nm]["isoforms"]["f1_within3"]
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
