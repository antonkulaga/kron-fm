"""Where a KronGenome checkpoint's predictions come from, on the held-out benchmark windows (chr8/chr9, 131,072 bases
each, 1 Mb of context centred on each): masked-base loss by sequence class, and ablations of the core and the conv
tower's skips. Inference only; one card.

    python scripts/kron_dna/diag/ablate.py classes CKPT BENCH GENES OUT.json [--passes 4] [--limit N]
    python scripts/kron_dna/diag/ablate.py core CKPT BENCH [--windows 6]          # all core layers skipped / core zeroed
    python scripts/kron_dna/diag/ablate.py layers CKPT BENCH [--windows 32]       # each core layer skipped: DNase r, bits
    python scripts/kron_dna/diag/ablate.py skips CKPT BENCH [--windows 6]         # each encoder skip zeroed: bits
    python scripts/kron_dna/diag/ablate.py joint CKPT BENCH OUT.json [--sets 1-5 7-9 0-9] [--windows 32]
    python scripts/kron_dna/diag/ablate.py reset CKPT BENCH OUT.json [--reach 8192 32768 131072] [--windows 32]
    python scripts/kron_dna/diag/ablate.py cpg CKPT BENCH OUT.json [--windows 32] [--passes 2]
    python scripts/kron_dna/diag/ablate.py spans CKPT BENCH OUT.json [--lens 1024 4096 16384] [--windows 16]
    python scripts/kron_dna/diag/ablate.py flanks CKPT BENCH OUT.json [--keep 0 32768 131072] [--windows 32]

``flanks`` (Astra 6.2's flank replacement): keep the scored window plus KEEP bases each side, replace everything
beyond with the same stretch of another benchmark window (real sequence, window i takes the flanks of window i + n/2), and
measure how far every head's output on the scored window moves (relative L2 of raw outputs), and DNase Pearson
against the frozen targets, paired over windows. It tests use of distant DNA through every path, not only the core.

``spans``: masked bases inside contiguous blocks (every base of a block MASK, blocks covering about 10 % of the central
window, placed at random with a fixed seed), scored for the intact model, with every core layer skipped, and with the
core output zeroed: whether long context matters once the neighbours are hidden too.

``joint``: sets of core layers skipped together; ``reset``: the core's recurrence restarted at REACH bases on each side
of the central window (segment ids; the conv path, its context and the heads unchanged), so the core sees only that
much context. Both give DNase Pearson and masked bits per condition with 95 % bootstrap intervals of the paired
difference to the intact model, windows resampled together (1,000 draws). ``cpg``: at masked bases whose left
neighbour is a visible C, the predicted P(G) against the observed G frequency in bins of the prediction (and the
same for P(C) left of a visible G), by repeat flag: calibration over every CpG opportunity, not only realised CpGs.

BENCH is an xhead_bench folder (windows.csv, targets.json, target_dnase_128.npy); GENES the gene-structure label store
(tss/polya/utr5/cds/utr3/ncexon/intron, per strand). Masks: 15 % scattered MASK over the central window only, seeded
by the window's index, the flanks unmasked. A skipped core layer is the identity (each block is residual); a zeroed
skip or core output is a forward hook returning zeros. Everything runs in eval mode under bf16 autocast, as the
benchmark does.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from kron_core.paths import interim_dir
from kron_dna import genome as G
from kron_dna import xheads as X

WIN, CTX = 131072, 1048576
FL = (CTX - WIN) // 2
CLASSES = ["CDS", "5'UTR", "3'UTR", "ncRNA exon", "promoter (TSS +-1 kb)", "intron", "intergenic"]


class Skip(torch.nn.Module):
    """A core layer switched off: the residual block's identity."""
    def forward(self, h, cu):
        return h


class Genome:
    def __init__(self, name: str = "hg38"):
        src = interim_dir("genome", name)
        self.bases = np.memmap(src / "bases.u8", dtype=np.uint8, mode="r")
        self.repeat = np.memmap(src / "repeat.u8", dtype=np.uint8, mode="r")
        self.contigs = pl.read_parquet(src / "contigs.parquet")

    def codes(self, chrom: str, lo: int, hi: int) -> np.ndarray:
        """Base codes of chrom[lo:hi); outside the gap-free contigs N (4)."""
        out = np.full(hi - lo, 4, dtype=np.int64)
        for r in self.contigs.filter((pl.col("chrom") == chrom) & (pl.col("start") < hi)
                                     & (pl.col("end") > lo)).iter_rows(named=True):
            s, e = max(lo, r["start"]), min(hi, r["end"])
            out[s - lo:e - lo] = self.bases[r["offset"] + s - r["start"]:r["offset"] + e - r["start"]]
        return out


def classes_of(genes, off: int) -> np.ndarray:
    """Per base of the window at layout offset ``off``: the class index in CLASSES (priority CDS first)."""
    gid = list(genes.ids)
    pad = 1000                                     # start sites up to 1 kb outside the window still make promoters
    g = genes.window(off - pad, WIN + 2 * pad, WIN + 2 * pad).astype(bool)
    ch = lambda n: g[:, gid.index(n + "_plus")] | g[:, gid.index(n + "_minus")]  # noqa: E731
    tss = (np.convolve(ch("tss").astype(np.int32), np.ones(2001, np.int32), "same") > 0)[pad:pad + WIN]
    g = g[pad:pad + WIN]
    cls = np.full(WIN, 6, np.int8)
    for k, m in [(5, ch("intron")), (4, tss), (3, ch("ncexon")), (2, ch("utr3")), (1, ch("utr5")), (0, ch("cds"))]:
        cls[m] = k
    return cls


class Runner:
    def __init__(self, ckpt: str, heads=()):
        self.dev = torch.device("cuda")
        torch.cuda.init()
        self.model, self.state = G.load_checkpoint(ckpt)
        self.model = self.model.to(self.dev).eval()
        self.model.x_active = set(heads)
        self.model.x_span = (FL, FL + WIN) if heads else None
        self.unit = 2 ** self.model.config.n_down

    def reset_segments(self, reach: int) -> torch.Tensor:
        """Segment ids restarting the core's recurrence ``reach`` bases each side of the central window."""
        seg = torch.zeros(1, CTX // self.unit, dtype=torch.long, device=self.dev)
        lo, hi = (FL - reach) // self.unit, (FL + WIN + reach) // self.unit
        seg[:, lo:hi] = 1
        seg[:, hi:] = 2
        return seg

    def run(self, seq: torch.Tensor, seg: torch.Tensor | None = None) -> torch.Tensor:
        seg = seg if seg is not None else torch.zeros(1, CTX // self.unit, dtype=torch.long, device=self.dev)
        ctx = torch.zeros(1, dtype=torch.long, device=self.dev) if self.model.config.n_contexts > 0 else None
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return self.model(seq[None], seg, ctx)[0][0, FL:FL + WIN].float()

    def masked(self, codes: np.ndarray, seed: int):
        """(positions in the window, bits per masked base, hit per masked base) for one mask draw."""
        g = torch.Generator(device=self.dev).manual_seed(seed)
        seq = torch.from_numpy(codes).to(self.dev)
        inp, lab = G.mask_bases(seq[FL:FL + WIN], 0.15, generator=g)
        s2 = seq.clone()
        s2[FL:FL + WIN] = inp
        lg = self.run(s2)
        m = lab >= 0
        bits = F.cross_entropy(lg[m], lab[m], reduction="none") / np.log(2)
        return m.nonzero().squeeze(1).cpu().numpy(), bits.cpu().numpy(), (lg[m].argmax(-1) == lab[m]).cpu().numpy()


def windows(bench: str, limit: int = 0) -> pl.DataFrame:
    w = pl.read_csv(f"{bench}/windows.csv")
    return w.head(limit) if limit else w


def cmd_classes(a):
    win, gen, r = windows(a.bench, a.limit), Genome(), Runner(a.ckpt)
    genes = X.open_store(a.genes)
    acc = {c: [0.0, 0, 0] for c in range(7)}
    rep_acc, cpg = {v: [0.0, 0] for v in (0, 1)}, {v: [0.0, 0] for v in (0, 1)}
    gc_acc, allbits, per_win, t0 = {q: [0.0, 0] for q in range(4)}, [], [], time.time()
    for w, row in enumerate(win.iter_rows(named=True)):
        off = int(row["bases_offset"])
        codes = gen.codes(row["chrom"], row["start"] - FL, row["end"] + FL)
        cls, rp, mid = classes_of(genes, off), gen.repeat[off:off + WIN] > 0, codes[FL:FL + WIN]
        gc = np.isin(mid, (1, 2)).reshape(-1, 128).mean(1).repeat(128)
        cp = np.zeros(WIN, bool)                     # C of a CpG and its G, no wrap at the window's ends
        dinuc = (mid[:-1] == 1) & (mid[1:] == 2)
        cp[:-1] |= dinuc
        cp[1:] |= dinuc
        ws, wn = 0.0, 0
        for k in range(a.passes):
            pos, bits, hit = r.masked(codes, 1000 * w + k)
            for c in range(7):
                m = cls[pos] == c
                acc[c][0] += bits[m].sum(); acc[c][1] += m.sum(); acc[c][2] += hit[m].sum()
            for v in (0, 1):
                m = rp[pos] == v; rep_acc[v][0] += bits[m].sum(); rep_acc[v][1] += m.sum()
                m = cp[pos] == v; cpg[v][0] += bits[m].sum(); cpg[v][1] += m.sum()
            q = np.clip((gc[pos] * 4).astype(int), 0, 3)
            for v in range(4):
                m = q == v; gc_acc[v][0] += bits[m].sum(); gc_acc[v][1] += m.sum()
            ws += bits.sum(); wn += bits.size; allbits.append(bits.astype(np.float16))
        per_win.append(dict(window=row["window_id"], chrom=row["chrom"], start=row["start"], bits=ws / wn))
        if w % 8 == 0:
            print(f"window {w + 1}/{win.height} {ws / wn:.3f} bits, {time.time() - t0:.0f} s", flush=True)
    ab = np.concatenate(allbits).astype(np.float32); tot = ab.sum(); srt = np.sort(ab)[::-1]
    res = dict(ckpt=a.ckpt, step=int(r.state.get("step", -1)), windows=win.height, passes=a.passes,
               overall_bits=float(ab.mean()),
               by_class=[dict(cls=CLASSES[c], bases=int(v[1]), bits=v[0] / max(v[1], 1), acc=v[2] / max(v[1], 1),
                              share_of_loss=v[0] / tot) for c, v in acc.items()],
               by_repeat=[dict(repeat=bool(v), bits=x[0] / max(x[1], 1), n=int(x[1])) for v, x in rep_acc.items()],
               by_gc=[dict(gc=f"{25 * v}-{25 * v + 25}%", bits=x[0] / max(x[1], 1), n=int(x[1])) for v, x in gc_acc.items()],
               by_cpg=[dict(cpg=bool(v), bits=x[0] / max(x[1], 1), n=int(x[1])) for v, x in cpg.items()],
               loss_concentration={f"top{p}%": float(srt[:int(len(srt) * p / 100)].sum() / tot) for p in (1, 5, 10, 25)},
               per_window=per_win)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k != "per_window"}, indent=1, default=float))


def cmd_core(a):
    win, gen, r = windows(a.bench, a.windows), Genome(), Runner(a.ckpt, heads=("dnase",))
    orig = list(r.model.core)
    rows = []
    for w, row in enumerate(win.iter_rows(named=True)):
        codes = gen.codes(row["chrom"], row["start"] - FL, row["end"] + FL)
        out = {}
        for mode in ("base", "skip_all", "zero_core"):
            r.model.core = torch.nn.ModuleList([Skip() for _ in orig] if mode == "skip_all" else orig)
            hook = r.model.core_norm.register_forward_hook(lambda m, i, o: torch.zeros_like(o)) \
                if mode == "zero_core" else None
            pos, bits, _ = r.masked(codes, w)
            out[mode] = (float(bits.mean()), r.model.last_ntv3, r.model.last_x["dnase"][0].float().clone())
            if hook is not None:
                hook.remove()
        r.model.core = torch.nn.ModuleList(orig)
        rel = lambda x, y: float((x - y).norm() / (y.norm() + 1e-9))  # noqa: E731
        for mode, (b, _, dn) in out.items():
            rows.append(dict(window=w, mode=mode, bits=b, dnase_change=rel(dn, out["base"][2])))
    df = pl.DataFrame(rows).group_by("mode").agg(pl.col("bits").mean(), pl.col("dnase_change").mean())
    print(df)


def cmd_layers(a):
    win, gen, r = windows(a.bench, a.windows), Genome(), Runner(a.ckpt, heads=("dnase",))
    meta = json.load(open(f"{a.bench}/targets.json"))
    ds = X.open_store(meta["dnase_store"]); consts = ds.consts(r.dev)
    tgt = np.load(f"{a.bench}/target_dnase_128.npy")[:win.height]
    seqs = [gen.codes(row["chrom"], row["start"] - FL, row["end"] + FL) for row in win.iter_rows(named=True)]

    def evaluate():
        pr = np.zeros(tgt.shape, np.float32); bits = []
        for w, codes in enumerate(seqs):
            r.run(torch.from_numpy(codes).to(r.dev))                  # DNase on the unmasked window
            pd = r.model.last_x["dnase"][0].float()
            p_log, _ = X.eval_pair(ds, pd, torch.zeros_like(pd), consts)
            pr[w] = p_log.cpu().numpy().reshape(-1, 128, p_log.shape[-1]).mean(1)
            bits.append(float(r.masked(codes, w)[1].mean()))
        t, p = tgt.reshape(-1, tgt.shape[-1]), pr.reshape(-1, tgt.shape[-1])
        return float(np.mean([np.corrcoef(t[:, j], p[:, j])[0, 1] for j in range(t.shape[1])])), float(np.mean(bits))

    r0, b0 = evaluate(); print(f"all layers: DNase r {r0:.4f}, bits {b0:.4f}", flush=True)
    for i in range(len(r.model.core)):
        keep = r.model.core[i]; r.model.core[i] = Skip()
        rr, bb = evaluate(); r.model.core[i] = keep
        print(f"without core {i}: DNase r {rr:.4f} ({rr - r0:+.4f}), bits {bb:.4f} ({bb - b0:+.5f})", flush=True)


def cmd_skips(a):
    win, gen, r = windows(a.bench, a.windows), Genome(), Runner(a.ckpt)
    seqs = [gen.codes(row["chrom"], row["start"] - FL, row["end"] + FL) for row in win.iter_rows(named=True)]
    bits = lambda: float(np.mean([r.masked(c, w)[1].mean() for w, c in enumerate(seqs)]))  # noqa: E731
    b0 = bits(); print(f"all: {b0:.4f}")
    for i in range(r.model.config.n_down):
        h = r.model.skip_proj[i].register_forward_hook(lambda m, i_, o: torch.zeros_like(o))
        b = bits(); h.remove()
        level = 2 ** (r.model.config.n_down - 1 - i)   # decode() reads the skips coarsest first (skip_proj[0])
        print(f"skip_proj[{i}] ({level} bp level) zeroed: {b:.4f} ({b - b0:+.4f})", flush=True)


def _dnase_setup(a, r):
    meta = json.load(open(f"{a.bench}/targets.json"))
    ds = X.open_store(meta["dnase_store"])
    return ds, ds.consts(r.dev)


def _condition(r, gen, rows, ds, consts, seg_fn=None):
    """Per window: DNase 128-bp log1p predictions [1024, T] and masked bits, under the model as it is now set up."""
    preds, bits = [], []
    for w, codes in enumerate(rows):
        seq = torch.from_numpy(codes).to(r.dev)
        seg = seg_fn() if seg_fn else torch.zeros(1, CTX // r.unit, dtype=torch.long, device=r.dev)
        ctx = torch.zeros(1, dtype=torch.long, device=r.dev) if r.model.config.n_contexts > 0 else None
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            r.model(seq[None], seg, ctx)
            pd = r.model.last_x["dnase"][0].float()
            p_log, _ = X.eval_pair(ds, pd, torch.zeros_like(pd), consts)
        preds.append(p_log.cpu().numpy().reshape(-1, 128, p_log.shape[-1]).mean(1))
        g = torch.Generator(device=r.dev).manual_seed(w)
        inp, lab = G.mask_bases(seq[FL:FL + WIN], 0.15, generator=g)
        s2 = seq.clone(); s2[FL:FL + WIN] = inp
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            lg = r.model(s2[None], seg, ctx)[0][0, FL:FL + WIN].float()
        m = lab >= 0
        bits.append(float((F.cross_entropy(lg[m], lab[m]) / np.log(2)).item()))
    return np.stack(preds), np.array(bits)


def _pooled_r(t, p):
    t, p = t.reshape(-1, t.shape[-1]), p.reshape(-1, p.shape[-1])
    return float(np.mean([np.corrcoef(t[:, j], p[:, j])[0, 1] for j in range(t.shape[1]) if t[:, j].std() > 0]))


def _paired(tgt, base, cond, n_boot=1000, seed=0):
    """Point difference and 95 % bootstrap interval (windows resampled together) of DNase r and bits, cond - base."""
    rng = np.random.default_rng(seed); n = tgt.shape[0]
    d_r = _pooled_r(tgt, cond[0]) - _pooled_r(tgt, base[0]); d_b = float(cond[1].mean() - base[1].mean())
    br, bb = [], []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        br.append(_pooled_r(tgt[i], cond[0][i]) - _pooled_r(tgt[i], base[0][i])); bb.append(float((cond[1][i] - base[1][i]).mean()))
    return dict(d_dnase_r=d_r, d_dnase_r_ci=[float(np.quantile(br, .025)), float(np.quantile(br, .975))],
                d_bits=d_b, d_bits_ci=[float(np.quantile(bb, .025)), float(np.quantile(bb, .975))])


def _layer_set(spec: str, n: int) -> list[int]:
    out = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return [i for i in out if i < n]


def cmd_joint(a):
    win, gen, r = windows(a.bench, a.windows), Genome(), Runner(a.ckpt, heads=("dnase",))
    ds, consts = _dnase_setup(a, r)
    tgt = np.load(f"{a.bench}/target_dnase_128.npy")[:win.height]
    rows = [gen.codes(row["chrom"], row["start"] - FL, row["end"] + FL) for row in win.iter_rows(named=True)]
    base = _condition(r, gen, rows, ds, consts)
    res = dict(ckpt=a.ckpt, windows=win.height, base_dnase_r=_pooled_r(tgt, base[0]), base_bits=float(base[1].mean()), sets={})
    orig = list(r.model.core)
    for spec in a.sets:
        drop = _layer_set(spec, len(orig))
        r.model.core = torch.nn.ModuleList([Skip() if i in drop else l for i, l in enumerate(orig)])
        res["sets"][spec] = _paired(tgt, base, _condition(r, gen, rows, ds, consts))
        r.model.core = torch.nn.ModuleList(orig)
        print(spec, json.dumps(res["sets"][spec]), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


def cmd_reset(a):
    win, gen, r = windows(a.bench, a.windows), Genome(), Runner(a.ckpt, heads=("dnase",))
    ds, consts = _dnase_setup(a, r)
    tgt = np.load(f"{a.bench}/target_dnase_128.npy")[:win.height]
    rows = [gen.codes(row["chrom"], row["start"] - FL, row["end"] + FL) for row in win.iter_rows(named=True)]
    base = _condition(r, gen, rows, ds, consts)
    res = dict(ckpt=a.ckpt, windows=win.height, base_dnase_r=_pooled_r(tgt, base[0]), base_bits=float(base[1].mean()), reach={})
    bins = CTX // r.unit
    for reach in a.reach:
        lo, hi = (FL - reach) // r.unit, (FL + WIN + reach) // r.unit
        seg = torch.zeros(1, bins, dtype=torch.long, device=r.dev); seg[:, lo:hi] = 1; seg[:, hi:] = 2
        res["reach"][str(reach)] = _paired(tgt, base, _condition(r, gen, rows, ds, consts, seg_fn=lambda: seg))
        print(reach, json.dumps(res["reach"][str(reach)]), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


def cmd_cpg(a):
    win, gen, r = windows(a.bench, a.windows), Genome(), Runner(a.ckpt)
    edges = np.linspace(0, 1, 11)
    acc = {(k, rp): np.zeros((10, 3)) for k in ("P(G) after C", "P(C) before G") for rp in (0, 1)}
    real = {k: [0.0, 0] for k in ("CpG G", "CpG C")}
    for w, row in enumerate(win.iter_rows(named=True)):
        codes = gen.codes(row["chrom"], row["start"] - FL, row["end"] + FL)
        off = int(row["bases_offset"]); rp = gen.repeat[off:off + WIN] > 0
        for k in range(a.passes):
            g = torch.Generator(device=r.dev).manual_seed(1000 * w + k)
            seq = torch.from_numpy(codes).to(r.dev)
            inp, lab = G.mask_bases(seq[FL:FL + WIN], 0.15, generator=g)
            s2 = seq.clone(); s2[FL:FL + WIN] = inp
            prob = torch.softmax(r.run(s2), -1).cpu().numpy()
            vis = inp.cpu().numpy(); lab = lab.cpu().numpy(); mid = codes[FL:FL + WIN]
            m = lab >= 0
            pos = np.flatnonzero(m[1:-1]) + 1
            for name, nb, want, sel in (("P(G) after C", pos - 1, 2, vis[pos - 1] == 1), ("P(C) before G", pos + 1, 1, vis[pos + 1] == 2)):
                p_ = pos[sel]; pr = prob[p_, want]; hit = (mid[p_] == want).astype(float)
                b = np.clip(np.digitize(pr, edges) - 1, 0, 9)
                for rv in (0, 1):
                    mm = rp[p_] == rv
                    np.add.at(acc[(name, rv)], b[mm], np.stack([np.ones(mm.sum()), pr[mm], hit[mm]], 1))
                rl = "CpG G" if want == 2 else "CpG C"
                real[rl][0] += float(-np.log2(np.clip(pr[hit > 0], 1e-9, 1)).sum()); real[rl][1] += int(hit.sum())
        if w % 8 == 0:
            print(f"window {w + 1}/{win.height}", flush=True)
    out = dict(ckpt=a.ckpt, windows=win.height, passes=a.passes, realised_bits={k: v[0] / max(v[1], 1) for k, v in real.items()},
               calibration=[dict(kind=k, repeat=bool(rv), bin=f"{edges[i]:.1f}-{edges[i + 1]:.1f}", n=int(t[i, 0]),
                                 mean_pred=float(t[i, 1] / max(t[i, 0], 1)), observed=float(t[i, 2] / max(t[i, 0], 1)))
                            for (k, rv), t in acc.items() for i in range(10) if t[i, 0] > 0],
               overall={f"{k} repeat={rv}": dict(mean_pred=float(t[:, 1].sum() / max(t[:, 0].sum(), 1)),
                                                 observed=float(t[:, 2].sum() / max(t[:, 0].sum(), 1)), n=int(t[:, 0].sum()))
                        for (k, rv), t in acc.items()})
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out["overall"], indent=1), json.dumps(out["realised_bits"]))


def cmd_spans(a):
    win, gen, r = windows(a.bench, a.windows), Genome(), Runner(a.ckpt)
    rows = [gen.codes(row["chrom"], row["start"] - FL, row["end"] + FL) for row in win.iter_rows(named=True)]
    orig = list(r.model.core)
    res = dict(ckpt=a.ckpt, windows=win.height, lens={})
    for L in a.lens:
        modes = ["intact", "core layers skipped", "core zeroed"] + [f"core reset {d // 1024} kb" for d in a.reach]
        out = {m: [] for m in modes}
        for w, codes in enumerate(rows):
            rng = np.random.default_rng(10_000 * w + L)
            n_blk = max(1, int(0.1 * WIN / L)); starts = np.sort(rng.choice(WIN // L, n_blk, replace=False)) * L
            at = np.zeros(WIN, bool)
            for s0 in starts:
                at[s0:s0 + L] = True
            at &= codes[FL:FL + WIN] < 4
            seq = torch.from_numpy(codes).to(r.dev); s2 = seq.clone()
            mid = s2[FL:FL + WIN]; mid[torch.from_numpy(at).to(r.dev)] = G.MASK_CHANNEL
            lab = seq[FL:FL + WIN][torch.from_numpy(at).to(r.dev)]
            for mode in out:
                r.model.core = torch.nn.ModuleList([Skip() for _ in orig] if mode == "core layers skipped" else orig)
                hook = r.model.core_norm.register_forward_hook(lambda m_, i_, o: torch.zeros_like(o)) \
                    if mode == "core zeroed" else None
                seg = r.reset_segments(int(mode.split()[2]) * 1024) if mode.startswith("core reset") else None
                lg = r.run(s2, seg)[torch.from_numpy(at).to(r.dev)]
                if hook is not None:
                    hook.remove()
                out[mode].append(float((F.cross_entropy(lg, lab) / np.log(2)).item()))
            r.model.core = torch.nn.ModuleList(orig)
        res["lens"][str(L)] = {m: float(np.mean(v)) for m, v in out.items()}
        print(L, json.dumps(res["lens"][str(L)]), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


def cmd_flanks(a):
    win = windows(a.bench, a.windows); gen = Genome()
    r = Runner(a.ckpt)
    heads = [h[0] for h in r.model.config.xheads]
    r.model.x_active = set(heads); r.model.x_span = (FL, FL + WIN)
    meta = json.load(open(f"{a.bench}/targets.json")); ds = X.open_store(meta["dnase_store"]); consts = ds.consts(r.dev)
    tgt = np.load(f"{a.bench}/target_dnase_128.npy")[:win.height]
    rows = [gen.codes(row["chrom"], row["start"] - FL, row["end"] + FL) for row in win.iter_rows(named=True)]
    seg = torch.zeros(1, CTX // r.unit, dtype=torch.long, device=r.dev)
    ctx = torch.zeros(1, dtype=torch.long, device=r.dev) if r.model.config.n_contexts > 0 else None

    def outputs(codes):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            r.model(torch.from_numpy(codes).to(r.dev)[None], seg, ctx)
        o = {h: r.model.last_x[h][0].float().clone() for h in heads}
        if r.model.last_ntv3 is not None:
            o["ntv3"] = r.model.last_ntv3[0, FL:FL + WIN].float().clone()
        pd = o["dnase"]; p_log, _ = X.eval_pair(ds, pd, torch.zeros_like(pd), consts)
        return o, p_log.cpu().numpy().reshape(-1, 128, p_log.shape[-1]).mean(1)

    base = [outputs(c) for c in rows]
    res = dict(ckpt=a.ckpt, windows=win.height, keep={})
    for keep in a.keep:
        lo, hi = FL - keep, FL + WIN + keep
        moves = {k: [] for k in base[0][0]}; preds = []
        for i, c in enumerate(rows):
            donor = rows[(i + len(rows) // 2) % len(rows)]   # a distant window, never an overlapping neighbour
            sw = donor.copy(); sw[lo:hi] = c[lo:hi]
            o, d128 = outputs(sw)
            preds.append(d128)
            for k in o:
                moves[k].append(float((o[k] - base[i][0][k]).norm() / (base[i][0][k].norm() + 1e-9)))
        p = np.stack(preds); b = np.stack([x[1] for x in base])
        res["keep"][str(keep)] = dict(output_move={k: float(np.mean(v)) for k, v in moves.items()},
                                      **_paired(tgt, (b, np.zeros(len(rows))), (p, np.zeros(len(rows)))))
        print(keep, json.dumps(res["keep"][str(keep)]), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("classes"); p.add_argument("ckpt"); p.add_argument("bench"); p.add_argument("genes")
    p.add_argument("out"); p.add_argument("--passes", type=int, default=4); p.add_argument("--limit", type=int, default=0)
    for name, n in (("core", 6), ("layers", 32), ("skips", 6)):
        p = sub.add_parser(name); p.add_argument("ckpt"); p.add_argument("bench")
        p.add_argument("--windows", type=int, default=n)
    p = sub.add_parser("joint"); p.add_argument("ckpt"); p.add_argument("bench"); p.add_argument("out")
    p.add_argument("--sets", nargs="+", default=["1-5", "7-9", "0-9"]); p.add_argument("--windows", type=int, default=32)
    p = sub.add_parser("reset"); p.add_argument("ckpt"); p.add_argument("bench"); p.add_argument("out")
    p.add_argument("--reach", nargs="+", type=int, default=[8192, 32768, 131072]); p.add_argument("--windows", type=int, default=32)
    p = sub.add_parser("cpg"); p.add_argument("ckpt"); p.add_argument("bench"); p.add_argument("out")
    p.add_argument("--windows", type=int, default=32); p.add_argument("--passes", type=int, default=2)
    p = sub.add_parser("spans"); p.add_argument("ckpt"); p.add_argument("bench"); p.add_argument("out")
    p.add_argument("--lens", nargs="+", type=int, default=[1024, 4096, 16384]); p.add_argument("--windows", type=int, default=16)
    p.add_argument("--reach", nargs="*", type=int, default=[8192, 32768, 131072])
    p = sub.add_parser("flanks"); p.add_argument("ckpt"); p.add_argument("bench"); p.add_argument("out")
    p.add_argument("--keep", nargs="+", type=int, default=[0, 32768, 131072]); p.add_argument("--windows", type=int, default=32)
    a = ap.parse_args()
    dict(classes=cmd_classes, core=cmd_core, layers=cmd_layers, skips=cmd_skips, joint=cmd_joint, reset=cmd_reset,
         cpg=cmd_cpg, spans=cmd_spans, flanks=cmd_flanks)[a.cmd](a)


if __name__ == "__main__":
    main()
