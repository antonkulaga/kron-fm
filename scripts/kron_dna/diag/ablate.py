"""Where a KronGenome checkpoint's predictions come from, on the held-out benchmark windows (chr8/chr9, 131,072 bases
each, 1 Mb of context centred on each): masked-base loss by sequence class, and ablations of the core and the conv
tower's skips. Inference only; one card.

    python scripts/kron_dna/diag/ablate.py classes CKPT BENCH GENES OUT.json [--passes 4] [--limit N]
    python scripts/kron_dna/diag/ablate.py core CKPT BENCH [--windows 6]          # all core layers skipped / core zeroed
    python scripts/kron_dna/diag/ablate.py layers CKPT BENCH [--windows 32]       # each core layer skipped: DNase r, bits
    python scripts/kron_dna/diag/ablate.py skips CKPT BENCH [--windows 6]         # each encoder skip zeroed: bits

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
    g = genes.window(off, WIN, WIN).astype(bool)
    ch = lambda n: g[:, gid.index(n + "_plus")] | g[:, gid.index(n + "_minus")]  # noqa: E731
    tss = np.convolve(ch("tss").astype(np.int32), np.ones(2001, np.int32), "same") > 0
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

    def run(self, seq: torch.Tensor) -> torch.Tensor:
        seg = torch.zeros(1, CTX // self.unit, dtype=torch.long, device=self.dev)
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
        cp = ((mid == 1) & (np.roll(mid, -1) == 2)) | ((mid == 2) & (np.roll(mid, 1) == 1))
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
        print(f"skip {i} ({2 ** i} bp level) zeroed: {b:.4f} ({b - b0:+.4f})", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("classes"); p.add_argument("ckpt"); p.add_argument("bench"); p.add_argument("genes")
    p.add_argument("out"); p.add_argument("--passes", type=int, default=4); p.add_argument("--limit", type=int, default=0)
    for name, n in (("core", 6), ("layers", 32), ("skips", 6)):
        p = sub.add_parser(name); p.add_argument("ckpt"); p.add_argument("bench")
        p.add_argument("--windows", type=int, default=n)
    a = ap.parse_args()
    dict(classes=cmd_classes, core=cmd_core, layers=cmd_layers, skips=cmd_skips)[a.cmd](a)


if __name__ == "__main__":
    main()
