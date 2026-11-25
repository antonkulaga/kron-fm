"""Timing pilot for an Evo 2 teacher on coding bases: how fast Evo 2 scores, and whether reading the spliced coding
sequence gives the per-base distributions that reading the gene in its genomic context gives.

usage: python evo2_teacher_pilot.py GTF GENOME_DIR OUT.json [--chrom chr21] [--genes 8] [--model evo2_7b]
       [--window 8192] [--budget-min 22]

Runs in this folder's environment (see evo2_clinvar.py), on one card. GENOME_DIR is a prepared genome
(``bases.u8`` + ``contigs.parquet``). Per gene (the transcript with the longest CDS, spans up to 300 kb), in the
gene's own direction:

- **spliced**: the CDS exons joined; windows of ``--window`` bases with half overlap, each base scored with at
  least half a window of left context (all of it for a CDS shorter than a window);
- **genomic**: the gene from 2 kb before its CDS to its end, introns included, scored the same way.

For every coding base both give Evo 2's log-probabilities of A, C, G, T as the next base. OUT.json: bases per second
of each mode, peak memory, and over the coding bases the mean log-likelihood of the true base under each, the share
of bases whose most likely base agrees between the two, and the Pearson correlation of the true base's
log-likelihood. Each mode stops at its time budget.
"""
import gzip
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

args = [a for a in sys.argv[1:] if not a.startswith("--")]
opt = {sys.argv[i][2:]: sys.argv[i + 1] for i in range(1, len(sys.argv) - 1) if sys.argv[i].startswith("--")}
gtf, genome, out = Path(args[0]), Path(args[1]), Path(args[2])
chrom, n_genes, name = opt.get("chrom", "chr21"), int(opt.get("genes", 8)), opt.get("model", "evo2_7b")
window, budget = int(opt.get("window", 8192)), float(opt.get("budget-min", 22)) * 60
torch.cuda.init()                                          # the card shows a process at once
COMP = bytes.maketrans(b"ACGTN", b"TGCAN")

# ---- genes: per transcript its CDS exons; the longest CDS per gene; spans up to 300 kb
tx: dict = {}
gid_re, tid_re = re.compile(r'gene_id "([^"]+)"'), re.compile(r'transcript_id "([^"]+)"')
with gzip.open(gtf, "rt") as f:
    for line in f:
        if line.startswith("#"):
            continue
        c = line.split("\t", 9)
        if c[0] != chrom or c[2] not in ("CDS", "stop_codon"):
            continue
        t = tx.setdefault(tid_re.search(c[8]).group(1), dict(gene=gid_re.search(c[8]).group(1), strand=c[6], ex=[]))
        t["ex"].append((int(c[3]) - 1, int(c[4])))
best: dict = {}
for t in tx.values():
    ex = sorted(t["ex"])
    merged = [list(ex[0])]
    for a, b in ex[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    t["ex"], t["cds"] = merged, sum(b - a for a, b in merged)
    if merged[-1][1] - merged[0][0] <= 300_000 and t["cds"] >= 600 and t["cds"] > best.get(t["gene"], dict(cds=0))["cds"]:
        best[t["gene"]] = t
genes = sorted(best.values(), key=lambda t: t["ex"][0][0])
genes = [genes[i] for i in np.linspace(0, len(genes) - 1, n_genes).round().astype(int)]

contigs = pl.read_parquet(genome / "contigs.parquet").filter(pl.col("chrom") == chrom).to_dicts()
bases = np.memmap(genome / "bases.u8", dtype=np.uint8, mode="r")


def fetch(lo: int, hi: int) -> bytes:
    """Forward-strand bases lo:hi of the chromosome (N where the prepared genome has a gap)."""
    buf = np.full(hi - lo, 4, dtype=np.uint8)
    for s in contigs:
        a, b = max(lo, s["start"]), min(hi, s["end"])
        if a < b:
            buf[a - lo:b - lo] = bases[s["offset"] + a - s["start"]:s["offset"] + b - s["start"]]
    return bytes(np.frombuffer(b"ACGTN", dtype=np.uint8)[buf])


def oriented(seq: bytes, strand: str) -> bytes:
    return seq if strand == "+" else seq.translate(COMP)[::-1]


from evo2 import Evo2  # noqa: E402

model = Evo2(name)
ACGT = [ord(x) for x in "ACGT"]


@torch.no_grad()
def score(seq: bytes) -> np.ndarray:
    """``[len, 4]`` float32: log-probabilities of A, C, G, T as base i given the bases before it (row 0: zeros)."""
    n = len(seq)
    out_lp = np.zeros((n, 4), dtype=np.float32)
    starts = [0] if n <= window else list(range(0, n - window + window // 2, window // 2))
    for s in starts:
        piece = seq[s:s + window]
        ids = torch.tensor(model.tokenizer.tokenize(piece.decode()), dtype=torch.int).unsqueeze(0).to("cuda:0")
        o = model(ids)                                    # (outputs, embeddings); outputs[0]: [1, len, vocab]
        logits = o[0][0] if isinstance(o, tuple) else o
        lp = torch.log_softmax(logits[0].float(), -1)[:, ACGT].cpu().numpy()       # row j predicts base j + 1
        keep_from = 0 if s == 0 else window // 2                                    # later windows: their second half
        a, b = s + max(keep_from, 1), min(s + len(piece), n)
        out_lp[a:b] = lp[a - 1 - s:b - 1 - s]
    return out_lp


res = dict(model=name, chrom=chrom, window=window, genes=[], modes={})
store: dict = {"spliced": [], "genomic": [], "true": []}
for mode in ("spliced", "genomic"):
    t0, n_scored, torch_peak = time.time(), 0, 0
    torch.cuda.reset_peak_memory_stats()
    for g in genes:
        if time.time() - t0 > budget:
            break
        ex, strand = g["ex"], g["strand"]
        if mode == "spliced":
            seq = oriented(b"".join(fetch(a, b) for a, b in ex), strand)
            lp = score(seq)
            cds_lp = lp
        else:
            lo, hi = ex[0][0] - (2000 if strand == "+" else 0), ex[-1][1] + (2000 if strand == "-" else 0)
            seq = oriented(fetch(lo, hi), strand)
            lp = score(seq)
            idx = np.concatenate([np.arange(a, b) for a, b in ex])
            idx = (idx - lo) if strand == "+" else (hi - 1 - idx)[::-1]
            cds_lp = lp[np.sort(idx)]
        n_scored += len(seq)
        store[mode].append(cds_lp)
        if mode == "spliced":
            cds = oriented(b"".join(fetch(a, b) for a, b in ex), strand)
            store["true"].append(np.frombuffer(cds.translate(bytes.maketrans(b"ACGTN", bytes([0, 1, 2, 3, 4]))), dtype=np.uint8))
            res["genes"].append(dict(gene=g["gene"], strand=strand, cds=g["cds"], span=ex[-1][1] - ex[0][0], exons=len(ex)))
        print(f"{mode} {g['gene']} {len(seq):,} bases, {time.time() - t0:.0f} s so far", flush=True)
    dt = time.time() - t0
    res["modes"][mode] = dict(genes=len(store[mode]), bases=n_scored, seconds=round(dt, 1), bases_per_second=round(n_scored / dt, 1),
                              peak_gib=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2))
k = min(len(store["spliced"]), len(store["genomic"]))
if k:
    sp, ge, tr = (np.concatenate(store[m][:k]) for m in ("spliced", "genomic", "true"))
    ok = (tr < 4) & (np.abs(sp).sum(1) > 0) & (np.abs(ge).sum(1) > 0)
    sp, ge, tr = sp[ok], ge[ok], tr[ok]
    row = np.arange(len(tr))
    res["compare"] = dict(coding_bases=int(len(tr)), genes=k,
                          mean_loglik_true_spliced=float(sp[row, tr].mean()), mean_loglik_true_genomic=float(ge[row, tr].mean()),
                          argmax_agree=float((sp.argmax(1) == ge.argmax(1)).mean()),
                          argmax_is_true_spliced=float((sp.argmax(1) == tr).mean()), argmax_is_true_genomic=float((ge.argmax(1) == tr).mean()),
                          pearson_loglik_true=float(np.corrcoef(sp[row, tr], ge[row, tr])[0, 1]),
                          mean_kl_genomic_to_spliced=float((np.exp(ge) * (ge - sp)).sum(1).mean()))
out.write_text(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if k != "genes"}, indent=1))
