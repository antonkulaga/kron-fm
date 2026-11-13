"""Evo 2 7B over whole mitochondrial genomes: per-base log-likelihood of each base given the bases before it, and the
mean embedding (layer blocks.28.mlp.l3) over each gene, rRNA, tRNA and the D-loop.

usage: python evo2_mito.py MITOGENOMES.fa FEATURES.parquet OUT_DIR [WINDOW]

A genome that fails (a cuFFT internal error at some lengths) is read again in 8,192-base windows, else skipped and
named in the log. A genome longer than WINDOW (default 20,000) is read in windows of WINDOW stepped by WINDOW - 2,000; each base takes
its likelihood and embedding from the window where it has the most left context. Writes OUT_DIR/ll-NNN.parquet
(accession, ll: list of float32, NaN for the first base) and OUT_DIR/emb-NNN.npz (accession, gene, kind, emb [n, 4096]
float16) per 100 genomes; resumable by part. Runs in this folder's environment (see evo2_clinvar.py).
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch

fasta, features, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
window = int(sys.argv[4]) if len(sys.argv) > 4 else 20_000
overlap = 2_000
layer = "blocks.28.mlp.l3"
out.mkdir(parents=True, exist_ok=True)

seqs, name = {}, None
for line in open(fasta):
    line = line.strip()
    if line.startswith(">"):
        name = line[1:].split()[0]
        seqs[name] = []
    elif name:
        seqs[name].append(line.upper())
seqs = {k: "".join(v) for k, v in seqs.items()}
feat = pl.read_parquet(features).filter((pl.col("end") - pl.col("start")) < 7_000)
accs = sorted(seqs)
print(f"{len(accs):,} genomes, {feat.height:,} features", flush=True)

from evo2 import Evo2  # noqa: E402

model = Evo2("evo2_7b")
torch.cuda.init()


def run(seq: str, window: int = window):
    """(ll [L] float32, emb [L, d] float16) for one sequence, in windows when it is longer than the window."""
    L = len(seq)
    ll = np.full(L, np.nan, dtype=np.float32)
    emb = None
    starts = [0] if L <= window else list(range(0, L - overlap, window - overlap))
    for k, a in enumerate(starts):
        b = min(L, a + window)
        ids = torch.tensor(model.tokenizer.tokenize(seq[a:b]), dtype=torch.int).unsqueeze(0).to("cuda:0")
        logits, embs = model(ids, return_embeddings=True, layer_names=[layer])
        logits = logits[0] if isinstance(logits, tuple) else logits
        lp = torch.log_softmax(logits[0, :-1].float(), -1).gather(-1, ids[0, 1:].long().unsqueeze(-1))[:, 0]
        e = embs[layer][0].to(torch.float16).cpu().numpy()
        if emb is None:
            emb = np.zeros((L, e.shape[1]), dtype=np.float16)
        keep = 0 if k == 0 else overlap                      # later windows give bases after their overlap
        ll[a + 1 + max(keep - 1, 0):b] = lp[max(keep - 1, 0):].cpu().numpy()
        emb[a + keep:b] = e[keep:]
    return ll, emb


per = 100
for part in range(0, len(accs), per):
    tag = f"{part // per:03d}"
    if (out / f"emb-{tag}.npz").exists():
        continue
    lls, e_acc, e_gene, e_kind, e_vec = [], [], [], [], []
    for acc in accs[part:part + per]:
        try:
            ll, emb = run(seqs[acc])
        except RuntimeError as err:                      # a cuFFT internal error at some lengths: retry in 8 kb windows
            print(f"{acc} ({len(seqs[acc]):,} bp): {err}; again in 8,192-base windows", flush=True)
            torch.cuda.empty_cache()
            try:
                ll, emb = run(seqs[acc], 8_192)
            except RuntimeError as err2:
                print(f"{acc}: skipped ({err2})", flush=True)
                torch.cuda.empty_cache()
                continue
        lls.append({"accession": acc, "ll": ll.tolist()})
        for r in feat.filter(pl.col("accession") == acc).iter_rows(named=True):
            e_acc.append(acc); e_gene.append(r["gene"]); e_kind.append(r["kind"])
            e_vec.append(emb[r["start"]:r["end"]].astype(np.float32).mean(0).astype(np.float16))
    pl.DataFrame(lls, schema={"accession": pl.Utf8, "ll": pl.List(pl.Float32)}).write_parquet(out / f"ll-{tag}.parquet")
    np.savez(out / f"emb-{tag}.npz", accession=np.array(e_acc), gene=np.array(e_gene), kind=np.array(e_kind),
             emb=np.stack(e_vec))
    print(f"part {tag}: {min(part + per, len(accs)):,} of {len(accs):,} genomes, peak "
          f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
print("done", flush=True)
