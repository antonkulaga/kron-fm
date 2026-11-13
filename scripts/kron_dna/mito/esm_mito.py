"""ESM-2 over the 13 mitochondrial proteins of MitoAge's alignments: for each species and protein, the log-probability
ESM-2 gives the observed residue at each position (one unmasked pass), placed back on the alignment's columns, and the
mean last-layer embedding.

usage: uv run python scripts/kron_dna/mito/esm_mito.py ALIGNMENTS.parquet OUT_DIR [MODEL] [BATCH]

Writes OUT_DIR/<gene>.npz: slug, accession, logp [n, columns] float16 (NaN at gaps), emb [n, d] float16. Proteins are
read as MitoAge translated them (each species' genetic code); a stop (*) ends the sequence, a letter ESM-2 lacks (J) becomes X; longer than
1,022 residues is cut. Needs the `baselines` group (fair-esm) and a GPU.
"""
import sys
from pathlib import Path

import esm
import numpy as np
import polars as pl
import torch

aln, out = Path(sys.argv[1]), Path(sys.argv[2])
model_name = sys.argv[3] if len(sys.argv) > 3 else "esm2_t33_650M_UR50D"
batch = int(sys.argv[4]) if len(sys.argv) > 4 else 16
out.mkdir(parents=True, exist_ok=True)
net, alphabet = getattr(esm.pretrained, model_name)()
net = net.eval().cuda().half()
conv = alphabet.get_batch_converter()
last = net.num_layers
prot = pl.read_parquet(aln).filter(pl.col("kind") == "protein")
for (gene,), g in sorted(prot.group_by("gene"), key=lambda x: x[0]):
    if (out / f"{gene}.npz").exists():
        continue
    rows = g.sort("slug").to_dicts()
    width = len(rows[0]["aligned"])
    logp = np.full((len(rows), width), np.nan, dtype=np.float16)
    emb = np.zeros((len(rows), net.embed_dim), dtype=np.float16)
    for b0 in range(0, len(rows), batch):
        chunk = rows[b0:b0 + batch]
        cols, seqs = [], []
        for r in chunk:
            c = [i for i, ch in enumerate(r["aligned"]) if ch != "-"]
            s = "".join(r["aligned"][i] for i in c)
            cut = s.find("*")
            if cut >= 0:
                s, c = s[:cut], c[:cut]
            s, c = s[:1022], c[:1022]
            s = "".join(ch if ch in alphabet.tok_to_idx else "X" for ch in s)   # J and other codes ESM lacks
            seqs.append((r["slug"], s))
            cols.append(c)
        _, _, toks = conv(seqs)
        toks = toks.cuda()
        with torch.no_grad():
            res = net(toks, repr_layers=[last])
        lp = torch.log_softmax(res["logits"].float(), -1).gather(-1, toks.unsqueeze(-1))[..., 0].cpu().numpy()
        rep = res["representations"][last].float().cpu().numpy()
        for j, (c, (_, s)) in enumerate(zip(cols, seqs)):
            n = len(s)
            logp[b0 + j, c] = lp[j, 1:n + 1]
            emb[b0 + j] = rep[j, 1:n + 1].mean(0)
    np.savez(out / f"{gene}.npz", slug=np.array([r["slug"] for r in rows]),
             accession=np.array([r["accession"] for r in rows]), logp=logp, emb=emb)
    print(f"{gene}: {len(rows):,} species, {width} columns", flush=True)
print("done", flush=True)
