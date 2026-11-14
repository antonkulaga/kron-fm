"""ESM-2 embeddings of the proteins in a FASTA, per residue and mean-pooled (last layer, mean over residues without
BOS/EOS, one unmasked pass, fp16), written beside each other:

  PREFIX.index.parquet   id, length, offset (row of its first residue in PREFIX.residues.npy), truncated
  PREFIX.mean.npy        [n, d] float16, row i = protein i of the index
  PREFIX.residues.npy    [sum of lengths, d] float16; protein i is rows offset .. offset + length (np.load(mmap_mode="r"))

usage: uv run python scripts/kron_dna/mito/esm_embed.py IN.faa PREFIX [MODEL] [TOKENS_PER_BATCH]

A stop (*) ends a sequence, letters ESM-2 lacks become X, and sequences longer than 1,022 residues are cut (flagged in
`truncated`). Needs the `baselines` group (fair-esm) and a GPU.
"""
import sys

import esm
import numpy as np
import polars as pl
import torch

fasta, prefix = sys.argv[1], sys.argv[2]
model_name = sys.argv[3] if len(sys.argv) > 3 else "esm2_t33_650M_UR50D"
budget = int(sys.argv[4]) if len(sys.argv) > 4 else 8192
net, alphabet = getattr(esm.pretrained, model_name)()
net = net.eval().cuda().half()
conv = alphabet.get_batch_converter()
last = net.num_layers

ids, seqs, trunc = [], [], []
name, buf = None, []
for line in list(open(fasta)) + [">"]:
    line = line.strip()
    if line.startswith(">"):
        if name is not None:
            s = "".join(buf).upper().split("*")[0]
            s = "".join(c if c in alphabet.tok_to_idx else "X" for c in s)
            ids.append(name); trunc.append(len(s) > 1022); seqs.append(s[:1022])
        name, buf = line[1:].split()[0] if len(line) > 1 else None, []
    else:
        buf.append(line)
lengths = np.array([len(s) for s in seqs])
offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])
res = np.lib.format.open_memmap(f"{prefix}.residues.npy", mode="w+", dtype=np.float16, shape=(int(lengths.sum()), net.embed_dim))
mean = np.zeros((len(seqs), net.embed_dim), dtype=np.float16)
order = np.argsort(-lengths)
i = 0
while i < len(order):
    j = i
    while j < len(order) and (j - i + 1) * (lengths[order[i]] + 2) <= budget:
        j += 1
    j = max(j, i + 1)
    batch = order[i:j]
    _, _, toks = conv([(ids[k], seqs[k]) for k in batch])
    with torch.no_grad():
        rep = net(toks.cuda(), repr_layers=[last])["representations"][last].float().cpu().numpy()
    for b, k in enumerate(batch):
        L = lengths[k]
        e = rep[b, 1:L + 1]
        res[offsets[k]:offsets[k] + L] = e.astype(np.float16)
        mean[k] = e.mean(0).astype(np.float16)
    i = j
res.flush()
np.save(f"{prefix}.mean.npy", mean)
pl.DataFrame({"id": ids, "length": lengths, "offset": offsets, "truncated": trunc}).write_parquet(f"{prefix}.index.parquet")
print(f"{len(ids):,} proteins, {int(lengths.sum()):,} residues, {sum(trunc)} truncated -> {prefix}.*", flush=True)
