"""NVIDIA CodonFM (Encodon) zero-shot on the kron-dna ClinVar variants, with NVIDIA's own mutation recipe:
the CDS's first 2,046 codons with <CLS>/<SEP>, the variant's codon masked, log p(alt) - log p(ref) at it
(lower = more damaging). Variants past codon 2,044 are left out (NaN), as their recipe cannot place them.
Runs in the CodonFM clone's own environment (github.com/NVIDIA-Digital-Bio/CodonFM at v1.1, built in a scratch folder
outside this repository; nothing of its code is copied here), from the clone's folder, with the published Encodon
safetensors from NVIDIA's Hugging Face release. CDS_VARIANTS comes from scripts/kron_dna/export_cds_variants.py.

usage: python codonfm_clinvar.py CDS_VARIANTS.parquet MODEL.safetensors OUT.parquet NAME
"""
import sys

import numpy as np
import polars as pl
import torch

import xformers.ops as xops


def _sdpa(query, key, value, op=None, attn_bias=None, p=0.0, scale=None):
    """xformers' memory_efficient_attention, (b, s, h, d) layout and additive bias, through PyTorch's own kernel: the
    installed xformers wheel has no CUDA build, so CodonFM's attention layer would find no operator."""
    out = torch.nn.functional.scaled_dot_product_attention(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), attn_mask=attn_bias, dropout_p=p,
        scale=scale)
    return out.transpose(1, 2)


def _check():
    """The swap must equal softmax(q k^T / sqrt(d) + bias) v on random inputs, padding masked as CodonFM masks it."""
    g = torch.Generator(device="cuda").manual_seed(0)
    q, k, v = (torch.randn(2, 7, 3, 16, device="cuda", generator=g) for _ in range(3))
    keep = torch.tensor([[1] * 7, [1] * 5 + [0] * 2], device="cuda", dtype=torch.float32)
    bias = ((1.0 - keep) * torch.finfo(torch.float32).min)[:, None, None, :].repeat(1, 3, 7, 1)
    ref = torch.softmax(torch.einsum("bqhd,bkhd->bhqk", q, k) / 16 ** 0.5 + bias, dim=-1)
    ref = torch.einsum("bhqk,bkhd->bqhd", ref, v)
    assert torch.allclose(_sdpa(q, k, v, attn_bias=bias), ref, atol=1e-5), "attention swap differs"


xops.memory_efficient_attention = _sdpa
_check()

from src.data.metadata import MetadataFields as F
from src.data.preprocess.mutation_pred import mlm_process_item
from src.inference.encodon import EncodonInference

inp, weights, out, name = sys.argv[1:5]
CONTEXT = 2048
t = pl.read_parquet(inp)
inf = EncodonInference(model_path=weights, task_type="mutation_prediction")
inf.configure_model()
inf.model.to("cuda").eval()
tok = inf.tokenizer
scores = np.full(t.height, np.nan)
rows = [(i, r) for i, r in enumerate(t.iter_rows(named=True)) if r["codon_index"] + 1 < CONTEXT - 1]
print(f"{len(rows):,} of {t.height:,} variants inside the {CONTEXT - 2}-codon context", flush=True)
for start in range(0, len(rows), 16):
    chunk = rows[start:start + 16]
    items = [mlm_process_item(r["cds"], r["codon_index"], r["ref_codon"], r["alt_codon"], CONTEXT, tok) for _, r in chunk]
    batch = {k: torch.as_tensor(np.stack([it[k] for it in items])).to("cuda") for k in items[0]}
    with torch.no_grad():
        res = inf.predict_mutation(batch)
    for (i, _), lr in zip(chunk, res.likelihood_ratios):
        scores[i] = -float(lr)                       # theirs is ref - alt; ours is alt - ref
    if start % 1600 == 0:
        print(f"{start + len(chunk):,} scored", flush=True)
t.select("id", "gene", "consequence", "label", "in_training").with_columns(pl.Series(name, scores)).write_parquet(out)
print("done", out, flush=True)
