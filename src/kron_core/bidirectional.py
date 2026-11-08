"""Bidirectional KronLSTM blocks over packed documents, and a document-masked attention block.

A masked-language model needs every position to read both sides. Bio-xLSTM gets both directions from
one pass because an mLSTM's gates depend only on the current step, but the delta rule's erase term
makes the state depend on the order of the writes, so the backward direction is a second scan over
the flipped sequence (as in Caduceus's BiMamba). The two outputs are summed. With
``tie_directions`` both scans share one mixer, as Caduceus found best at a given depth; otherwise each
direction has its own.

Every block reads a ``[B, T, D]`` batch of packed rows together with ``cu_seqlens`` of the flattened
``B * T`` tokens (:func:`kron_core.segments.cu_seqlens_from_doc_ids`), so the recurrence restarts at
every document in both directions.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from kron_core import blocks
from kron_core.blocks import SwiGLU
from kron_core.segments import reverse_cu_seqlens


class BiKronLSTMBlock(nn.Module):
    """Pre-norm residual block: RMSNorm -> forward + backward KronLSTM scans, RMSNorm -> SwiGLU."""

    def __init__(self, d_model: int, n_heads: int, ff_units: int, dropout: float, layer_idx: int,
                 tie_directions: bool = False) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} must be divisible by n_heads {n_heads}")
        head_dim = d_model // n_heads
        if not 16 <= head_dim <= 256:
            raise ValueError(f"KronLSTM's chunk kernels need a head dimension in [16, 256], got {head_dim}")
        self.layer_idx = layer_idx
        self.tie_directions = bool(tie_directions)
        self.mixer_norm = nn.RMSNorm(d_model, eps=1e-5)
        # looked up through the module so a test can swap in a CPU stand-in
        self.forward_mixer = blocks.kronlstm_mixer(d_model, n_heads, layer_idx)
        self.backward_mixer = None if self.tie_directions else blocks.kronlstm_mixer(d_model, n_heads, layer_idx)
        self.mlp_norm = nn.RMSNorm(d_model, eps=1e-5)
        self.mlp = SwiGLU(d_model, ff_units, dropout)
        self.drop = nn.Dropout(dropout)

    def mix(self, h: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        """``[B, T, D]`` normed input -> the summed forward and backward scans, ``[B, T, D]``."""
        batch, steps, width = h.shape
        flat = h.reshape(1, batch * steps, width)
        fwd, _, _ = self.forward_mixer(flat, cu_seqlens=cu_seqlens)
        backward = self.forward_mixer if self.backward_mixer is None else self.backward_mixer
        bwd, _, _ = backward(flat.flip(1), cu_seqlens=reverse_cu_seqlens(cu_seqlens))
        return (fwd + bwd.flip(1)).reshape(batch, steps, width)

    def forward(self, h: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        h = h + self.drop(self.mix(self.mixer_norm(h), cu_seqlens))
        return h + self.drop(self.mlp(self.mlp_norm(h)))


class SegmentAttentionBlock(nn.Module):
    """Pre-norm residual block: RMSNorm -> bidirectional self-attention within each document, RMSNorm -> SwiGLU.

    No positional encoding: the KronLSTM blocks before it already carry position, as in the
    recurrent-attention hybrids (Jamba, HybriDNA). The attention mask comes from
    :func:`kron_core.segments.same_segment_mask`, so packed documents never read each other.
    """

    def __init__(self, d_model: int, n_heads: int, ff_units: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} must be divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.attn_norm = nn.RMSNorm(d_model, eps=1e-5)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = float(dropout)
        self.mlp_norm = nn.RMSNorm(d_model, eps=1e-5)
        self.mlp = SwiGLU(d_model, ff_units, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, segment_mask: torch.Tensor) -> torch.Tensor:
        batch, steps, width = h.shape
        q, k, v = self.qkv(self.attn_norm(h)).view(batch, steps, 3, self.n_heads, -1).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=segment_mask,
                                           dropout_p=self.attn_dropout if self.training else 0.0)
        h = h + self.drop(self.out(o.transpose(1, 2).reshape(batch, steps, width)))
        return h + self.drop(self.mlp(self.mlp_norm(h)))
