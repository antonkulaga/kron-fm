"""KronGenome: a base-resolution genome model with a KronLSTM core over binned sequence (the multi-track pilot).

The shape follows AlphaGenome's U-Net (Avsec et al., Nature 2026): convolutions at single-base resolution halve the
length ``n_down`` times (128-bp bins for 7), a bidirectional KronLSTM core runs over the bins, and a decoder brings
the representation back to single bases with skip connections from the encoder. The core's cost is linear in length,
so a window can be a whole gap-free contig: segments (``cu_seqlens``) keep the recurrence from crossing a gap.

Stage 1 trains masked-base prediction (no tracks needed); stage 2 adds track heads on the same representations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from kron_core.bidirectional import BiKronLSTMBlock

#: Input channels: A, C, G, T, N (unknown or gap), MASK.
BASES = "ACGTN"
N_IN = 6
MASK_CHANNEL = 5


@dataclass
class KronGenomeConfig:
    d_model: int = 384          # the core's width (at bin resolution)
    n_heads: int = 6
    n_layers: int = 8           # KronLSTM blocks in the core
    ff_units: int = 1152
    dropout: float = 0.1
    stem_channels: int = 96     # channels at single-base resolution
    n_down: int = 7             # 2**7 = 128-bp bins

    def to_dict(self) -> dict:
        return asdict(self)


def one_hot(seq: torch.Tensor) -> torch.Tensor:
    """``[B, L]`` base indices (0..4 for A, C, G, T, N; 5 for a masked base) -> ``[B, 6, L]`` float."""
    return F.one_hot(seq.long(), N_IN).transpose(1, 2).float()


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 5) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, c_in)
        self.conv = nn.Conv1d(c_in, c_out, k, padding=k // 2)
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.skip(x) + self.conv(F.gelu(self.norm(x)))


class KronGenome(nn.Module):
    """``[B, L]`` bases (L a multiple of ``2**n_down``) -> ``[B, L, 4]`` masked-base logits and the core states.

    ``segments`` (``[B, L // 2**n_down]`` int, optional) gives each bin a segment id; the recurrence restarts at every
    change of id, so windows that span a gap are split into independent pieces. Default: one segment per row."""

    def __init__(self, config: KronGenomeConfig) -> None:
        super().__init__()
        self.config = c = config
        widths = [c.stem_channels + (c.d_model - c.stem_channels) * i // c.n_down for i in range(c.n_down + 1)]
        self.stem = nn.Conv1d(N_IN, widths[0], 15, padding=7)
        self.down = nn.ModuleList(ConvBlock(widths[i], widths[i + 1]) for i in range(c.n_down))
        self.core = nn.ModuleList(BiKronLSTMBlock(c.d_model, c.n_heads, c.ff_units, c.dropout, i)
                                  for i in range(c.n_layers))
        self.core_norm = nn.RMSNorm(c.d_model, eps=1e-5)
        self.up = nn.ModuleList(ConvBlock(widths[i + 1], widths[i]) for i in reversed(range(c.n_down)))
        self.head = nn.Conv1d(widths[0], 4, 1)

    def forward(self, seq: torch.Tensor, segments: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.config
        b, length = seq.shape
        if length % (2 ** c.n_down):
            raise ValueError(f"length {length} is not a multiple of {2 ** c.n_down}")
        x = self.stem(one_hot(seq))
        skips = []
        for block in self.down:
            skips.append(x)
            x = F.max_pool1d(block(x), 2)
        h = x.transpose(1, 2)                                       # [B, bins, d_model]
        bins = h.shape[1]
        if segments is None:
            segments = torch.arange(b, device=seq.device).unsqueeze(1).expand(b, bins)
        else:
            segments = segments + torch.arange(b, device=seq.device).unsqueeze(1) * (segments.max() + 1)
        flat = segments.reshape(-1)
        starts = torch.nonzero(torch.diff(flat, prepend=flat[:1] - 1)).squeeze(1)
        cu = torch.cat([starts, torch.tensor([flat.numel()], device=seq.device)]).to(torch.int32)
        h = h.reshape(1, b * bins, -1)
        for layer in self.core:
            h = layer(h, cu)
        core = self.core_norm(h).reshape(b, bins, -1)
        x = core.transpose(1, 2)
        for block, skip in zip(self.up, reversed(skips)):
            x = block(F.interpolate(x, scale_factor=2, mode="nearest")) + skip
        return self.head(x).transpose(1, 2), core

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def mask_bases(seq: torch.Tensor, prob: float = 0.15, generator: torch.Generator | None = None
               ) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask ``prob`` of the known bases (A, C, G, T; never N): ``(inputs, labels)`` with labels -100 elsewhere."""
    known = seq < 4
    pick = (torch.rand(seq.shape, generator=generator, device=seq.device) < prob) & known
    inputs = seq.masked_fill(pick, MASK_CHANNEL)
    labels = seq.masked_fill(~pick, -100)
    return inputs, labels


_CODE = bytes.maketrans(b"ACGTNacgtn", b"\x00\x01\x02\x03\x04\x00\x01\x02\x03\x04")


def encode_bases(seq: str) -> np.ndarray:
    """A DNA string -> uint8 codes 0..4 (A, C, G, T, N); any other letter (IUPAC ambiguity) becomes N."""
    raw = seq.encode("ascii")
    codes = np.frombuffer(raw.translate(_CODE), dtype=np.uint8).copy()
    codes[codes > 4] = 4
    return codes


def contigs_of(codes: np.ndarray, min_gap: int = 50, min_len: int = 10_000) -> list[tuple[int, int]]:
    """Gap-free stretches of a chromosome: ``(start, end)`` between runs of at least ``min_gap`` N, at least
    ``min_len`` long. Short N runs inside a contig stay, as N bases."""
    n = (codes == 4).astype(np.int8)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], n, [0]])))
    gaps = [(s, e) for s, e in zip(edges[::2], edges[1::2]) if e - s >= min_gap]
    out, start = [], 0
    for s, e in gaps + [(len(codes), len(codes))]:
        if s - start >= min_len:
            out.append((start, s))
        start = e
    return out
