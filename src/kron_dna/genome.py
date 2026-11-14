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
    n_tracks: int = 0           # stage 2: per-base track logits beside the masked-base head (TRACKS)
    arch: str = "v1"            # "v2": per-position RMS norm, two convs per block, scaled projected decoder skips,
                                # MLP heads (see ConvBlock2); "v1" checkpoints load unchanged
    head_hidden: int = 0        # v2: hidden width of the MLP heads (0 = 2 x stem_channels)
    n_epigenomes: int = 0       # cell-type conditioning: activity per 128-bp bin in each epigenome, one shared head
                                # FiLM-modulated by a learned epigenome embedding (Bolero-style)
    chunk_ratio: float = 0.0    # learned chunking (H-Net): > 0 replaces fixed pooling past 2**n_down bp by a router
                                # that picks chunk starts, ~1 in chunk_ratio positions on average (ratio loss)
    local_width: int = 0        # two-pathway: a light KronLSTM at single bases beside the long-range core (0: off)
    local_layers: int = 2
    local_span: int = 8192      # the local track's state restarts every this many bases (bounded memory)
    chunk_length_feature: bool = False  # with chunking: tell the core each chunk's log-length (time-aware gating)
    jepa: bool = False          # a predictor of an EMA target's core states from the masked input's (JEPA-DNA)
    n_contexts: int = 0         # v2: learned context embeddings added to the core's input bins: 0 nuclear, else
                                # the organelle's NCBI genetic code table (2 vertebrate mito, 5 invertebrate, ...)

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


class ChannelRMSNorm(nn.Module):
    """RMS norm over channels at each position of ``[B, C, L]``: unlike GroupNorm(1), no statistic depends on the
    window's length or content elsewhere, so a 4 kb chunk, a 128 kb window and a 1 Mb window are normalised alike."""

    def __init__(self, c: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        return (x32 * torch.rsqrt(x32.pow(2).mean(1, keepdim=True) + self.eps)).to(x.dtype) * self.weight[:, None]


class ConvBlock2(nn.Module):
    """v2: two pre-norm k=5 convolutions with GELU and one residual (NTv3's encoder block), per-position norm."""

    def __init__(self, c_in: int, c_out: int, k: int = 5) -> None:
        super().__init__()
        self.norm1, self.conv1 = ChannelRMSNorm(c_in), nn.Conv1d(c_in, c_out, k, padding=k // 2)
        self.norm2, self.conv2 = ChannelRMSNorm(c_out), nn.Conv1d(c_out, c_out, k, padding=k // 2)
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.gelu(self.norm1(x)))
        return self.skip(x) + self.conv2(F.gelu(self.norm2(h)))


class MLPHead(nn.Module):
    """v2: per-position two-layer head (norm, 1x1, GELU, 1x1), one per task, so tasks do not share one linear map."""

    def __init__(self, c_in: int, hidden: int, c_out: int) -> None:
        super().__init__()
        self.norm, self.fc1, self.fc2 = ChannelRMSNorm(c_in), nn.Conv1d(c_in, hidden, 1), nn.Conv1d(hidden, c_out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(self.norm(x))))


class KronGenome(nn.Module):
    """``[B, L]`` bases (L a multiple of ``2**n_down``) -> ``[B, L, 4]`` masked-base logits, the core states and, with
    ``n_tracks``, ``[B, L, n_tracks]`` track logits (else None).

    ``segments`` (``[B, L // 2**n_down]`` int, optional) gives each bin a segment id; the recurrence restarts at every
    change of id, so windows that span a gap are split into independent pieces. Default: one segment per row."""

    def __init__(self, config: KronGenomeConfig) -> None:
        super().__init__()
        self.config = c = config
        widths = [c.stem_channels + (c.d_model - c.stem_channels) * i // max(c.n_down, 1) for i in range(c.n_down + 1)]
        if c.arch == "v3":
            self._init_v3(c)
            return
        if c.arch not in ("v1", "v2"):
            raise ValueError(f"unknown arch {c.arch!r}")
        v2 = c.arch == "v2"
        block = ConvBlock2 if v2 else ConvBlock
        self.stem = nn.Conv1d(N_IN, widths[0], 15, padding=7)
        self.down = nn.ModuleList(block(widths[i], widths[i + 1]) for i in range(c.n_down))
        self.core = nn.ModuleList(BiKronLSTMBlock(c.d_model, c.n_heads, c.ff_units, c.dropout, i)
                                  for i in range(c.n_layers))
        self.core_norm = nn.RMSNorm(c.d_model, eps=1e-5)
        self.up = nn.ModuleList(block(widths[i + 1], widths[i]) for i in reversed(range(c.n_down)))
        if v2:   # NTv3's decoder: upsample, block, add a 1x1-projected skip times a learned scale, a second block
            self.skip_proj = nn.ModuleList(nn.Conv1d(widths[i], widths[i], 1) for i in reversed(range(c.n_down)))
            self.skip_scale = nn.Parameter(torch.ones(c.n_down))
            self.up2 = nn.ModuleList(block(widths[i], widths[i]) for i in reversed(range(c.n_down)))
            hidden = c.head_hidden or 2 * widths[0]
            self.head = MLPHead(widths[0], hidden, 4)
            self.tracks = MLPHead(widths[0], hidden, c.n_tracks) if c.n_tracks else None
        else:
            self.head = nn.Conv1d(widths[0], 4, 1)
            self.tracks = nn.Conv1d(widths[0], c.n_tracks, 1) if c.n_tracks else None
        self.context = nn.Embedding(c.n_contexts, c.d_model) if c.n_contexts else None
        self.jepa_pred = nn.Sequential(nn.RMSNorm(c.d_model, eps=1e-5), nn.Linear(c.d_model, 2 * c.d_model), nn.GELU(),
                                       nn.Linear(2 * c.d_model, c.d_model)) if c.jepa else None
        self.last_jepa = None
        if c.n_epigenomes:
            hid = 256
            self.epi_in = nn.Sequential(nn.RMSNorm(c.d_model, eps=1e-5), nn.Linear(c.d_model, hid))
            self.epi_film = nn.Embedding(c.n_epigenomes, 2 * hid)     # per epigenome: scale and shift
            nn.init.zeros_(self.epi_film.weight)
            self.epi_out = nn.Linear(hid, 1)
        self.last_epi = None
        if c.chunk_ratio:
            if c.jepa:
                raise ValueError("learned chunking does not combine with jepa yet")
            self.route_q = nn.Linear(c.d_model, c.d_model, bias=False)
            self.route_k = nn.Linear(c.d_model, c.d_model, bias=False)
            if c.chunk_length_feature:
                self.chunk_len = nn.Linear(1, c.d_model)
                nn.init.zeros_(self.chunk_len.weight)
                nn.init.zeros_(self.chunk_len.bias)
        self.last_ratio_loss = None
        self.checkpoint_core = False
        if c.local_width:                 # local track: 1-bp recurrence, fused gated into the decoder's finest level
            self.local_in = nn.Conv1d(widths[0], c.local_width, 1)
            heads = max(1, c.local_width // 32)
            self.local_core = nn.ModuleList(BiKronLSTMBlock(c.local_width, heads, 2 * c.local_width, c.dropout, i)
                                            for i in range(c.local_layers))
            self.local_out = nn.Conv1d(c.local_width, widths[0], 1)
            self.local_gate = nn.Parameter(torch.zeros(1))      # recompute each core layer in backward (memory for longer windows)
        self.last_chunks = None
        if self.context is not None:
            nn.init.zeros_(self.context.weight)              # a new context starts as the nuclear model

    def _init_v3(self, c) -> None:
        """v3, no conv tower and no fixed bins: one-hot -> a k=15 embedding -> a light 1-bp KronLSTM (the local
        track, state restarting every ``local_span`` bases) -> an H-Net router on its states picks chunk starts at
        single-base resolution (~1 in ``chunk_ratio``) -> the full core over the chunks -> every base takes its chunk's
        state (straight-through), fused with its local state -> per-base MLP heads."""
        if not (c.local_width and c.chunk_ratio):
            raise ValueError("arch v3 needs local_width and chunk_ratio")
        w = c.local_width
        self.stem = nn.Conv1d(N_IN, w, 15, padding=7)
        heads = max(1, w // 32)
        self.local_core = nn.ModuleList(BiKronLSTMBlock(w, heads, 2 * w, c.dropout, i) for i in range(c.local_layers))
        self.to_core = nn.Linear(w, c.d_model)
        self.route_q = nn.Linear(c.d_model, c.d_model, bias=False)
        self.route_k = nn.Linear(c.d_model, c.d_model, bias=False)
        nn.init.eye_(self.route_q.weight)          # start from "neighbours alike, few boundaries": at 1 bp a random
        nn.init.eye_(self.route_k.weight)          # router would cut every 2nd base and swamp the core
        self.chunk_len = nn.Linear(1, c.d_model) if c.chunk_length_feature else None
        if self.chunk_len is not None:
            nn.init.zeros_(self.chunk_len.weight); nn.init.zeros_(self.chunk_len.bias)
        self.core = nn.ModuleList(BiKronLSTMBlock(c.d_model, c.n_heads, c.ff_units, c.dropout, i)
                                  for i in range(c.n_layers))
        self.core_norm = nn.RMSNorm(c.d_model, eps=1e-5)
        self.from_core = nn.Linear(c.d_model, w)
        self.fuse_norm = ChannelRMSNorm(w)
        hidden = c.head_hidden or 2 * w
        self.head = MLPHead(w, hidden, 4)
        self.tracks = MLPHead(w, hidden, c.n_tracks) if c.n_tracks else None
        self.context = nn.Embedding(c.n_contexts, c.d_model) if c.n_contexts else None
        if self.context is not None:
            nn.init.zeros_(self.context.weight)
        self.jepa_pred = nn.Sequential(nn.RMSNorm(c.d_model, eps=1e-5), nn.Linear(c.d_model, 2 * c.d_model), nn.GELU(),
                                       nn.Linear(2 * c.d_model, c.d_model)) if c.jepa else None
        self.last_jepa = self.last_epi = self.last_ratio_loss = self.last_chunks = None
        self.checkpoint_core = False
        if c.n_epigenomes:
            hid = 256
            self.epi_in = nn.Sequential(nn.RMSNorm(c.d_model, eps=1e-5), nn.Linear(c.d_model, hid))
            self.epi_film = nn.Embedding(c.n_epigenomes, 2 * hid)
            nn.init.zeros_(self.epi_film.weight)
            self.epi_out = nn.Linear(hid, 1)

    def _forward_v3(self, seq, segments, context):
        c = self.config
        b, length = seq.shape
        dev = seq.device
        if segments is None:
            segments = torch.zeros(b, length, dtype=torch.long, device=dev)
        segments = segments + torch.arange(b, device=dev).unsqueeze(1) * (segments.max() + 1)
        x = self.stem(one_hot(seq)).transpose(1, 2).reshape(1, b * length, -1)         # [1, B*L, w]
        flat_seg = segments.reshape(-1)
        pos = torch.arange(b * length, device=dev)
        cut = (torch.diff(flat_seg, prepend=flat_seg[:1] - 1) != 0) | (pos % length % c.local_span == 0)
        cu_l = torch.cat([torch.nonzero(cut).squeeze(1), torch.tensor([b * length], device=dev)]).to(torch.int32)
        ck = self.checkpoint_core and torch.is_grad_enabled()
        for layer in self.local_core:
            x = torch.utils.checkpoint.checkpoint(layer, x, cu_l, use_reentrant=False) if ck else layer(x, cu_l)
        local = x.reshape(b, length, -1)                                                # [B, L, w]
        h = self.to_core(local)
        if self.context is not None:
            ctx = context.long() if context is not None else torch.zeros(b, dtype=torch.long, device=dev)
            h = h + self.context(ctx).unsqueeze(1).to(h.dtype)
        q, k = self.route_q(h), self.route_k(h)
        cos = F.cosine_similarity(q[:, 1:], k[:, :-1], dim=-1)
        p = torch.cat([torch.ones(b, 1, device=dev, dtype=cos.dtype), 0.5 * (1 - cos)], 1).float().clamp(0, 1)
        seg_start = torch.diff(segments, dim=1, prepend=segments[:, :1] - 1) != 0
        bound = (p >= 0.5) | seg_start
        flat_b = bound.reshape(-1)
        chunk_id = torch.cumsum(flat_b.long(), 0) - 1
        chunks = h.reshape(b * length, -1)[flat_b]
        seg_c = flat_seg[flat_b]
        starts = torch.nonzero(torch.diff(seg_c, prepend=seg_c[:1] - 1)).squeeze(1)
        cu = torch.cat([starts, torch.tensor([seg_c.numel()], device=dev)]).to(torch.int32)
        if self.chunk_len is not None:
            lengths = torch.bincount(chunk_id, minlength=chunks.shape[0]).float()
            chunks = chunks + self.chunk_len(torch.log(lengths).unsqueeze(-1)).to(chunks.dtype)
        core_c = self.run_core(chunks.unsqueeze(0), cu, checkpoint=ck)[0]
        up = core_c[chunk_id].reshape(b, length, -1)
        conf = torch.where(bound, p, 1 - p)
        up = up * (conf + (1 - conf).detach()).unsqueeze(-1).to(up.dtype)
        if self.training:
            ratio = c.chunk_ratio
            f, gm = flat_b.float().mean(), p.mean()
            self.last_ratio_loss = ratio / (ratio - 1) * ((ratio - 1) * f * gm + (1 - f) * (1 - gm))
        self.last_chunks = float(flat_b.float().mean())
        # JEPA in v3: predict the EMA target's per-base chunk states (comparable per base, whatever the chunking)
        self.last_jepa = self.jepa_pred(up) if self.jepa_pred is not None and self.training else None
        if c.n_epigenomes:                      # cell types per 128-bp bin: the chunk states pooled over each bin
            self.last_epi = self.epigenome_logits(up.reshape(b, length // 128, 128, -1).mean(2))
        fused = self.fuse_norm((local + self.from_core(up)).transpose(1, 2))           # [B, w, L]
        tracks = self.tracks(fused).transpose(1, 2) if self.tracks is not None else None
        return self.head(fused).transpose(1, 2), up, tracks

    def forward(self, seq: torch.Tensor, segments: torch.Tensor | None = None, context: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        c = self.config
        if c.arch == "v3":
            return self._forward_v3(seq, segments, context)
        b, length = seq.shape
        if length % (2 ** c.n_down):
            raise ValueError(f"length {length} is not a multiple of {2 ** c.n_down}")
        x, skips = self.encode(seq)
        h = x.transpose(1, 2)                                       # [B, bins, d_model]
        if self.context is not None:                                # context: [B]; none given = nuclear (0)
            ctx = context.long() if context is not None else torch.zeros(b, dtype=torch.long, device=seq.device)
            h = h + self.context(ctx).unsqueeze(1).to(h.dtype)
        bins = h.shape[1]
        if self.config.chunk_ratio:
            return self._forward_chunked(h, skips, segments)
        if segments is None:
            segments = torch.arange(b, device=seq.device).unsqueeze(1).expand(b, bins)
        else:
            segments = segments + torch.arange(b, device=seq.device).unsqueeze(1) * (segments.max() + 1)
        flat = segments.reshape(-1)
        starts = torch.nonzero(torch.diff(flat, prepend=flat[:1] - 1)).squeeze(1)
        cu = torch.cat([starts, torch.tensor([flat.numel()], device=seq.device)]).to(torch.int32)
        core = self.run_core(h.reshape(1, b * bins, -1), cu,
                             checkpoint=self.checkpoint_core and torch.is_grad_enabled()).reshape(b, bins, -1)
        # inside forward, so data-parallel training syncs the predictor's gradients
        self.last_jepa = self.jepa_pred(core) if self.jepa_pred is not None and self.training else None
        self.last_epi = self.epigenome_logits(core) if self.config.n_epigenomes else None
        x = self.add_local(self.decode(core.transpose(1, 2), skips), skips)
        tracks = self.tracks(x).transpose(1, 2) if self.tracks is not None else None   # [B, L, n_tracks]
        return self.head(x).transpose(1, 2), core, tracks

    def _forward_chunked(self, h: torch.Tensor, skips: list[torch.Tensor], segments: torch.Tensor | None):
        """H-Net dynamic chunking between the encoder and the core: a boundary probability per position from the
        cosine dissimilarity of neighbouring states; positions with p >= 0.5 (and every segment start) begin a chunk;
        the core runs over the chunks (packed, so windows may differ in chunk count); every position takes its chunk's
        core state, multiplied by a straight-through confidence so the router learns; then the usual decoder."""
        b, n, d = h.shape
        dev = h.device
        if segments is None:
            segments = torch.zeros(b, n, dtype=torch.long, device=dev)
        segments = segments + torch.arange(b, device=dev).unsqueeze(1) * (segments.max() + 1)
        q, k = self.route_q(h), self.route_k(h)
        cos = F.cosine_similarity(q[:, 1:], k[:, :-1], dim=-1)
        p = torch.cat([torch.ones(b, 1, device=dev, dtype=cos.dtype), 0.5 * (1 - cos)], 1).float().clamp(0, 1)
        seg_start = torch.diff(segments, dim=1, prepend=segments[:, :1] - 1) != 0
        bound = (p >= 0.5) | seg_start
        flat_b = bound.reshape(-1)
        chunk_id = torch.cumsum(flat_b.long(), 0) - 1                       # [b*n]: each position's chunk
        chunks = h.reshape(b * n, d)[flat_b]                                # [C, d]: the chunk-start states
        seg_c = segments.reshape(-1)[flat_b]
        starts = torch.nonzero(torch.diff(seg_c, prepend=seg_c[:1] - 1)).squeeze(1)
        cu = torch.cat([starts, torch.tensor([seg_c.numel()], device=dev)]).to(torch.int32)
        if self.config.chunk_length_feature:                               # a recurrent step over a long chunk
            lengths = torch.bincount(chunk_id, minlength=chunks.shape[0]).float()   # should forget more: say how long
            chunks = chunks + self.chunk_len(torch.log(lengths).unsqueeze(-1)).to(chunks.dtype)
        core_c = self.run_core(chunks.unsqueeze(0), cu, checkpoint=self.checkpoint_core and torch.is_grad_enabled())[0]
        up = core_c[chunk_id].reshape(b, n, d)
        conf = torch.where(bound, p, 1 - p)
        up = up * (conf + (1 - conf).detach()).unsqueeze(-1).to(up.dtype)   # value 1, gradient to the router
        if self.training:                                                   # H-Net's ratio loss: mean boundary rate
            ratio = self.config.chunk_ratio                                 # near 1 / ratio, in hard and soft form
            f, gm = flat_b.float().mean(), p.mean()
            self.last_ratio_loss = ratio / (ratio - 1) * ((ratio - 1) * f * gm + (1 - f) * (1 - gm))
        self.last_chunks = float(flat_b.float().mean())
        # the cell-type head reads the core's states at the encoder's resolution (2**n_down bp per position)
        self.last_epi = self.epigenome_logits(up) if self.config.n_epigenomes else None
        x = self.add_local(self.decode(up.transpose(1, 2), skips), skips)
        tracks = self.tracks(x).transpose(1, 2) if self.tracks is not None else None
        return self.head(x).transpose(1, 2), up, tracks

    def add_local(self, x: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        """The local track: a light bidirectional KronLSTM over every base of the stem's output, its state restarting
        every ``local_span`` bases, added to the decoder's 1-bp output through a learned gate (zero at start)."""
        c = self.config
        if not c.local_width:
            return x
        stem = skips[0] if skips else x                             # [B, C0, L] at 1 bp
        b, _, length = stem.shape
        h = self.local_in(stem).transpose(1, 2).reshape(1, b * length, -1)
        span = min(c.local_span, length)
        starts = torch.arange(0, b * length, span, device=x.device)
        cu = torch.cat([starts, torch.tensor([b * length], device=x.device)]).to(torch.int32)
        for layer in self.local_core:
            h = (torch.utils.checkpoint.checkpoint(layer, h, cu, use_reentrant=False)
                 if self.checkpoint_core and torch.is_grad_enabled() else layer(h, cu))
        h = h.reshape(b, length, -1).transpose(1, 2)
        return x + self.local_gate * self.local_out(h)

    def epigenome_logits(self, core: torch.Tensor) -> torch.Tensor:
        """``[B, bins, d_model]`` core states -> ``[B, bins, n_epigenomes]`` activity logits: one head, its hidden
        layer scaled and shifted per epigenome (FiLM), so a new cell type is one more embedding row."""
        h = self.epi_in(core)                                        # [B, bins, hid]
        scale, shift = self.epi_film.weight.chunk(2, dim=-1)        # [E, hid] each, zero at start
        h = F.gelu(h.unsqueeze(2) * (1 + scale) + shift)            # [B, bins, E, hid]
        return self.epi_out(h).squeeze(-1)

    def encode(self, seq: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """``[B, L]`` bases -> ``[B, d_model, L / 2**n_down]`` bins and the encoder's skips, finest first."""
        x = self.stem(one_hot(seq))
        skips = []
        for block in self.down:
            skips.append(x)
            x = F.max_pool1d(block(x), 2)
        return x, skips

    def run_core(self, h: torch.Tensor, cu: torch.Tensor, checkpoint: bool = False) -> torch.Tensor:
        """The KronLSTM core over ``[1, N, d_model]`` packed bins; ``checkpoint`` recomputes each layer in backward."""
        for layer in self.core:
            h = torch.utils.checkpoint.checkpoint(layer, h, cu, use_reentrant=False) if checkpoint else layer(h, cu)
        return self.core_norm(h)

    def decode(self, x: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        """``[B, d_model, bins]`` -> ``[B, stem_channels, L]`` through the decoder and the encoder's skips."""
        if self.config.arch == "v2":
            for i, (block, skip) in enumerate(zip(self.up, reversed(skips))):
                x = block(F.interpolate(x, scale_factor=2, mode="nearest"))
                x = self.up2[i](x + self.skip_scale[i] * self.skip_proj[i](skip))
            return x
        for block, skip in zip(self.up, reversed(skips)):
            x = block(F.interpolate(x, scale_factor=2, mode="nearest")) + skip
        return x

    def forward_contig(self, seq: torch.Tensor, train_at: list[int], chunk: int = 1 << 20, halo: int = 4096,
                       context: int | None = None) -> list[tuple[torch.Tensor, torch.Tensor | None]]:
        """A whole contig in one recurrence. ``seq`` is ``[L]`` (L a multiple of ``2**n_down``). The encoder runs
        chunk by chunk (each with ``halo`` bases of context on both sides; gradient only for scored chunks), the core over
        every bin of the contig at once (each layer recomputed in backward), and the decoder only on the chunks
        that start at ``train_at``: returns per chunk ``(logits [1, len, 4], tracks [1, len, n_tracks] or None)``.
        Memory grows with the contig only at bin resolution. The encoder's GroupNorm takes its statistics per
        chunk, as training on windows of ``chunk`` bases does, so a chunk of the window length matches training."""
        unit = 2 ** self.config.n_down
        if seq.numel() % unit or chunk % unit or halo % unit:
            raise ValueError(f"length, chunk and halo must be multiples of {unit}")
        length = seq.numel()
        grad = torch.is_grad_enabled()
        enc = lambda piece: self.encode(piece)[0]                   # noqa: E731
        span = lambda s: (max(0, s - halo), min(length, s + chunk + halo))  # noqa: E731
        bins, scored = [], set(train_at)
        for s in range(0, length, chunk):
            lo, hi = span(s)
            piece = seq[lo:hi].unsqueeze(0)
            if grad and s in scored:            # the encoder learns through the core from the scored chunks only
                x = torch.utils.checkpoint.checkpoint(enc, piece, use_reentrant=False)
            else:
                with torch.no_grad():
                    x = enc(piece)
            a = (s - lo) // unit
            bins.append(x[:, :, a:a + (min(s + chunk, length) - s) // unit])
        h = torch.cat(bins, 2).transpose(1, 2)
        if self.context is not None:                                # none given = nuclear (0)
            h = h + self.context.weight[int(context or 0)].to(h.dtype)
        cu = torch.tensor([0, h.shape[1]], dtype=torch.int32, device=seq.device)
        core = self.run_core(h, cu, checkpoint=grad).transpose(1, 2)    # [1, d_model, bins]
        def heads(bins_, piece, a, b):
            _, skips = self.encode(piece)
            x = self.decode(bins_, skips)
            logits = self.head(x)[:, :, a:b].transpose(1, 2)
            if self.tracks is None:
                return (logits,)
            return logits, self.tracks(x)[:, :, a:b].transpose(1, 2)

        outs = []
        for s in train_at:                                          # one chunk's activations alive at a time
            lo, hi = span(s)
            args = (core[:, :, lo // unit:hi // unit], seq[lo:hi].unsqueeze(0), s - lo, s - lo + min(chunk, length - s))
            out = torch.utils.checkpoint.checkpoint(heads, *args, use_reentrant=False) if grad else heads(*args)
            outs.append((out[0], out[1] if len(out) > 1 else None))
        return outs

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def mask_bases(seq: torch.Tensor, prob: float = 0.15, generator: torch.Generator | None = None,
               bert: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask ``prob`` of the known bases (A, C, G, T; never N): ``(inputs, labels)`` with labels -100 elsewhere.
    ``bert``: of the picked bases 80 % become MASK, 10 % a random base and 10 % stay, so the model must also check
    bases it can see (BERT's and NTv3's scheme)."""
    known = seq < 4
    pick = (torch.rand(seq.shape, generator=generator, device=seq.device) < prob) & known
    labels = seq.masked_fill(~pick, -100)
    if not bert:
        return seq.masked_fill(pick, MASK_CHANNEL), labels
    u = torch.rand(seq.shape, generator=generator, device=seq.device)
    rand_base = torch.randint(0, 4, seq.shape, generator=generator, device=seq.device, dtype=seq.dtype)
    inputs = torch.where(pick & (u < 0.8), torch.full_like(seq, MASK_CHANNEL), seq)
    inputs = torch.where(pick & (u >= 0.8) & (u < 0.9), rand_base, inputs)
    return inputs, labels


def reverse_complement(codes: np.ndarray) -> np.ndarray:
    """Base codes (0..3 A, C, G, T; 4 N) of the other strand, read 5' to 3'."""
    out = codes[::-1].copy()
    known = out < 4
    out[known] = 3 - out[known]
    return out


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


#: Bits of the per-base label byte (genome-labels).
LABEL_BITS = {"promoter": 0, "enhancer": 1, "CTCF_binding_site": 2, "open_chromatin_region": 3,
              "exon": 4, "donor": 5, "acceptor": 6, "CDS": 7}


def paint(index, chrom: str, starts: np.ndarray, ends: np.ndarray, out: np.ndarray, value,
          mode: str = "or") -> int:
    """Write ``value`` over the 0-based half-open ``[starts, ends)`` of ``chrom`` into ``out``, an array laid out like
    the contig bases (``index`` rows: chrom, start, end, offset). ``value`` is one number or one per interval.
    ``mode`` "or" sets bits, "max" keeps the larger. Returns how many intervals touched a contig."""
    values = np.broadcast_to(np.asarray(value, dtype=np.uint8), (len(starts),))
    rows = index.filter(index["chrom"] == chrom).sort("start")
    cs, ce, co = (rows[c].to_numpy() for c in ("start", "end", "offset"))
    touched = 0
    for s, e, v in zip(starts, ends, values):
        i = max(0, int(np.searchsorted(cs, s, side="right")) - 1)
        while i < len(cs) and cs[i] < e:
            lo, hi = max(s, cs[i]), min(e, ce[i])
            if lo < hi:
                seg = out[co[i] + lo - cs[i]: co[i] + hi - cs[i]]
                if mode == "or":
                    seg |= v
                else:
                    np.maximum(seg, v, out=seg)
                touched += 1
            i += 1
    return touched


def splice_sites(exons):
    """GENCODE exons (polars: chrom, start, end 1-based closed, strand, transcript_id) -> two frames of 0-based
    half-open 2-bp intervals, ``(donors, acceptors)``: the intron's first two bases and last two, by strand."""
    import polars as pl
    introns = (exons.sort("transcript_id", "start")
               .with_columns(nxt=pl.col("start").shift(-1).over("transcript_id"))
               .filter(pl.col("nxt").is_not_null())
               .select("chrom", "strand", i0=pl.col("end").cast(pl.Int64), i1=pl.col("nxt").cast(pl.Int64) - 1))
    head = introns.select("chrom", "strand", start=pl.col("i0"), end=pl.col("i0") + 2)      # 0-based [end, end+2)
    tail = introns.select("chrom", "strand", start=pl.col("i1") - 2, end=pl.col("i1"))
    plus, minus = pl.col("strand") == "+", pl.col("strand") == "-"
    donors = pl.concat([head.filter(plus), tail.filter(minus)]).unique()
    acceptors = pl.concat([tail.filter(plus), head.filter(minus)]).unique()
    return donors, acceptors


#: Finer gene structure of ``genome-labels-extra`` (``labels_extra.u8``), from GENCODE basic, strand-aware.
EXTRA_BITS = {"utr5": 0, "utr3": 1, "start_codon": 2, "stop_codon": 3, "tss": 4, "lncRNA_exon": 5,
              "gene_plus": 6, "gene_minus": 7}


def gene_structure(gt):
    """GENCODE features (polars: chrom, start, end 1-based closed, type, strand, transcript_id, transcript_type) ->
    ``{EXTRA_BITS name: frame of 0-based half-open intervals (chrom, start, end)}``. A UTR before the transcript's
    first CDS base in transcription order is 5', after its last is 3'; the TSS is the transcript's first base; genes
    on each strand cover their transcripts' spans."""
    import polars as pl
    z = lambda f: f.select("chrom", start=pl.col("start").cast(pl.Int64) - 1, end=pl.col("end").cast(pl.Int64))
    cds = gt.filter(pl.col("type") == "CDS").group_by("transcript_id").agg(c0=pl.col("start").min(), c1=pl.col("end").max())
    utr = gt.filter(pl.col("type") == "UTR").join(cds, on="transcript_id", how="inner")
    before = pl.col("end") < pl.col("c0")                     # left of the CDS on the genome
    after = pl.col("start") > pl.col("c1")
    plus = pl.col("strand") == "+"
    five = utr.filter((plus & before) | (~plus & after))
    three = utr.filter((plus & after) | (~plus & before))
    tx = gt.filter(pl.col("type") == "transcript")
    tss = tx.select("chrom", start=pl.when(plus).then(pl.col("start")).otherwise(pl.col("end")),
                    end=pl.when(plus).then(pl.col("start")).otherwise(pl.col("end")))
    exons = gt.filter((pl.col("type") == "exon") & (pl.col("transcript_type") == "lncRNA"))
    return {"utr5": z(five), "utr3": z(three), "start_codon": z(gt.filter(pl.col("type") == "start_codon")),
            "stop_codon": z(gt.filter(pl.col("type") == "stop_codon")), "tss": z(tss), "lncRNA_exon": z(exons),
            "gene_plus": z(tx.filter(plus)), "gene_minus": z(tx.filter(~plus))}


def paint_ids(index, chrom: str, starts: np.ndarray, ends: np.ndarray, out: np.ndarray, ids: np.ndarray) -> int:
    """Like :func:`paint` for integer ids (any dtype of ``out``): write ``ids[i]`` over interval i, later intervals
    over earlier ones. Returns how many intervals touched a contig."""
    rows = index.filter(index["chrom"] == chrom).sort("start")
    cs, ce, co = (rows[c].to_numpy() for c in ("start", "end", "offset"))
    touched = 0
    for s, e, v in zip(starts, ends, ids):
        i = max(0, int(np.searchsorted(cs, s, side="right")) - 1)
        while i < len(cs) and cs[i] < e:
            lo, hi = max(s, cs[i]), min(e, ce[i])
            if lo < hi:
                out[co[i] + lo - cs[i]: co[i] + hi - cs[i]] = v
                touched += 1
            i += 1
    return touched


def read_genbank(lines):
    """Records of a GenBank flat file (an iterable of lines): dicts of accession (VERSION), organism, lineage (list),
    circular, transl_table (the most common /transl_table of its CDS, 1 if none) and sequence (upper case; empty for a
    CON record, which has no ORIGIN)."""
    from collections import Counter
    rec, section, tables, seq = None, None, Counter(), []
    for line in lines:
        if line.startswith("LOCUS"):
            rec = {"accession": line.split()[1], "organism": "", "lineage": [], "circular": " circular " in line}
            section, tables, seq = None, Counter(), []
        elif rec is None:
            continue
        elif line.startswith("VERSION"):
            rec["accession"] = line.split()[1]
        elif line.startswith("  ORGANISM"):
            rec["organism"], section = line[12:].strip(), "lineage"
        elif section == "lineage" and line.startswith(" " * 12):
            rec["lineage"] += [x.strip().rstrip(".") for x in line.strip().split(";") if x.strip().rstrip(".")]
        elif line.startswith("ORIGIN"):
            section = "origin"
        elif line.startswith("//"):
            rec["transl_table"] = tables.most_common(1)[0][0] if tables else 1
            rec["sequence"] = "".join(seq).upper()
            yield rec
            rec, section = None, None
        elif section == "origin":
            seq.append("".join(line.split()[1:]))
        else:
            if line[:1] != " ":
                section = None
            s = line.strip()
            if s.startswith("/transl_table="):
                tables[int(s.split("=")[1])] += 1


#: Stage-2 track outputs: the label bits in order, then the activity share.
TRACKS = [*LABEL_BITS, "activity"]
TRACK_GROUPS = {"all": TRACKS,
                "regulatory": ["promoter", "enhancer", "CTCF_binding_site", "open_chromatin_region", "activity"],
                "gene": ["exon", "donor", "acceptor", "CDS"]}


def track_mask(spec: str) -> list[bool]:
    """Which of ``TRACKS`` the loss trains, from a group name of ``TRACK_GROUPS`` or comma-separated track names.
    The head keeps all tracks (so checkpoints stay interchangeable); the others are still scored, untrained."""
    names = TRACK_GROUPS.get(spec) or [t.strip() for t in spec.split(",") if t.strip()]
    unknown = sorted(set(names) - set(TRACKS))
    if unknown or not names:
        raise ValueError(f"unknown tracks {unknown or spec!r}; groups {list(TRACK_GROUPS)}, tracks {TRACKS}")
    return [t in names for t in TRACKS]


def track_targets(labels: torch.Tensor, activity: torch.Tensor) -> torch.Tensor:
    """``[B, L]`` uint8 label bytes and activity (0..255) -> ``[B, L, len(TRACKS)]`` float targets in [0, 1]."""
    bits = torch.arange(len(LABEL_BITS), device=labels.device)
    out = ((labels.long().unsqueeze(-1) >> bits) & 1).float()
    return torch.cat([out, activity.float().unsqueeze(-1) / 255], dim=-1)


def ranking_scores(scores: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    """``(AUROC, average precision)`` of 1-D ``scores`` against 0/1 ``labels`` (ties broken by order); NaN without
    both classes. Runs on whatever device the tensors are on."""
    labels = labels.float()
    pos = labels.sum()
    neg = labels.numel() - pos
    if pos == 0 or neg == 0:
        return float("nan"), float("nan")
    order = torch.argsort(scores, descending=True)
    hit = labels[order]
    tp = torch.cumsum(hit, 0)
    ap = (tp / torch.arange(1, hit.numel() + 1, device=hit.device) * hit).sum() / pos
    auroc = ((1 - hit) * (pos - tp)).sum() / (pos * neg)        # positives ranked below each negative
    return float(1 - auroc), float(ap)
