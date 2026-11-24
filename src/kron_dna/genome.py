"""KronGenome: a base-resolution genome model with a KronLSTM core over binned sequence (the multi-track pilot).

The shape follows AlphaGenome's U-Net (Avsec et al., Nature 2026): convolutions at single-base resolution halve the
length ``n_down`` times (128-bp bins for 7), a bidirectional KronLSTM core runs over the bins, and a decoder brings
the representation back to single bases with skip connections from the encoder. The core's cost is linear in length,
so a window can be a whole gap-free contig: segments (``cu_seqlens``) keep the recurrence from crossing a gap.

Stage 1 trains masked-base prediction (no tracks needed); stage 2 adds track heads on the same representations.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from kron_core.bidirectional import BiKronLSTMBlock

#: Input channels: A, C, G, T, N (unknown or gap), MASK.
BASES = "ACGTN"
N_IN = 6
MASK_CHANNEL = 5

#: Context ids (rows of ``KronGenome.context``), fixed here so that runs started at different times agree on what
#: an id means. A row is added to the core's input bins; every row starts at zero. A checkpoint with fewer rows loads
#: into a larger table through :func:`expand_state_dict` (the new rows start at zero, or at the nuclear row).
#:
#: ===========  ==========================================================================================
#: 0            nuclear reference sequence (every nuclear window today)
#: 1 - 33       organelle genomes: the id IS the NCBI genetic-code table (2 vertebrate mito, 5 invertebrate
#:              mito, 11 plastid ...); unused table numbers stay reserved for their table
#: 34 - 63      reserved: further sequence classes (plastid vs mitochondrion split, viral, synthetic, ...)
#: 64 - 191     species (128 slots); the reference species keeps 0, a species id is ``64 + index`` in a
#:              species list that a run must store beside its checkpoint
#: 192 - 223    assemblies / haplotypes of one species (e.g. T2T-CHM13 vs hg38, hap1 vs hap2)
#: 224 - 255    cell states / conditions (tissue, age bin, treatment)
#: ===========  ==========================================================================================
#:
#: A window may carry several ids at once (``context`` of shape ``[B, K]``; their rows are summed and ids < 0 are
#: ignored), e.g. ``[64 + k, 192 + a]`` for species k on assembly a. ``n_contexts = 256`` provisions every range.
CONTEXT_RANGES = {"nuclear": (0, 1), "organelle": (1, 34), "sequence_class": (34, 64), "species": (64, 192),
                  "assembly": (192, 224), "cell_state": (224, 256)}
N_CONTEXTS_ORGANELLE = 34      # what --mito needs: ids 0..33
N_CONTEXTS_ALL = 256


def context_id(kind: str, index: int = 0) -> int:
    """The context id of the ``index``-th slot of range ``kind`` (see ``CONTEXT_RANGES``); for "organelle" the index
    is the genetic-code table itself (1..33)."""
    lo, hi = CONTEXT_RANGES[kind]
    cid = index if kind == "organelle" else lo + index
    if not lo <= cid < hi:
        raise ValueError(f"context {kind!r} index {index} falls outside ids {lo}..{hi - 1}")
    return cid


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
                                # MLP heads (see ConvBlock2); "v1" checkpoints load unchanged; "v2l": v2 with
                                # NTv3's lighter conv layout (one conv per decoder step, pool right after the stem;
                                # see KronGenome._init_v2l_convs)
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
    jepa_pred_kind: str = "mlp"  # the JEPA predictor: "mlp" (per bin, as before) or "context" (a mask embedding
                                # at hidden bins, then bidirectional KronLSTM layers over the core states and a residual
                                # readout that starts as the identity, so it begins as the masked core: JepaContextPredictor)
    jepa_pred_layers: int = 2   # context: KronLSTM layers of the predictor
    n_contexts: int = 0         # v2: learned context embeddings added to the core's input bins: 0 nuclear, else
                                # the organelle's NCBI genetic code table (2 vertebrate mito, 5 invertebrate, ...)
    n_ntv3: int = 0             # quantitative per-base assay tracks (NTv3 bigwigs): one output per track, predicted
                                # in the same log1p units the targets are stored in, beside the masked-base head
    n_in_extra: int = 0         # reserved per-base input channels after the 6 base channels (repeat mask, variant
                                # dosage, methylation, ...): zero-initialised stem weights, so absent = zero = today
    xheads: list = field(default_factory=list)   # further heads, each its own module: [name, outputs] or
                                # [name, outputs, bin] in registration order (v2/v2l). bin 1 (default) predicts per
                                # base; bin k (a power of two up to 2**n_down) predicts per k bases from the mean of
                                # the decoder states over them, so memory falls k-fold and thousands of outputs fit.
                                # A head added to a trained model is new parameters only; the inherited heads keep
                                # their keys, shapes and outputs
    xhead_hidden: int = 0       # hidden width of the further heads (0: as the other MLP heads), their memory knob
    xhead_kind: str = "mlp"     # the further heads' kind: "mlp" (MLPHead, per position) or "kron" (KronHead: a
                                # per-base readout plus bidirectional KronLSTM layers over xhead_bin-base means)
    xhead_layers: int = 2       # kron: KronLSTM layers per head
    xhead_bin: int = 32         # kron: bases per step of the heads' KronLSTM
    xhead_width: int = 128      # kron: the heads' KronLSTM width

    def to_dict(self) -> dict:
        return asdict(self)


class JepaContextPredictor(nn.Module):
    """A JEPA predictor that reads context (genome-train --jepa-pred context): a learned mask embedding is added at the
    hidden bins, ``layers`` bidirectional KronLSTM blocks run over the core states of each packed row, and a residual
    readout zero-initialised, so at the start it returns the core states as they are (the masked core, the
    baseline it must beat)."""

    def __init__(self, c: "KronGenomeConfig") -> None:
        super().__init__()
        self.mask = nn.Parameter(torch.zeros(c.d_model))
        self.layers = nn.ModuleList(BiKronLSTMBlock(c.d_model, c.n_heads, c.ff_units, c.dropout, c.n_layers + i)
                                    for i in range(c.jepa_pred_layers))
        self.out_norm = nn.RMSNorm(c.d_model, eps=1e-5)
        self.out = nn.Linear(c.d_model, c.d_model)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, core: torch.Tensor, cu: torch.Tensor, hidden: torch.Tensor | None = None) -> torch.Tensor:
        h = core if hidden is None else core + hidden.unsqueeze(-1).to(core.dtype) * self.mask.to(core.dtype)
        for layer in self.layers:
            h = layer(h, cu)
        return core + self.out(self.out_norm(h))


def make_jepa_predictor(c: "KronGenomeConfig") -> nn.Module | None:
    """The config's JEPA predictor, or None without ``jepa``."""
    if not c.jepa:
        return None
    if c.jepa_pred_kind == "context":
        if c.arch == "v3" or c.chunk_ratio:
            raise ValueError("jepa_pred_kind context needs arch v1, v2 or v2l without learned chunking")
        return JepaContextPredictor(c)
    if c.jepa_pred_kind != "mlp":
        raise ValueError(f"jepa_pred_kind must be mlp or context, got {c.jepa_pred_kind!r}")
    return nn.Sequential(nn.RMSNorm(c.d_model, eps=1e-5), nn.Linear(c.d_model, 2 * c.d_model), nn.GELU(),
                         nn.Linear(2 * c.d_model, c.d_model))


def one_hot(seq: torch.Tensor) -> torch.Tensor:
    """``[B, L]`` base indices (0..4 for A, C, G, T, N; 5 for a masked base) -> ``[B, 6, L]`` float."""
    return F.one_hot(seq.long(), N_IN).transpose(1, 2).float()


def base_channels(seq: torch.Tensor) -> torch.Tensor:
    """The stem's base input: integer codes ``[B, L]`` -> :func:`one_hot`; a float ``[B, 6, L]`` of per-base channel
    weights (fractional bases, e.g. 0.5 A + 0.5 G at a heterozygous site) passes through as float32."""
    if seq.is_floating_point():
        if seq.dim() != 3 or seq.shape[1] != N_IN:
            raise ValueError(f"float base input must be [B, {N_IN}, L], got {tuple(seq.shape)}")
        return seq.float()
    if seq.dim() != 2:
        raise ValueError(f"integer base input must be [B, L] codes, got {tuple(seq.shape)}")
    return one_hot(seq)


#: Names of the stem's input channels, in order: the 6 base channels, then the ``n_in_extra`` reserved ones. A
#: reserved channel genome-train actually feeds gets a name here (its index into ``extra`` is the value); the others
#: are recorded as ``reserved<i>`` in a run's record (``genome-train`` writes the registry into every checkpoint).
BASE_INPUT_CHANNELS = ("A", "C", "G", "T", "N", "MASK")
EXTRA_INPUT_CHANNELS = {"repeat": 0}    # --repeat-channel: 1 = lower-case (soft-masked) base in the source FASTA


def input_channel_registry(n_in_extra: int, fed: tuple[str, ...] = ()) -> dict[str, int]:
    """Name -> stem input column: the base channels 0..5, then ``n_in_extra`` reserved columns, named after
    ``EXTRA_INPUT_CHANNELS`` when listed in ``fed`` and ``reserved<i>`` otherwise."""
    reg = {n: i for i, n in enumerate(BASE_INPUT_CHANNELS)}
    names = {EXTRA_INPUT_CHANNELS[f]: f for f in fed}
    for i in range(n_in_extra):
        reg[names.get(i, f"reserved{i}")] = N_IN + i
    return reg


def soft_base_channels(inputs: torch.Tensor, targets: torch.Tensor | None, rate: float,
                       generator: torch.Generator | None = None) -> torch.Tensor:
    """Training-time fractional bases (``genome-train --soft-base-rate``): integer ``inputs`` ``[B, L]`` (after
    masking) -> ``[B, 6, L]`` float32. Of the positions that show a base (A, C, G, T) and are not masked-LM targets
    (``targets == -100``; ``None``: no targets), each is mixed with probability ``rate``: 0.5 of its base and 0.5 of
    one of the other three, drawn uniformly, i.e. what :func:`haplotype_channels` gives at a heterozygous site.
    N, MASK and every target position keep their one-hot, so the loss and its targets are unchanged. Draws two
    ``[B, L]`` random tensors from ``generator`` (``None``: the global one); call it only when ``rate > 0``."""
    x = one_hot(inputs)
    free = inputs < 4
    if targets is not None:
        free = free & (targets == -100)
    pick = (torch.rand(inputs.shape, generator=generator, device=inputs.device) < rate) & free
    shift = torch.randint(1, 4, inputs.shape, generator=generator, device=inputs.device)
    other = (inputs.long().clamp(max=3) + shift) % 4                            # one of the other three bases
    mixed = 0.5 * x + 0.5 * F.one_hot(other, N_IN).transpose(1, 2).float()
    return torch.where(pick.unsqueeze(1), mixed, x)


def haplotype_channels(hap1, hap2) -> torch.Tensor:
    """Two aligned haplotypes of base codes (0..4 A, C, G, T, N; 5 MASK; numpy or torch, ``[..., L]``) ->
    ``[..., 6, L]`` float32 weights, the mean of their one-hots: 1.0 where they agree, 0.5 + 0.5 at a heterozygous
    site. Feed to ``KronGenome.forward`` in place of integer codes."""
    a, b = (torch.as_tensor(np.asarray(h) if not torch.is_tensor(h) else h).long() for h in (hap1, hap2))
    if a.shape != b.shape:
        raise ValueError(f"haplotypes differ in shape: {tuple(a.shape)} vs {tuple(b.shape)}")
    if (a.min() < 0) or (a.max() >= N_IN) or (b.min() < 0) or (b.max() >= N_IN):
        raise ValueError(f"base codes must lie in 0..{N_IN - 1}")
    return 0.5 * (F.one_hot(a, N_IN) + F.one_hot(b, N_IN)).movedim(-1, -2).float()


#: IUPAC nucleotide codes as weights over A, C, G, T (equal shares). N and anything unknown (gap, '.', '-')
#: go to the N channel, as :func:`encode_bases` does, so an all-N stretch looks to the model as it always has.
IUPAC = {"A": "A", "C": "C", "G": "G", "T": "T", "U": "T", "R": "AG", "Y": "CT", "S": "CG", "W": "AT", "K": "GT",
         "M": "AC", "B": "CGT", "D": "AGT", "H": "ACT", "V": "ACG"}
_IUPAC_TABLE = np.zeros((256, N_IN), dtype=np.float32)
_IUPAC_TABLE[:, 4] = 1.0                                       # default: N
for _k, _v in IUPAC.items():
    for _case in (_k, _k.lower()):
        _IUPAC_TABLE[ord(_case)] = 0.0
        for _base in _v:
            _IUPAC_TABLE[ord(_case), "ACGT".index(_base)] = 1.0 / len(_v)


def iupac_channels(seq: str) -> torch.Tensor:
    """A DNA string with IUPAC ambiguity codes -> ``[6, L]`` float32 base weights (R = 0.5 A + 0.5 G, B = 1/3 each
    of C, G, T, ...; N and unknown letters -> the N channel; lower case = upper case). Add a batch axis with
    ``.unsqueeze(0)`` for ``KronGenome.forward``."""
    codes = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    return torch.from_numpy(_IUPAC_TABLE[codes].T.copy())


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


class ResConv(nn.Module):
    """v2l: one pre-norm k=5 convolution with GELU and a residual (1x1-projected when the width changes); the unit
    of NTv3's stem branch and decoder steps, with KronGenome's per-position norm."""

    def __init__(self, c_in: int, c_out: int, k: int = 5) -> None:
        super().__init__()
        self.norm, self.conv = ChannelRMSNorm(c_in), nn.Conv1d(c_in, c_out, k, padding=k // 2)
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.skip(x) + self.conv(F.gelu(self.norm(x)))


class MLPHead(nn.Module):
    """v2: per-position two-layer head (norm, 1x1, GELU, 1x1), one per task, so tasks do not share one linear map."""

    def __init__(self, c_in: int, hidden: int, c_out: int) -> None:
        super().__init__()
        self.norm, self.fc1, self.fc2 = ChannelRMSNorm(c_in), nn.Conv1d(c_in, hidden, 1), nn.Conv1d(hidden, c_out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(self.norm(x))))


class KronHead(nn.Module):
    """A per-base linear readout plus ``layers`` bidirectional KronLSTM layers over ``bin``-base means of the states
    (each row its own document), whose outputs are added back at every base of their bin. Its context is the stretch
    it is given: in training the supervised span, in a split forward each card's piece."""

    def __init__(self, c: int, t: int, bin: int = 32, d: int = 128, layers: int = 2) -> None:
        super().__init__()
        self.bin, self.n_out = bin, t
        self.norm, self.local, self.down = ChannelRMSNorm(c), nn.Conv1d(c, t, 1), nn.Conv1d(c, d, 1)
        self.blocks = nn.ModuleList(BiKronLSTMBlock(d, 4, 2 * d, 0.0, i) for i in range(layers))
        self.out = nn.Conv1d(d, t, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm(x)
        p = F.avg_pool1d(self.down(z), self.bin)                       # [B, d, n]
        b, d, n = p.shape
        if n == 0:                                                     # shorter than one step: the readout alone
            return self.local(z)
        cu = torch.arange(0, b * n + 1, n, device=x.device, dtype=torch.int32)
        h = p.transpose(1, 2).reshape(1, b * n, d)
        for block in self.blocks:
            h = block(h, cu)
        h = h.reshape(b, n, d).transpose(1, 2)
        wide = torch.repeat_interleave(self.out(h), self.bin, dim=-1)
        return self.local(z) + F.pad(wide, (0, x.shape[-1] - wide.shape[-1]))


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
        if c.arch not in ("v1", "v2", "v2l"):
            raise ValueError(f"unknown arch {c.arch!r} (v1, v2, v2l, v3)")
        light = c.arch == "v2l"
        if light and (c.chunk_ratio or c.local_width):
            raise ValueError("arch v2l does not support learned chunking (chunk_ratio) or the local track "
                             "(local_width); use arch v2 for those")
        if light and c.n_down < 2:
            raise ValueError("arch v2l needs n_down >= 2 (a pool after the stem plus at least one tower block)")
        v2 = c.arch in ("v2", "v2l")
        block = ConvBlock2 if v2 else ConvBlock
        if light:
            self._init_v2l_encoder(c)
        else:            # module order kept as before, so a seed builds the same v1/v2 weights as it always did
            self.stem = self._stem_conv(widths[0])
            self.down = nn.ModuleList(block(widths[i], widths[i + 1]) for i in range(c.n_down))
        self.core = nn.ModuleList(BiKronLSTMBlock(c.d_model, c.n_heads, c.ff_units, c.dropout, i)
                                  for i in range(c.n_layers))
        self.core_norm = nn.RMSNorm(c.d_model, eps=1e-5)
        if light:
            self._init_v2l_decoder(c)
        else:
            self.up = nn.ModuleList(block(widths[i + 1], widths[i]) for i in reversed(range(c.n_down)))
        if v2:   # NTv3's decoder: upsample, block, add a 1x1-projected skip times a learned scale, a second block
            if not light:
                self.skip_proj = nn.ModuleList(nn.Conv1d(widths[i], widths[i], 1) for i in reversed(range(c.n_down)))
                self.skip_scale = nn.Parameter(torch.ones(c.n_down))
                self.up2 = nn.ModuleList(block(widths[i], widths[i]) for i in reversed(range(c.n_down)))
            hidden = c.head_hidden or 2 * widths[0]
            self.head = MLPHead(widths[0], hidden, 4)
            self.tracks = MLPHead(widths[0], hidden, c.n_tracks) if c.n_tracks else None
        else:
            self.head = nn.Conv1d(widths[0], 4, 1)
            self.tracks = nn.Conv1d(widths[0], c.n_tracks, 1) if c.n_tracks else None
        self.ntv3 = (MLPHead(widths[0], c.head_hidden or 2 * widths[0], c.n_ntv3) if v2
                     else nn.Conv1d(widths[0], c.n_ntv3, 1)) if c.n_ntv3 else None
        self.last_ntv3 = None
        self._init_xheads(c, widths[0])
        self.context = nn.Embedding(c.n_contexts, c.d_model) if c.n_contexts else None
        self.jepa_pred = make_jepa_predictor(c)
        self.jepa_hidden, self.jepa_decode_pred = None, False
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

    def _init_xheads(self, c: KronGenomeConfig, width: int) -> None:
        """The further per-base heads of ``config.xheads``. ``x_active`` (not saved) names the heads the next forward
        runs: None runs all, a set only those, so a head registered ahead of its data costs no forward."""
        rows = [(str(h[0]), int(h[1]), int(h[2]) if len(h) > 2 else 1) for h in c.xheads]
        names = [n for n, _, _ in rows]
        if len(set(names)) != len(names) or any(not n.isidentifier() for n in names):
            raise ValueError(f"xheads need distinct identifier names, got {names}")
        if c.xheads and (c.arch not in ("v2", "v2l") or c.chunk_ratio or c.local_width):
            raise ValueError("xheads need arch v2 or v2l without learned chunking or a local track")
        for n, _, b in rows:
            if b < 1 or b & (b - 1) or b > 2 ** c.n_down:
                raise ValueError(f"xhead {n}: bin {b} must be a power of two up to {2 ** c.n_down}")
        if c.xhead_kind == "mlp":
            hidden = c.xhead_hidden or c.head_hidden or 2 * width
            self.xheads = nn.ModuleDict({n: MLPHead(width, hidden, k) for n, k, _ in rows})
        elif c.xhead_kind == "kron":
            kb = c.xhead_bin
            if kb < 1 or kb & (kb - 1) or c.xhead_layers < 1 or c.xhead_width % 4 or not 64 <= c.xhead_width <= 1024:
                raise ValueError(f"kron xheads: step {kb} must be a power of two, layers {c.xhead_layers} at least 1, "
                                 f"width {c.xhead_width} a multiple of 4 in [64, 1024] (4 KronLSTM heads of 16-256)")
            # a binned head reads bin means; its KronLSTM steps over xhead_bin bases, or one bin when the bin is wider
            step = {n: max(1, kb // b) for n, _, b in rows}
            for n, _, b in rows:                      # a span is a multiple of 2**n_down: whole steps in every bin
                if b * step[n] > 2 ** c.n_down:
                    raise ValueError(f"xhead {n}: bin {b} x KronLSTM step {step[n]} exceeds {2 ** c.n_down} bases")
            self.xheads = nn.ModuleDict({n: KronHead(width, k, step[n], c.xhead_width, c.xhead_layers)
                                         for n, k, _ in rows})
        else:
            raise ValueError(f"xhead_kind must be mlp or kron, got {c.xhead_kind!r}")
        self.xhead_bin = {n: b for n, _, b in rows}
        self.x_active: set[str] | None = None
        self.checkpoint_xheads = False                # recompute each head's inside in backward (not saved)
        self.x_span: tuple[int, int] | None = None    # window bases the heads run on (not saved; a multiple of
        self.last_x: dict[str, torch.Tensor] = {}     # 2**n_down at both ends); last_x_at: the bases last_x covers
        self.last_x_at: tuple[int, int] = (0, 0)

    def run_xheads(self, x: torch.Tensor, heads: nn.ModuleDict | None = None, at: int = 0
                   ) -> tuple[dict[str, torch.Tensor], tuple[int, int]]:
        """The active further heads on decoder states ``x`` [B, C, L] that cover window bases ``at:at + L`` (``at``
        and ``L`` multiples of 2**n_down, as windows and split pieces are). Returns ``({name: [B, n / bin,
        outputs]}, (lo, hi))``: the heads run on bases ``lo:hi`` only, the part of ``x`` inside ``x_span`` (all of
        it when ``x_span`` is None), since a head is per position (or per bin) and costs memory at every base it
        runs on, while training supervises only part of a window. A binned head reads the mean state of each bin.
        ``heads``: a split replica's copies."""
        lo, hi = at, at + x.shape[-1]
        if self.x_span is not None:
            lo, hi = max(lo, int(self.x_span[0])), min(hi, int(self.x_span[1]))
            hi = max(hi, lo)
            x = x[..., lo - at:hi - at]
        out, pooled = {}, {1: x}
        for n, h in (self.xheads if heads is None else heads).items():
            if self.x_active is not None and n not in self.x_active:
                continue
            b = self.xhead_bin[n]
            if hi == lo:                                  # nothing of the span in this stretch
                out[n] = x.new_zeros(x.shape[0], 0, h.n_out if isinstance(h, KronHead) else h.fc2.out_channels)
                continue
            if b not in pooled:
                pooled[b] = F.avg_pool1d(x, b)
            if self.checkpoint_xheads and torch.is_grad_enabled():    # keep the input and output, recompute inside
                out[n] = torch.utils.checkpoint.checkpoint(h, pooled[b], use_reentrant=False).transpose(1, 2)
            else:
                out[n] = h(pooled[b]).transpose(1, 2)
        return out, (lo, hi)

    def _stem_conv(self, width: int) -> nn.Conv1d:
        """The k=15 stem over the 6 base channels plus ``n_in_extra`` reserved ones. The base part is drawn exactly as
        before (same shapes, same random numbers); the reserved input channels' weights are appended as zeros without
        touching the random stream, so a seed builds the same model whatever ``n_in_extra`` is."""
        stem = nn.Conv1d(N_IN, width, 15, padding=7)
        extra = self.config.n_in_extra
        if extra:
            w = stem.weight.detach()
            stem.weight = nn.Parameter(torch.cat([w, w.new_zeros(w.shape[0], extra, w.shape[2])], 1))
            stem.in_channels = N_IN + extra
        return stem

    def _stem_in(self, seq: torch.Tensor, extra: torch.Tensor | None) -> torch.Tensor:
        """The stem applied to bases (integer codes or float channel weights) and the optional reserved channels
        ``extra`` ``[B, n_in_extra, L]``. Without ``extra`` only the base columns of the weight are used, which is
        the same convolution as a model built without reserved channels (bit for bit), not a sum with zeros."""
        x = base_channels(seq)
        n_extra = self.config.n_in_extra
        if extra is None:
            if not n_extra:
                return self.stem(x)
            return F.conv1d(x, self.stem.weight[:, :N_IN].contiguous(), self.stem.bias, padding=self.stem.padding)
        if not n_extra:
            raise ValueError("extra input channels given, but the model has n_in_extra = 0")
        if extra.dim() != 3 or extra.shape[1] != n_extra or extra.shape[0] != x.shape[0] or extra.shape[2] != x.shape[2]:
            raise ValueError(f"extra must be [B, {n_extra}, L] = {(x.shape[0], n_extra, x.shape[2])}, "
                             f"got {tuple(extra.shape)}")
        return self.stem(torch.cat([x, extra.to(x.dtype)], 1))

    def _v2l_widths(self, c) -> list[int]:
        """v2l width at 2**i bp for i = 0..n_down: the stem's width at 1 and 2 bp, then KronGenome's linear
        schedule from stem_channels to d_model spread over the n_down - 1 tower blocks."""
        n = c.n_down - 1
        return [c.stem_channels] + [c.stem_channels + (c.d_model - c.stem_channels) * j // n for j in range(n + 1)]

    def _init_v2l_encoder(self, c) -> None:
        """v2l, following NTv3's Methods 4.1.1 with KronGenome's per-position norm and widths: a k=15 stem plus one
        residual pre-norm k=5 conv at single bases, a max-pool right after it, then n_down - 1 two-conv residual
        blocks (ConvBlock2), each followed by a max-pool. The core still sees 2**n_down-bp bins; skips are at 1, 2,
        ..., 2**(n_down - 1) bp, as in v1/v2."""
        r = self._v2l_widths(c)
        self.stem = self._stem_conv(r[0])
        self.stem_res = ResConv(r[0], r[0])
        self.down = nn.ModuleList(ConvBlock2(r[j], r[j + 1]) for j in range(1, c.n_down))

    def _init_v2l_decoder(self, c) -> None:
        """v2l decoder, one level per halving, coarsest first: ResConv at the coarse resolution (to the finer level's
        width), nearest-neighbour upsampling, times a learned scalar (initialised to one), plus the 1x1-projected
        skip, then a ResConv at the fine resolution."""
        r = self._v2l_widths(c)
        self.up = nn.ModuleList(ResConv(r[i + 1], r[i]) for i in reversed(range(c.n_down)))
        self.skip_proj = nn.ModuleList(nn.Conv1d(r[i], r[i], 1) for i in reversed(range(c.n_down)))
        self.skip_scale = nn.Parameter(torch.ones(c.n_down))
        self.up2 = nn.ModuleList(ResConv(r[i], r[i]) for i in reversed(range(c.n_down)))

    def _init_v3(self, c) -> None:
        """v3, no conv tower and no fixed bins: one-hot -> a k=15 embedding -> a light 1-bp KronLSTM (the local
        track, state restarting every ``local_span`` bases) -> an H-Net router on its states picks chunk starts at
        single-base resolution (~1 in ``chunk_ratio``) -> the full core over the chunks -> every base takes its chunk's
        state (straight-through), fused with its local state -> per-base MLP heads."""
        if not (c.local_width and c.chunk_ratio):
            raise ValueError("arch v3 needs local_width and chunk_ratio")
        w = c.local_width
        self.stem = self._stem_conv(w)
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
        self.ntv3 = MLPHead(w, hidden, c.n_ntv3) if c.n_ntv3 else None
        self.last_ntv3 = None
        self._init_xheads(c, w)
        self.context = nn.Embedding(c.n_contexts, c.d_model) if c.n_contexts else None
        if self.context is not None:
            nn.init.zeros_(self.context.weight)
        self.jepa_pred = make_jepa_predictor(c)
        self.jepa_hidden, self.jepa_decode_pred = None, False
        self.last_jepa = self.last_epi = self.last_ratio_loss = self.last_chunks = None
        self.checkpoint_core = False
        if c.n_epigenomes:
            hid = 256
            self.epi_in = nn.Sequential(nn.RMSNorm(c.d_model, eps=1e-5), nn.Linear(c.d_model, hid))
            self.epi_film = nn.Embedding(c.n_epigenomes, 2 * hid)
            nn.init.zeros_(self.epi_film.weight)
            self.epi_out = nn.Linear(hid, 1)

    def _forward_v3(self, seq, segments, context, extra=None):
        c = self.config
        b, length = seq.shape[0], seq.shape[-1]
        dev = seq.device
        if segments is None:
            segments = torch.zeros(b, length, dtype=torch.long, device=dev)
        segments = segments + torch.arange(b, device=dev).unsqueeze(1) * (segments.max() + 1)
        x = self._stem_in(seq, extra).transpose(1, 2).reshape(1, b * length, -1)       # [1, B*L, w]
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
            h = h + self.context_rows(context, b, dev).unsqueeze(1).to(h.dtype)
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
        self.last_ntv3 = self.ntv3(fused).transpose(1, 2) if self.ntv3 is not None else None
        return self.head(fused).transpose(1, 2), up, tracks

    def context_rows(self, context: torch.Tensor | None, b: int, device) -> torch.Tensor:
        """``[B, d_model]`` context vectors: ``context`` ``[B]`` ids (none = nuclear, 0), or ``[B, K]`` ids whose rows
        are summed, ids < 0 ignored (e.g. species and assembly at once; see ``CONTEXT_RANGES``)."""
        if context is None:
            return self.context(torch.zeros(b, dtype=torch.long, device=device))
        ctx = context.long()
        if ctx.dim() == 1:
            return self.context(ctx)
        if ctx.dim() != 2:
            raise ValueError(f"context must be [B] or [B, K] ids, got {tuple(ctx.shape)}")
        rows = self.context(ctx.clamp_min(0)) * (ctx >= 0).unsqueeze(-1).to(self.context.weight.dtype)
        return rows.sum(1)

    def forward(self, seq: torch.Tensor, segments: torch.Tensor | None = None, context: torch.Tensor | None = None,
                extra: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """``seq``: integer base codes ``[B, L]`` or float channel weights ``[B, 6, L]`` (see :func:`base_channels`);
        ``context``: ``[B]`` or ``[B, K]`` ids (see ``CONTEXT_RANGES``); ``extra``: ``[B, n_in_extra, L]`` reserved
        input channels (none = zeros)."""
        c = self.config
        if c.arch == "v3":
            return self._forward_v3(seq, segments, context, extra)
        b, length = seq.shape[0], seq.shape[-1]
        if length % (2 ** c.n_down):
            raise ValueError(f"length {length} is not a multiple of {2 ** c.n_down}")
        x, skips = self.encode(seq, extra)
        h = x.transpose(1, 2)                                       # [B, bins, d_model]
        if self.context is not None:                                # context: [B]; none given = nuclear (0)
            h = h + self.context_rows(context, b, seq.device).unsqueeze(1).to(h.dtype)
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
        self.last_jepa = self.predict_jepa(core, cu) if self.jepa_pred is not None and self.training else None
        self.last_epi = self.epigenome_logits(core) if self.config.n_epigenomes else None
        x = self.add_local(self.decode(self.core_to_decode(core).transpose(1, 2), skips), skips)
        tracks = self.tracks(x).transpose(1, 2) if self.tracks is not None else None   # [B, L, n_tracks]
        self.last_ntv3 = self.ntv3(x).transpose(1, 2) if self.ntv3 is not None else None
        self.last_x, self.last_x_at = self.run_xheads(x)
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
        self.last_ntv3 = self.ntv3(x).transpose(1, 2) if self.ntv3 is not None else None
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
        # restart points per row (audit 2026-11-18, TR06): global multiples of span let one recurrent segment straddle
        # two independent rows whenever length is not a multiple of span (packed rows make such lengths)
        per_row = torch.arange(0, length, span, device=x.device)
        starts = (torch.arange(b, device=x.device).unsqueeze(1) * length + per_row).reshape(-1)
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

    def encode(self, seq: torch.Tensor, extra: torch.Tensor | None = None) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """``[B, L]`` bases (or ``[B, 6, L]`` weights) -> ``[B, d_model, L / 2**n_down]`` bins and the encoder's
        skips, finest first."""
        x = self._stem_in(seq, extra)
        if self.config.arch == "v2l":              # stem branch at 1 bp, pool, then the tower (pool after each block)
            x = self.stem_res(x)
            skips = [x]
            x = F.max_pool1d(x, 2)
            for block in self.down:
                skips.append(x)
                x = F.max_pool1d(block(x), 2)
            return x, skips
        skips = []
        for block in self.down:
            skips.append(x)
            x = F.max_pool1d(block(x), 2)
        return x, skips

    def predict_jepa(self, core: torch.Tensor, cu: torch.Tensor | None = None) -> torch.Tensor:
        """The JEPA predictor on core states ``[B, bins, d]``: per bin (``mlp``), or reading the packed rows ``cu`` with
        the hidden bins ``self.jepa_hidden`` [B, bins] bool (None: no mask embedding) marked (``context``)."""
        if isinstance(self.jepa_pred, JepaContextPredictor):
            if cu is None:
                b_, n_ = core.shape[:2]
                cu = torch.arange(0, b_ * n_ + 1, n_, device=core.device, dtype=torch.int32)
            return self.jepa_pred(core, cu, self.jepa_hidden)
        return self.jepa_pred(core)

    def core_to_decode(self, core: torch.Tensor) -> torch.Tensor:
        """The core states the decoder reads: ``core``, or with ``jepa_decode_pred`` (genome-train --jepa-decode-pred,
        training only) the predictor's states at the hidden bins, so the decoder and the heads train on them."""
        if not (self.training and self.jepa_decode_pred and self.last_jepa is not None and self.jepa_hidden is not None):
            return core
        return torch.where(self.jepa_hidden.unsqueeze(-1), self.last_jepa.to(core.dtype), core)

    def run_core(self, h: torch.Tensor, cu: torch.Tensor, checkpoint: bool = False) -> torch.Tensor:
        """The KronLSTM core over ``[1, N, d_model]`` packed bins; ``checkpoint`` recomputes each layer in backward."""
        for layer in self.core:
            h = torch.utils.checkpoint.checkpoint(layer, h, cu, use_reentrant=False) if checkpoint else layer(h, cu)
        return self.core_norm(h)

    def decode(self, x: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        """``[B, d_model, bins]`` -> ``[B, stem_channels, L]`` through the decoder and the encoder's skips."""
        if self.config.arch == "v2l":              # conv at the coarse level, upsample x scale + skip, conv at fine
            for i, (block, skip) in enumerate(zip(self.up, reversed(skips))):
                x = torch.repeat_interleave(block(x), 2, dim=-1)
                x = self.up2[i](self.skip_scale[i] * x + self.skip_proj[i](skip))
            return x
        if self.config.arch == "v2":
            for i, (block, skip) in enumerate(zip(self.up, reversed(skips))):
                x = block(F.interpolate(x, scale_factor=2, mode="nearest"))
                x = self.up2[i](x + self.skip_scale[i] * self.skip_proj[i](skip))
            return x
        for block, skip in zip(self.up, reversed(skips)):
            x = block(F.interpolate(x, scale_factor=2, mode="nearest")) + skip
        return x

    def forward_contig(self, seq: torch.Tensor, train_at: list[int], chunk: int = 1 << 20, halo: int = 4096,
                       context: int | list[int] | None = None, extra: torch.Tensor | None = None
                       ) -> list[tuple[torch.Tensor, torch.Tensor | None]]:
        """A whole contig in one recurrence. ``seq`` is ``[L]`` integer codes or ``[6, L]`` float base weights (L a
        multiple of ``2**n_down``); ``extra`` the optional ``[n_in_extra, L]`` reserved channels; ``context`` one id or
        a list of ids whose rows are summed. The encoder runs
        chunk by chunk (each with ``halo`` bases of context on both sides; gradient only for scored chunks), the core over
        every bin of the contig at once (each layer recomputed in backward), and the decoder only on the chunks
        that start at ``train_at``: returns per chunk ``(logits [1, len, 4], tracks [1, len, n_tracks] or None)``.
        Memory grows with the contig only at bin resolution. The encoder's GroupNorm takes its statistics per
        chunk, as training on windows of ``chunk`` bases does, so a chunk of the window length matches training."""
        unit = 2 ** self.config.n_down
        length = seq.shape[-1]
        if length % unit or chunk % unit or halo % unit:
            raise ValueError(f"length, chunk and halo must be multiples of {unit}")
        grad = torch.is_grad_enabled()
        cut = lambda t, lo, hi: None if t is None else t[..., lo:hi].unsqueeze(0)   # noqa: E731
        enc = lambda piece, piece_x: self.encode(piece, piece_x)[0]                # noqa: E731
        span = lambda s: (max(0, s - halo), min(length, s + chunk + halo))  # noqa: E731
        bins, scored = [], set(train_at)
        for s in range(0, length, chunk):
            lo, hi = span(s)
            piece, piece_x = cut(seq, lo, hi), cut(extra, lo, hi)
            if grad and s in scored:            # the encoder learns through the core from the scored chunks only
                x = torch.utils.checkpoint.checkpoint(enc, piece, piece_x, use_reentrant=False)
            else:
                with torch.no_grad():
                    x = enc(piece, piece_x)
            a = (s - lo) // unit
            bins.append(x[:, :, a:a + (min(s + chunk, length) - s) // unit])
        h = torch.cat(bins, 2).transpose(1, 2)
        if self.context is not None:                                # none given = nuclear (0)
            if isinstance(context, (list, tuple)):
                ids = [int(i) for i in context if int(i) >= 0]
                h = h + self.context.weight[ids].sum(0).to(h.dtype)
            else:
                h = h + self.context.weight[int(context or 0)].to(h.dtype)
        cu = torch.tensor([0, h.shape[1]], dtype=torch.int32, device=seq.device)
        core = self.run_core(h, cu, checkpoint=grad).transpose(1, 2)    # [1, d_model, bins]
        def heads(bins_, piece, piece_x, a, b):
            _, skips = self.encode(piece, piece_x)
            x = self.decode(bins_, skips)
            logits = self.head(x)[:, :, a:b].transpose(1, 2)
            if self.tracks is None:
                return (logits,)
            return logits, self.tracks(x)[:, :, a:b].transpose(1, 2)

        outs = []
        for s in train_at:                                          # one chunk's activations alive at a time
            lo, hi = span(s)
            args = (core[:, :, lo // unit:hi // unit], cut(seq, lo, hi), cut(extra, lo, hi), s - lo,
                    s - lo + min(chunk, length - s))
            out = torch.utils.checkpoint.checkpoint(heads, *args, use_reentrant=False) if grad else heads(*args)
            outs.append((out[0], out[1] if len(out) > 1 else None))
        return outs

    #: what a piece of a split window runs on its own device (see :meth:`forward_split`)
    SPLIT_LOCAL = ("stem", "stem_res", "down", "up", "up2", "skip_proj", "head", "tracks", "ntv3", "xheads")

    def split_replica(self, device, force: bool = False) -> "KronGenome":
        """A shallow copy of this model for :meth:`forward_split` whose conv path and per-base heads read
        differentiable copies (``Tensor.to``) of this model's parameters on ``device``, as DataParallel's
        ``replicate`` builds them, but one copy per parameter: gradients flow back into THIS model's parameters
        through autograd, and a parameter whose copy takes no part in the loss gets no gradient at all (as
        unsplit; a broadcast would hand it a zero). Nothing is registered: no new parameter, no new state-dict
        key. Rebuild it after every optimizer step (the copies are taken when it is built). On the model's own
        device it is the model itself, unless ``force`` (tests: real copies on one device)."""
        device = _device(device)
        if next(self.parameters()).device == device and not force:
            return self

        def rep(m: nn.Module, children) -> nn.Module:
            r = m._replicate_for_data_parallel()          # shallow: own (empty) _parameters, copied dicts
            r.__dict__["_compiled_call_impl"] = None      # a compiled call would run the original's weights
            for k, p in m._parameters.items():
                object.__setattr__(r, k, None if p is None else p.to(device, copy=force))
            for k, buf in m._buffers.items():
                r._buffers[k] = None if buf is None else buf.to(device)
            for k, child in m._modules.items():
                if child is not None and (children is None or k in children):
                    r._modules[k] = rep(child, None)
            return r

        return rep(self, set(self.SPLIT_LOCAL))

    def forward_split(self, seq: torch.Tensor, segments: torch.Tensor | None = None,
                      context: torch.Tensor | None = None, extra: torch.Tensor | None = None, *, devices,
                      halo: int | None = None, by: str | None = None, core_only: bool = False
                      ) -> tuple[list["SplitPiece"], torch.Tensor]:
        """:meth:`forward` with the window's conv path divided over ``devices`` (the first is this model's own):
        the batch is cut into ``len(devices)`` pieces, by rows when the batch divides evenly (``by="rows"``),
        else along the length (``by="length"``: contiguous stretches on 2**n_down boundaries, each encoded with
        ``halo`` bases of real sequence on both sides, default :func:`split_halo`). Piece k's encoder, decoder
        and per-base heads run on device k; only its core-input bins (cropped to its own stretch) go to the first
        device, where the core, the cell-type head and the JEPA predictor run over the whole batch as in
        :meth:`forward`, and the core states of its stretch plus halo come back. Every output equals
        :meth:`forward`'s up to floating-point round-off (v2 and v2l normalise per position only).

        Returns ``(pieces, core)``: :class:`SplitPiece` per device with its rows, its stretch and its outputs on
        its device (``logits`` [b, n, 4], ``tracks``, ``ntv3`` [b, n, n_ntv3]), and ``core`` [B, bins, d_model] on
        the first device. ``last_jepa``/``last_epi`` are set as by forward; ``last_ntv3`` is None (it is in the
        pieces; :func:`gather_split` joins them). ``core_only``: stop after the core (no decoder, no heads)."""
        c = self.config
        if c.arch not in ("v2", "v2l") or c.chunk_ratio or c.local_width:
            raise ValueError("forward_split needs arch v2 or v2l without learned chunking or a local track (v1's "
                             "GroupNorm takes statistics over the whole window, so a piece would see other ones)")
        devices = [_device(d) for d in devices]
        dev0 = next(self.parameters()).device
        if devices[0] != dev0:
            raise ValueError(f"the first split device must be the model's own ({dev0}), got {devices[0]}")
        b, length = seq.shape[0], seq.shape[-1]
        unit = 2 ** c.n_down
        halo = split_halo(c) if halo is None else int(halo)
        plan = split_plan(b, length, len(devices), unit, halo, by)
        by = plan[0][0]
        reps = {}
        enc = []                                                      # per piece: (skips, its core-input bins)
        for k, (_, rows, start, end, lo, hi) in enumerate(plan):
            d = devices[k]
            if d not in reps:
                reps[d] = self.split_replica(d)
            piece = seq[rows][..., lo:hi].to(d)
            piece_x = None if extra is None else extra[rows][..., lo:hi].to(d)
            x, skips = reps[d].encode(piece, piece_x)
            a = (start - lo) // unit
            enc.append((None if core_only else skips, x[:, :, a:a + (end - start) // unit].to(dev0)))
        x = torch.cat([e[1] for e in enc], 2 if by == "length" else 0)
        h = x.transpose(1, 2)                                         # [B, bins, d_model], as in forward
        if self.context is not None:
            h = h + self.context_rows(context, b, dev0).unsqueeze(1).to(h.dtype)
        bins = h.shape[1]
        if segments is None:
            segments = torch.arange(b, device=dev0).unsqueeze(1).expand(b, bins)
        else:
            segments = segments + torch.arange(b, device=dev0).unsqueeze(1) * (segments.max() + 1)
        flat = segments.reshape(-1)
        starts = torch.nonzero(torch.diff(flat, prepend=flat[:1] - 1)).squeeze(1)
        cu = torch.cat([starts, torch.tensor([flat.numel()], device=dev0)]).to(torch.int32)
        core = self.run_core(h.reshape(1, b * bins, -1), cu,
                             checkpoint=self.checkpoint_core and torch.is_grad_enabled()).reshape(b, bins, -1)
        self.last_jepa = self.predict_jepa(core, cu) if self.jepa_pred is not None and self.training else None
        self.last_epi = self.epigenome_logits(core) if self.config.n_epigenomes else None
        self.last_ntv3 = None
        self.last_x = {}
        if core_only:
            return [], core
        pieces = []
        core_dec = self.core_to_decode(core)
        for k, (_, rows, start, end, lo, hi) in enumerate(plan):
            d, r = devices[k], reps[devices[k]]
            skips = enc[k][0]
            y = r.decode(core_dec[rows, lo // unit:hi // unit].transpose(1, 2).to(d), skips)
            y = y[:, :, start - lo:end - lo]                          # heads are per position: crop first
            pieces.append(SplitPiece(rows=rows, start=start, end=end, device=d, logits=r.head(y).transpose(1, 2),
                                     tracks=r.tracks(y).transpose(1, 2) if r.tracks is not None else None,
                                     ntv3=r.ntv3(y).transpose(1, 2) if r.ntv3 is not None else None,
                                     **dict(zip(("x", "x_at"), self.run_xheads(y, r.xheads, at=start)))))
            enc[k] = None                                             # drop this function's hold on the skips
        return pieces, core

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def expand_state_dict(state: dict, model: KronGenome, new_contexts: str = "zero") -> tuple[dict, list[str]]:
    """A checkpoint's ``state`` made to fit ``model`` where the model has MORE reserved room, function-preserving:

    - ``stem.weight`` with fewer input channels: the missing input channels (``n_in_extra``) get zero weights;
    - ``context.weight`` with fewer rows (or none at all): the missing rows are ``new_contexts`` = "zero" (a new
      id adds nothing to the core's input, i.e. it starts as the checkpoint WITHOUT any context row) or "nuclear"
      (a copy of row 0, so a new id starts exactly as the trained nuclear model; needs row 0 in the checkpoint).

    Old ids keep their rows, so every input the checkpoint understood gives the same outputs. Anything else that
    differs in shape raises: this is not ``strict=False``, nothing is dropped silently. Returns the new state dict
    and one note per change. Load the result with ``model.load_state_dict(..., strict=True)`` (see
    :func:`load_expanded`)."""
    if new_contexts not in ("zero", "nuclear"):
        raise ValueError("new_contexts must be 'zero' or 'nuclear'")
    want = model.state_dict()
    out, notes = dict(state), []
    key = "stem.weight"
    if key in out and key in want and out[key].shape != want[key].shape:
        old, new = out[key], want[key]
        if old.dim() == 3 and old.shape[0] == new.shape[0] and old.shape[2] == new.shape[2] and old.shape[1] < new.shape[1]:
            out[key] = torch.cat([old, old.new_zeros(old.shape[0], new.shape[1] - old.shape[1], old.shape[2])], 1)
            notes.append(f"{key}: input channels {old.shape[1]} -> {new.shape[1]} (new channels zero)")
        else:
            raise ValueError(f"{key}: checkpoint {tuple(old.shape)} cannot be padded to {tuple(new.shape)}")
    key = "context.weight"
    if key in want:
        new = want[key]
        old = out.get(key)
        if old is None:
            if new_contexts == "nuclear":
                raise ValueError("new_contexts='nuclear' needs a context table in the checkpoint")
            out[key] = new.new_zeros(new.shape)
            notes.append(f"{key}: none in checkpoint -> {new.shape[0]} zero rows")
        elif old.shape != new.shape:
            if old.dim() != 2 or old.shape[1] != new.shape[1] or old.shape[0] > new.shape[0]:
                raise ValueError(f"{key}: checkpoint {tuple(old.shape)} cannot be padded to {tuple(new.shape)}")
            n_new = new.shape[0] - old.shape[0]
            fill = old[:1].expand(n_new, -1).clone() if new_contexts == "nuclear" else old.new_zeros(n_new, old.shape[1])
            out[key] = torch.cat([old, fill], 0)
            notes.append(f"{key}: rows {old.shape[0]} -> {new.shape[0]} (new rows {new_contexts})")
    # a context table the model lacks is left in place: strict loading reports it as unexpected, strict=False
    # (genome-train --init) drops it, as before
    return out, notes


def load_expanded(model: KronGenome, state: dict, strict: bool = True, new_contexts: str = "zero"):
    """``model.load_state_dict`` through :func:`expand_state_dict`: returns ``(notes, load result)``."""
    padded, notes = expand_state_dict(state, model, new_contexts=new_contexts)
    return notes, model.load_state_dict(padded, strict=strict)


def load_checkpoint(path, map_location="cpu", new_contexts: str = "zero", **overrides) -> tuple[KronGenome, dict]:
    """A KronGenome from a ``genome-train`` checkpoint (or run directory), its config updated by ``overrides`` (e.g.
    ``n_in_extra=8, n_contexts=256``) and its weights loaded through :func:`load_expanded`, strictly. Returns the
    model (on CPU, in eval mode) and the raw checkpoint dict."""
    from pathlib import Path
    path = Path(path)
    if path.is_dir():
        path = path / "checkpoint.pt"
    state = torch.load(path, map_location=map_location, weights_only=False)
    model = KronGenome(KronGenomeConfig(**{**state["config"], **overrides}))
    notes, _ = load_expanded(model, state["model"], strict=True, new_contexts=new_contexts)
    state["expand_notes"] = notes
    return model.eval(), state


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


def parse_block_len(spec: str, unit: int) -> tuple[int, int]:
    """``genome-train --jepa-block-len LO:HI`` (bases) -> ``(lo, hi)`` in bins of ``unit`` bases. Both must be
    positive multiples of ``unit`` with ``LO <= HI``; a single number means ``LO = HI``."""
    parts = spec.split(":")
    try:
        lo, hi = (int(parts[0]), int(parts[-1])) if len(parts) in (1, 2) else (0, 0)
    except ValueError:
        lo = hi = 0
    if lo <= 0 or hi < lo or lo % unit or hi % unit:
        raise ValueError(f"block lengths must be LO:HI in bases, positive multiples of {unit} with LO <= HI, "
                         f"got {spec!r}")
    return lo // unit, hi // unit


def place_blocks(n: int, share: float, lo: int, hi: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Contiguous blocks to hide in a run of ``n`` bins: ``[(start, length), ...]`` in bins, sorted, non-overlapping.

    The run's quota is ``k = round(share * n)`` bins (at most ``n - 1``, so something stays visible), and the blocks'
    lengths sum to ``k`` exactly: the realised share is within half a bin (``0.5 / n``) of ``share``. Lengths are
    drawn log-uniformly from the integers ``lo..hi`` (``P(l) ~ log((l + 1) / l)``) until the quota is filled; the
    last draw is cut to what is left, and if that leaves it shorter than ``lo`` the bins are moved between it and the
    block before it so that both stay inside ``lo..hi`` (always possible when ``hi >= 2 * lo - 1``; otherwise the two
    are merged into one block of at most ``hi + lo - 1``). So every block lies in ``lo..hi`` except that a run whose
    whole quota is below ``lo`` gets one block of ``k`` bins, and at most two blocks per run are not plain
    log-uniform draws. The blocks are put in random order and placed uniformly at random among all placements that
    leave at least one visible bin between neighbours (two blocks never touch; if the run is too full for that,
    touching is allowed)."""
    k = min(int(math.floor(share * n + 0.5)), n - 1)
    if k <= 0:
        return []
    lo_, hi_ = min(lo, k), min(hi, k)
    lens, left = [], k
    while left > 0:
        l_ = min(max(int(math.exp(rng.uniform(math.log(lo_), math.log(hi_ + 1)))), lo_), hi_, left)
        lens.append(l_)
        left -= l_
    if len(lens) > 1 and lens[-1] < lo_:                       # the cut-off last block is too short
        r_ = lens[-1]
        if lens[-2] + r_ > hi_ and lens[-2] - (lo_ - r_) >= lo_:
            lens[-2] -= lo_ - r_
            lens[-1] = lo_
        else:
            lens.pop()
            lens[-1] += r_
    m = len(lens)
    lens = [lens[i] for i in rng.permutation(m)]
    free = n - k
    gap = 1 if free >= m - 1 else 0                            # visible bins kept between neighbouring blocks
    spare = free - gap * (m - 1)
    cuts = np.sort(rng.choice(spare + m, size=m, replace=False))   # stars and bars: spare bins before each block
    out, done = [], 0
    for i in range(m):
        start = int(cuts[i]) - i + gap * i + done
        out.append((start, lens[i]))
        done += lens[i]
    return out


def block_mask(segments: np.ndarray, seq: np.ndarray | None, unit: int, share: float, lo: int, hi: int,
               rng: np.random.Generator) -> np.ndarray:
    """Hidden bins for block JEPA (``genome-train --jepa-block-share``): ``segments`` ``[B, bins]`` (0 = sequence,
    anything else = padding), ``seq`` ``[B, bins * unit]`` base codes -> ``[B, bins]`` bool. Each row is treated on
    its own, and within a row each maximal run of sequence bins gets :func:`place_blocks` with its own quota, so any
    number of rows of any length works. Padding bins are never hidden, and neither is a sequence bin that carries
    padding itself: one that ends in N and is followed by padding or by the end of the row (the last bin of a source
    whose length is not a multiple of ``unit``)."""
    segments = np.asarray(segments)
    ok = segments == 0
    if seq is not None:
        ends_n = np.asarray(seq)[:, unit - 1::unit] == 4
        before_pad = np.ones_like(ok)
        before_pad[:, :-1] = ~ok[:, 1:]
        ok = ok & ~(ends_n & before_pad)
    out = np.zeros(ok.shape, dtype=bool)
    for b in range(ok.shape[0]):
        edges = np.flatnonzero(np.diff(np.concatenate([[False], ok[b], [False]]).astype(np.int8)))
        for a, z in zip(edges[::2], edges[1::2]):               # each run [a, z) of sequence bins
            for start, length in place_blocks(int(z - a), share, lo, hi, rng):
                out[b, a + start:a + start + length] = True
    return out


def hide_blocks(inputs: torch.Tensor, targets: torch.Tensor | None, blocks: torch.Tensor, unit: int,
                extra: torch.Tensor | None = None, exclude: bool = True):
    """The online (context) input of block JEPA. ``inputs`` ``[B, L]`` integer base codes (after
    :func:`mask_bases`), ``blocks`` ``[B, L / unit]`` bool hidden bins: every base of a hidden bin becomes MASK,
    whatever it was (a base, N, a BERT replacement). ``exclude``: the masked-LM ``targets`` inside blocks become
    -100, so the head is not scored on blanked input. ``extra`` ``[B, C, L]`` per-base side channels (the repeat
    mask) are zeroed inside blocks, the value a window without that channel has. Returns ``(inputs, targets, extra,
    at)`` with ``at`` ``[B, L]`` bool, the hidden bases. Apply it before :func:`soft_base_channels`, which only mixes
    positions that still show a base, so no fractional base is drawn inside a block."""
    at = blocks.repeat_interleave(unit, dim=1)
    inputs = inputs.masked_fill(at, MASK_CHANNEL)
    if exclude and targets is not None:
        targets = targets.masked_fill(at, -100)
    if extra is not None:
        extra = extra.masked_fill(at.unsqueeze(1), 0.0)
    return inputs, targets, extra, at


def core_rank(z: torch.Tensor) -> dict:
    """Effective rank of core states ``z`` ``[N, d]`` (rows centred): ``entropy_rank`` = exp of the entropy of the
    covariance eigenvalue shares, ``participation_ratio`` = (sum ev)^2 / sum ev^2 (both in 1..d), as in
    ``v21/jepa_diag.py``."""
    z = z.double()
    z = z - z.mean(0)
    ev = torch.linalg.eigvalsh(z.T @ z / max(z.shape[0] - 1, 1)).clamp(min=0)
    if not bool(torch.isfinite(ev).all()) or float(ev.sum()) <= 0:
        return dict(entropy_rank=float("nan"), participation_ratio=float("nan"))
    p = ev / ev.sum()
    ent = float(-(p[p > 0] * p[p > 0].log()).sum())
    return dict(entropy_rank=math.exp(ent), participation_ratio=float(ev.sum() ** 2 / (ev ** 2).sum()))


def jepa_guard_metrics(clean: torch.Tensor, target: torch.Tensor, online: torch.Tensor, pred: torch.Tensor,
                       hidden: torch.Tensor) -> dict:
    """Collapse monitoring of ``genome-train --jepa-guard`` on fixed held-out windows. ``clean`` ``[W, B, d]``: the
    online model's core states of the clean windows (the states whose rank and spread are watched); ``target``: the
    JEPA target's states of the same windows; ``online``: the online model's states with 15% scattered MASK plus
    hidden blocks; ``pred``: the predictor on ``online``; ``hidden`` ``[W, B]`` bool. Returns the effective ranks
    (:func:`core_rank` of all W*B states), per-dimension std (min, mean, median), the JEPA loss (1 - cosine) on hidden
    and visible bins, the copy baseline (online state vs target, 1 - cosine) on hidden and visible bins, and top-1
    retrieval of each hidden bin's prediction among the target states of its window (chance 1/B)."""
    clean, target, online, pred = (t.float() for t in (clean, target, online, pred))
    w, b, d = clean.shape
    z = clean.reshape(-1, d)
    out = core_rank(z)
    sd = z.std(0)
    out.update(std_min=float(sd.min()), std_mean=float(sd.mean()), std_median=float(sd.median()))
    vis = ~hidden

    def mean_on(x, m):
        return float(x[m].mean()) if bool(m.any()) else float("nan")

    d_pred = 1 - F.cosine_similarity(pred, target, dim=-1)
    d_copy = 1 - F.cosine_similarity(online, target, dim=-1)
    out.update(jepa_hidden=mean_on(d_pred, hidden), jepa_visible=mean_on(d_pred, vis),
               copy_hidden=mean_on(d_copy, hidden), copy_visible=mean_on(d_copy, vis))
    sim = F.normalize(pred, dim=-1) @ F.normalize(target, dim=-1).transpose(1, 2)        # [W, B, B]
    top = sim.argmax(-1) == torch.arange(b, device=sim.device)
    out.update(top1_hidden=mean_on(top.float(), hidden), top1_chance=1.0 / b, windows=w)
    return out


def jepa_guard_new() -> dict:
    """The state of ``genome-train --jepa-guard`` (saved in the checkpoint as ``jepa_guard``): ``mult`` the factor on
    the --jepa-reg weight, ``max_rank`` the running maximum of the entropy rank and ``std_mins`` the per-evaluation
    minimum per-dimension std, both over the evaluations that did not trigger; ``hits`` consecutive triggered
    evaluations; ``evals`` evaluations seen; ``stopped`` set when the guard stopped the run."""
    return dict(mult=1.0, max_rank=None, std_mins=[], hits=0, evals=0, stopped=False)


def jepa_guard_update(state: dict, rank: float, std_min: float, act: bool, rank_frac: float = 0.5,
                      std_frac: float = 0.05, cap: float = 8.0) -> dict:
    """One evaluation of the collapse guard (updates ``state`` in place, returns what happened).

    Triggered when the entropy ``rank`` is below ``rank_frac`` x its running maximum, or ``std_min`` (the smallest
    per-dimension std of the core states) below ``std_frac`` x the running median of earlier minima, or either is not
    finite. The first evaluation only sets the references; a triggered evaluation does not enter them (a collapse
    does not lower its own bar). At the second consecutive triggered evaluation, and at each further one, ``act``
    doubles ``mult`` (at most ``cap``); a triggered evaluation with ``mult`` already at ``cap`` (and two or more in a
    row) sets ``stopped``. Without ``act`` (``--jepa-guard monitor``) the same is computed and reported, nothing
    changes but the references. ``action``: ok, triggered, raise, stop or would-raise."""
    prev_max = state["max_rank"]
    prev_med = float(np.median(state["std_mins"])) if state["std_mins"] else None
    bad = not (math.isfinite(rank) and math.isfinite(std_min))
    trig_rank = bad or (prev_max is not None and rank < rank_frac * prev_max)
    trig_std = bad or (prev_med is not None and std_min < std_frac * prev_med)
    trig = trig_rank or trig_std
    state["evals"] += 1
    if not trig:
        state["max_rank"] = rank if prev_max is None else max(prev_max, rank)
        state["std_mins"].append(std_min)
    state["hits"] = state["hits"] + 1 if trig else 0
    action = "triggered" if trig else "ok"
    if state["hits"] >= 2:
        if not act:
            action = "would-raise"
        elif state["mult"] < cap:
            state["mult"] = min(state["mult"] * 2.0, cap)
            action = "raise"
        else:
            state["stopped"] = True
            action = "stop"
    return dict(action=action, trigger_rank=bool(trig_rank), trigger_std=bool(trig_std), ref_max_rank=prev_max,
                ref_median_std_min=prev_med, mult=state["mult"], hits=state["hits"])


def poisson_multinomial_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, scale_weight: float = 0.2,
                             eps: float = 1e-7) -> torch.Tensor:
    """NTv3's (and Borzoi's, AlphaGenome's) track objective for non-negative ``pred`` and ``target`` ``[B, L, T]``
    under a 0/1 ``mask`` broadcastable to them (the supervised positions and track channels): a Poisson term on each
    (window, track)'s total over its supervised positions, divided by their number and averaged over the pairs that
    have any, plus the multinomial cross-entropy of the profile within them, averaged over supervised values;
    ``shape + scale_weight * scale`` (NTv3 Methods 4.3.6, where the weight is 0.2)."""
    mask = mask.to(pred.dtype).expand_as(pred)
    pred, target = pred * mask, target * mask
    n_pos = mask.sum(1)                                                  # [B, T]
    tot_p, tot_t = pred.sum(1), target.sum(1)
    scale = (tot_p - tot_t * torch.log(tot_p + eps)) / (n_pos + eps)
    has = n_pos > 0
    scale = scale[has].mean() if bool(has.any()) else pred.sum() * 0
    counts, want = pred + eps * mask, target + eps * mask
    p = counts / (counts.sum(1, keepdim=True) + eps)
    log_p = torch.log(torch.where(mask > 0, p, torch.ones_like(p)))     # 0 outside the mask, and no 0/0 gradient
    shape = -(want * log_p).sum() / mask.sum().clamp_min(1)
    return shape + scale_weight * scale


def _device(d) -> torch.device:
    """A device with its index ("cuda" -> "cuda:<current>"), so that devices compare as tensors' devices do."""
    d = torch.device(d)
    return torch.device("cuda", torch.cuda.current_device()) if d.type == "cuda" and d.index is None else d


@dataclass
class SplitPiece:
    """One device's share of a split window (:meth:`KronGenome.forward_split`): batch ``rows``, bases
    ``start:end`` of the window, and its outputs on ``device``."""
    rows: slice
    start: int
    end: int
    device: torch.device
    logits: torch.Tensor | None = None       # [b, end - start, 4]
    tracks: torch.Tensor | None = None       # [b, end - start, n_tracks]
    ntv3: torch.Tensor | None = None         # [b, end - start, n_ntv3]
    x: dict | None = None                    # the active further heads: {name: [b, n / bin, outputs]} over the
    x_at: tuple[int, int] | None = None      # window bases x_at[0]:x_at[1], the stretch's share of the heads' span


def split_reach(config: KronGenomeConfig) -> int:
    """An upper bound, in bases, on how far the zero padding at a piece's cut reaches into the conv path (v2,
    v2l): beyond it every core-input bin and every decoder output of the piece equals the unsplit window's. Each
    k = 5 convolution at 2**i bp adds 2 x 2**i, the k = 15 stem 7; every pooling or nearest upsampling step adds
    a cell (2**i) to stay conservative. The core's states are exact everywhere (the core sees the whole window),
    so the decoder is reached only through the encoder skips and its own padding at the cut."""
    nd, arch = config.n_down, config.arch
    if arch not in ("v2", "v2l"):
        raise ValueError(f"split_reach covers arch v2 and v2l, not {arch!r}")
    skip = []
    c = 7                                                    # the stem
    if arch == "v2":
        for i in range(nd):
            skip.append(c)
            c += 2 * (2 * 2 ** i) + 2 ** i                  # ConvBlock2 at 2**i bp, then the pool
    else:
        c += 2                                               # stem_res (k = 5 at 1 bp)
        skip.append(c)
        c += 1                                               # pool
        for j in range(1, nd):
            skip.append(c)
            c += 2 * (2 * 2 ** j) + 2 ** j
    bins_reach, d = c, 0
    for i in reversed(range(nd)):
        if arch == "v2":                                     # upsample, ConvBlock2, + skip, ConvBlock2
            d = max(d + 2 ** i + 4 * 2 ** i, skip[i]) + 4 * 2 ** i
        else:                                                # ResConv at 2**(i+1), upsample, + skip, ResConv
            d = max(d + 2 * 2 ** (i + 1) + 2 ** i, skip[i]) + 2 * 2 ** i
    return max(bins_reach, d)


def split_halo(config: KronGenomeConfig) -> int:
    """The default halo of :meth:`KronGenome.forward_split`: :func:`split_reach` rounded up to 2**n_down."""
    unit = 2 ** config.n_down
    return -(-split_reach(config) // unit) * unit


def split_plan(batch: int, length: int, n: int, unit: int, halo: int, by: str | None = None) -> list[tuple]:
    """``n`` pieces of a ``[batch, length]`` window: ``(by, rows, start, end, lo, hi)`` each, where the piece owns
    bases ``start:end`` of ``rows`` and is encoded over ``lo:hi`` (its stretch plus ``halo`` each side, clipped
    to the window). ``by``: "rows" (whole windows, no halo; needs ``batch % n == 0``), "length" (contiguous
    stretches on ``unit`` boundaries), None: rows when the batch divides evenly, else length; or "balanced"
    (--short-sources pack, whose packs may hold an odd number of short rows): rows when the batch divides
    evenly, else whichever of uneven whole rows (``ceil(batch / n)`` rows on the busiest piece, no halo) and
    length pieces (with their halo) puts fewer encoded bases on the busiest piece; ties go to length."""
    if n < 1:
        raise ValueError("split over at least one device")
    if length % unit or halo % unit or halo < 0:
        raise ValueError(f"length and halo must be non-negative multiples of {unit}")
    if by == "balanced":
        if batch % n == 0:
            by = "rows"
        else:
            bins_ = length // unit
            cuts_ = [round(i * bins_ / n) * unit for i in range(n + 1)]
            by_len = max(min(length, e + halo) - max(0, s - halo) for s, e in zip(cuts_[:-1], cuts_[1:])) * batch
            by_row = -(-batch // n) * length if batch >= n else None
            if by_row is not None and by_row < by_len:
                edges = [round(i * batch / n) for i in range(n + 1)]        # uneven: sizes differ by at most one
                return [("rows", slice(a, b), 0, length, 0, length) for a, b in zip(edges[:-1], edges[1:])]
            by = "length"
    by = by or ("rows" if batch % n == 0 else "length")
    if by == "rows":
        if batch % n:
            raise ValueError(f"a split by rows needs a batch divisible by {n}, got {batch}")
        k = batch // n
        return [("rows", slice(i * k, (i + 1) * k), 0, length, 0, length) for i in range(n)]
    if by != "length":
        raise ValueError(f"split by 'rows' or 'length', not {by!r}")
    bins = length // unit
    if bins < n:
        raise ValueError(f"a window of {length} bases has fewer than {n} bins of {unit}")
    cuts = [round(i * bins / n) * unit for i in range(n + 1)]
    return [("length", slice(0, batch), s, e, max(0, s - halo), min(length, e + halo))
            for s, e in zip(cuts[:-1], cuts[1:])]


def gather_split(pieces: list[SplitPiece], field: str, device) -> torch.Tensor | None:
    """One output of :meth:`KronGenome.forward_split` (``field``: logits, tracks or ntv3) joined into the whole
    ``[B, L, ...]`` on ``device``, as :meth:`KronGenome.forward` returns it (differentiable)."""
    parts = [(p, getattr(p, field)) for p in pieces]
    if any(t is None for _, t in parts):
        return None
    b = max(p.rows.stop for p, _ in parts)
    length = max(p.end for p, _ in parts)
    t0 = parts[0][1]
    out = t0.new_zeros((b, length) + tuple(t0.shape[2:]), device=device)
    for p, t in parts:
        out[p.rows, p.start:p.end] = t.to(device)
    return out


def poisson_multinomial_parts(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                              eps: float = 1e-7) -> dict[str, torch.Tensor]:
    """The sums :func:`poisson_multinomial_loss` needs over one stretch of positions of ``[b, n, T]`` (one piece
    of a split window): per (row, track) the supervised count, the predicted and target totals, the summed
    counts of the profile and the summed target weights; and over all values the target-weighted log count and
    the mask's sum. Sum them over the stretches (rows stay rows) and :func:`poisson_multinomial_combine` gives the
    loss of the whole window, up to round-off (log p = log count - log total, rather than log of a quotient)."""
    mask = mask.to(pred.dtype).expand_as(pred)
    pred, target = pred * mask, target * mask
    counts, want = pred + eps * mask, target + eps * mask
    return dict(n_pos=mask.sum(1), tot_p=pred.sum(1), tot_t=target.sum(1), cnt=counts.sum(1), want=want.sum(1),
                wlog=(want * torch.log(torch.where(mask > 0, counts, torch.ones_like(counts)))).sum(),
                msum=mask.sum())


def poisson_multinomial_combine(parts: list[tuple[slice, dict]], batch: int, device, scale_weight: float = 0.2,
                                eps: float = 1e-7) -> torch.Tensor:
    """:func:`poisson_multinomial_loss` of a whole window from the :func:`poisson_multinomial_parts` of its
    pieces, each given with its batch ``rows``; computed on ``device``."""
    keys = ("n_pos", "tot_p", "tot_t", "cnt", "want")
    acc = {k: None for k in keys}
    wlog = msum = 0.0
    for rows, pt in parts:
        for k in keys:
            v = pt[k].to(device)
            full = v.new_zeros((batch,) + tuple(v.shape[1:]))
            full[rows] = v
            acc[k] = full if acc[k] is None else acc[k] + full
        wlog = wlog + pt["wlog"].to(device)
        msum = msum + pt["msum"].to(device)
    n_pos, tot_p, tot_t = acc["n_pos"], acc["tot_p"], acc["tot_t"]
    scale = (tot_p - tot_t * torch.log(tot_p + eps)) / (n_pos + eps)
    has = n_pos > 0
    scale = scale[has].mean() if bool(has.any()) else tot_p.sum() * 0
    shape = -(wlog - (acc["want"] * torch.log(acc["cnt"] + eps)).sum()) / msum.clamp_min(1)
    return shape + scale_weight * scale


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
