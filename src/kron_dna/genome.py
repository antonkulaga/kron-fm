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
    """``[B, L]`` bases (L a multiple of ``2**n_down``) -> ``[B, L, 4]`` masked-base logits, the core states and, with
    ``n_tracks``, ``[B, L, n_tracks]`` track logits (else None).

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
        self.tracks = nn.Conv1d(widths[0], c.n_tracks, 1) if c.n_tracks else None

    def forward(self, seq: torch.Tensor, segments: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        c = self.config
        b, length = seq.shape
        if length % (2 ** c.n_down):
            raise ValueError(f"length {length} is not a multiple of {2 ** c.n_down}")
        x, skips = self.encode(seq)
        h = x.transpose(1, 2)                                       # [B, bins, d_model]
        bins = h.shape[1]
        if segments is None:
            segments = torch.arange(b, device=seq.device).unsqueeze(1).expand(b, bins)
        else:
            segments = segments + torch.arange(b, device=seq.device).unsqueeze(1) * (segments.max() + 1)
        flat = segments.reshape(-1)
        starts = torch.nonzero(torch.diff(flat, prepend=flat[:1] - 1)).squeeze(1)
        cu = torch.cat([starts, torch.tensor([flat.numel()], device=seq.device)]).to(torch.int32)
        core = self.run_core(h.reshape(1, b * bins, -1), cu).reshape(b, bins, -1)
        x = self.decode(core.transpose(1, 2), skips)
        tracks = self.tracks(x).transpose(1, 2) if self.tracks is not None else None   # [B, L, n_tracks]
        return self.head(x).transpose(1, 2), core, tracks

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
        for block, skip in zip(self.up, reversed(skips)):
            x = block(F.interpolate(x, scale_factor=2, mode="nearest")) + skip
        return x

    def forward_contig(self, seq: torch.Tensor, train_at: list[int], chunk: int = 1 << 20, halo: int = 4096
                       ) -> list[tuple[torch.Tensor, torch.Tensor | None]]:
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
