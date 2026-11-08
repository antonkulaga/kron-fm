"""The KronLSTM residual block and the recurrent-state cache every kron-fm model shares.

Moved verbatim from glucose_forecasting_metabonet's ``kronlstm/model.py`` (commit 6e9bbbe91), where the
glucose forecaster (now :mod:`krontime.glucose`) defined them; only the vendored layer's namespace
changed. Module and parameter names are unchanged, so the forecaster's checkpoints load as they were.

``kronlstm_mixer`` is looked up through this module's globals when a block is built, so a test can
replace it with a CPU stand-in that keeps the layer's call (``mixer(h, attention_mask=mask)`` or
``mixer(h, cu_seqlens=...)``, returning ``(out, _, _)``) and its padding rule.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.gate_up = nn.Linear(d_model, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(self.drop(F.silu(gate) * up))


def kronlstm_mixer(d_model: int, n_heads: int, layer_idx: int) -> nn.Module:
    """NX-AI's KronLSTM token mixer from the vendored kernels, imported only when a model is built.

    It is called as ``mixer(h, attention_mask=mask)`` and returns ``(out, _, _)``. A step whose
    mask is 0 is taken out of the sequence: it reads nothing, nothing reads it, and its output is
    zero. The model's tests replace this function with a CPU stand-in that keeps that call and
    that rule, so the decode layout is tested on machines with no CUDA device.
    """
    from kron_core.vendor.fla.layers.kronlstm import KronLSTM

    head_dim = d_model // n_heads
    return KronLSTM(hidden_size=d_model, k_head_dim=head_dim, v_head_dim=head_dim,
                    num_k_heads=n_heads, num_v_heads=n_heads, layer_idx=layer_idx)


class LayerStates:
    """Every layer's recurrent memory and short-convolution state for a batch of streams (C66).

    It is the cache the vendored layer reads and writes (``past_key_values``): indexed by the
    layer, each entry a dict of ``recurrent_state`` ``(N, H, K, V)`` and ``conv_state``
    ``(N, D, W)``, or ``None`` before the layer has read anything, which the layer takes as zeros.
    The layer calls :meth:`update` after every piece it reads, so the same object carries a stream
    from one piece of a person's record to the next. ``offset`` is accepted and ignored: nothing
    here reads absolute positions.
    """

    def __init__(self, layers: Sequence[dict[str, torch.Tensor] | None]) -> None:
        self.layers = list(layers)

    @classmethod
    def empty(cls, n_layers: int) -> "LayerStates":
        return cls([None] * n_layers)

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, layer_idx: int) -> dict[str, torch.Tensor] | None:
        return self.layers[layer_idx]

    def update(self, layer_idx: int, offset: int = 0, **state: torch.Tensor | None) -> None:
        self.layers[layer_idx] = {k: v for k, v in state.items()}

    def _map(self, fn) -> "LayerStates":
        return LayerStates([None if s is None else {k: (None if v is None else fn(v)) for k, v in s.items()}
                            for s in self.layers])

    def clone(self) -> "LayerStates":
        """A copy a later read cannot change, gradients kept."""
        return self._map(torch.Tensor.clone)

    def detach(self) -> "LayerStates":
        """The same state cut from the graph: the next chunk carries it, its gradient stops here."""
        return self._map(torch.Tensor.detach)

    def select(self, rows: torch.Tensor) -> "LayerStates":
        """The streams at ``rows`` (an index tensor), for decoding several origins of one stream."""
        return self._map(lambda v: v.index_select(0, rows))

    def reset(self, rows: torch.Tensor) -> "LayerStates":
        """Zero the streams where ``rows`` (a bool mask) is true: they start a new stretch of record."""
        if not bool(rows.any()):
            return self

        def zero(v: torch.Tensor) -> torch.Tensor:
            keep = (~rows).to(v.dtype).view(-1, *([1] * (v.dim() - 1)))
            return v * keep

        return self._map(zero)

    @staticmethod
    def cat(parts: Sequence["LayerStates"]) -> "LayerStates":
        """The streams of several states stacked along the batch, layer by layer."""
        layers = []
        for entries in zip(*(p.layers for p in parts), strict=True):
            if any(e is None for e in entries):
                if not all(e is None for e in entries):
                    raise ValueError("cannot stack a layer that has read nothing with one that has")
                layers.append(None)
                continue
            layers.append({k: (None if any(e[k] is None for e in entries) else torch.cat([e[k] for e in entries]))
                           for k in entries[0]})
        return LayerStates(layers)


class KronLSTMBlock(nn.Module):
    """Pre-norm residual block: RMSNorm -> KronLSTM token mixer, RMSNorm -> SwiGLU."""

    def __init__(self, d_model: int, n_heads: int, ff_units: int, dropout: float, layer_idx: int) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} must be divisible by n_heads {n_heads}")
        head_dim = d_model // n_heads
        if not 16 <= head_dim <= 256:
            raise ValueError(f"KronLSTM's chunk kernels need a head dimension in [16, 256], got {head_dim}")
        self.layer_idx = layer_idx
        self.mixer_norm = nn.RMSNorm(d_model, eps=1e-5)
        self.mixer = kronlstm_mixer(d_model, n_heads, layer_idx)
        self.mlp_norm = nn.RMSNorm(d_model, eps=1e-5)
        self.mlp = SwiGLU(d_model, ff_units, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, attention_mask: torch.Tensor | None,
                states: LayerStates | None = None) -> torch.Tensor:
        """``states`` (C66), when given, is read as this layer's starting memory and left holding its end."""
        if states is None:
            mixed, _, _ = self.mixer(self.mixer_norm(h), attention_mask=attention_mask)
        else:
            mixed, _, _ = self.mixer(self.mixer_norm(h), attention_mask=attention_mask, past_key_values=states,
                                     use_cache=True)
        h = h + self.drop(mixed)
        return h + self.drop(self.mlp(self.mlp_norm(h)))

    def forward_pieces(self, h: torch.Tensor, attention_mask: torch.Tensor | None, states: LayerStates,
                       cuts: Sequence[int]) -> tuple[torch.Tensor, list[dict[str, torch.Tensor] | None]]:
        """:meth:`forward` with the recurrence read in pieces ending at each of ``cuts``.

        ``cuts`` are increasing step counts; the state after the first ``c`` steps is copied out for
        each, so a decoder can start from the memory at an origin. Only the token mixer is
        sequential, so the pieces are mixed one after another and the rest of the block runs once.
        """
        normed = self.mixer_norm(h)
        steps = h.size(1)
        bounds = [0, *[int(c) for c in cuts], steps]
        if any(later < earlier for earlier, later in zip(bounds, bounds[1:])):
            raise ValueError(f"cuts {list(cuts)} must increase within the {steps} steps")
        outs, snaps = [], []
        for j, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
            if b > a:
                mask = None if attention_mask is None else attention_mask[:, a:b]
                if mask is not None and not bool(mask.any()):
                    # No stream reaches into this piece (every lane has ended): nothing to read, the
                    # memory stays as it is, and the fused norm's backward cannot take an empty input.
                    outs.append(torch.zeros_like(normed[:, a:b]))
                else:
                    if mask is not None and bool(mask.all()):
                        mask = None
                    out, _, _ = self.mixer(normed[:, a:b], attention_mask=mask, past_key_values=states,
                                           use_cache=True)
                    outs.append(out)
            if j < len(cuts):
                state = states[self.layer_idx]
                snaps.append(None if state is None else {k: (None if v is None else v.clone())
                                                         for k, v in state.items()})
        h = h + self.drop(torch.cat(outs, dim=1) if outs else torch.zeros_like(h))
        return h + self.drop(self.mlp(self.mlp_norm(h))), snaps


def _read_block(blk: KronLSTMBlock, h: torch.Tensor, mask: torch.Tensor | None, n_layers: int,
                cuts: tuple[int, ...]) -> tuple[torch.Tensor, list]:
    """One block over a stream from zero memory, with its states at ``cuts``: what a checkpoint recomputes."""
    states = LayerStates.empty(n_layers)
    if cuts:
        return blk.forward_pieces(h, mask, states, cuts)
    return blk(h, mask, states), []

