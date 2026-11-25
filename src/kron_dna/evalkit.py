"""What every scorer of ``scripts/kron_dna`` shares to run on any card and across many: the compute precision
(``--dtype``: bf16 on an A100, fp16 or fp32 where bf16 is emulated, as on Turing) and the split of the work into
shards (``--shard i/n``: rows ``k % n == i``; each shard writes its part under ``shards/``, and a ``merge`` step joins
them once all ``n`` are there)."""
from __future__ import annotations

import contextlib
import math

import torch
import typer

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}


def autocast(dev: torch.device, dtype: str = "bf16"):
    """The scorers' autocast: ``bf16``/``fp16`` mixed precision on a card, none for ``fp32`` or on the CPU."""
    if dtype not in DTYPES:
        raise typer.BadParameter(f"--dtype {dtype!r}: one of {list(DTYPES)}")
    if dev.type != "cuda" or DTYPES[dtype] is None:
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=DTYPES[dtype])


def parse_shard(spec: str) -> tuple[int, int]:
    """``"i/n"`` -> ``(i, n)``; ``""`` -> ``(0, 1)``, the whole job."""
    if not spec:
        return 0, 1
    try:
        i, n = (int(x) for x in spec.split("/"))
    except ValueError:
        raise typer.BadParameter(f"--shard {spec!r}: expected i/n, e.g. 3/16") from None
    if not (n >= 1 and 0 <= i < n):
        raise typer.BadParameter(f"--shard {spec!r}: need 0 <= i < n")
    return i, n


def mine(k: int, shard: tuple[int, int]) -> bool:
    return k % shard[1] == shard[0]


def tag(shard: tuple[int, int]) -> str:
    return f"{shard[0]}-of-{shard[1]}"


# --------------------------------------------------------------------------------- the memory-capped forward

def capped_ok(model) -> bool:
    c = model.config
    return c.arch in ("v2", "v2l") and not c.chunk_ratio


def core_capped(model, seq: torch.Tensor, ctx: torch.Tensor | None, chunk: int) -> torch.Tensor:
    """The core states ``[1, bins, d_model]`` of a window ``seq`` ``[L]``, its encoder run ``chunk`` bases at a time
    (each with :func:`genome.split_halo` bases of real sequence on both sides, its own stretch kept), so the per-base
    activations never span the whole window. Equal to :meth:`KronGenome.forward`'s core up to round-off (v2/v2l
    normalise per position only)."""
    from kron_dna import genome as G
    unit = 2 ** model.config.n_down
    if chunk % unit:
        raise typer.BadParameter(f"--chunk must be a multiple of {unit}")
    L, halo = seq.shape[-1], G.split_halo(model.config)
    parts = []
    for s in range(0, L, chunk):
        e = min(L, s + chunk)
        lo, hi = G.piece_bounds(model.config, s, e, L, halo)     # where a restarting recurrence restarts
        x, _ = model.encode(seq[None, lo:hi])
        parts.append(x[..., (s - lo) // unit:(e - lo) // unit])
    h = torch.cat(parts, dim=-1).transpose(1, 2)
    if model.context is not None:
        h = h + model.context_rows(ctx, 1, seq.device).unsqueeze(1).to(h.dtype)
    cu = torch.tensor([0, h.shape[1]], dtype=torch.int32, device=seq.device)
    return model.run_core(h, cu)


def decode_span(model, seq: torch.Tensor, core: torch.Tensor, span: tuple[int, int]) -> dict:
    """The further heads on bases ``span`` of the window, from its core states: the encoder's skips and the decoder
    run only over the span plus a halo (aligned to the local track's restart points, so it restarts where a whole
    forward would). Sets and returns ``model.last_x``; ``model.last_x_at`` is in window coordinates."""
    from kron_dna import genome as G
    c = model.config
    unit, L, halo = 2 ** c.n_down, seq.shape[-1], G.split_halo(c)
    lo, hi = G.piece_bounds(c, span[0], span[1], L, halo)       # the local track and the in-tower blocks restart
    _, skips = model.encode(seq[None, lo:hi])
    x = model.add_local(model.decode(core[:, lo // unit:hi // unit].transpose(1, 2), skips), skips)
    keep = model.x_span
    model.x_span = (span[0] - lo, span[1] - lo)
    try:
        last, at = model.run_xheads(x)
    finally:
        model.x_span = keep
    model.last_x, model.last_x_at = last, (at[0] + lo, at[1] + lo)
    return last


def forward_capped(model, seq: torch.Tensor, ctx: torch.Tensor | None, span: tuple[int, int], chunk: int) -> dict:
    """The further heads' outputs on ``span`` as :meth:`KronGenome.forward` gives them with ``x_span = span``, at the
    memory of a ``chunk``-base window plus the span (no main-head outputs)."""
    return decode_span(model, seq, core_capped(model, seq, ctx, chunk), span)
