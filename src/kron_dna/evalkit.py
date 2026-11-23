"""What every scorer of ``scripts/kron_dna`` shares to run on any card and across many: the compute precision
(``--dtype``: bf16 on an A100, fp16 or fp32 where bf16 is emulated, as on Turing) and the split of the work into
shards (``--shard i/n``: rows ``k % n == i``; each shard writes its part under ``shards/``, and a ``merge`` step joins
them once all ``n`` are there)."""
from __future__ import annotations

import contextlib

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
