"""Shorten the memory of one band of a checkpoint's core channels, to ask which band a score depends on.

    uv run python scripts/kron_dna/diag/slow_band.py IN.pt OUT.pt --lo-kb 30 --hi-kb 100 [--to-kb 2] [--bin 128]

Every decay channel of the core's KronLSTM mixers (both directions) whose half-life with the input term at zero,
``ln 2 / (exp(A_log) * softplus(dt_bias))`` steps of ``--bin`` bases, lies in ``[lo, hi)`` gets the half-life
``--to-kb`` instead (its ``dt_bias`` only; the head's rate and every other weight stay). No training: score OUT.pt
beside IN.pt with the same scorer. A score that falls says the model reads that band; one that does not move says the
band is carried and not used. Bands that together cover the slow heads' range (30-100, 100-300, 300-1000 kb) show
which part of it is worth keeping. OUT.pt records what was done under ``slow_band``.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import torch
import torch.nn.functional as F
import typer

KEY = re.compile(r"core\.\d+\.(forward|backward)_mixer\.A_log$")


def shorten(state: dict, lo: float, hi: float, to: float) -> dict[str, int]:
    """In place on a model state dict; half-lives in steps. Returns per mixer how many channels were changed."""
    done: dict[str, int] = {}
    for k in sorted(x for x in state if KEY.match(x)):
        a, dt = state[k].double(), state[k[:-len("A_log")] + "dt_bias"]
        heads, per = a.numel(), dt.numel() // a.numel()
        rate = a.exp().repeat_interleave(per)
        half = math.log(2.0) / (rate * F.softplus(dt.double()))
        hit = (half >= lo) & (half < hi)
        y = math.log(2.0) / (to * rate[hit])                          # softplus(dt) wanted
        dt[hit] = torch.log(torch.expm1(y)).to(dt.dtype)
        done[k[:-len(".A_log")]] = int(hit.sum())
    return done


def main(inp: Path, out: Path, lo_kb: float = typer.Option(...), hi_kb: float = typer.Option(...), to_kb: float = 2.0,
         bin: int = 128) -> None:
    if not 0 < to_kb < lo_kb < hi_kb:
        raise typer.BadParameter("0 < to-kb < lo-kb < hi-kb")
    s = torch.load(inp, map_location="cpu", weights_only=False)
    done = shorten(s["model"], lo_kb * 1000 / bin, hi_kb * 1000 / bin, to_kb * 1000 / bin)
    if not done:
        raise typer.BadParameter("no core KronLSTM mixer in this checkpoint")
    s["slow_band"] = dict(lo_kb=lo_kb, hi_kb=hi_kb, to_kb=to_kb, bin=bin, channels=sum(done.values()))
    torch.save(s, out)
    typer.echo(f"{sum(done.values())} channels in {len(done)} mixers: {lo_kb:g}-{hi_kb:g} kb -> {to_kb:g} kb; "
               f"per mixer {min(done.values())} to {max(done.values())}; wrote {out}")


if __name__ == "__main__":
    typer.run(main)
