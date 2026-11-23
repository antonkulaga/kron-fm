"""A per-base track store as a pre-binned store: per cell of BIN bases on the genome's global grid, two float16
values per channel, for a further head that predicts at BIN bases (``genome-train --xhead-bin``).

    uv run python scripts/kron_dna/bin_track_store.py SRC_STORE OUT_DIR [--bin 128] [--part K --parts N]

Why: a per-base store of T channels is T bytes per base; a 1 Mb training window reads T MB of it at random, and
once the page cache is full every step waits on the disk (the trainer measured D at 6-7 s/step instead of 2.7 with
the 196-track DNase store, 600 GB). Binned at 128 bases it is 32-fold smaller and a window reads one cell per
128 bases.

Cells are ``[k * BIN, (k + 1) * BIN)`` of the global base layout (the store's ``dense.u8`` rows), so a training
window must start on the grid; ``genome-train`` snaps primary-genome windows to it when a pre-binned head is in
use. Per cell and channel:

- ``target.f16``: the mean over the cell's bases of the per-base training target (``xheads.targets_of``: decoded
  signal over the track's non-zero mean, the 0.75 power for RNA-seq, the soft clip above 10), exactly the target a
  binned head trains on from the per-base store;
- ``log1p.f16``: the mean over the cell's bases of log1p signal, what its held-out Pearson compares against.

Both ``[cells, T]`` float16, base-major like ``dense.u8``. ``binned.json`` carries the source's record (ids,
scales, non-zero means, assays, tracks) plus the bin and the grid. Kill-safe and parallel: ``--parts N`` jobs each
take every Nth block of cells, write into one pair of files (page-aligned, disjoint blocks) and mark each finished
block; a rerun skips marked blocks; the part that finds every block marked writes ``binned.json`` last.
"""
from __future__ import annotations

import json
import os
import socket
from pathlib import Path

import numpy as np
import typer

PART = f".part.{socket.gethostname()}.{os.getpid()}"


def targets_np(u8: np.ndarray, scales: np.ndarray, nz: np.ndarray, rna: np.ndarray) -> np.ndarray:
    """numpy twin of :func:`kron_dna.xheads.targets_of` (float32), on the shared transforms."""
    from kron_dna.prep.transforms import decode_u8, targets_np as tr
    return tr(decode_u8(u8, scales) / nz, rna).astype(np.float32)


def main(src: Path, out: Path, bin: int = 128, part: int = 0, parts: int = 1,
         block_cells: int = typer.Option(8192, help="cells per block (one unit of work and of resume)")) -> None:
    j = json.loads((src / "dense.json").read_text())
    total, t = int(j["total_bases"]), int(j["n_tracks"])
    dense = np.memmap(src / "dense.u8", dtype=np.uint8, mode="r", shape=(total, t))
    scales = np.asarray(j["scales"], dtype=np.float32)
    nz = np.asarray(j["nonzero_mean"], dtype=np.float32)
    rna = np.array(["RNA-seq" in a for a in j["assays"]], dtype=bool)
    if (block_cells * t * 2) % 4096:
        raise typer.BadParameter("block_cells x channels x 2 bytes must be a multiple of 4096 (page-aligned blocks)")
    cells = -(-total // bin)
    n_blocks = -(-cells // block_cells)
    out.mkdir(parents=True, exist_ok=True)
    marks = out / "blocks"
    marks.mkdir(exist_ok=True)
    files = {}
    for name in ("target", "log1p"):
        path = out / f"{name}.f16"
        if not path.exists():                          # one creator; the others open what it made
            tmp = Path(str(path) + PART)
            np.memmap(tmp, dtype=np.float16, mode="w+", shape=(cells, t)).flush()
            try:
                os.link(tmp, path)                     # atomic, fails if another part made it first
            except FileExistsError:
                pass
            tmp.unlink()
        files[name] = np.memmap(path, dtype=np.float16, mode="r+", shape=(cells, t))
    done = 0
    for blk in range(part, n_blocks, parts):
        if (marks / str(blk)).exists():
            continue
        c0, c1 = blk * block_cells, min((blk + 1) * block_cells, cells)
        a, b = c0 * bin, min(c1 * bin, total)
        u8 = np.asarray(dense[a:b])
        n_full = (b - a) // bin
        tgt = targets_np(u8, scales, nz, rna)
        lg = u8.astype(np.float32) / 255.0 * scales
        for arr, name in ((tgt, "target"), (lg, "log1p")):
            v = np.empty((c1 - c0, t), dtype=np.float32)
            v[:n_full] = arr[:n_full * bin].reshape(n_full, bin, t).mean(1)
            if n_full < c1 - c0:                       # the genome's last, partial cell
                v[n_full:] = arr[n_full * bin:].mean(0)
            files[name][c0:c1] = v.astype(np.float16)
        for f in files.values():
            f.flush()
        (marks / str(blk)).touch()
        done += 1
        if done % 50 == 0:
            typer.echo(f"part {part}/{parts}: {done} blocks this run, at block {blk} of {n_blocks}")
    if all((marks / str(k)).exists() for k in range(n_blocks)) and not (out / "binned.json").exists():
        meta = dict(j, bin=bin, cells=cells, grid=f"cell k = global bases [k*{bin}, (k+1)*{bin}) of the store's layout",
                    files={"target": "mean of xheads.targets_of over the cell, float16 [cells, T]",
                           "log1p": "mean of log1p signal over the cell, float16 [cells, T]"},
                    source=str(src), source_record_sha256=__import__("hashlib").sha256(
                        (src / "dense.json").read_bytes()).hexdigest())
        Path(str(out / "binned.json") + PART).write_text(json.dumps(meta, indent=1))
        os.replace(str(out / "binned.json") + PART, out / "binned.json")
        typer.echo(f"all {n_blocks} blocks done; wrote {out / 'binned.json'}")
    else:
        left = sum(not (marks / str(k)).exists() for k in range(n_blocks))
        typer.echo(f"part {part}/{parts} finished its blocks; {left} of {n_blocks} blocks still open elsewhere")


if __name__ == "__main__":
    typer.run(main)
