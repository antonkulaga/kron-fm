"""A per-bin map of where a genome has something to learn: annotation and measured signal per cell of BIN bases,
for tilting masked-base placement and for weighting track losses toward bins that are not empty.

    uv run python scripts/kron_dna/focus_map.py OUT_DIR STORE [STORE ...] [--name hg38] [--bin 128] [--signal 1,4]

Cells are ``[k * BIN, (k + 1) * BIN)`` of the prepared genome's global base layout (``bases.u8``), the grid of the
pre-binned track stores. ``OUT_DIR/focus.u8`` is ``[cells, K]`` uint8, base-major; ``OUT_DIR/focus.json`` names the
K columns:

- one per annotation bit of ``labels.u8`` (``genome.LABEL_BITS``: promoter, enhancer, CTCF_binding_site,
  open_chromatin_region, exon, donor, acceptor, CDS): the number of the cell's bases that carry the bit, as
  ``round(255 * bases / BIN)``;
- one per STORE (a pre-binned track store at the same BIN, ``target.f16``) and threshold of ``--signal``: the share
  of its channels whose cell target is at least the threshold, as ``ceil(255 * share)`` (zero only when no channel
  reaches it). The target is the signal over the track's non-zero mean, so 1 means "at or above this track's
  typical signal where it has any", which for ChIP-seq is still background nearly everywhere (96 % of cells of
  the human genome); 4 means four times that, closer to a peak.

Nothing is thresholded further: the training loop chooses which columns count and how much. The json also gives,
per column, the share of cells that are non-zero, over the whole genome and over the training split.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer

from kron_core.paths import interim_dir
from kron_dna.genome import LABEL_BITS


def main(out: Path, stores: list[Path], name: str = "hg38", bin: int = 128, signal: str = "1,4",
         step_cells: int = 1 << 18) -> None:
    genome = interim_dir("genome", name)
    labels = np.memmap(genome / "labels.u8", dtype=np.uint8, mode="r")
    total = len(labels)
    cells = -(-total // bin)
    metas = []
    for s in stores:
        j = json.loads((s / "binned.json").read_text())
        if int(j["bin"]) != bin or int(j["cells"]) != cells:
            raise typer.BadParameter(f"{s}: bin {j['bin']}, {j['cells']} cells; the map has bin {bin}, {cells} cells")
        metas.append((s.name, np.memmap(s / "target.f16", dtype=np.float16, mode="r", shape=(cells, len(j["ids"])))))
    thrs = [float(x) for x in str(signal).split(",") if x.strip()]
    cols = [f"label:{b}" for b in LABEL_BITS] + [f"signal>={x:g}:{n}" for n, _ in metas for x in thrs]
    if len(set(cols)) != len(cols):
        raise typer.BadParameter("two stores share a folder name")
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "focus.u8.part"
    mm = np.memmap(tmp, dtype=np.uint8, mode="w+", shape=(cells, len(cols)))
    for a in range(0, cells, step_cells):
        z = min(a + step_cells, cells)
        lab = np.asarray(labels[a * bin:min(z * bin, total)])
        if len(lab) < (z - a) * bin:                                   # the last cell may be short
            lab = np.concatenate([lab, np.zeros((z - a) * bin - len(lab), dtype=np.uint8)])
        lab = lab.reshape(z - a, bin)
        for k, b in enumerate(LABEL_BITS.values()):
            n = ((lab >> b) & 1).sum(1)
            mm[a:z, k] = np.round(255.0 * n / bin).astype(np.uint8)
        k = len(LABEL_BITS)
        for _, t in metas:
            piece = np.asarray(t[a:z])
            for x in thrs:
                mm[a:z, k] = np.ceil(255.0 * (piece >= x).sum(1) / piece.shape[1]).astype(np.uint8)
                k += 1
        typer.echo(f"cells {z:,} of {cells:,}")
    mm.flush()
    split_path = genome / "ntv3" / "split.u8"
    train = None
    if split_path.exists():                                            # a cell is training when its first base is
        train = np.asarray(np.memmap(split_path, dtype=np.uint8, mode="r")[::bin][:cells]) == 1
    nz = (np.asarray(mm) > 0)
    share = {c: {"all": round(float(nz[:, k].mean()), 5),
                 **({"train": round(float(nz[train, k].mean()), 5)} if train is not None else {})}
             for k, c in enumerate(cols)}
    del mm
    tmp.replace(out / "focus.u8")
    (out / "focus.json").write_text(json.dumps({
        "genome": name, "bin": bin, "cells": cells, "total_bases": total, "columns": cols, "signal_thresholds": thrs,
        "label_columns": "round(255 * bases of the cell carrying the bit / bin)",
        "signal_columns": "ceil(255 * share of the store's channels with cell target >= the threshold)",
        "stores": [n for n, _ in metas], "nonzero_share": share,
        "grid": f"cell k = global bases [k*{bin}, (k+1)*{bin}) of the genome's layout"}, indent=1))
    typer.echo(f"wrote {out / 'focus.u8'}: {cells:,} cells x {len(cols)} columns")
    for c, v in share.items():
        typer.echo(f"  {c}: non-zero in {100 * v['all']:.2f}% of cells")


if __name__ == "__main__":
    typer.run(main)
