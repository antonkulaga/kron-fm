"""scripts/kron_dna/bin_track_store.py: a per-base store as cells of BIN bases, in parts, resumable."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from kron_dna import xheads as X
from test_kron_dna_multigenome import genomes  # noqa: F401  (a fixture)
from test_kron_dna_xheads_cli import make_store

spec = importlib.util.spec_from_file_location(
    "bin_track_store", Path(__file__).resolve().parents[1] / "scripts" / "kron_dna" / "bin_track_store.py")
BT = importlib.util.module_from_spec(spec)
spec.loader.exec_module(BT)


def test_cells_hold_the_bin_means_of_the_per_base_targets(genomes, tmp_path):
    store = make_store(tmp_path, genomes["hgt"], n=3)
    j = json.loads((store / "dense.json").read_text())
    j["assays"] = ["DNase-seq", "RNA-seq", "DNase-seq"]                  # one RNA channel: the 0.75 power
    (store / "dense.json").write_text(json.dumps(j))
    out = tmp_path / "binned"
    BT.main(store, out, bin=4, part=0, parts=2, block_cells=2048)       # half the blocks: no record yet
    assert not (out / "binned.json").exists()
    BT.main(store, out, bin=4, part=1, parts=2, block_cells=2048)
    meta = json.loads((out / "binned.json").read_text())
    st = X.open_store(store)
    cells, t = meta["cells"], len(st.ids)
    tgt = np.memmap(out / "target.f16", dtype=np.float16, mode="r", shape=(cells, t))
    lg = np.memmap(out / "log1p.f16", dtype=np.float16, mode="r", shape=(cells, t))
    consts = st.consts("cpu")
    at, n = 160, 64                                                       # a window on the grid, 16 cells
    want = torch.from_numpy(np.asarray(st.dense[at:at + n]))[None]
    y, _ = X.binned_targets(want, torch.ones(1, n), consts, 4)
    assert np.allclose(np.asarray(tgt[at // 4:(at + n) // 4], dtype=np.float32), y[0].numpy(), rtol=2e-3, atol=2e-3)
    w = X.pool_bins(want.float() / 255.0 * consts[0], 4)
    assert np.allclose(np.asarray(lg[at // 4:(at + n) // 4], dtype=np.float32), w[0].numpy(), rtol=2e-3, atol=2e-3)
    assert meta["bin"] == 4 and meta["ids"] == st.ids
