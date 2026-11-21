"""The ONNX export converts a pre-binned track head (``BinnedTrackStore``, kind ``tracks_binned``) to natural units like
a per-base track head; before this, ``x_constants`` skipped it and the graph sent its raw output through the sigmoid
meant for binary heads (found on KronGenome F, 21 Nov 2026: ``x_dnase196b`` and ``x_rna462b`` read as probabilities)."""
import json

import numpy as np

from kron_dna import genome_onnx as GO


def _binned_store(path, ids, cells=4, bin_=128):
    path.mkdir()
    j = dict(ids=ids, cells=cells, bin=bin_, scales=[1.0] * len(ids), nonzero_mean=[0.5 + i for i in range(len(ids))],
             assays=["DNase-seq"] + ["total RNA-seq"] * (len(ids) - 1), genome="test", total_bases=cells * bin_)
    (path / "binned.json").write_text(json.dumps(j))
    for n in ("target", "log1p"):
        np.zeros((cells, len(ids)), dtype=np.float16).tofile(path / f"{n}.f16")
    return path


def test_x_constants_include_binned_track_stores(tmp_path):
    store = _binned_store(tmp_path / "rna-bin128", ["a_P", "a_M", "b"])
    record = {"xheads": {"stores": {"rna": {"path": str(store), "ids": ["a_P", "a_M", "b"]}}}}
    consts, ids = GO.x_constants(record)
    assert ids == {"rna": ["a_P", "a_M", "b"]}
    assert "rna" in consts, "a binned track head must get the natural-units conversion, not the binary sigmoid"
    nz, rna = consts["rna"]
    assert np.allclose(nz, [0.5, 1.5, 2.5])
    assert rna.tolist() == [False, True, True]
