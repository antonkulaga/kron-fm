"""focus_map.py: per cell, how many bases carry each annotation bit and how many tracks of each store have signal."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from kron_dna.genome import LABEL_BITS

spec = importlib.util.spec_from_file_location(
    "focus_map", Path(__file__).resolve().parents[1] / "scripts" / "kron_dna" / "focus_map.py")
FM = importlib.util.module_from_spec(spec)
spec.loader.exec_module(FM)


def test_map_counts_annotated_bases_and_tracks_with_signal(tmp_path, monkeypatch):
    bin_, total = 8, 8 * 5 + 3                                        # six cells, the last one short
    cells = 6
    genome = tmp_path / "genome"
    (genome / "ntv3").mkdir(parents=True)
    lab = np.zeros(total, dtype=np.uint8)
    lab[0:8] = 1 << LABEL_BITS["CDS"]                                 # cell 0: all coding
    lab[8:12] = (1 << LABEL_BITS["exon"]) | (1 << LABEL_BITS["promoter"])   # cell 1: half exon and promoter
    lab[40:43] = 1 << LABEL_BITS["enhancer"]                          # the short last cell
    lab.tofile(genome / "labels.u8")
    split = np.ones(total, dtype=np.uint8)
    split[16:] = 0                                                    # cells 0 and 1 train
    split.tofile(genome / "ntv3" / "split.u8")
    store = tmp_path / "dnase-bin8"
    store.mkdir()
    t = np.zeros((cells, 3), dtype=np.float16)
    t[2] = [1.0, 0.99, 5.0]                                           # two channels at or above 1.0
    t[4] = [0.2, 0.0, 0.0]                                            # below the threshold: empty
    t.tofile(store / "target.f16")
    (store / "binned.json").write_text(json.dumps({"bin": bin_, "cells": cells, "ids": ["a", "b", "c"]}))
    monkeypatch.setattr(FM, "interim_dir", lambda *a: genome)
    out = tmp_path / "focus"
    FM.main(out, [store], name="x", bin=bin_, signal="1,4", step_cells=4)
    j = json.loads((out / "focus.json").read_text())
    m = np.fromfile(out / "focus.u8", dtype=np.uint8).reshape(cells, len(j["columns"]))
    col = {c: m[:, k] for k, c in enumerate(j["columns"])}
    assert col["label:CDS"].tolist() == [255, 0, 0, 0, 0, 0]
    assert col["label:exon"].tolist() == [0, 128, 0, 0, 0, 0] and col["label:promoter"].tolist() == [0, 128, 0, 0, 0, 0]
    assert col["label:enhancer"].tolist() == [0, 0, 0, 0, 0, round(255 * 3 / 8)]
    assert col["signal>=1:dnase-bin8"].tolist() == [0, 0, 170, 0, 0, 0]          # two of three channels
    assert col["signal>=4:dnase-bin8"].tolist() == [0, 0, 85, 0, 0, 0]           # one of three
    assert j["nonzero_share"]["label:CDS"] == {"all": round(1 / 6, 5), "train": 0.5}
    (store / "binned.json").write_text(json.dumps({"bin": 16, "cells": cells, "ids": ["a", "b", "c"]}))
    with pytest.raises(Exception):
        FM.main(out, [store], name="x", bin=bin_)
