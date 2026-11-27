"""genome-train --contact-stores/--contact-weight: the contact head trains on the core's states of primary rows, and
its loss joins the total (CPU, a stub model that hands back core states)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import torch

from kron_dna import cli
from kron_dna import contacts as C
from kron_dna import genome as G
import test_kron_dna_multigenome as MGT
from test_kron_dna_multigenome import SMALL, genomes, run  # noqa: F401  (genomes is a fixture)

CELL, DIAGS = 256, 3


class StubWithCore(MGT.StubGenome):
    """The stub, plus core states per 128-bp bin and the contact head the config asks for."""

    def __init__(self, config):
        super().__init__(config)
        self.core_proj = torch.nn.Linear(4, 16)
        tracks = int(getattr(config, "contact_tracks", 0) or 0)
        self.contact_head = (C.ContactHead(16, 8, int(config.contact_reach), tracks) if tracks else None)

    def forward(self, seq, segments=None, context=None):
        logits, _, _ = super().forward(seq, segments, context)
        b, length = seq.shape[0], seq.shape[-1]
        core = self.core_proj(logits.reshape(b, length // 128, 128, -1).mean(2))
        return logits, core, None


def contact_store(path: Path, chroms: dict[str, int], scale: float) -> Path:
    """One store over ``chroms`` (name: length) in cells of CELL bases, DIAGS diagonals, values ``scale * (1 + d)``."""
    path.mkdir(parents=True)
    rows, off = [], 0
    for c, n in chroms.items():
        k = -(-n // CELL)
        rows.append((c, n, k, off, 0))
        off += k
    m = np.tile(scale * (1 + np.arange(DIAGS, dtype=np.float32)), (off, 1)).astype(np.float16)
    m.tofile(path / "contacts.f16")
    pl.DataFrame(rows, schema=["chrom", "length", "cells", "offset", "masked"], orient="row").write_parquet(
        path / "chroms.parquet")
    (path / "hic.json").write_text(json.dumps(dict(bin=CELL, diagonals=DIAGS)))
    return path


def test_the_contact_head_trains_and_reports(genomes, tmp_path, monkeypatch):
    MGT.cpu_only(monkeypatch)
    monkeypatch.setattr(G, "KronGenome", StubWithCore)
    chroms = {"chr1": 6000, "chr2": 4000, "chr3": 700, "chr8": 2600, "chr9": 1900}
    stores = [contact_store(tmp_path / "hic" / n, chroms, s) for n, s in (("a", 1.0), ("b", 2.0))]
    out = tmp_path / "run"
    args = SMALL + ["--steps", "4", "--eval-every", "2", "--batch-size", "2",
                    "--contact-stores", ",".join(map(str, stores)), "--contact-weight", "0.5"]
    res, _ = run(cli, monkeypatch, genomes["root"], out, args)
    assert res.exit_code == 0, res.output + repr(res.exception)
    text = MGT.flat(res.output)
    assert "contacts: 2 maps (a, b), 256-base cells, 3 diagonals, weight 0.5" in text
    assert "contact head: train MSE" in text
    ck = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert ck["config"]["contact_tracks"] == 2 and ck["config"]["contact_reach"] == DIAGS
    head = [k for k in ck["model"] if k.startswith("contact_head.")]
    assert head and all(torch.isfinite(ck["model"][k]).all() for k in head)
    mse = [float(x) for x in text.split("contact head: train MSE ")[1:] for x in [x.split()[0]]]
    assert mse and all(np.isfinite(mse))


def test_contact_weight_needs_stores(genomes, tmp_path, monkeypatch):
    MGT.cpu_only(monkeypatch)
    res, _ = run(cli, monkeypatch, genomes["root"], tmp_path / "run",
                 SMALL + ["--steps", "1", "--contact-weight", "0.5"])
    assert res.exit_code != 0 and "--contact-weight needs --contact-stores" in MGT.flat(res.output + repr(res.exception))
