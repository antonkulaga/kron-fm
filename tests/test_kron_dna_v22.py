"""v2.2 = --short-sources pack (branch no-pad) + block JEPA (branch jepa-block): the two together.

With pack a training micro-batch reaches loss_on as several parts of different row lengths; block JEPA places its
blocks per row inside loss_on, so each part's rows get their own blocks whatever their length and number.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from kron_dna import genome as G

import test_kron_dna_multigenome as MGT  # noqa: F401
from test_kron_dna_multigenome import run
from test_kron_dna_jepa_block import JEPA, SMALLJ, jgenomes, spy_on, stub_jepa  # noqa: F401  (jgenomes: a fixture)


def test_block_jepa_with_short_sources_pack(jgenomes, tmp_path, monkeypatch):
    root = jgenomes["root"]
    stub_jepa(monkeypatch)
    from kron_dna import cli as new
    blocks = []
    spy_on(monkeypatch, "block_mask", blocks)
    args = SMALLJ + JEPA + ["--bert-mask", "--mito", "mt", "--mito-share", "0.5", "--n-contexts", "64",
                            "--n-in-extra", "2", "--soft-base-rate", "0.05", "--repeat-channel",
                            "--repeat-channel-drop", "0.0", "--rc", "--steps", "10", "--eval-every", "5",
                            "--batch-size", "4", "--short-sources", "pack", "--jepa-block-share", "0.25",
                            "--jepa-block-len", "256:1024", "--jepa-log-every", "1"]
    res, ev = run(new, monkeypatch, root, tmp_path / "pack", args)
    assert res.exit_code == 0, res.output + repr(res.exception)
    train = [e for e in ev if e[0] == "forward" and e[1]]
    assert len(train) == len(blocks) > 10                    # some micro-batches came as more than one part
    lengths = set()
    n_hidden = n_seq = 0
    for e, ((seg, seq, unit, share, lo, hi, _), _, m) in zip(train, blocks):
        assert (unit, share, lo, hi) == (128, 0.25, 2, 8) and m.shape == seg.shape
        assert seq.shape == (seg.shape[0], seg.shape[1] * 128) and tuple(e[3].shape[::2]) == seq.shape
        lengths.add(seq.shape[1])
        assert not m[seg != 0].any()                         # padding is never hidden
        for row in range(m.shape[0]):
            run_ = seg[row] == 0
            n_run = int(run_.sum())
            assert run_[:n_run].all()                        # one run of sequence bins, then padding
            partial = n_run > 0 and seq[row, n_run * 128 - 1] == 4
            n_ok = n_run - int(partial)
            assert m[row].sum() == min(int(math.floor(0.25 * n_ok + 0.5)), max(n_ok - 1, 0)), (n_run, n_ok)
            n_hidden += int(m[row].sum())
            n_seq += n_run
        at = torch.from_numpy(m).repeat_interleave(128, dim=1)
        inside = e[3].transpose(1, 2)[at]                    # the online input inside blocks: MASK only
        assert (inside[:, G.MASK_CHANNEL] == 1).all() and (inside.sum(-1) == 1).all()
        assert (e[4].transpose(1, 2)[at] == 0).all()
    assert len(lengths) >= 2 and max(lengths) == 4096 and min(lengths) < 4096, lengths
    assert 0.15 < n_hidden / n_seq < 0.30
    lines = (tmp_path / "pack" / "jepa.csv").read_text().strip().split("\n")
    head = lines[0].split(",")
    rows = [dict(zip(head, map(float, line.split(",")))) for line in lines[1:]]
    assert [r["step"] for r in rows] == list(range(1, 11))
    assert all(math.isfinite(r["jepa_hidden"]) and math.isfinite(r["train_total"]) and 0.1 < r["hidden_share"] < 0.35
               for r in rows)
