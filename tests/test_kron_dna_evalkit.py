"""kron_dna.evalkit: the memory-capped forward gives the whole forward's further-head outputs (encoder in chunks with
the model's halo, core over every bin, decoder on the head span only); shards cover every row once."""
from __future__ import annotations

import pytest
import torch

import kron_core.blocks
from kron_dna import evalkit as EK


def test_capped_forward_equals_the_whole_forward(monkeypatch):
    from kron_core.testing import CausalMixer
    monkeypatch.setattr(kron_core.blocks, "kronlstm_mixer", CausalMixer)
    from test_kron_dna_split import small
    m = small("v2", xheads=[["dnase", 3], ["genes", 14]], xhead_kind="kron", xhead_bin=4, xhead_width=64).eval()
    L, span = 2048, (512, 1024)
    g = torch.Generator().manual_seed(0)
    seq = torch.randint(0, 5, (L,), generator=g)
    ctx = torch.zeros(1, dtype=torch.long)
    m.x_span = span
    with torch.no_grad():
        _, core, _ = m(seq[None], torch.zeros(1, L // 16, dtype=torch.long), ctx)
        whole = {k: v.clone() for k, v in m.last_x.items()}
        at = m.last_x_at
        c2 = EK.core_capped(m, seq, ctx, chunk=256)
        assert torch.allclose(c2, core, atol=1e-5)
        got = EK.forward_capped(m, seq, ctx, span, chunk=256)
    assert m.last_x_at == at == span
    for k in whole:
        assert torch.allclose(got[k], whole[k], rtol=1e-5, atol=1e-4), (k, float((got[k] - whole[k]).abs().max()))


def test_shards_cover_every_row_once():
    n = 5
    rows = [k for i in range(n) for k in range(23) if EK.mine(k, (i, n))]
    assert sorted(rows) == list(range(23))
    assert EK.parse_shard("") == (0, 1) and EK.parse_shard("3/16") == (3, 16)
    with pytest.raises(Exception):
        EK.parse_shard("16/16")
