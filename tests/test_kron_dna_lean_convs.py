"""genome.LEAN_CONV (genome-train --lean-convs): norm+GELU recomputed in backward and index-free pooling give the
same outputs and the same gradients as the stock path, on the real KronGenome (CPU core)."""
from __future__ import annotations

import pytest
import torch

import kron_core.blocks
from kron_core.testing import CausalMixer
from kron_dna import genome as G

import test_kron_dna_jepa_block as JB


@pytest.mark.parametrize("arch", ["v2", "v2l"])
def test_lean_matches_stock(monkeypatch, arch):
    monkeypatch.setattr(kron_core.blocks, "kronlstm_mixer", CausalMixer)
    torch.manual_seed(0)
    cfg = G.KronGenomeConfig(d_model=32, n_heads=2, n_layers=1, ff_units=64, stem_channels=16, n_down=3, arch=arch,
                             n_ntv3=3)
    model = JB.CpuCoreGenome(cfg).train()
    seq = torch.randint(0, 4, (2, 512))
    seg = torch.zeros(2, 512 // 8, dtype=torch.long)

    def run():
        model.zero_grad(set_to_none=True)
        logits, _, _ = model(seq, seg, None)
        loss = logits.float().pow(2).mean() + (model.last_ntv3.float().pow(2).mean() if getattr(model, "last_ntv3", None)
                                              is not None else 0.0)
        loss.backward()
        return logits.detach().clone(), {n: p.grad.detach().clone() for n, p in model.named_parameters()
                                         if p.grad is not None}

    monkeypatch.setattr(G, "LEAN_CONV", False)
    out0, g0 = run()
    monkeypatch.setattr(G, "LEAN_CONV", True)
    out1, g1 = run()
    assert torch.equal(out0, out1)
    assert g0.keys() == g1.keys() and len(g0) > 10
    for n in g0:
        assert torch.allclose(g0[n], g1[n], rtol=1e-5, atol=1e-7), n


def test_pool2_matches_max_pool_on_odd_lengths(monkeypatch):
    x = torch.randn(2, 3, 9)
    monkeypatch.setattr(G, "LEAN_CONV", False)
    a = G.pool2(x)
    monkeypatch.setattr(G, "LEAN_CONV", True)
    assert torch.equal(a, G.pool2(x))
