"""Further per-base heads (``KronGenomeConfig.xheads``): each its own module beside the inherited heads.

A head registered on a trained model adds parameters and nothing else: the parent's weights load into the same keys,
every inherited output is the parent's, an inactive head runs no forward, and a split window gives the unsplit
outputs.
"""
from __future__ import annotations

import pytest
import torch

from kron_dna import genome as G

from test_kron_dna_split import batch_for, forced_replicas, small


def with_heads(parent: G.KronGenome, heads: list) -> G.KronGenome:
    cfg = G.KronGenomeConfig(**{**parent.config.to_dict(), "xheads": heads})
    torch.manual_seed(1)
    return G.KronGenome(cfg)


@pytest.mark.parametrize("arch", ["v2", "v2l"])
def test_new_heads_are_the_only_new_parameters_and_the_parent_outputs_stand(device, arch):
    parent = small(arch).to(device).eval()
    child = with_heads(parent, [["dnase", 13], ["splice", 2]]).to(device).eval()
    missing, unexpected = child.load_state_dict(parent.state_dict(), strict=False)
    assert not unexpected and missing and all(k.startswith("xheads.") for k in missing)
    assert {k.split(".")[1] for k in missing} == {"dnase", "splice"}
    seq, seg, ctx, extra = batch_for(2, 2048, device)
    with torch.no_grad():
        logits_p, core_p, _ = parent(seq, seg, ctx, extra)
        ntv3_p = parent.last_ntv3
        logits_c, core_c, _ = child(seq, seg, ctx, extra)
    assert torch.equal(logits_p, logits_c) and torch.equal(core_p, core_c) and torch.equal(ntv3_p, child.last_ntv3)
    assert child.last_x["dnase"].shape == (2, 2048, 13) and child.last_x["splice"].shape == (2, 2048, 2)
    assert parent.last_x == {}


def test_config_round_trips_and_a_checkpoint_without_heads_loads(device):
    child = with_heads(small("v2"), [["dnase", 3]])
    again = G.KronGenome(G.KronGenomeConfig(**child.config.to_dict()))
    again.load_state_dict(child.state_dict())                        # strict: same keys
    old = {k: v for k, v in small("v2").config.to_dict().items() if k != "xheads"}
    assert G.KronGenomeConfig(**old).xheads == []                    # a config saved before xheads existed
    narrow = G.KronGenome(G.KronGenomeConfig(**(child.config.to_dict() | {"xhead_hidden": 6})))
    assert narrow.xheads["dnase"].fc1.out_channels == 6 and narrow.head.fc1.out_channels == child.head.fc1.out_channels


def test_an_inactive_head_runs_no_forward_and_gets_no_gradient(device):
    m = with_heads(small("v2"), [["dnase", 3], ["splice", 2]]).to(device).train()
    seq, seg, ctx, extra = batch_for(2, 1024, device)
    m.x_active = {"dnase"}
    m(seq, seg, ctx, extra)
    assert set(m.last_x) == {"dnase"}
    m.last_x["dnase"].sum().backward()
    assert all(p.grad is None for p in m.xheads["splice"].parameters())
    assert all(p.grad is not None for p in m.xheads["dnase"].parameters())
    m.x_active = set()
    m(seq, seg, ctx, extra)
    assert m.last_x == {}


@pytest.mark.parametrize("bad", [[["a", 1], ["a", 2]], [["not a name", 1]]])
def test_head_names_are_checked(bad):
    with pytest.raises(ValueError):
        with_heads(small("v2"), bad)


def test_heads_refuse_v1():
    with pytest.raises(ValueError):
        G.KronGenome(G.KronGenomeConfig(d_model=32, n_heads=2, n_layers=1, ff_units=64, stem_channels=16, n_down=3,
                                        xheads=[["dnase", 2]]))


@pytest.mark.parametrize("arch", ["v2", "v2l"])
@pytest.mark.parametrize("b,length,by", [(2, 2048, "length"), (4, 1024, None)])
def test_split_window_gives_the_unsplit_head_outputs(device, monkeypatch, arch, b, length, by):
    forced_replicas(monkeypatch)
    m = with_heads(small(arch), [["dnase", 5], ["splice", 2]]).to(device).eval()
    m.x_active = {"dnase"}
    seq, seg, ctx, extra = batch_for(b, length, device)
    with torch.no_grad():
        m(seq, seg, ctx, extra)
        want = m.last_x["dnase"]
        pieces, _ = m.forward_split(seq, seg, ctx, extra, devices=[device, device], by=by)
    assert m.last_x == {} and all(set(p.x) == {"dnase"} for p in pieces)
    got = torch.zeros_like(want)
    for p in pieces:
        got[p.rows, p.start:p.end] = p.x["dnase"].to(device)
    assert torch.allclose(got, want, rtol=1e-4, atol=1e-5)


def test_split_window_sends_gradients_to_the_head_and_the_trunk(device, monkeypatch):
    forced_replicas(monkeypatch)
    m = with_heads(small("v2"), [["dnase", 5]]).to(device).train()
    seq, seg, ctx, extra = batch_for(2, 2048, device)
    pieces, _ = m.forward_split(seq, seg, ctx, extra, devices=[device, device], by="length")
    sum(p.x["dnase"].float().pow(2).sum().to(device) for p in pieces).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.xheads.parameters())
    assert m.stem.weight.grad is not None and float(m.stem.weight.grad.abs().sum()) > 0
