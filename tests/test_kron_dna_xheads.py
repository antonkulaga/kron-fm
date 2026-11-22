"""Further per-base heads (``KronGenomeConfig.xheads``): each its own module beside the inherited heads.

A head registered on a trained model adds parameters and nothing else: the parent's weights load into the same keys,
every inherited output is the parent's, an inactive head runs no forward, and a split window gives the unsplit
outputs.
"""
from __future__ import annotations

import pytest
import torch

from kron_dna import genome as G

from test_kron_dna_split import _restore_precision, batch_for, forced_replicas, small  # noqa: F401


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


@pytest.mark.parametrize("arch", ["v2", "v2l"])
def test_a_binned_head_reads_the_mean_state_of_each_bin(device, monkeypatch, arch):
    forced_replicas(monkeypatch)
    m = with_heads(small(arch), [["base", 3], ["b4", 3, 4], ["b16", 7, 16]]).to(device).eval()
    m.xheads["b4"].load_state_dict(m.xheads["base"].state_dict())    # the same map, so the two can be compared
    seq, seg, ctx, extra = batch_for(2, 2048, device)
    with torch.no_grad():
        m(seq, seg, ctx, extra)
        got = dict(m.last_x)
        assert got["base"].shape == (2, 2048, 3) and got["b4"].shape == (2, 512, 3) and got["b16"].shape == (2, 128, 7)
        # the head is applied to the mean state: a linear probe of the states shows it without a second forward
        probe = torch.nn.Conv1d(m.xheads["base"].fc1.in_channels, 1, 1).to(device)
        keep = {}
        hook = m.xheads["b4"].norm.register_forward_hook(lambda mod, inp, out: keep.setdefault("in", inp[0]))
        hook1 = m.xheads["base"].norm.register_forward_hook(lambda mod, inp, out: keep.setdefault("x", inp[0]))
        m(seq, seg, ctx, extra)
        hook.remove()
        hook1.remove()
        assert torch.allclose(keep["in"], keep["x"].reshape(2, -1, 512, 4).mean(-1), rtol=1e-5, atol=1e-6)
        for by in ("length", None):
            pieces, _ = m.forward_split(seq, seg, ctx, extra, devices=[device, device], by=by)
            for name, k in (("b4", 4), ("b16", 16)):
                whole = torch.zeros_like(got[name])
                for p in pieces:
                    whole[p.rows, p.start // k:p.end // k] = p.x[name].to(device)
                assert torch.allclose(whole, got[name], rtol=1e-4, atol=1e-5), (name, by)
    del probe


@pytest.mark.parametrize("bad", [3, 0, 32])
def test_a_bin_must_be_a_power_of_two_within_the_core_step(bad):
    with pytest.raises(ValueError):                                  # small(): n_down 4, so bins up to 16
        with_heads(small("v2"), [["x", 2, bad]])


# ---------------------------------------------------------------------------------------- KronLSTM heads (kind kron)

def kron_heads(parent: G.KronGenome, heads: list, **kw) -> G.KronGenome:
    """``parent`` with KronLSTM heads; small() has n_down 4, so a step of 4 bases fits a 16-base core step."""
    cfg = G.KronGenomeConfig(**{**parent.config.to_dict(), "xheads": heads, "xhead_kind": "kron", "xhead_bin": 4,
                                "xhead_width": 64, **kw})
    torch.manual_seed(1)
    return G.KronGenome(cfg)


@pytest.mark.parametrize("layers", [1, 2])
def test_kron_heads_predict_per_base_and_learn_with_the_trunk(device, layers):
    parent = small("v2")
    m = kron_heads(parent, [["dnase", 13], ["splice", 4]], xhead_layers=layers).to(device).train()
    missing, unexpected = m.load_state_dict(parent.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("xheads.") for k in missing)
    assert len(m.xheads["dnase"].blocks) == layers
    seq, seg, ctx, extra = batch_for(2, 1024, device)
    m(seq, seg, ctx, extra)
    assert m.last_x["dnase"].shape == (2, 1024, 13) and m.last_x["splice"].shape == (2, 1024, 4)
    (m.last_x["dnase"].sum() + m.last_x["splice"].sum()).backward()
    # A_log only on the card: the CPU stand-in for the KronLSTM layer does not use it (the trunk's alike)
    used = [p for n, p in m.xheads.named_parameters() if device.type == "cuda" or not n.endswith("A_log")]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in used)
    assert m.stem.weight.grad is not None and float(m.stem.weight.grad.abs().sum()) > 0
    again = G.KronGenome(G.KronGenomeConfig(**m.config.to_dict()))
    again.load_state_dict(m.state_dict())                            # the config rebuilds the same kron heads


def test_kron_heads_read_context_beyond_one_step_and_run_on_the_span(device):
    m = kron_heads(small("v2"), [["dnase", 3]]).to(device).eval()
    seq, seg, ctx, extra = batch_for(1, 1024, device)
    with torch.no_grad():
        m(seq, seg, ctx, extra)
        base = m.last_x["dnase"].clone()
        far = seq.clone()
        far[:, 900:] = (far[:, 900:] + 1) % 4                       # bases far from the first step change
        m(far, seg, ctx, extra)
    # the per-position MLP head sees only its own decoder state; the KronLSTM head reads along the stretch, so its
    # first step can change although its own states hardly do (the trunk's own reach is the same for both kinds)
    assert not torch.equal(base[:, :4], m.last_x["dnase"][:, :4])
    m.x_span = (256, 768)
    with torch.no_grad():
        m(seq, seg, ctx, extra)
    assert m.last_x["dnase"].shape == (1, 512, 3) and m.last_x_at == (256, 768)
    m.x_span = (512, 512)                                            # an empty span: an empty output of the right width
    with torch.no_grad():
        m(seq, seg, ctx, extra)
    assert m.last_x["dnase"].shape == (1, 0, 3)


@pytest.mark.parametrize("kw", [dict(xhead_bin=3), dict(xhead_bin=32), dict(xhead_layers=0), dict(xhead_width=32), dict(xhead_kind="kan")])
def test_kron_head_settings_are_checked(kw):
    with pytest.raises(ValueError):
        kron_heads(small("v2"), [["x", 2]], **kw)
    with pytest.raises(ValueError):                                  # a bin wider than the 16-base core step
        kron_heads(small("v2"), [["x", 2, 32]])



def test_a_binned_kron_head_steps_once_per_bin_when_the_bin_is_wide(device):
    m = kron_heads(small("v2"), [["dnase", 3, 16], ["splice", 4]], xhead_bin=4).to(device).eval()
    assert m.xheads["dnase"].bin == 1 and m.xheads["splice"].bin == 4
    seq, seg, ctx, extra = batch_for(2, 1024, device)
    with torch.no_grad():
        m(seq, seg, ctx, extra)
    assert m.last_x["dnase"].shape == (2, 64, 3) and m.last_x["splice"].shape == (2, 1024, 4)


def test_binned_loss_and_eval_use_bin_means_of_the_per_base_targets():
    from kron_dna import xheads as X
    g = torch.Generator().manual_seed(0)
    want = torch.randint(0, 256, (2, 64, 3), generator=g, dtype=torch.uint8)
    ok = torch.ones(2, 64)
    ok[0, 5] = 0                                                       # one base off: its whole bin is unsupervised
    consts = (torch.full((3,), 3.0), torch.ones(3), torch.zeros(3, dtype=torch.bool))
    pred = torch.randn(2, 4, 3, generator=g)
    y, m = X.binned_targets(want, ok, consts, 16)
    assert torch.allclose(y, X.targets_of(want, *consts).reshape(2, 4, 16, 3).mean(2))
    assert m[0].tolist() == [0, 1, 1, 1] and m[1].tolist() == [1, 1, 1, 1]
    want_loss = G.poisson_multinomial_loss(torch.nn.functional.softplus(pred), y, m.unsqueeze(-1))
    assert torch.allclose(X.head_loss(pred, want, ok, consts, "tracks", 16), want_loss)
    p_log, w_log = X.eval_pair(type("S", (), {"kind": "tracks"})(), pred, want, consts, 16)
    assert p_log.shape == w_log.shape == (2, 4, 3)
    assert torch.allclose(w_log, (want.float() / 255 * 3.0).reshape(2, 4, 16, 3).mean(2))
    with pytest.raises(ValueError):
        X.head_loss(pred, want, ok, (torch.ones(3),), "binary", 16)


def test_recomputed_kron_heads_give_the_same_outputs_and_gradients(device):
    torch.manual_seed(0)
    m = kron_heads(small("v2"), [["dnase", 3], ["rna", 2, 16]], xhead_bin=4).to(device).train()
    seq = torch.randint(0, 4, (2, 256), device=device)
    grads = []
    for ck in (False, True):
        m.zero_grad(set_to_none=True)
        m.checkpoint_xheads = ck
        torch.manual_seed(2)                                          # the same draws (dropout, masks) in both passes
        m(seq)
        loss = sum((v.float() ** 2).mean() for v in m.last_x.values())
        loss.backward()
        grads.append({n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None})
        outs = {n: v.detach().clone() for n, v in m.last_x.items()}
        if not ck:
            first = outs
    assert grads[0].keys() == grads[1].keys() and any(n.startswith("xheads.") for n in grads[0])
    for n in grads[0]:
        torch.testing.assert_close(grads[1][n], grads[0][n], rtol=1e-4, atol=1e-6)
    for n in first:
        torch.testing.assert_close(outs[n], first[n])
