"""Further per-base heads (``KronGenomeConfig.xheads``): each its own module beside the inherited heads.

A head registered on a trained model adds parameters and nothing else: the parent's weights load into the same keys,
every inherited output is the parent's, an inactive head runs no forward, and a split window gives the unsplit
outputs.
"""
from __future__ import annotations

import pytest
import numpy as np
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


# ------------------------------------------------------------------------- further genomes (--xhead-species)

def test_a_channel_mask_scores_only_the_rows_channels():
    """A masked loss equals the loss over the channels each row has; masked scorers count each channel's rows only."""
    from kron_dna import xheads as XH
    torch.manual_seed(0)
    pred, want = torch.randn(2, 64, 4), torch.rand(2, 64, 4)
    ok = torch.ones(2, 64)
    cm = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=torch.bool)
    m = ok.unsqueeze(-1) * cm.float().unsqueeze(1)
    want_t = torch.stack([want[..., 0] * 0 + want[..., 0]], -1)       # pre-binned targets: [..., 0] is the target
    got = XH.head_loss(pred, torch.stack([want, want], -1), ok, (), "tracks_binned", 1, cm)
    ref = G.poisson_multinomial_loss(torch.nn.functional.softplus(pred), want, m)
    torch.testing.assert_close(got, ref)
    lab = (torch.rand(2, 64, 4) < 0.1).float()
    pw = (torch.ones(4),)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(pred, lab, pos_weight=pw[0], reduction="none")
    torch.testing.assert_close(XH.head_loss(pred, lab, ok, pw, "binary", 1, cm), (bce * m).sum() / m.sum())
    p = XH.Pearson(4, bin=8)
    p.add(pred, want, cm)
    q = XH.Pearson(4, bin=8)
    q.add(pred[:1], want[:1])
    torch.testing.assert_close(torch.as_tensor(p.value()[:2]), torch.as_tensor(q.value()[:2]))
    assert p.n.tolist() == [8.0, 8.0, 8.0, 8.0]
    ap = XH.AveragePrecision(4)
    ap.add(pred, lab, cm)
    assert all(len(torch.cat(ap.pos[k])) == int(lab[1 if k >= 2 else 0, :, k].sum()) for k in range(4))
    del want_t


class _Part:
    """A store stand-in for SpeciesStores: binary or pre-binned, with a record carrying each track's metadata."""

    def __init__(self, tmp, name, ids, kind, meta=None, genome="g"):
        import json as _j
        self.path, self.ids, self.kind, self.genome = tmp / name, list(ids), kind, genome
        self.path.mkdir(parents=True)
        self.swap = np.arange(len(ids))
        for k, i in enumerate(ids):
            if i.endswith(("_P", "_M")):
                self.swap[k] = self.ids.index(i[:-1] + ("M" if i.endswith("_P") else "P"))
        self.positives = np.arange(1, len(ids) + 1)
        self.bin = 128
        self.identity = dict(ids=list(ids))
        if kind == "tracks_binned":
            (self.path / "binned.json").write_text(_j.dumps(dict(tracks=meta)))

    def window(self, at, take, width, flip=False):
        return np.full((width, len(self.ids)), 7, dtype=np.uint8)

    def empty(self, width):
        return np.zeros((width, len(self.ids)), dtype=np.uint8)

    def consts(self, device):
        return (torch.arange(len(self.ids), dtype=torch.float32) + (100 if self.genome == "mm" else 0),)


def test_species_stores_share_matching_channels_and_append_the_rest(tmp_path):
    from kron_dna import xheads as XH
    hs = _Part(tmp_path, "h", ["E1", "E2", "R_P", "R_M"], "tracks_binned",
               [dict(assay="DNase-seq", biosample="liver"), dict(assay="DNase-seq", biosample="K562"),
                dict(assay="polyA plus RNA-seq", biosample="liver"), dict(assay="polyA plus RNA-seq", biosample="liver")])
    ms = _Part(tmp_path, "m", ["F1", "F2", "S_P", "S_M"], "tracks_binned",
               [dict(assay="DNase-seq", biosample="liver"), dict(assay="DNase-seq", biosample="forebrain"),
                dict(assay="total RNA-seq", biosample="liver"), dict(assay="total RNA-seq", biosample="liver")],
               genome="mm")
    s = XH.SpeciesStores(hs, {"mm": ms})
    assert s.ids == ["E1", "E2", "R_P", "R_M", "mm:F2"] and s.shared == {"mm": 3}
    assert s.cols["mm"].tolist() == [0, 4, 2, 3] and s.swap.tolist() == [0, 1, 3, 2, 4]
    assert s.mask("mm").tolist() == [True, False, True, True, True] and s.mask(None).tolist()[:4] == [True] * 4
    w = s.window(0, 4, 4, genome="mm")
    assert w.shape == (4, 5) and (w[:, [0, 2, 3, 4]] == 7).all() and (w[:, 1] == 0).all()
    assert s.consts("cpu")[0].tolist() == [0, 1, 2, 3, 101]           # the primary's on shared channels
    assert s.identity["species"]["mm"]["channels"] == [0, 4, 2, 3]
    twice = _Part(tmp_path, "m2", ["F1", "F1b"], "tracks_binned",
                  [dict(assay="DNase-seq", biosample="liver"), dict(assay="DNase-seq", biosample="liver")], genome="mm")
    t2 = XH.SpeciesStores(hs, {"mm": twice})                      # two liver experiments: one shares, one is new
    assert t2.cols["mm"].tolist() == [0, 4] and t2.ids[4] == "mm:F1b" and t2.shared == {"mm": 1}
    lab_h, lab_m = (_Part(tmp_path, n, ["donor_plus", "donor_minus"], "binary") for n in ("lh", "lm"))
    lab_h.swap = lab_m.swap = np.array([1, 0])
    b = XH.SpeciesStores(lab_h, {"mm": lab_m})
    assert b.ids == ["donor_plus", "donor_minus"] and b.positives.tolist() == [2, 4]
    with pytest.raises(ValueError, match="one kind"):
        XH.SpeciesStores(hs, {"mm": lab_m})
