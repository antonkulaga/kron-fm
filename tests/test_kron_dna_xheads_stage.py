"""kron_dna.xheads: the track store, a further head's loss (whole and from split pieces) and the stage's schedule."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch

from kron_dna import genome as G
from kron_dna import xheads as X

from test_kron_dna_split import _restore_precision, batch_for, forced_replicas, small, smooth_routing  # noqa: F401


def write_store(path, total=4096, ids=("a", "b_P", "b_M"), assays=("DNase-seq", "total RNA-seq", "total RNA-seq")):
    rng = np.random.default_rng(0)
    path.mkdir(parents=True)
    dense = rng.integers(0, 256, size=(total, len(ids)), dtype=np.uint8)
    dense[rng.random((total, len(ids))) < 0.6] = 0
    dense.tofile(path / "dense.u8")
    (path / "dense.json").write_text(json.dumps(dict(
        ids=list(ids), total_bases=total, n_tracks=len(ids), scales=[2.0, 3.0, 3.5], assays=list(assays),
        nonzero_mean=[0.8, 1.5, 1.2], mean_stride=64, encoding="uint8 = log1p(signal)/log1p_scale*255, clipped",
        genome="toy", tracks=[dict(unobserved_contigs=[]), dict(unobserved_contigs=["chrY:0-10"]), {}])))
    return dense


def consts(store, device="cpu"):
    return tuple(torch.as_tensor(v).to(device) for v in (store.scales, store.nonzero_mean, store.is_rna))


def test_store_reads_windows_and_flips_strand_partners(tmp_path):
    dense = write_store(tmp_path / "s")
    st = X.TrackStore.open(tmp_path / "s")
    assert st.ids == ["a", "b_P", "b_M"] and st.swap.tolist() == [0, 2, 1] and st.is_rna.tolist() == [False, True, True]
    w = st.window(100, 50, 64)
    assert w.shape == (64, 3) and np.array_equal(w[:50], dense[100:150]) and not w[50:].any()
    f = st.window(100, 50, 64, flip=True)
    assert np.array_equal(f[:50], dense[100:150][::-1][:, [0, 2, 1]]) and not f[50:].any()
    assert st.unobserved[1] == ["chrY:0-10"]


def test_store_identity_catches_another_store(tmp_path):
    write_store(tmp_path / "s")
    write_store(tmp_path / "t", ids=("a", "b_M", "b_P"))
    s, t = X.TrackStore.open(tmp_path / "s"), X.TrackStore.open(tmp_path / "t")
    assert s.differences(json.loads(json.dumps(s.identity))) == []
    assert any(d.startswith("ids") for d in s.differences(t.identity))
    (tmp_path / "s" / "dense.u8").write_bytes(b"\0" * 10)
    with pytest.raises(ValueError):
        X.TrackStore.open(tmp_path / "s")


def test_targets_invert_to_log1p_signal(tmp_path):
    write_store(tmp_path / "s")
    st = X.TrackStore.open(tmp_path / "s")
    c = consts(st)
    want = torch.from_numpy(st.window(0, 512, 512)).unsqueeze(0)
    back = X.signal_of(X.targets_of(want, *c), c[1], c[2])
    assert torch.allclose(back, want.float() / 255.0 * c[0], rtol=1e-4, atol=1e-4)


def test_supervised_keeps_the_training_split_the_centre_and_visible_bases():
    split = torch.tensor([[1, 1, 1, 1, 4, 1, 1, 1], [0, 1, 2, 1, 1, 3, 1, 1]])
    assert X.supervised(split).tolist() == [[1, 1, 1, 1, 0, 1, 1, 1], [0, 1, 0, 1, 1, 0, 1, 1]]
    assert X.supervised(split, center=0.5).tolist() == [[0, 0, 1, 1, 0, 1, 0, 0], [0, 0, 0, 1, 1, 0, 0, 0]]
    skip = torch.zeros(2, 8, dtype=torch.bool)
    skip[0, 2] = True
    assert X.supervised(split, skip=skip)[0].tolist() == [1, 1, 0, 1, 0, 1, 1, 1]


UPSTREAM = ("stem", "stem_res", "down", "core", "context")


def same_gradients(m, g_whole, device):
    """Every parameter's split gradient against the whole window's, as a share of the gradient's own scale: 1e-4,
    as test_kron_dna_split. On CUDA the parameters upstream of the core get 5e-3: what reaches them from a head's
    loss alone passes back through the KronLSTM kernels, which are not bit-repeatable (a whole pass against itself
    differs by 1e-5 there) and amplify a 2e-7 change of the stem weights to 1e-4..4e-4, and to about 1e-3 when a
    test process compiled them before TRITON_F32_DEFAULT=ieee was set. Measured 18 Nov 2026 on an A100: with IEEE
    kernels split and whole differ by 3e-6..1e-5, the same as two whole passes; the heads and the decoder agree
    to 1e-6 either way. The CPU stand-in is exact, so there the 1e-4 holds for all."""
    for n, g in g_whole.items():
        loose = device.type == "cuda" and n.split(".")[0] in UPSTREAM
        scale = float(g.abs().max()) + 1e-12
        assert float((m.get_parameter(n).grad - g).abs().max()) <= (5e-3 if loose else 1e-4) * scale + 1e-9, n


@pytest.mark.parametrize("by,b,length", [("length", 2, 2048), (None, 4, 1024)])
def test_split_loss_and_gradient_equal_the_whole_window(device, monkeypatch, tmp_path, by, b, length):
    forced_replicas(monkeypatch)
    smooth_routing(monkeypatch)       # as test_kron_dna_split: a max-pool's near-tie must not reroute a gradient
    write_store(tmp_path / "s", total=b * length)
    st = X.TrackStore.open(tmp_path / "s")
    cfg = small("v2").config.to_dict() | {"xheads": [["meas", 3]]}
    torch.manual_seed(2)
    m = G.KronGenome(G.KronGenomeConfig(**cfg)).to(device).eval()      # no dropout: two forwards of one function
    seq, seg, ctx, extra = batch_for(b, length, device)
    want = torch.stack([torch.from_numpy(st.window(r * length, length, length)) for r in range(b)])
    split = torch.ones(b, length, dtype=torch.long)
    split[0, :100] = 4
    ok = X.supervised(split, center=0.75)
    m(seq, seg, ctx, extra)
    whole = X.head_loss(m.last_x["meas"], want.to(device), ok.to(device), consts(st, device))
    whole.backward()
    g_whole = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    m.zero_grad()
    pieces, _ = m.forward_split(seq, seg, ctx, extra, devices=[device, device], by=by)
    parts = X.split_head_loss(pieces, "meas", want, ok, lambda d: consts(st, d), device)
    parts.backward()
    assert torch.allclose(parts, whole, rtol=1e-4, atol=1e-6)
    same_gradients(m, g_whole, device)


def test_pearson_matches_numpy_at_bins():
    g = torch.Generator().manual_seed(0)
    x, y = torch.rand(2, 1024, 3, generator=g), torch.rand(2, 1024, 3, generator=g)
    y[..., 0] = x[..., 0] * 2 + 1
    y[..., 2] = 0.5
    p = X.Pearson(3, bin=128)
    p.add(x[:1], y[:1])
    p.add(x[1:], y[1:])
    r = p.value()
    xb, yb = (v.reshape(2, 8, 128, 3).mean(2).reshape(-1, 3).numpy() for v in (x, y))
    assert abs(r[0] - 1.0) < 1e-5 and abs(r[1] - np.corrcoef(xb[:, 1], yb[:, 1])[0, 1]) < 1e-5 and np.isnan(r[2])
    assert np.isnan(X.Pearson(3).value()).all()


PLAN = dict(heads=["dnase", "splice"], head_only=3, lr_inherited=4e-5, lr_head=4e-4, warmup_inherited=4,
            warmup_head=2)


def test_stage_goes_head_only_then_joint_with_a_rate_per_group():
    st = X.Stage(X.StagePlan(**PLAN))
    assert st.state["phase"] == "head-only" and st.active == [] and st.lr_head("dnase") == 0.0
    st.activate("dnase", {"ids": ["a"]})
    assert st.lr_inherited() == 0.0 and st.lr_head("dnase") == pytest.approx(2e-4) and st.lr_head("splice") == 0.0
    assert st.stepped(["dnase"]) is None and st.lr_head("dnase") == pytest.approx(4e-4)
    assert st.stepped([]) is None                               # a step without labels moves no head's count
    assert st.state["updates"]["dnase"] == 1 and st.state["stage_step"] == 2
    assert st.stepped(["dnase"]) is None and st.stepped(["dnase"]) == "joint"
    assert st.inherited_trains and st.lr_inherited() == pytest.approx(1e-5)
    for want in (2e-5, 3e-5, 4e-5, 4e-5):
        st.stepped(["dnase"])
        assert st.lr_inherited() == pytest.approx(want)
    # a head whose data arrives later starts its own warmup while the others keep their schedules
    st.activate("splice", {"ids": ["donor", "acceptor"]})
    assert st.state["active"]["splice"] == st.state["stage_step"] == 8
    assert st.lr_head("splice") == pytest.approx(2e-4) and st.lr_head("dnase") == pytest.approx(4e-4)
    assert st.stepped(["dnase", "splice"]) is None and st.lr_head("splice") == pytest.approx(4e-4)


def test_stage_resumes_to_the_same_phase_rates_and_stores():
    a = X.Stage(X.StagePlan(**PLAN))
    a.activate("dnase", {"ids": ["a"]})
    for _ in range(5):
        a.stepped(["dnase"])
    saved = json.loads(json.dumps(a.state))                     # what a checkpoint holds
    b = X.Stage(X.StagePlan(**PLAN), copy.deepcopy(saved))
    b.activate("dnase", {"ids": ["a"]})                         # the same store again: accepted, not re-activated
    assert b.state == saved
    for _ in range(4):
        assert (a.lr_inherited(), a.lr_head("dnase"), a.state["phase"]) == \
               (b.lr_inherited(), b.lr_head("dnase"), b.state["phase"])
        a.stepped(["dnase"])
        b.stepped(["dnase"])
    with pytest.raises(ValueError):
        b.activate("dnase", {"ids": ["other"]})
    with pytest.raises(ValueError):
        X.Stage(X.StagePlan(**(PLAN | {"heads": ["dnase"]})), copy.deepcopy(saved))
    with pytest.raises(ValueError):
        b.activate("unknown", {})


def test_stage_scales_the_run_schedule_and_can_skip_the_head_only_phase():
    st = X.Stage(X.StagePlan(**(PLAN | {"head_only": 0, "warmup_inherited": 1, "warmup_head": 1})))
    assert st.state["phase"] == "joint"
    st.activate("dnase", {})
    assert st.lr_inherited() == pytest.approx(4e-5) and st.lr_inherited(0.5) == pytest.approx(2e-5)
    assert st.lr_head("dnase", 0.25) == pytest.approx(1e-4) and st.lr_head("splice", 0.25) == 0.0
    opt = torch.optim.AdamW([dict(params=[torch.nn.Parameter(torch.zeros(1))], role=r)
                             for r in ("inherited", "dnase", "splice")], lr=1.0)
    assert X.set_rates(opt, st, ["dnase"], 0.5) == {"inherited": pytest.approx(2e-5), "dnase": pytest.approx(2e-4),
                                                    "splice": 0.0}


def test_groups_cover_every_parameter_once_and_rates_follow_the_stage(device):
    cfg = small("v2").config.to_dict() | {"xheads": [["dnase", 3], ["splice", 2]]}
    m = G.KronGenome(G.KronGenomeConfig(**cfg)).to(device)
    groups = X.param_groups(m)
    seen = [id(p) for g in groups for p in g["params"]]
    assert len(seen) == len(set(seen)) == len(list(m.parameters()))
    assert [g["role"] for g in groups] == ["inherited"] * 2 + ["dnase"] * 2 + ["splice"] * 2
    opt = torch.optim.AdamW(groups, lr=0.0)
    st = X.Stage(X.StagePlan(**PLAN))
    st.activate("dnase", {})
    X.freeze(m, st)
    assert not m.stem.weight.requires_grad and all(p.requires_grad for p in m.xheads.parameters())
    rates = X.set_rates(opt, st, ["dnase"])
    assert rates == {"inherited": 0.0, "dnase": pytest.approx(2e-4), "splice": 0.0}
    assert [g["lr"] for g in opt.param_groups] == [0.0, 0.0, rates["dnase"], rates["dnase"], 0.0, 0.0]
    before = copy.deepcopy(m.state_dict())
    m.x_active = {"dnase"}
    seq, seg, ctx, extra = batch_for(1, 512, device)
    m(seq, seg, ctx, extra)
    m.last_x["dnase"].pow(2).mean().backward()
    opt.step()
    moved = {n for n, v in m.state_dict().items() if not torch.equal(v, before[n])}
    assert moved and all(n.startswith("xheads.dnase.") for n in moved)
    for _ in range(3):
        st.stepped(["dnase"])
    X.freeze(m, st)
    assert m.stem.weight.requires_grad and X.set_rates(opt, st, [])["dnase"] == 0.0


# ------------------------------------------------------------------------------------- 0/1 labels (splice sites)

BITS = {"donor_plus": 0, "acceptor_plus": 1, "donor_minus": 2, "acceptor_minus": 3}


def write_labels(path, total=4096, rate=0.02):
    rng = np.random.default_rng(1)
    path.mkdir(parents=True)
    lab = np.zeros(total, dtype=np.uint8)
    for b in range(4):
        lab[rng.random(total) < rate] |= 1 << b
    lab.tofile(path / "splice.u8")
    (path / "splice.json").write_text(json.dumps(dict(
        genome="toy", total_bases=total, bits=BITS, definition="one base per site, per strand",
        check={k: dict(sites=int(((lab >> b) & 1).sum())) for k, b in BITS.items()}, splice_sha256="x")))
    return lab


def test_label_store_reads_bits_and_exchanges_strands(tmp_path):
    lab = write_labels(tmp_path / "s")
    st = X.open_store(tmp_path / "s")
    assert st.kind == "binary" and st.ids == list(BITS) and st.swap.tolist() == [2, 3, 0, 1]
    w = st.window(10, 100, 128)
    assert w.shape == (128, 4) and not w[100:].any()
    for b in range(4):
        assert np.array_equal(w[:100, b], (lab[10:110] >> b) & 1)
    f = st.window(10, 100, 128, flip=True)
    assert np.array_equal(f[:100], w[:100][::-1][:, [2, 3, 0, 1]])
    assert st.differences(json.loads(json.dumps(st.identity))) == []
    assert st.differences(st.identity | {"definition": "two bases"}) != []
    (w_,) = st.consts("cpu")
    assert w_.shape == (4,) and float(w_.min()) >= 1 and float(w_.max()) <= 100
    write_store(tmp_path / "t")
    assert X.open_store(tmp_path / "t").kind == "tracks"
    with pytest.raises(ValueError):
        X.open_store(tmp_path)


@pytest.mark.parametrize("by,b,length", [("length", 2, 2048), (None, 4, 1024)])
def test_binary_split_loss_and_gradient_equal_the_whole_window(device, monkeypatch, tmp_path, by, b, length):
    forced_replicas(monkeypatch)
    smooth_routing(monkeypatch)
    write_labels(tmp_path / "s", total=b * length)
    st = X.open_store(tmp_path / "s")
    cfg = small("v2").config.to_dict() | {"xheads": [["splice", 4]]}
    torch.manual_seed(2)
    m = G.KronGenome(G.KronGenomeConfig(**cfg)).to(device).eval()
    seq, seg, ctx, extra = batch_for(b, length, device)
    want = torch.stack([torch.from_numpy(st.window(r * length, length, length)) for r in range(b)])
    split = torch.ones(b, length, dtype=torch.long)
    split[0, :100] = 4
    ok = X.supervised(split, center=0.75)
    m(seq, seg, ctx, extra)
    whole = X.head_loss(m.last_x["splice"], want.to(device), ok.to(device), st.consts(device), "binary")
    whole.backward()
    g_whole = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    m.zero_grad()
    pieces, _ = m.forward_split(seq, seg, ctx, extra, devices=[device, device], by=by)
    parts = X.split_head_loss(pieces, "splice", want, ok, st.consts, device, "binary")
    parts.backward()
    assert torch.allclose(parts, whole, rtol=1e-4, atol=1e-6)
    same_gradients(m, g_whole, device)
    none = X.head_loss(m.last_x.get("splice", parts.new_zeros(b, length, 4)), want.to(device),
                       torch.zeros_like(ok).to(device), st.consts(device), "binary")
    assert float(none) == 0.0                                    # nothing supervised: zero, not NaN


def brute_ap(x, y):
    order = np.argsort(-x, kind="stable")
    hit = y[order]
    prec = np.cumsum(hit) / np.arange(1, len(hit) + 1)
    return float(prec[hit].mean())


def test_average_precision_is_exact_with_everything_kept_and_a_lower_bound_otherwise():
    g = torch.Generator().manual_seed(0)
    y = torch.rand(2, 4000, 3, generator=g) < 0.02
    x = torch.randn(2, 4000, 3, generator=g) + 2.5 * y.float()
    y[..., 2] = False
    full = X.AveragePrecision(3, keep=1.0)
    full.add(x[:1], y[:1].to(torch.uint8))
    full.add(x[1:], y[1:].to(torch.uint8))
    r = full.value()
    for k in (0, 1):
        assert r[k] == pytest.approx(brute_ap(x[..., k].reshape(-1).numpy(), y[..., k].reshape(-1).numpy()), abs=1e-9)
    assert np.isnan(r[2]) and full.n == int(y.sum())
    part = X.AveragePrecision(3, keep=0.05)
    part.add(x[:1], y[:1].to(torch.uint8))
    part.add(x[1:], y[1:].to(torch.uint8))
    p = part.value()
    assert all(p[k] <= r[k] + 1e-9 and p[k] > 0.8 * r[k] for k in (0, 1))
    perfect = X.AveragePrecision(1)
    perfect.add(torch.tensor([[[5.0], [4.0], [-1.0], [-2.0]]]), torch.tensor([[[1], [1], [0], [0]]], dtype=torch.uint8))
    assert perfect.value()[0] == pytest.approx(1.0)
