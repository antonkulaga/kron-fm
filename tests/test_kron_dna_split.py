"""Split-window training (``genome-train --split-cards``): one window's conv path divided over several devices.

Model level: :meth:`KronGenome.forward_split` against :meth:`KronGenome.forward` for v2 and v2l, by length (with a
halo) and by rows, in eval mode (every output) and in train mode (loss and every parameter's gradient), on the CPU
with two logical pieces on one device (the replicas forced to take real copies, so gradients cross a copy as they
do between cards), and on two CUDA devices when there are two. The halo's bound is shown to matter: without a halo
the outputs differ.

CLI level: ``genome-train --bench --bench-grads`` with the long runs' options (bert masks, reverse complement, mito,
--genome, --ntv3-extra with the Poisson-multinomial loss, cell types, JEPA with SIGReg, soft bases, the repeat
channel, core checkpointing, grad accumulation) gives the same first-step loss and gradients split and unsplit, for
batch 1 (by length), an even batch (by rows) and an odd batch (by length); and a checkpoint written unsplit resumes
split (and back) with the same metrics.
"""
from __future__ import annotations

import json
import shutil

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from kron_dna import genome as G

import test_kron_dna_multigenome as MGT
from test_kron_dna_multigenome import cpu_only, genomes, run  # noqa: F401  (genomes is a fixture)

REAL_GENOME = G.KronGenome
REAL_MAX_POOL = F.max_pool1d


class _SmoothRoutedPool(torch.autograd.Function):
    """max_pool1d(x, 2) exactly in the forward pass; in the backward pass each pooled gradient is shared between the
    pair by sigmoid((a - b) / tau), tau = 1e-3 of the pair's size, instead of going wholly to the larger element.

    Why the gradient tests need it: max-pool's gradient is discontinuous where the two elements of a pair are
    (near-)equal. Two computations that differ by fp32 round-off (any change of kernel shape does it, e.g. a batch
    of 1 vs 2; measured: forward outputs within 5e-7) can then send one pair's gradient to the other base, and a
    single such flip moves the stem's and first block's weight gradients by ~5e-3 of their maximum (found on the
    cluster's torch 2.11 CPU: one flipped pair, no exact tie; see test_unsplit_max_pool_routing_is_discontinuous_at_near_ties). Both
    routings are valid subgradients; the unsplit model alone flips the same way. Routing smoothly, identically in
    both runs, keeps every comparison continuous, so round-off stays round-off and anything the split itself got
    wrong would still show at full size."""

    @staticmethod
    def forward(ctx, x):
        a, b = x[..., 0::2], x[..., 1::2]
        ctx.save_for_backward(a, b)
        return torch.maximum(a, b)

    @staticmethod
    def backward(ctx, g):
        a, b = ctx.saved_tensors
        tau = 1e-3 * torch.maximum(a.abs(), b.abs()) + 1e-12
        w = torch.sigmoid((a - b) / tau)
        out = g.new_empty(g.shape[:-1] + (2 * g.shape[-1],))
        out[..., 0::2] = g * w
        out[..., 1::2] = g * (1 - w)
        return out


def smooth_pool(x, kernel_size, *args, **kwargs):
    if kernel_size == 2 and not args and not kwargs and x.shape[-1] % 2 == 0:
        return _SmoothRoutedPool.apply(x)
    return REAL_MAX_POOL(x, kernel_size, *args, **kwargs)


def smooth_routing(monkeypatch):
    """Gradient tests: the model's max-pools route their gradient continuously (see _SmoothRoutedPool)."""
    monkeypatch.setattr(G.F, "max_pool1d", smooth_pool)


@pytest.fixture(autouse=True)
def _restore_precision():
    """Tests here may turn TF32 off; put both switches back afterwards."""
    prev, cudnn_tf32 = torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32
    yield
    torch.set_float32_matmul_precision(prev)
    torch.backends.cudnn.allow_tf32 = cudnn_tf32


# ------------------------------------------------------------------------------------------------------- the plan

def test_split_plan():
    p = G.split_plan(1, 16 * 128, 2, 128, 256)
    assert [(q[0], q[2], q[3], q[4], q[5]) for q in p] == [("length", 0, 1024, 0, 1280), ("length", 1024, 2048, 768, 2048)]
    p = G.split_plan(3, 3 * 128 * 5, 3, 128, 128, by="length")
    assert [q[2] % 128 for q in p] == [0, 0, 0] and p[-1][3] == 1920 and all(q[1] == slice(0, 3) for q in p)
    p = G.split_plan(4, 1024, 2, 128, 256)
    assert [(q[0], q[1], q[4], q[5]) for q in p] == [("rows", slice(0, 2), 0, 1024), ("rows", slice(2, 4), 0, 1024)]
    assert G.split_plan(4, 1024, 2, 128, 256, by="length")[0][0] == "length"
    with pytest.raises(ValueError):
        G.split_plan(3, 1024, 2, 128, 128, by="rows")
    with pytest.raises(ValueError):
        G.split_plan(1, 1024, 2, 128, 100)
    # the reach bound: v2 and v2l at 128-bp bins, and the halo is a multiple of the bin
    assert G.split_reach(G.KronGenomeConfig(arch="v2")) == 1145
    assert G.split_reach(G.KronGenomeConfig(arch="v2l")) == 889
    assert G.split_halo(G.KronGenomeConfig(arch="v2")) == 1152 and G.split_halo(G.KronGenomeConfig(arch="v2l")) == 896
    with pytest.raises(ValueError):
        G.split_reach(G.KronGenomeConfig(arch="v1"))


def test_forward_split_refuses_v1():
    m = G.KronGenome(G.KronGenomeConfig(d_model=32, n_heads=2, n_layers=1, stem_channels=8, n_down=4, arch="v1"))
    with pytest.raises(ValueError, match="GroupNorm"):
        m.forward_split(torch.zeros(1, 256, dtype=torch.long), devices=["cpu", "cpu"])


# ------------------------------------------------------------------------------- the Poisson-multinomial pieces

@pytest.mark.parametrize("cuts", [[0, 1000], [0, 1, 999, 1000], [0, 333, 1000]])
def test_poisson_parts_combine_to_the_loss(cuts):
    g = torch.Generator().manual_seed(3)
    pred = torch.rand(3, 1000, 7, generator=g) * 2 + 1e-3
    tgt = torch.rand(3, 1000, 7, generator=g) * 4 * (torch.rand(3, 1000, 7, generator=g) > 0.7)
    mask = (torch.rand(3, 1000, 1, generator=g) > 0.4).float() * (torch.rand(3, 1, 7, generator=g) > 0.3).float()
    mask[1] = 0                                                       # a row with no supervised value at all
    ref = G.poisson_multinomial_loss(pred, tgt, mask)
    by_len = [(slice(0, 3), G.poisson_multinomial_parts(pred[:, a:b], tgt[:, a:b], mask[:, a:b]))
              for a, b in zip(cuts[:-1], cuts[1:])]
    assert torch.allclose(G.poisson_multinomial_combine(by_len, 3, "cpu"), ref, rtol=1e-6, atol=0)
    by_rows = [(slice(r, r + 1), G.poisson_multinomial_parts(pred[r:r + 1], tgt[r:r + 1], mask[r:r + 1]))
               for r in range(3)]
    assert torch.allclose(G.poisson_multinomial_combine(by_rows, 3, "cpu"), ref, rtol=1e-6, atol=0)
    pg = pred.clone().requires_grad_(True)                            # and its gradient
    G.poisson_multinomial_loss(pg, tgt, mask).backward()
    ps = pred.clone().requires_grad_(True)
    G.poisson_multinomial_combine([(slice(0, 3), G.poisson_multinomial_parts(ps[:, a:b], tgt[:, a:b], mask[:, a:b]))
                                   for a, b in zip(cuts[:-1], cuts[1:])], 3, "cpu").backward()
    assert torch.allclose(ps.grad, pg.grad, rtol=1e-5, atol=1e-9)
    zero = torch.zeros_like(mask)                                     # nothing supervised: zero, not NaN
    assert float(G.poisson_multinomial_combine([(slice(0, 3), G.poisson_multinomial_parts(pred, tgt, zero))],
                                               3, "cpu")) == float(G.poisson_multinomial_loss(pred, tgt, zero)) == 0.0


# --------------------------------------------------------------------------------------------- the model level

def small(arch, **kw):
    cfg = G.KronGenomeConfig(d_model=32, n_heads=2, n_layers=2, ff_units=64, stem_channels=8, n_down=4, arch=arch,
                             n_ntv3=5, n_epigenomes=3, n_contexts=8, n_in_extra=2, jepa=True, **kw)
    torch.manual_seed(1)
    m = G.KronGenome(cfg)
    with torch.no_grad():                    # zero-initialised rows and FiLM would hide a missing contribution
        for p in m.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return m


def batch_for(b, length, device, soft=False):
    g = torch.Generator().manual_seed(b * 1000 + length)
    seq = torch.randint(0, 6, (b, length), generator=g)
    if soft:                                                          # fractional bases [B, 6, L]
        seq = G.soft_base_channels(seq.clamp(max=4), None, 0.3, generator=g)
    seg = torch.zeros(b, length // 16, dtype=torch.long)
    seg[0, -5:] = 1                                                   # padding: its own segment
    return (seq.to(device), seg.to(device), torch.randint(0, 8, (b, 2), generator=g).to(device),
            torch.randn(b, 2, length, generator=g).to(device))


CASES = [(1, 4096, None), (3, 2048, None), (2, 2048, "length"), (4, 1024, None)]


def forced_replicas(monkeypatch):
    """forward_split takes real parameter copies even on one device, as it does on a second card. On CUDA, TF32
    is turned off: cuDNN's convolutions default to it (~1e-3 relative), which would hide round-off behind it."""
    torch.backends.cudnn.allow_tf32 = False                           # restored by _restore_precision
    torch.set_float32_matmul_precision("highest")
    monkeypatch.setenv("TRITON_F32_DEFAULT", "ieee")                 # and in the KronLSTM kernels' tl.dot
    real = G.KronGenome.split_replica
    monkeypatch.setattr(G.KronGenome, "split_replica", lambda self, d, force=False: real(self, d, force=True))


@pytest.mark.parametrize("arch", ["v2", "v2l"])
@pytest.mark.parametrize("b,length,by", CASES)
def test_forward_split_equals_forward_eval(device, monkeypatch, arch, b, length, by):
    forced_replicas(monkeypatch)
    m = small(arch).to(device).eval()
    seq, seg, ctx, extra = batch_for(b, length, device, soft=b == 3)
    with torch.no_grad():
        logits, core, _ = m(seq, seg, ctx, extra)
        ntv3, epi = m.last_ntv3, m.last_epi
        pieces, core_s = m.forward_split(seq, seg, ctx, extra, devices=[device, device], by=by)
        assert pieces[0].rows == (slice(0, b // 2) if b % 2 == 0 and by is None else slice(0, b))
        tol = dict(rtol=1e-4, atol=1e-5)
        assert torch.allclose(G.gather_split(pieces, "logits", device), logits, **tol)
        assert torch.allclose(G.gather_split(pieces, "ntv3", device), ntv3, **tol)
        assert torch.allclose(core_s, core, **tol) and torch.allclose(m.last_epi, epi, **tol)
        assert m.last_ntv3 is None
        _, core_only = m.forward_split(seq, seg, ctx, extra, devices=[device, device], by=by, core_only=True)
        assert torch.allclose(core_only, core, **tol)
        if (by or ("rows" if b % 2 == 0 else "length")) == "length":  # the halo is what makes it exact
            bad, _ = m.forward_split(seq, seg, ctx, extra, devices=[device, device], by=by, halo=0)
            assert not torch.allclose(G.gather_split(bad, "logits", device), logits, rtol=1e-2, atol=1e-2)


def split_loss(m, out, targets, want, mask, epi_t, want_core):
    """A training loss with every head the long runs train: masked bases, Poisson tracks, cell types, JEPA."""
    logits, ntv3, core = out
    loss = F.cross_entropy(logits.float().reshape(-1, 4), targets.reshape(-1), ignore_index=-100)
    loss = loss + 0.5 * G.poisson_multinomial_loss(F.softplus(ntv3.float()), want, mask)
    loss = loss + F.binary_cross_entropy_with_logits(m.last_epi.float(), epi_t)
    loss = loss + (1 - F.cosine_similarity(m.last_jepa.float(), want_core, dim=-1)).mean()
    return loss + core.float().pow(2).mean() * 1e-3


@pytest.mark.parametrize("arch", ["v2", "v2l"])
@pytest.mark.parametrize("b,length,by", CASES)
def test_forward_split_gradients_equal_forward(device, monkeypatch, arch, b, length, by):
    """Train mode (dropout on, same seed): the loss and every parameter's gradient agree; a parameter that gets no
    gradient unsplit gets none split either. Max-pool gradients are routed continuously (see _SmoothRoutedPool)."""
    forced_replicas(monkeypatch)
    smooth_routing(monkeypatch)
    seq, seg, ctx, extra = batch_for(b, length, device)
    g = torch.Generator().manual_seed(9)
    targets = torch.where(torch.rand(b, length, generator=g) < 0.15, torch.randint(0, 4, (b, length), generator=g),
                          torch.full((b, length), -100)).to(device)
    want = (torch.rand(b, length, 5, generator=g) * 3).to(device)
    mask = (torch.rand(b, length, 1, generator=g) > 0.3).float().to(device)
    epi_t = (torch.rand(b, length // 16, 3, generator=g) > 0.5).float().to(device)
    want_core = torch.randn(b, length // 16, 32, generator=g).to(device)
    res = []
    for split in (False, True):
        m = small(arch).to(device).train()
        m.checkpoint_core = split                          # recomputation in backward changes nothing either
        torch.manual_seed(5)
        if split:
            pieces, core = m.forward_split(seq, seg, ctx, extra, devices=[device, device], by=by)
            out = (G.gather_split(pieces, "logits", device), G.gather_split(pieces, "ntv3", device), core)
        else:
            logits, core, _ = m(seq, seg, ctx, extra)
            out = (logits, m.last_ntv3, core)
        loss = split_loss(m, out, targets, want, mask, epi_t, want_core)
        loss.backward()
        res.append((float(loss.detach()), {n: p.grad for n, p in m.named_parameters()}))
    (l0, g0), (l1, g1) = res
    assert l1 == pytest.approx(l0, rel=1e-5)
    assert g0.keys() == g1.keys()
    for n in g0:
        assert (g0[n] is None) == (g1[n] is None), n
        if g0[n] is not None:
            scale = float(g0[n].abs().max()) + 1e-12
            # on the card the KronLSTM kernels are not bit-repeatable and their round-off grows through the layers:
            # split and unsplit differ by up to ~1.1e-3 of a gradient's scale there (measured 19 Nov, e.g.
            # down.1.norm2.weight), against ~1e-6 on the CPU, so the card gets 5e-3 and the CPU keeps 1e-4
            rel = 5e-3 if device.type == "cuda" else 1e-4
            assert float((g0[n] - g1[n]).abs().max()) <= rel * scale + 1e-9, n


def cpu_mixer(monkeypatch):
    import kron_core.blocks
    from kron_core.testing import CausalMixer
    monkeypatch.setattr(kron_core.blocks, "kronlstm_mixer", CausalMixer)


def test_smooth_routed_pool_forward_is_max_pool():
    x = torch.randn(3, 5, 64, dtype=torch.float64, requires_grad=True)
    y = smooth_pool(x, 2)
    assert torch.equal(y, F.max_pool1d(x, 2))
    y.sum().backward()                                   # every pooled gradient is shared out in full
    assert torch.allclose(x.grad[..., 0::2] + x.grad[..., 1::2], torch.ones(3, 5, 32, dtype=torch.float64))
    far = torch.tensor([[[1.0, 2.0, 5.0, -1.0]]], requires_grad=True)
    smooth_pool(far, 2).sum().backward()                 # away from ties it is max-pool's own routing
    assert torch.allclose(far.grad, torch.tensor([[[0.0, 1.0, 1.0, 0.0]]]))


def test_unsplit_max_pool_routing_is_discontinuous_at_near_ties():
    """The effect the gradient tests guard against, without any split: a pair within round-off of a tie sends its
    whole gradient to whichever element is larger by one ulp."""
    a = torch.tensor([[[1.0, 1.0 + 2 ** -23]]], requires_grad=True)
    b = torch.tensor([[[1.0 + 2 ** -23, 1.0]]], requires_grad=True)
    F.max_pool1d(a, 2).sum().backward()
    F.max_pool1d(b, 2).sum().backward()
    assert torch.equal(a.grad, torch.tensor([[[0.0, 1.0]]])) and torch.equal(b.grad, torch.tensor([[[1.0, 0.0]]]))


def test_unused_head_gets_no_gradient_split_either(monkeypatch):
    """A head whose output is not in the loss (e.g. the track head on a window without tracks) has gradient None
    unsplit; the replica's copies must not hand its parameters a zero (AdamW would then decay them)."""
    forced_replicas(monkeypatch)
    cpu_mixer(monkeypatch)
    m = small("v2").train()
    seq, seg, ctx, extra = batch_for(1, 2048, "cpu")
    pieces, core = m.forward_split(seq, seg, ctx, extra, devices=["cpu", "cpu"])
    sum(p.logits.float().sum() for p in pieces).backward()
    assert all(p.grad is None for p in m.ntv3.parameters())
    assert m.stem.weight.grad is not None and m.stem.weight.grad.abs().sum() > 0
    assert m.down[0].conv1.weight.grad.abs().sum() > 0 and m.head.fc2.weight.grad.abs().sum() > 0


def test_split_adds_no_state(monkeypatch):
    cpu_mixer(monkeypatch)
    m = small("v2l")
    keys = list(m.state_dict())
    seq, seg, ctx, extra = batch_for(1, 1024, "cpu")
    m.forward_split(seq, seg, ctx, extra, devices=["cpu", "cpu"])
    rep = m.split_replica("cpu", force=True)
    assert list(m.state_dict()) == keys and rep.stem.weight is not m.stem.weight
    for name in G.KronGenome.SPLIT_LOCAL:            # the conv path's copies are plain tensors, not parameters
        if getattr(m, name, None) is not None:
            assert list(getattr(rep, name).parameters()) == [], name
    assert rep.core is m.core and "skip_scale" not in dict(rep.named_parameters(recurse=False))


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason="needs two CUDA devices")
@pytest.mark.parametrize("arch", ["v2", "v2l"])
def test_forward_split_two_cards(arch):
    """Two real cards: outputs on each piece's own card, gradients back on the model's card, equal to unsplit."""
    pytest.importorskip("triton")
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    import os
    os.environ["TRITON_F32_DEFAULT"] = "ieee"
    d0, d1 = torch.device("cuda", 0), torch.device("cuda", 1)
    seq, seg, ctx, extra = batch_for(1, 4096, d0)
    res = []
    for split in (False, True):
        m = small(arch).to(d0).train()
        torch.manual_seed(5)
        if split:
            pieces, core = m.forward_split(seq, seg, ctx, extra, devices=[d0, d1])
            assert [p.device for p in pieces] == [d0, d1] and pieces[1].logits.device == d1
            loss = sum(p.logits.float().pow(2).sum().to(d0) + p.ntv3.float().pow(2).sum().to(d0) for p in pieces)
        else:
            logits, core, _ = m(seq, seg, ctx, extra)
            loss = logits.float().pow(2).sum() + m.last_ntv3.float().pow(2).sum()
        loss = loss / 4096 + core.float().pow(2).mean()
        loss.backward()
        res.append((float(loss.detach()), {n: p.grad for n, p in m.named_parameters()}))
    assert res[1][0] == pytest.approx(res[0][0], rel=1e-4)
    for n, g in res[0][1].items():
        g1 = res[1][1][n]
        assert (g is None) == (g1 is None), n
        if g is not None:
            assert g1.device == d0
            assert float((g - g1).abs().max()) <= 1e-3 * (float(g.abs().max()) + 1e-12) + 1e-8, n


# ----------------------------------------------------------------------------------------------- the CLI level

def make_recipe_data(gd):
    """What the long runs feed besides bases: dense NTv3 tracks with non-zero means (alphagenome target, Poisson)
    for the primary and one further species, stranded pairs for --rc, the Regulatory Build cell-type tables, and
    repeat.u8 for every genome."""
    rng = np.random.default_rng(11)
    root, hg = gd["root"], gd["hgt"]
    n_h = int(np.fromfile(hg / "bases.u8", dtype=np.uint8).size)
    nd = root / "hg_ntv3"
    nd.mkdir()
    ids = ["A1", "R1_P", "R1_M", "H1"]
    tracks = [{"file_id": "A1", "assay": "ATAC-seq", "log1p_scale": 5.0},
              {"file_id": "R1_P", "assay": "total RNA-seq", "log1p_scale": 6.0},
              {"file_id": "R1_M", "assay": "total RNA-seq", "log1p_scale": 6.0},
              {"file_id": "H1", "assay": "Histone ChIP-seq", "log1p_scale": 4.0}]
    (nd / "tracks.json").write_text(json.dumps({"tracks": tracks}))
    dense = (rng.integers(0, 255, (n_h, 4)) * (rng.random((n_h, 4)) < 0.4)).astype(np.uint8)
    dense.tofile(nd / "dense.u8")
    (nd / "dense.json").write_text(json.dumps({"ids": ids, "total_bases": n_h, "n_tracks": 4,
                                               "nonzero_mean": [0.7, 1.3, 1.1, 0.4]}))
    split = np.ones(n_h, dtype=np.uint8)
    split[700:2500] = 3
    split.tofile(nd / "split.u8")
    xa = gd["gA"]
    n_x = int(np.fromfile(xa / "bases.u8", dtype=np.uint8).size)
    (xa / "ntv3").mkdir()
    (rng.integers(0, 255, (n_x, 3)) * (rng.random((n_x, 3)) < 0.5)).astype(np.uint8).tofile(xa / "ntv3" / "dense.u8")
    xs = np.ones(n_x, dtype=np.uint8)
    xs[2048:4096] = 3
    xs.tofile(xa / "ntv3" / "split.u8")
    (xa / "ntv3" / "dense.json").write_text(json.dumps({"ids": ["X1", "X2", "X3"], "total_bases": n_x, "n_tracks": 3,
                                                        "scales": [5.0, 5.0, 5.0], "assays": ["ATAC-seq"] * 3,
                                                        "nonzero_mean": [0.9, 0.5, 1.2]}))
    reg = np.full(n_h, -1, dtype=np.int32)                            # 6 elements, 5 epigenomes
    for k, (a, b) in enumerate([(100, 400), (1500, 1700), (3000, 3600), (5000, 5100), (8000, 8300), (11000, 11500)]):
        reg[a:b] = k
    reg.tofile(hg / "regfeat.i32")
    np.save(hg / "regfeat_active.npy", rng.random((6, 5)) < 0.5)
    for name in ("hgt", "gA", "gB", "mt"):
        b = np.fromfile(gd[name] / "bases.u8", dtype=np.uint8)
        (b == 0).astype(np.uint8).tofile(gd[name] / "repeat.u8")
    return nd


def recipe(nd, arch):
    return ["--name", "hgt", "--val-chroms", "chr8,chr9", "--arch", arch, "--n-down", "4", "--d-model", "96",
            "--n-layers", "2", "--stem-channels", "8", "--warmup", "1", "--seed", "5", "--bert-mask", "--rc",
            "--epigenome-weight", "1.0", "--epigenome-focus", "0.05", "--ntv3-dir", str(nd), "--ntv3-assays", "all",
            "--ntv3-target", "alphagenome", "--ntv3-loss", "poisson", "--ntv3-center", "0.375", "--ntv3-weight",
            "0.5", "--ntv3-extra", "gA", "--ntv3-extra-share", "0.3", "--mito", "mt", "--mito-share", "0.1",
            "--genome", "gB:0.1:64", "--n-contexts", "256", "--n-in-extra", "8", "--soft-base-rate", "0.05",
            "--repeat-channel", "--repeat-channel-drop", "0.5", "--jepa-weight", "1.0", "--jepa-ema", "0.0",
            "--jepa-reg", "sigreg"]


def real_model_cpu(monkeypatch):
    """genome_train on the CPU with the REAL KronGenome (the KronLSTM stand-in on CPU) and every piece forced
    to take real parameter copies."""
    cpu_only(monkeypatch)
    monkeypatch.setattr(G, "KronGenome", REAL_GENOME)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    import kron_core.blocks
    from kron_core.testing import CausalMixer
    monkeypatch.setattr(kron_core.blocks, "kronlstm_mixer", CausalMixer)
    forced_replicas(monkeypatch)


@pytest.mark.parametrize("arch", ["v2", "v2l"])
@pytest.mark.parametrize("window,tokens,accum", [(2048, 2048, 1), (1024, 4096, 1), (1024, 3072, 1), (1024, 4096, 2)])
def test_cli_split_first_step_loss_and_gradients(genomes, tmp_path, monkeypatch, arch, window, tokens, accum):
    """--bench --bench-grads: the first optimizer step's loss and every parameter's gradient, split over two pieces
    and unsplit, with the long runs' options; batch 1 (by length), 4 (by rows), 3 (by length), and 2 x 2 accumulated."""
    nd = make_recipe_data(genomes)
    real_model_cpu(monkeypatch)
    smooth_routing(monkeypatch)
    from kron_dna import cli
    args = recipe(nd, arch) + ["--curriculum", f"{window}:10", "--tokens-per-step", str(tokens),
                               "--grad-accum", str(accum), "--bench", "--checkpoint-core"]
    out = {}
    for tag, extra in (("one", []), ("split", ["--split-cards", "2"])):
        res, _ = run(cli, monkeypatch, genomes["root"], tmp_path / tag, args + extra
                     + ["--bench-grads", str(tmp_path / f"{tag}.pt")])
        assert res.exit_code == 0, res.output + repr(res.exception)
        out[tag] = torch.load(tmp_path / f"{tag}.pt")
        if extra:
            assert "split windows over 2 cards" in res.output and "split over 2 cards" in res.output
    a, b = out["one"], out["split"]
    assert len(a["losses"]) == accum and b["losses"] == pytest.approx(a["losses"], rel=1e-5)
    assert a["grads"].keys() == b["grads"].keys()
    n_grad = 0
    for n, g in a["grads"].items():
        assert (g is None) == (b["grads"][n] is None), n
        if g is not None:
            n_grad += 1
            assert float((g - b["grads"][n]).abs().max()) <= 1e-4 * (float(g.abs().max()) + 1e-12) + 1e-9, n
    assert n_grad > 50


@pytest.mark.parametrize("loss_args", [["--ntv3-loss", "mse", "--ntv3-target", "log1p", "--ntv3-focus", "0.3"],
                                       ["--ntv3-loss", "mse", "--ntv3-target", "alphagenome"]])
def test_cli_split_mse_track_losses(genomes, tmp_path, monkeypatch, loss_args):
    """The other --ntv3-loss forms (mse with per-species channels, with and without --ntv3-focus): same first-step
    loss and gradients split (batch 1 by length, batch 3 by length) as unsplit."""
    nd = make_recipe_data(genomes)
    real_model_cpu(monkeypatch)
    smooth_routing(monkeypatch)
    from kron_dna import cli
    args = recipe(nd, "v2")
    i = args.index("--ntv3-loss")
    args = args[:i] + args[i + 2:]
    i = args.index("--ntv3-target")
    args = args[:i] + args[i + 2:] + loss_args
    for tokens in (2048, 3072):
        base = args + ["--curriculum", f"{2048 if tokens == 2048 else 1024}:10", "--tokens-per-step", str(tokens),
                       "--bench"]
        out = {}
        for tag, extra in (("one", []), ("split", ["--split-cards", "2"])):
            res, _ = run(cli, monkeypatch, genomes["root"], tmp_path / f"{tag}{tokens}",
                         base + extra + ["--bench-grads", str(tmp_path / f"{tag}{tokens}.pt")])
            assert res.exit_code == 0, res.output + repr(res.exception)
            out[tag] = torch.load(tmp_path / f"{tag}{tokens}.pt")
        a, b = out["one"], out["split"]
        assert b["losses"] == pytest.approx(a["losses"], rel=1e-5)
        for n, g in a["grads"].items():
            assert (g is None) == (b["grads"][n] is None), n
            if g is not None:
                assert float((g - b["grads"][n]).abs().max()) <= 1e-4 * (float(g.abs().max()) + 1e-12) + 1e-9, n


def test_cli_resume_unsplit_checkpoint_split_and_back(genomes, tmp_path, monkeypatch):
    """A checkpoint written unsplit resumes with --split-cards 2 (same keys, same optimizer state) and gives the
    metrics an unsplit resumption gives; the split run's checkpoint then resumes unsplit."""
    nd = make_recipe_data(genomes)
    real_model_cpu(monkeypatch)
    from kron_dna import cli
    base = recipe(nd, "v2") + ["--tokens-per-step", "2048", "--eval-every", "2"]
    res, _ = run(cli, monkeypatch, genomes["root"], tmp_path / "start", base + ["--curriculum", "1024:2,2048:4", "--stop-step", "2"])
    assert res.exit_code == 0, res.output + repr(res.exception)
    cur = ["--curriculum", "1024:2,2048:4", "--stop-step", "4"]
    for tag in ("one", "split"):
        shutil.copytree(tmp_path / "start", tmp_path / tag)
        res, _ = run(cli, monkeypatch, genomes["root"], tmp_path / tag,
                     base + cur + (["--split-cards", "2"] if tag == "split" else []))
        assert res.exit_code == 0, res.output + repr(res.exception)
        assert "resumed at step 2" in res.output
    rows = {t: [r.split(",") for r in (tmp_path / t / "metrics.csv").read_text().strip().split("\n")[1:]]
            for t in ("one", "split")}
    assert [r[0] for r in rows["one"]] == [r[0] for r in rows["split"]] == ["2", "4"]
    head = (tmp_path / "one" / "metrics.csv").read_text().split("\n", 1)[0].split(",")
    for ra, rb in zip(rows["one"], rows["split"]):
        for h, x, y in zip(head, ra, rb):
            if h not in ("hours", "lr", "step") and x and y:
                assert float(y) == pytest.approx(float(x), rel=2e-4, abs=1e-6), h
    sa = torch.load(tmp_path / "one" / "checkpoint.pt", weights_only=False)
    sb = torch.load(tmp_path / "split" / "checkpoint.pt", weights_only=False)
    assert sa["step"] == sb["step"] == 4 and sa["model"].keys() == sb["model"].keys()
    for k in sa["model"]:
        assert sa["model"][k].shape == sb["model"][k].shape
        assert float((sa["model"][k] - sb["model"][k]).abs().max()) < 5e-3, k   # 2 AdamW steps at lr 3e-4
    m, _ = G.load_checkpoint(tmp_path / "split")                      # loads strictly, as any checkpoint
    assert m.n_parameters() == sum(v.numel() for v in sb["model"].values())
    res, _ = run(cli, monkeypatch, genomes["root"], tmp_path / "split", base + ["--curriculum", "1024:2,2048:4",
                                                                               "--stop-step", "5"])
    assert res.exit_code == 0 and "resumed at step 4" in res.output, res.output + repr(res.exception)


@pytest.mark.parametrize("extra,words", [(["--arch", "v1"], "--arch v1/v3"), (["--compile-convs"], "--compile-convs"),
                                         (["--whole-contigs"], "--whole-contigs"),
                                         (["--split-halo", "100"], "--split-halo"),
                                         (["--bench-grads", "x.pt"], None)])
def test_cli_split_refusals(genomes, tmp_path, monkeypatch, extra, words):
    real_model_cpu(monkeypatch)
    from kron_dna import cli
    args = MGT.SMALL + ["--steps", "1"] + extra + ([] if words is None else ["--split-cards", "2"])
    res, _ = run(cli, monkeypatch, genomes["root"], tmp_path / "r", args)
    assert res.exit_code != 0
    text = MGT.flat(res.output)
    assert (words or "--bench-grads needs --bench") in text, text


# ------------------------------------------------------------------- the local track (a 1-bp KronLSTM) split too

LOCAL = dict(local_width=16, local_layers=1, local_span=512, dropout=0.0)   # the track's dropout masks follow the piece sizes


def test_split_plan_aligns_to_the_local_span():
    plan = G.split_plan(1, 4096, 2, 16, 48, "length", align=512)
    assert [(s, e, lo, hi) for _, _, s, e, lo, hi in plan] == [(0, 2048, 0, 2560), (2048, 4096, 1536, 4096)]
    plan = G.split_plan(1, 3072, 2, 16, 0, "length", align=512)          # 6 segments: cut at 1536
    assert [(s, e) for _, _, s, e, _, _ in plan] == [(0, 1536), (1536, 3072)]
    assert G.split_plan(4, 1000 // 16 * 16, 2, 16, 0, None, align=512)[0][0] == "rows"   # rows need no alignment
    with pytest.raises(ValueError, match="local track"):
        G.split_plan(1, 2000 // 16 * 16, 2, 16, 0, "length", align=512)


@pytest.mark.parametrize("b,length,by", [(1, 4096, None), (2, 2048, "length"), (3, 2048, None)])
def test_forward_split_with_local_track_equals_forward(device, monkeypatch, b, length, by):
    """Eval and train mode: with a local track the split cuts at its restart points, so logits, tracks, core and
    every gradient (the track's included) agree with the unsplit forward."""
    forced_replicas(monkeypatch)
    smooth_routing(monkeypatch)
    seq, seg, ctx, extra = batch_for(b, length, device)
    m = small("v2", **LOCAL).to(device).eval()
    with torch.no_grad():
        logits, core, _ = m(seq, seg, ctx, extra)
        ntv3 = m.last_ntv3
        pieces, core_s = m.forward_split(seq, seg, ctx, extra, devices=[device, device], by=by)
    tol = dict(rtol=1e-4, atol=1e-5)
    assert torch.allclose(G.gather_split(pieces, "logits", device), logits, **tol)
    assert torch.allclose(G.gather_split(pieces, "ntv3", device), ntv3, **tol)
    assert torch.allclose(core_s, core, **tol)
    g = torch.Generator().manual_seed(9)
    targets = torch.where(torch.rand(b, length, generator=g) < 0.15, torch.randint(0, 4, (b, length), generator=g),
                          torch.full((b, length), -100)).to(device)
    want = (torch.rand(b, length, 5, generator=g) * 3).to(device)
    mask = (torch.rand(b, length, 1, generator=g) > 0.3).float().to(device)
    epi_t = (torch.rand(b, length // 16, 3, generator=g) > 0.5).float().to(device)
    want_core = torch.randn(b, length // 16, 32, generator=g).to(device)
    res = []
    for split in (False, True):
        m = small("v2", **LOCAL).to(device).train()
        with torch.no_grad():
            m.local_gate.fill_(0.5)                    # the gate starts at zero, which would hide the track
        torch.manual_seed(5)
        if split:
            pieces, core = m.forward_split(seq, seg, ctx, extra, devices=[device, device], by=by)
            out = (G.gather_split(pieces, "logits", device), G.gather_split(pieces, "ntv3", device), core)
        else:
            logits, core, _ = m(seq, seg, ctx, extra)
            out = (logits, m.last_ntv3, core)
        split_loss(m, out, targets, want, mask, epi_t, want_core).backward()
        res.append({n: p.grad for n, p in m.named_parameters()})
    g0, g1 = res
    assert any(n.startswith("local_core.") and g0[n] is not None and float(g0[n].abs().max()) > 0 for n in g0)
    for n in g0:
        assert (g0[n] is None) == (g1[n] is None), n
        if g0[n] is not None:
            scale = float(g0[n].abs().max()) + 1e-12
            rel = 5e-3 if device.type == "cuda" else 1e-4
            assert float((g0[n] - g1[n]).abs().max()) <= rel * scale + 1e-9, n


def test_split_replica_copies_every_module_a_piece_runs(monkeypatch):
    """Regression (24 Nov, two real cards): the replica did not copy the local track's modules, so piece 2 read
    weights left on the first card. Every parameter of a module a piece runs must be a copy in the replica (the
    replica holds its copies as plain attributes, resolved here by name)."""
    import kron_core.blocks
    from kron_core.testing import CausalMixer
    monkeypatch.setattr(kron_core.blocks, "kronlstm_mixer", CausalMixer)
    m = small("v2", **LOCAL)
    r = m.split_replica("cpu", force=True)

    def resolve(root, dotted):
        for part in dotted.split("."):
            root = root._modules[part] if part in root._modules else getattr(root, part)
        return root

    piece = ("stem", "down", "up", "head", "ntv3", "local_in", "local_core", "local_out", "local_gate")
    names = [n for n, _ in m.named_parameters() if n.split(".")[0] in piece]
    assert any(n.startswith("local_core.") for n in names)
    for n in names:
        assert resolve(r, n) is not resolve(m, n), n
