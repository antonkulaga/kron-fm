"""kron-ppg's encoder, objectives and age clock, on the CPU stand-in and on the GPU kernels."""
from __future__ import annotations

import math

import pytest
import torch

from kron_core.blocks import LayerStates
from kron_ppg.model import AgeHead, KronPPG, KronPPGConfig, PPGAgeClock, patchify

SMALL = dict(d_model=32, n_heads=2, n_layers=2, ff_units=64, dropout=0.0, patch=5, horizon=3, proj_dim=16)


def small_model(device, **over) -> KronPPG:
    torch.manual_seed(0)
    return KronPPG(KronPPGConfig(**{**SMALL, **over})).to(device).eval()


def pulse(batch: int, samples: int, fs: float = 25.0, seed: int = 0) -> torch.Tensor:
    """Scaled pulse-like signals: a fundamental at 55-95 bpm, a harmonic and a little noise."""
    gen = torch.Generator().manual_seed(seed)
    t = torch.arange(samples) / fs
    hr = 55 + 40 * torch.rand(batch, 1, generator=gen)
    phase = 2 * math.pi * torch.rand(batch, 1, generator=gen)
    w = 2 * math.pi * hr / 60
    x = torch.sin(w * t + phase) + 0.4 * torch.sin(2 * w * t + 2 * phase + 0.8)
    return x + 0.05 * torch.randn(batch, samples, generator=gen)


def tol(device) -> float:
    # the GPU path runs the kernels under bfloat16 autocast; its rounding differs between chunkings
    return 5e-2 if device.type == "cuda" else 1e-5


def test_patchify_drops_the_tail_and_marks_any_invalid_sample():
    x = torch.arange(12.0).view(1, 12)
    valid = torch.ones(1, 12, dtype=torch.bool)
    valid[0, 6] = False
    patches, ok = patchify(x, 5, valid)
    assert patches.shape == (1, 2, 5)
    assert ok.tolist() == [[True, False]]
    assert patches[0, 1].abs().sum() == 0


def test_a_stream_read_in_chunks_gives_what_it_gives_read_whole(device):
    model = small_model(device)
    patches, _ = patchify(pulse(2, 400).to(device), 5)
    with torch.no_grad():
        whole = model.encode(patches)
        states = LayerStates.empty(len(model.blocks))
        parts = [model.encode(patches[:, a:b], states=states) for a, b in ((0, 23), (23, 51), (51, 80))]
    assert torch.allclose(torch.cat(parts, 1), whole, atol=tol(device), rtol=0)


def test_the_encoder_is_causal(device):
    model = small_model(device)
    patches, _ = patchify(pulse(1, 300).to(device), 5)
    changed = patches.clone()
    changed[:, 40:] = torch.randn_like(changed[:, 40:])
    with torch.no_grad():
        a, b = model.encode(patches), model.encode(changed)
    assert torch.allclose(a[:, :40], b[:, :40], atol=tol(device), rtol=0)
    assert not torch.allclose(a[:, 40:], b[:, 40:], atol=1e-3)


def test_an_invalid_patch_is_taken_out_of_the_recurrence(device):
    """Dropped patches leave the stream as if they were never there: zero output, unchanged memory."""
    model = small_model(device)
    patches, _ = patchify(pulse(1, 300).to(device), 5)
    keep = torch.ones(1, patches.size(1), dtype=torch.bool, device=device)
    keep[:, 20:26] = False
    garbage = patches.clone()
    garbage[:, 20:26] = 50.0
    with torch.no_grad():
        with_gap = model.encode(garbage, keep)
        without = model.encode(patches[:, keep[0]])
    assert with_gap[:, 20:26].abs().max() == 0
    assert torch.allclose(with_gap[:, keep[0]], without, atol=tol(device), rtol=0)


def test_next_patch_loss_reads_only_valid_pairs_and_training_lowers_it(device):
    torch.manual_seed(0)
    model = KronPPG(KronPPGConfig(**SMALL)).to(device).train()
    patches, ok = patchify(pulse(8, 250, seed=1).to(device), 5)
    ok[:, 10:14] = False
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(40):
        h = model.encode(patches, ok)
        loss = model.next_patch_loss(h, patches, ok)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert math.isfinite(losses[0]) and losses[-1] < 0.7 * losses[0]
    # a pair whose target is invalid contributes nothing: garbage there does not move the loss
    with torch.no_grad():
        h = model.encode(patches, ok)
        base = model.next_patch_loss(h, patches, ok)
        junk = patches.clone()
        junk[:, 10:14] = 1e3
        assert torch.allclose(model.next_patch_loss(h, junk, ok), base)


def test_subject_contrast_prefers_a_batch_whose_pairs_agree(device):
    model = small_model(device)
    subject = torch.tensor([0, 0, 1, 1, 2, 2], device=device)
    base = torch.randn(3, 32, device=device)
    agree = base.repeat_interleave(2, 0)
    disagree = torch.randn(6, 32, device=device)
    with torch.no_grad():
        assert model.subject_loss(agree, subject) < model.subject_loss(disagree, subject)
        assert model.subject_loss(agree, torch.arange(6, device=device)) == 0


def test_the_age_clock_streamed_in_chunks_reads_the_age_it_reads_whole(device):
    clock = PPGAgeClock(small_model(device), AgeHead(32).to(device)).eval()
    torch.nn.init.normal_(clock.head.net[-1].weight, std=0.5)
    clock.head.set_scale(torch.tensor([30.0, 70.0]))
    x = pulse(3, 450).to(device)
    with torch.no_grad():
        whole = clock(x)
        patches, ok = patchify(x, 5)
        states = LayerStates.empty(len(clock.encoder.blocks))
        pool_sum = torch.zeros(3, 32, device=device)
        pool_count = torch.zeros(3, 1, device=device)
        for a in range(0, patches.size(1), 25):
            age, pool_sum, pool_count = clock.stream(patches[:, a:a + 25], ok[:, a:a + 25], states,
                                                     pool_sum, pool_count)
    assert whole.shape == (3,)
    assert torch.allclose(age, whole, atol=50 * tol(device), rtol=0)


def test_an_untrained_head_predicts_the_training_mean():
    head = AgeHead(16)
    head.set_scale(torch.tensor([20.0, 40.0, 60.0]))
    assert torch.allclose(head(torch.randn(4, 16)), torch.full((4,), 40.0))


def test_a_head_dimension_below_the_kernels_floor_is_refused():
    with pytest.raises(ValueError, match="head dimension"):
        KronPPG(KronPPGConfig(**{**SMALL, "n_heads": 4}))
