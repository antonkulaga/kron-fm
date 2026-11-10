"""Wrist age evaluation: ridge head, grouped CV, baselines and nulls on synthetic people."""
import numpy as np
import torch

from kron_ppg.model import KronPPG, KronPPGConfig
from kron_ppg.wrist_age import (People, cross_validate, embed, evaluate, fit_ridge, linear_head, per_person,
                                spread_index)


def _people(n_per_source=(60, 40), signal=1.0, source_shift=(0.0, 30.0), seed=0, d=16):
    rng = np.random.default_rng(seed)
    rows = []
    for k, (n, shift) in enumerate(zip(n_per_source, source_shift)):
        age = rng.uniform(20, 40, n) + shift
        x = rng.normal(size=(n, d))
        x[:, 0] += signal * (age - age.mean()) / 5          # age, within source
        x[:, 1] += 10.0 * k                                 # the device's fingerprint
        rows.append((np.array([f"s{k}/p{i}" for i in range(n)]), np.full(n, f"s{k}"), age, x))
    return People(*(np.concatenate(c) for c in zip(*rows)), windows=np.ones(sum(n_per_source), int))


def test_ridge_recovers_a_linear_map_and_folds_into_one_layer():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 8)) * rng.uniform(0.5, 3, 8) + 5
    y = x @ np.arange(8.0) + 2
    r = fit_ridge(x, y)
    assert np.abs(r.predict(x) - y).mean() < 0.5
    w, b = linear_head(r)
    np.testing.assert_allclose(x @ w + b, r.predict(x), rtol=1e-9, atol=1e-9)


def test_age_signal_beats_both_baselines_and_the_null():
    res, _ = evaluate(_people(signal=1.0), folds=5, repeats=2, null=5)
    pooled = res["pooled"]
    assert pooled["mae"] < pooled["mae_source_median"] < pooled["mae_constant"]
    assert pooled["interval95"]["model_minus_source_median"][1] < 0
    assert res["null"]["min"] > pooled["mae"]
    assert set(res["leave_source_out"]) == {"s0", "s1"}


def test_a_device_fingerprint_alone_does_not_beat_the_source_median():
    res, _ = evaluate(_people(signal=0.0), folds=5, repeats=2, null=5)
    pooled = res["pooled"]
    assert pooled["mae"] < pooled["mae_constant"]           # it finds the source...
    assert pooled["interval95"]["model_minus_source_median"][0] > -1.0   # ...and nothing more
    assert pooled["mae"] > 0.9 * pooled["mae_source_median"]


def test_every_person_is_predicted_out_of_fold():
    p = _people(signal=1.0)
    cv = cross_validate(p, folds=5, repeats=1)
    leak = cross_validate(p, folds=5, repeats=1, age=p.age)
    np.testing.assert_allclose(cv["pred"], leak["pred"])
    assert np.isfinite(cv["pred"]).all()


def test_spread_index_and_per_person_embedding(device):
    subjects = np.array(["a/1"] * 10 + ["b/2"] * 3)
    idx = spread_index(subjects, cap=4)
    assert list(idx[:4]) == [0, 3, 6, 9] and list(idx[4:]) == [10, 11, 12]
    enc = KronPPG(KronPPGConfig(sample_rate=25.0, patch=5, d_model=128, n_heads=2, n_layers=1, ff_units=256)).to(device)

    class FakePool:
        def window(self, i):
            return np.sin(np.arange(250) / 4.0 + i).astype(np.float32)

    emb = embed(enc, FakePool(), idx, batch_size=4, device=device)
    assert emb.shape == (7, 128) and np.isfinite(emb).all()
    p = per_person(subjects[idx], np.array([30.0] * 4 + [50.0] * 3), emb)
    assert list(p.source) == ["a", "b"] and list(p.windows) == [4, 3] and list(p.age) == [30.0, 50.0]
    torch.testing.assert_close(torch.tensor(p.x[0]), torch.tensor(emb[:4].mean(0), dtype=torch.float64))


def test_handcrafted_reads_the_pulse_rate():
    from kron_ppg.wrist_age import handcrafted

    fs = 25.0
    t = np.arange(int(30 * fs)) / fs
    x = np.stack([np.sin(2 * np.pi * bpm / 60 * t) for bpm in (55, 72, 110)])
    f = handcrafted(x, fs)
    assert f.shape == (3, 12) and np.isfinite(f).all()
    np.testing.assert_allclose(f[:, 0], [55, 72, 110], rtol=0.05)
