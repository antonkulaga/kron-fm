"""Linear probes of frozen encoders, by person: ridge CV, bootstrap, end to end on synthetic windows."""
from __future__ import annotations

import numpy as np
import torch

from kron_ppg.model import KronPPG, KronPPGConfig
from kron_ppg.wrist_probe import bootstrap, cv_predict, probe

SMALL = dict(d_model=32, n_heads=2, n_layers=2, ff_units=64, horizon=2, proj_dim=16)


def test_cross_validated_ridge_recovers_a_linear_signal_out_of_fold():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((120, 10))
    y = x @ rng.standard_normal(10) + 0.1 * rng.standard_normal(120)
    pred, alphas = cv_predict(x, y, k=5, seed=0)
    assert np.isfinite(pred).all() and len(alphas) == 5
    assert np.corrcoef(pred, y)[0, 1] > 0.95
    noise = rng.standard_normal(120)
    pred0, _ = cv_predict(x, noise, k=5, seed=0)
    assert abs(np.corrcoef(pred0, noise)[0, 1]) < 0.35  # nothing to find: no leak from the held-out people


def test_bootstrap_interval_covers_the_statistic():
    v = np.random.default_rng(1).standard_normal(200)
    lo, hi = bootstrap(lambda i: float(v[i].mean()), len(v), 500, 0)
    assert lo < v.mean() < hi and hi - lo < 0.5


def tiny_variant(folder, n=24, fs=25.0, seconds=30.0):
    """``n`` people, half with a fast pulse and ``any_af`` 1, eight windows each."""
    from kron_ppg.data import write_windows

    rng = np.random.default_rng(0)
    t = np.arange(int(fs * seconds)) / fs
    windows, rows, subjects = [], [], []
    for i in range(n):
        af = i % 2
        hr = (110 if af else 60) + rng.normal(0, 3)
        subjects.append({"subject": f"p{i:02d}", "any_af": str(af), "age": str(30 + i)})
        w = np.stack([np.sin(2 * np.pi * hr / 60 * t + rng.uniform(0, 6)) + 0.05 * rng.standard_normal(len(t))
                      for _ in range(8)])
        windows.append(w)
        rows += [{"subject": f"p{i:02d}", "recording": f"p{i:02d}/0", "start_s": 30.0 * j, "hr_bpm": hr,
                  "periodicity": 0.9} for j in range(8)]
    write_windows(folder, windows, rows, subjects, {"source": "tiny", "sample_rate": fs, "seconds": seconds})
    return folder


def test_probe_compares_encoders_on_the_same_people(device, tmp_path):
    folder = tiny_variant(tmp_path / "tiny")
    torch.manual_seed(0)
    enc = KronPPG(KronPPGConfig(**SMALL))
    ckpt = tmp_path / "enc.pt"
    torch.save({"config": enc.config.to_dict(), "model": enc.state_dict()}, ckpt)
    r = probe(["tiny"], [("any_af", "binary"), ("age", "regression"), ("missing", "binary")],
              {"a": str(ckpt), "twin": f"untrained:{ckpt}"}, per_subject=4, k=3, boot=200, folders=[folder],
              device=device)
    assert r["people"] == 24 and r["windows_per_person_median"] == 4
    af = r["targets"]["any_af"]
    assert af["a"]["positive"] == 12 and 0.0 <= af["a"]["auc"] <= 1.0
    assert "auc_minus_a" in af["twin"] and len(af["twin"]["diff_ci95"]) == 2
    assert af["a"]["auc"] > 0.8  # the pulse rate separates the classes even through a random small encoder
    assert "mae" in r["targets"]["age"]["a"] and "skipped" in r["targets"]["missing"]
