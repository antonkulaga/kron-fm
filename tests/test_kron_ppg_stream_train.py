"""Stream training: truncated BPTT over whole streams, carried memory, causal scaling, a stream clock."""
from __future__ import annotations

import json
import math

import numpy as np
import polars as pl
import pytest

pytest.importorskip("scipy")

from kron_ppg.model import KronPPGConfig  # noqa: E402

SMALL = dict(d_model=32, n_heads=2, n_layers=2, ff_units=64, horizon=2, proj_dim=16)


def synthetic(seconds, fs, hr, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * fs)) / fs
    phase = 2 * np.pi * hr / 60 * t
    return 200 * (np.sin(phase) + 0.4 * np.sin(2 * phase + 0.6)) + 2 * rng.standard_normal(len(t))


def tiny_streams(tmp_path, n=24):
    """``n`` people, one or two streams each, ages 20-80, at 100 Hz, written as streams."""
    from kron_ppg import readers
    from kron_ppg.readers import Recording, SourceData
    from kron_ppg.streams import write_streams

    def fake(folder):
        recs, subjects = [], []
        for i in range(n):
            subjects.append({"subject": f"p{i:02d}", "age": str(20 + (60 * i) // n), "preop_htn": str(i % 2)})
            for j in range(1 + i % 2):
                x = synthetic(150 + 30 * j, 100.0, 55 + i, seed=10 * i + j)
                if j:
                    x[2000:4000] = np.nan
                recs.append(Recording(f"p{i:02d}", f"p{i:02d}/{j}", 100.0, x))
        return SourceData(subjects, iter(recs))

    readers.READERS["tinystream"] = fake
    try:
        (tmp_path / "in").mkdir(exist_ok=True)
        write_streams("tinystream", folder=tmp_path / "in", out=tmp_path / "streams", workers=1)
    finally:
        readers.READERS.pop("tinystream")
    return tmp_path / "streams"


def test_causal_scaling_uses_only_past_blocks_and_masks_beyond_the_clip(tmp_path):
    from kron_ppg.stream_train import StreamPool

    folder = tiny_streams(tmp_path, n=4)
    pool = StreamPool([folder])
    x, ok, w = pool.read(0, 0, 25 * 60)
    assert np.isfinite(x).all() and ok[:25 * 60].all() and (w > 0.5).mean() > 0.9
    assert 0.5 < np.percentile(np.abs(x), 75) < 2.0  # robust units, not ADC units
    x2, _, _ = pool.read(0, 25 * 30, 25 * 30)  # the same samples read from a later start: same scaling
    assert np.allclose(x2, x[25 * 30:25 * 60], atol=1e-5)
    sig = pool.sets[0].signal
    pool.sets[0].signal = np.array(sig)  # writable copy: put a spike in
    pool.sets[0].signal[25 * 100] = 1e6
    xs, oks, _ = pool.read(0, 0, 25 * 120)
    assert not oks[25 * 100] and xs[25 * 100] == 0.0


def test_stream_pretraining_and_the_stream_clock_run_end_to_end(device, tmp_path):
    from kron_ppg.stream_train import StreamClockArgs, StreamPretrainArgs, fit_stream_clock, pretrain_streams
    from kron_ppg.train import load_encoder

    folder = tiny_streams(tmp_path)
    config = KronPPGConfig(**SMALL)
    result = pretrain_streams(StreamPretrainArgs([folder], tmp_path / "pre", config, lanes=6, chunk_s=20.0,
                                                 pair_chunks=3, steps=6, eval_every=3, eval_lanes=4, eval_chunks=3,
                                                 device=str(device)))
    assert math.isfinite(result["best_val_next"]) and (tmp_path / "pre" / "checkpoint.pt").is_file()
    m = pl.read_parquet(tmp_path / "pre" / "metrics.parquet")
    assert m["step"].to_list() == [3, 6] and {"val_next_chunk0", "val_next_last"} <= set(m.columns)
    load_encoder(tmp_path / "pre" / "checkpoint.pt")  # the window trainer's loader takes it unchanged
    clock = fit_stream_clock(StreamClockArgs([folder], tmp_path / "clock", tmp_path / "pre" / "checkpoint.pt",
                                             config, lanes=6, chunk_s=20.0, steps=4, eval_every=2, eval_hours=0.05,
                                             eval_lanes=5, device=str(device)))
    assert clock["best_step"] in (2, 4)
    metrics = json.loads((tmp_path / "clock" / "metrics.json").read_text())
    assert "bias" in metrics and metrics["test"]["subjects"] >= 2
    pred = pl.read_parquet(tmp_path / "clock" / "predictions.parquet")
    assert pred["patches"].min() > 0 and set(pred["split"]) == {"train", "val", "test"}


def test_lanes_pair_one_person_and_reset_only_when_a_stream_changes(tmp_path):
    from kron_ppg.stream_train import Lanes, StreamPool

    pool = StreamPool([tiny_streams(tmp_path, n=6)])
    lanes = Lanes(pool, pool.keys("train") + pool.keys("val") + pool.keys("test"), 4, 25 * 20, 4,
                  np.random.default_rng(0))
    x, ok, w, reset, keys = lanes.batch()
    assert reset.all() and keys[0] == keys[1] and keys[2] == keys[3]
    if lanes.stream[0] == lanes.stream[1]:
        assert lanes.pos[0] != lanes.pos[1]
    resets = [lanes.batch()[3] for _ in range(6)]
    assert not all(r.all() for r in resets)


def test_memory_check_scores_identical_chunks_cold_and_warm(device, tmp_path):
    from kron_ppg.model import KronPPG
    from kron_ppg.stream_train import StreamPool, memory_check

    pool = StreamPool([tiny_streams(tmp_path, n=6)])
    model = KronPPG(KronPPGConfig(**SMALL)).to(device)
    rows = memory_check(model, pool, pool.keys("train") + pool.keys("val"), 4, 25 * 20, 3, device)
    assert [r["chunk"] for r in rows] == [0, 1, 2]
    assert abs(rows[0]["warm"] - rows[0]["cold"]) < 1e-5  # no context yet: the two are the same pass


def test_periodicity_pooling_gates_softly_and_the_clock_runs_with_it(device, tmp_path):
    from kron_ppg.stream_train import StreamClockArgs, _pool_weights, fit_stream_clock

    w = np.array([[0.2] * 5 + [0.675] * 5 + [1.0] * 5])
    ok = np.ones_like(w, bool)
    ok[0, 12] = False
    assert np.allclose(_pool_weights(w, ok, 5, "uniform"), [[1, 1, 0]])
    assert np.allclose(_pool_weights(w, ok, 5, "periodicity"), [[0, 0.5, 0]])
    folder = tiny_streams(tmp_path)
    result = fit_stream_clock(StreamClockArgs([folder], tmp_path / "clock", None, KronPPGConfig(**SMALL), lanes=6,
                                              chunk_s=20.0, steps=2, eval_every=2, eval_hours=0.05, eval_lanes=5,
                                              pool="periodicity", device=str(device)))
    assert math.isfinite(result["test"]["mae"])
