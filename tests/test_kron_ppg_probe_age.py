"""The pulse-exploration probe's hr_raw recordings, on a synthetic run (no real recording)."""
import numpy as np
import polars as pl

from kron_ppg.probe_age import OFFSET, probe_windows, read_hr_raw


def _pulse(t, hr_bpm=66.0):
    """A pulse wave, systole up: a sharp rise and a slower fall each beat."""
    phase = (t * hr_bpm / 60.0) % 1.0
    return np.where(phase < 0.15, phase / 0.15, np.exp(-(phase - 0.15) / 0.25))


def _run(tmp_path, seconds=200.0, flip=False, seed=0):
    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.uniform(0.038, 0.042, int(seconds * 25)))  # about 25 Hz with jitter
    w = _pulse(t)
    w = w - w.mean()
    on = t >= 1.0
    v0 = np.where(on, 2, 1)
    v1 = np.where(on, 2_720_000 - 4000 * w + 150 * rng.standard_normal(len(t)), OFFSET)  # light drops at systole
    ac = (40000 if flip else -40000) * w
    v5 = np.where(on, OFFSET + ac + 600 * rng.standard_normal(len(t)), OFFSET)
    zero = np.zeros(len(t), dtype=np.int64)
    cols = {"event_ns": (t * 1e9).astype(np.int64), "wall_ms": (t * 1e3).astype(np.int64), "accuracy": zero + 3,
            "v0": v0.astype(np.int64), "v1": v1.astype(np.int64), "v2": zero, "v3": zero, "v4": v1.astype(np.int64),
            "v5": v5.astype(np.int64)}
    for i in range(6, 16):
        cols[f"v{i}"] = zero + OFFSET
    run = tmp_path / f"run{int(flip)}"
    run.mkdir()
    pl.DataFrame(cols).write_csv(run / "hr_raw.csv")
    return run


def test_reader_puts_systole_up_whatever_the_ac_sign(tmp_path):
    for flip in (False, True):
        rec = read_hr_raw(_run(tmp_path, flip=flip))
        assert rec.sign == (-1.0 if not flip else 1.0)
        assert abs(rec.seconds_on_wrist - 199) < 3 and not rec.valid[:20].any()
        ok = rec.valid
        x = rec.x[ok] - np.nanmean(rec.x[ok])
        d = np.diff(x)
        # systole up: the rise is fast and the fall slow, so the derivative's upper tail is the longer one
        assert np.percentile(d, 99) > -np.percentile(d, 1)


def test_windows_pass_the_gate_at_the_pulse_rate(tmp_path):
    w, starts, tally = probe_windows(read_hr_raw(_run(tmp_path)))
    assert tally["accepted"] >= 5 and w.shape[1] == 750
    assert abs(tally["median_hr_bpm"] - 66) < 4 and tally["median_periodicity"] > 0.6
