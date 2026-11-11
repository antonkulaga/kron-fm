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


def test_a_short_run_still_gives_its_window(tmp_path):
    # 44 s with the first second off the wrist: one 30 s window starts at the first on-wrist sample
    w, _, tally = probe_windows(read_hr_raw(_run(tmp_path, seconds=44.0)))
    assert tally["candidates"] == 1 and tally["accepted"] == 1


def test_sdk_green_reads_upright_and_its_duplicate_stamps_spread(tmp_path):
    from kron_ppg.probe_age import read_sdk

    rng = np.random.default_rng(3)
    t = np.arange(0, 120, 0.04)
    green = 50000 - 800 * _pulse(t) + 20 * rng.standard_normal(len(t))  # light drops at systole
    ms = (np.floor(t * 1000 / 8) * 8).astype(np.int64)  # several samples share a millisecond stamp now and then
    run = tmp_path / "sdk"
    run.mkdir()
    pl.DataFrame({"timestamp_ms": ms, "wall_ms": ms, "green": green, "green_status": np.zeros(len(t), int),
                  "ir": green, "ir_status": np.zeros(len(t), int), "red": green,
                  "red_status": np.zeros(len(t), int)}).write_csv(run / "sdk_ppg.csv")
    rec = read_sdk(run, "sdk_ppg")
    assert rec is not None and rec.fs == 25.0 and rec.sign == -1.0
    d = np.diff(rec.x[rec.valid])
    assert np.percentile(d, 99) > -np.percentile(d, 1)  # systole up
    w, _, tally = probe_windows(rec)
    assert tally["accepted"] >= 3


def test_estimate_average_means_the_members_and_skips_a_clock_without_windows(monkeypatch):
    import kron_ppg.probe_age as pa

    ages = {"a": 40.0, "b": 50.0, "c": None}
    monkeypatch.setattr(pa, "estimate", lambda runs, m, device, seconds: {"model": str(m), "age": ages[str(m)],
                                                                          "windows": 3, "runs": [{"run": "r"}]})
    out = pa.estimate_average(["r"], ["a", "b", "c"], "cpu", reference_text="ref")
    assert out["age"] == 45.0 and out["members_spread_years"] == 10.0 and out["reference"] == "ref"
    assert [m["age"] for m in out["members"]] == [40.0, 50.0, None]
