"""genome-train's longer cooldown (--lr-total, --cooldown-from): a run resumed with a longer curriculum keeps its
learning rate continuous and decays it to zero at the new end, instead of a rewarm from raising the total alone."""
from __future__ import annotations

import math

import polars as pl
import pytest

from kron_dna import cli
from kron_dna.cli import genome_lr
from test_kron_dna_multigenome import SMALL, cpu_only, genomes, run  # noqa: F401  (genomes is a fixture)


def old_lr(step, lr, warmup, steps):                   # the formula genome-train used before the option
    return lr * min(1.0, (step + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(step / steps, 1.0)))


@pytest.mark.parametrize("warmup", [1, 2000])
def test_default_is_the_old_schedule(warmup):
    for step in [0, 1, 999, 2000, 100_000, 216_000, 399_999, 400_000, 450_000]:
        assert genome_lr(step, 4e-4, warmup, 400_000) == old_lr(step, 4e-4, warmup, 400_000)
        assert genome_lr(step, 4e-4, warmup, 400_000, lr_total=400_000) == old_lr(step, 4e-4, warmup, 400_000)


def test_cooldown_is_continuous_and_ends_at_zero():
    lr, warmup, S, old_total, new_total = 4e-4, 2000, 216_000, 400_000, 600_000
    f = lambda t: genome_lr(t, lr, warmup, new_total, lr_total=old_total, cooldown_from=S)  # noqa: E731
    for t in [0, 5_000, 100_000, S - 1]:                # before S: the original run's schedule, unchanged
        assert f(t) == old_lr(t, lr, warmup, old_total)
    assert f(S) == pytest.approx(old_lr(S, lr, warmup, old_total), rel=1e-12)     # no jump at the switch
    assert abs(f(S + 1) - f(S)) < 1e-9
    xs = [f(t) for t in range(S, new_total + 1, 1000)]
    assert all(a >= b for a, b in zip(xs, xs[1:]))      # monotone down to the end
    assert f(new_total) == pytest.approx(0.0, abs=1e-15)
    assert f(new_total + 10) == pytest.approx(0.0, abs=1e-15)
    raised = old_lr(S, lr, warmup, new_total)           # what raising --steps alone would do: a rewarm
    assert raised > 1.4 * f(S)


def test_rejects_a_cooldown_outside_the_curriculum(genomes, tmp_path, monkeypatch):
    cpu_only(monkeypatch)
    for extra, words in ((["--cooldown-from", "4"], "inside the curriculum"),
                         (["--cooldown-from", "2", "--lr-total", "1"], "--lr-total")):
        res, _ = run(cli, monkeypatch, genomes["root"], tmp_path / f"r{len(extra)}",
                     SMALL + ["--curriculum", "1024:4", "--eval-every", "2"] + extra)
        assert res.exit_code != 0 and words in (res.output + repr(res.exception)), res.output


def test_resume_with_longer_cooldown_logs_a_continuous_rate(genomes, tmp_path, monkeypatch):
    cpu_only(monkeypatch)                                # the stub model on the CPU (test_kron_dna_multigenome)
    out = tmp_path / "run"
    common = SMALL + ["--eval-every", "2", "--warmup", "2"]
    res, _ = run(cli, monkeypatch, genomes["root"], out, common + ["--curriculum", "1024:8", "--stop-step", "4"])
    assert res.exit_code == 0, res.output + repr(res.exception)
    res, _ = run(cli, monkeypatch, genomes["root"], out,
                 common + ["--curriculum", "1024:12", "--lr-total", "8", "--cooldown-from", "4"])
    assert res.exit_code == 0, res.output + repr(res.exception)
    m = pl.read_csv(out / "metrics.csv")
    got = dict(zip(m["step"].to_list(), m["lr"].to_list()))
    lr = 3e-4 if "--lr" not in common else float(common[common.index("--lr") + 1])
    for step, val in got.items():
        want = genome_lr(step - 1, lr, 2, 12, lr_total=8, cooldown_from=4) if step > 4 else \
            genome_lr(step - 1, lr, 2, 8)
        assert val == pytest.approx(want, rel=1e-3, abs=1e-9), (step, val, want)
    assert got[12] < got[4]
