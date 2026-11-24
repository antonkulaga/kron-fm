"""kron_dna.trackcompare: per-track scores, rescaling, window sets held by part of the models, and the means."""
from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from kron_dna import trackcompare as T


def make_target(n=6, bins=50, tracks=4, seed=0):
    return np.random.default_rng(seed).gamma(1.0, 2.0, (n, bins, tracks))


def test_metrics_are_one_for_a_perfect_prediction_and_ignore_scale_for_spearman():
    y = make_target()[..., 0]
    sp, pe = T.track_metrics(y, y)
    assert sp == pytest.approx(1.0) and pe == pytest.approx(1.0)
    assert T.track_metrics(10 * y, y)[0] == pytest.approx(1.0)
    assert T.track_metrics(10 * y, y)[1] < 1.0                       # log1p Pearson sees the units


def test_total_scale_restores_units_and_marks_bad_tracks():
    y = make_target()
    p = y * np.array([2.0, 0.5, 1.0, 1.0])
    p[..., 3] = 0.0
    s = T.total_scale(p, y)
    assert s[:3] == pytest.approx([0.5, 2.0, 1.0])
    assert np.isnan(s[3])


def test_score_skips_non_finite_tracks_and_checks_shapes():
    y = make_target()
    p = y.copy()
    p[0, 0, 2] = np.nan
    rows = T.score("m", "1 Mb", "all", p, y)
    assert [r["channel"] for r in rows] == [0, 1, 3]
    with pytest.raises(ValueError, match="differ"):
        T.score("m", "", "all", p[:3], y)


def test_compare_scores_partial_entries_only_on_sets_they_hold(tmp_path):
    y = make_target()
    np.save(tmp_path / "full.npy", y)
    np.save(tmp_path / "part.npy", 3 * y[[1, 4]])                    # other units, two windows
    manifest = [dict(model="full", context="1 Mb", path="full.npy"),
                dict(model="part", context="1 Mb", path="part.npy", windows=[1, 4], scale=True)]
    sets = {"all": list(range(6)), "held": [4, 1]}
    d = T.compare(manifest, y, sets, tmp_path)
    assert d.filter(pl.col("model") == "part")["window_set"].unique().to_list() == ["held"]
    part = d.filter(pl.col("model") == "part")
    assert part["pearson_log1p_128bp"].to_numpy() == pytest.approx(1.0)   # rescaled back to the target's units
    np.save(tmp_path / "bad.npy", y[:2])
    with pytest.raises(ValueError, match="2 windows"):
        T.compare([dict(model="b", path="bad.npy")], y, sets, tmp_path)


def test_aggregate_means_by_assay_all_and_shared():
    rows = [dict(model=m, context="", window_set="s", n_windows=1, channel=c, spearman_128bp=v, pearson_log1p_128bp=v)
            for m, chans in (("a", (0, 1, 2)), ("b", (0, 1))) for c, v in zip(chans, (0.2, 0.4, 0.9))]
    tracks = pl.DataFrame(dict(channel=[0, 1, 2], assay=["dnase", "dnase", "rna"]))
    a = T.aggregate(pl.DataFrame(rows), tracks)
    get = lambda m, assay: a.filter((pl.col("model") == m) & (pl.col("assay") == assay)).row(0, named=True)
    assert get("a", "all tracks")["pearson"] == pytest.approx(0.5)
    assert get("a", "shared tracks")["pearson"] == pytest.approx(0.3) and get("a", "shared tracks")["n_tracks"] == 2
    assert get("b", "dnase")["pearson"] == pytest.approx(0.3)
    ex = T.aggregate(pl.DataFrame(rows), tracks, exclude=[0])
    assert ex.filter((pl.col("model") == "a") & (pl.col("assay") == "all tracks"))["n_tracks"].item() == 2


def test_cli_writes_both_tables(tmp_path):
    from typer.testing import CliRunner
    y = make_target()
    np.save(tmp_path / "t.npy", y)
    np.save(tmp_path / "p.npy", y + 0.1)
    (tmp_path / "m.json").write_text(json.dumps([dict(model="m", context="131 kb", path="p.npy")]))
    pl.DataFrame(dict(channel=[0, 1, 2, 3], assay=["x", "x", "y", "y"])).write_csv(tmp_path / "tr.csv")
    out = tmp_path / "c.csv"
    r = CliRunner().invoke(T.app, [str(tmp_path / "m.json"), str(tmp_path / "t.npy"), str(tmp_path / "tr.csv"),
                                   "--exclude", "3", "--out", str(out)])
    assert r.exit_code == 0, r.output
    assert pl.read_csv(out).height == 4
    by = pl.read_csv(tmp_path / "c_by_assay.csv")
    assert by.filter(pl.col("assay") == "all tracks")["n_tracks"].item() == 3
