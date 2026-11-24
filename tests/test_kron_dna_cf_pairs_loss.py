"""genome-train --cf-pairs (JEPA step 2c): measured counterfactual pairs as a sign loss on a head's predicted
alt-minus-ref change. Units: the alternative window (substitution, long block, silenced element), the channels a
row reads, the held-out refusal and the split filter. End to end on the CPU with the stand-in model with heads."""
from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
import torch

from kron_dna import cfpairs as CF
from kron_dna import cli
import test_kron_dna_multigenome as MGT
import test_kron_dna_xheads_cli as XT
from test_kron_dna_multigenome import genomes, run  # noqa: F401  (genomes is a fixture)

LETTERS = "ACGTN"


def pair(**kw) -> CF.Pair:
    base = dict(chrom="c", pos=10, ref="A", alt="G", read_pos=12, head="h", channels=np.array([0]), sign=1.0,
                weight=1.0, del_span=None)
    return CF.Pair(**dict(base, **kw))


def test_alt_of_substitution_block_indel_and_silenced_element():
    seq = np.array([0, 1, 2, 3] * 8, dtype=np.int64)          # ACGT...
    got = CF.alt_of(seq, pair(pos=4, ref="A", alt="T"), 0)
    assert got[4] == 3 and np.array_equal(np.delete(got, 4), np.delete(seq, 4))
    long_ref = "".join(LETTERS[c] for c in seq[2:20])
    long_alt = "T" + long_ref[1:-1] + "A"
    got = CF.alt_of(seq, pair(pos=2, ref=long_ref, alt=long_alt), 0)
    assert got[2] == 3 and got[19] == 0 and len(got) == len(seq)          # a block longer than 64 bases is fine too
    assert CF.alt_of(seq, pair(pos=4, ref="C", alt="T"), 0) is None       # the reference does not hold ref
    got = CF.alt_of(seq, pair(pos=4, ref="AC", alt="A"), 0)               # a deletion keeps the window's length
    assert len(got) == len(seq) and got[-1] == 4
    got = CF.alt_of(seq, pair(ref="", alt="", del_span=(105, 110)), 100)  # silenced: N over the element
    assert (got[5:10] == 4).all() and np.array_equal(got[:5], seq[:5]) and np.array_equal(got[10:], seq[10:])


def test_channels_for():
    ids = ["a_P", "a_M", "b", "c_P", "c_M"]
    assert CF.channels_for(ids, ["b", "c_M", "zz"], None).tolist() == [2, 4]
    assert CF.channels_for(ids, [], "+").tolist() == [0, 2, 3]
    assert CF.channels_for(ids, [], "-").tolist() == [1, 2, 4]
    assert CF.channels_for(ids, None, None).tolist() == [0, 1, 2, 3, 4]


def table(rows) -> pl.DataFrame:
    return pl.DataFrame(rows, schema={"chrom": pl.Utf8, "pos": pl.Int64, "ref": pl.Utf8, "alt": pl.Utf8,
                                      "read_pos": pl.Int64, "head": pl.Utf8, "channel_ids": pl.List(pl.Utf8),
                                      "effect": pl.Float64, "weight": pl.Float64, "split": pl.Utf8}, orient="row")


def test_load_refuses_held_out_train_rows_and_keeps_train_only(tmp_path):
    ok = table([("chr1", 100, "A", "G", 120, "dnase", ["F0"], 1.0, 1.0, "train"),
                ("chr3", 100, "A", "G", 120, "dnase", ["F0"], -1.0, 1.0, "pair_test"),
                ("chr1", 300, "A", "G", 320, "other", [], 1.0, 1.0, "train"),
                ("chr1", 500, "A", "G", 520, "dnase", [], 0.0, 0.0, "train")])
    ok.write_parquet(tmp_path / "ok.parquet")
    cf = CF.CfPairs([tmp_path / "ok.parquet"], {"dnase": ["F0", "F1"]}, {"chr8", "chr9"}, 1024)
    assert len(cf.pairs) == 1 and cf.pairs[0].chrom == "chr1" and cf.pairs[0].sign == 1.0   # weight 0 and unknown head out
    bad = table([("chr8", 100, "A", "G", 120, "dnase", ["F0"], 1.0, 1.0, "train")])
    bad.write_parquet(tmp_path / "bad.parquet")
    with pytest.raises(ValueError, match="held-out"):
        CF.CfPairs([tmp_path / "bad.parquet"], {"dnase": ["F0"]}, {"chr8", "chr9"}, 1024)


def test_pairs_through_the_cli(genomes, tmp_path, monkeypatch):
    """A child run with a head and --cf-pairs: the pair loss runs every step on train rows only (both windows of a
    pair reach the model, the alternative differing at the variant), cf.csv is written, the run record carries the
    counts, and the head's parameters get the pair loss's gradient."""
    root, parent, store, child = XT.setup(genomes, tmp_path, monkeypatch)
    bases = np.fromfile(genomes["hgt"] / "bases.u8", dtype=np.uint8)
    idx = pl.read_parquet(genomes["hgt"] / "contigs.parquet")
    off = {r["chrom"]: r["offset"] for r in idx.iter_rows(named=True)}
    rows = []
    for k, (chrom, pos) in enumerate([("chr1", 1500), ("chr1", 3000), ("chr2", 2000), ("chr3", 300)]):
        ref = LETTERS[int(bases[off[chrom] + pos])]
        alt = "A" if ref != "A" else "C"
        rows.append((chrom, pos, ref, alt, pos + 40, "dnase", ["F1"], (-1.0) ** k, 1.0,
                     "pair_test" if chrom == "chr3" else "train"))
    table(rows).write_parquet(tmp_path / "pairs.parquet")
    XT.SEQS.clear()
    out = tmp_path / "cf"
    res, _ = run(cli, monkeypatch, root, out, child + ["--cf-pairs", str(tmp_path / "pairs.parquet"), "--cf-weight",
                                                       "1.0", "--cf-batch", "2", "--cf-window", "1024"])
    assert res.exit_code == 0, res.output + repr(res.exception)
    assert "--cf-pairs: 3 train pairs" in MGT.flat(res.output)
    log = pl.read_csv(out / "cf.csv")
    assert log.height == XT.STEPS - 4 and (log["pairs"] == 2).all()
    assert all(math.isfinite(v) for v in log["loss"].to_list())
    ck = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert ck["run_record"]["cf_pairs"]["pairs"] == 3
    # pairs reach the model as reference/alternative windows of --cf-window bases differing at one base
    cf_seqs = [s for s in XT.SEQS if s.shape[-1] == 1024 and s.shape[0] % 2 == 0]
    diffs = [int((s[0] != s[1]).sum()) for s in cf_seqs if s.shape[0] >= 2]
    assert diffs and min(diffs) == 1


def test_option_checks(genomes, tmp_path, monkeypatch):
    root, parent, store, child = XT.setup(genomes, tmp_path, monkeypatch)
    res, _ = run(cli, monkeypatch, root, tmp_path / "bad", child + ["--cf-weight", "1.0"])
    assert res.exit_code != 0 and "needs --cf-pairs" in MGT.flat(res.output)


def test_tables_drawn_evenly_and_crispri_absences_unsigned(tmp_path):
    big = table([("chr1", 100 + 10 * k, "A", "G", 120 + 10 * k, "dnase", ["F0"], 1.0, 1.0, "train") for k in range(90)])
    small = table([("chr2", 100 + 10 * k, "A", "G", 120 + 10 * k, "dnase", ["F0"], 1.0, 1.0, "train") for k in range(10)])
    small = small.with_columns(pl.Series("regulated", [k < 2 for k in range(10)]))
    big.write_parquet(tmp_path / "big.parquet")
    small.write_parquet(tmp_path / "small.parquet")
    cf = CF.CfPairs([tmp_path / "big.parquet", tmp_path / "small.parquet"], {"dnase": ["F0"]}, {"chr8"}, 1024)
    got = cf.sample(np.random.default_rng(0), 2000)
    share = sum(p.table == "small" for p in got) / len(got)
    assert 0.45 < share < 0.55
    signs = [p.sign for p in cf.pairs if p.table == "small"]
    assert signs.count(0.0) == 8 and signs.count(1.0) == 2
