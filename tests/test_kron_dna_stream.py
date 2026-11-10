"""The streaming, sharded prepare gives the same verdicts, tokens and rows as the in-memory one."""
from __future__ import annotations

import csv
import gzip
import json

import numpy as np

from kron_dna import codons as C
from kron_dna.data import PackedRows, documents, prepare
from kron_dna.stream import documents_np, encode_checked, prepare_stream


def _random_cds(rng: np.random.Generator) -> str:
    sense = [c for c in C.CODONS if C.GENETIC_CODE[c] != "*"]
    seq = rng.choice(["ATG", "GTG", "TTG"]) + "".join(rng.choice(sense, int(rng.integers(20, 60))))
    seq += rng.choice(C.STOP_CODONS)
    kind = rng.integers(0, 8)
    if kind == 1:
        seq = seq[:-1]
    elif kind == 2:
        seq = seq[:30] + "TAA" + seq[30:]
    elif kind == 3:
        seq = seq[:12] + "N" + seq[13:]
    elif kind == 4:
        seq = seq.lower().replace("t", "u")
    elif kind == 5:
        seq = seq[:-3] + "GGG"
    elif kind == 6:
        seq = seq[:60] + " \n" + seq[60:]
    return seq


def test_encode_checked_matches_check_cds_and_encode():
    rng = np.random.default_rng(0)
    for i in range(2000):
        seq = _random_cds(rng)
        for starts in (("ATG",), ("ATG", "GTG", "TTG")):
            reason, tokens = encode_checked(seq, starts)
            assert reason == C.check_cds(seq, starts=starts), seq
            if reason is None:
                assert tokens.dtype == np.uint8 and tokens.tolist() == C.encode(seq)


def test_documents_np_matches_documents():
    for n in (1, 5, 6, 7, 11, 12, 13, 30):
        tokens = [C.CLS_ID] + [C.FIRST_CODON_ID + i % 64 for i in range(n)] + [C.EOS_ID]
        for seq_len in (3, 5, 7, 8):
            got = [d.tolist() for d in documents_np(np.array(tokens, dtype=np.uint8), seq_len)]
            assert got == list(documents(tokens, seq_len)), (n, seq_len)


def test_prepare_stream_shards_each_split_like_prepare(tmp_path):
    rng = np.random.default_rng(1)
    path = tmp_path / "corpus.csv.gz"
    rows = [(_random_cds(rng), f"sp{i % 7}", ["bacteria", "plant"][i % 2], ["Train", "Test"][int(i % 5 == 0)])
            for i in range(600)]
    with gzip.open(path, "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["CDS", "Species", "Group", "Set"])
        w.writerows(rows)
    starts = ("ATG", "GTG", "TTG")
    for workers in (1, 2):
        out = tmp_path / f"stream{workers}"
        summary = prepare_stream(path, out, seq_len=64, starts=starts, shard_rows=16, workers=workers, chunk_size=50)
        assert set(summary["splits"]) == {"train", "test"} and summary["rows_read"] == 600
        for split, set_name in (("train", "Train"), ("test", "Test")):
            ref = prepare([path], tmp_path / f"ref-{split}", seq_len=64, starts=starts, keep={"Set": set_name})
            meta = json.loads((out / split / "meta.json").read_text())
            assert meta["rows"] == ref["rows"] and meta["sequences"] == ref["sequences"]
            assert meta["codons"] == ref["codons"] and meta["rejected"] == ref["rejected"]
            assert len(meta["shards"]) > 1 and all(s["rows"] <= 16 for s in meta["shards"])
            assert sum(meta["species"].values()) == meta["sequences"]
            got = PackedRows(out / split)
            want = np.load(tmp_path / f"ref-{split}" / "tokens.npy")
            assert np.array_equal(np.stack([got[i].numpy() for i in range(len(got))]), want)


def test_prepare_stream_holds_out_whole_species(tmp_path):
    from kron_dna.data import held_out

    rng = np.random.default_rng(2)
    path = tmp_path / "corpus.csv"
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["CDS", "Species", "Group", "Set"])
        for i in range(400):
            w.writerow(["ATG" + "GCC" * 40 + "TAA", f"sp{i % 40}", "plant", ["Train", "Test"][i % 4 == 0]])
    summary = prepare_stream(path, tmp_path / "out", seq_len=64, holdout_species=0.2, chunk_size=37)
    held = {f"sp{k}" for k in range(40) if held_out(f"sp{k}", 0.2)}
    assert 0 < len(held) < 40
    meta = json.loads((tmp_path / "out" / "species_holdout" / "meta.json").read_text())
    assert set(meta["species"]) == held and meta["sequences"] == 10 * len(held)
    for split in ("train", "test"):
        assert not held & set(json.loads((tmp_path / "out" / split / "meta.json").read_text())["species"])
    assert sum(v["sequences"] for v in summary["splits"].values()) == 400
