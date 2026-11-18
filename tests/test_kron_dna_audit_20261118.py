"""Regression tests for the independent code audit of 18 Nov 2026 (reviews/codex-20261118 on the cluster).

Each test drives the failure path the audit reproduced and asserts the repaired behaviour."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import typer


# --------------------------------------------------------------------------------- D1: genome-prepare publication

def _fasta(path, seq):
    path.write_text(">chr1\n" + seq + "\n")
    return path


def _prepare(tmp_path, monkeypatch, fasta, **kw):
    from kron_dna import cli
    monkeypatch.setenv("KRON_FM_DATA", str(tmp_path / "data"))
    cli.genome_prepare(fasta=fasta, min_gap=50, min_len=10, name="g", **kw)
    return tmp_path / "data" / "interim" / "genome" / "g"


def test_d1_prepare_publishes_a_consistent_pair(tmp_path, monkeypatch):
    out = _prepare(tmp_path, monkeypatch, _fasta(tmp_path / "a.fa", "ACGT" * 50), overwrite=False)
    idx = pl.read_parquet(out / "contigs.parquet")
    size = (out / "bases.u8").stat().st_size
    assert size == 200 and int((idx["offset"] + idx["end"] - idx["start"]).max()) == size
    assert not any(p.name.startswith(".prepare-") for p in out.iterdir())


def test_d1_existing_genome_needs_overwrite(tmp_path, monkeypatch):
    out = _prepare(tmp_path, monkeypatch, _fasta(tmp_path / "a.fa", "ACGT" * 50), overwrite=False)
    before = (out / "bases.u8").read_bytes(), (out / "contigs.parquet").read_bytes()
    with pytest.raises(typer.BadParameter):
        _prepare(tmp_path, monkeypatch, _fasta(tmp_path / "b.fa", "GGCC" * 60), overwrite=False)
    assert ((out / "bases.u8").read_bytes(), (out / "contigs.parquet").read_bytes()) == before


@pytest.mark.parametrize("failure", ["missing_fasta", "fails_while_writing"])
def test_d1_failed_rebuild_leaves_the_old_pair_byte_identical(tmp_path, monkeypatch, failure):
    from kron_dna import genome as G
    out = _prepare(tmp_path, monkeypatch, _fasta(tmp_path / "a.fa", "ACGT" * 50), overwrite=False)
    before = (out / "bases.u8").read_bytes(), (out / "contigs.parquet").read_bytes()
    if failure == "missing_fasta":
        fasta = tmp_path / "does_not_exist.fa"
    else:
        fasta = _fasta(tmp_path / "b.fa", "GGCC" * 60)
        def boom(*a, **k):
            raise RuntimeError("injected failure after parsing")
        monkeypatch.setattr(G, "contigs_of", boom)
    with pytest.raises(Exception):
        _prepare(tmp_path, monkeypatch, fasta, overwrite=True)
    assert ((out / "bases.u8").read_bytes(), (out / "contigs.parquet").read_bytes()) == before
    assert not any(p.name.startswith(".prepare-") or p.name.endswith(".prev") for p in out.iterdir())


def test_d1_overwrite_keeps_the_previous_pair(tmp_path, monkeypatch):
    out = _prepare(tmp_path, monkeypatch, _fasta(tmp_path / "a.fa", "ACGT" * 50), overwrite=False)
    old = (out / "bases.u8").read_bytes()
    _prepare(tmp_path, monkeypatch, _fasta(tmp_path / "b.fa", "GGCC" * 60), overwrite=True)
    assert (out / "bases.u8.prev").read_bytes() == old and (out / "bases.u8").stat().st_size == 240
