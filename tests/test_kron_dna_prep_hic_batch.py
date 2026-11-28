"""prep_hic_batch.py: which files of a selection are converted, and where their stores go."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "prep_hic_batch", Path(__file__).parents[1] / "scripts" / "kron_dna" / "prep_hic_batch.py")
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


def test_only_downloaded_files_without_a_store_are_planned(tmp_path):
    files, out = tmp_path / "files", tmp_path / "out"
    files.mkdir()
    rows = [dict(file="A", assembly="GRCh38", biosample="CD4-positive, alpha-beta T cell"),
            dict(file="B", assembly="mm10", biosample="ES-E14"),
            dict(file="C", assembly="GRCh38", biosample="K562"),          # still downloading
            dict(file="D", assembly="GRCh38", biosample="HepG2")]         # already converted
    for name in ("A.hic", "B.hic", "C.hic.part", "D.hic"):
        (files / name).write_bytes(b"")
    done = out / "hg38" / "encode-hepg2-5kb"
    done.mkdir(parents=True)
    (done / "hic.json").write_text("{}")
    todo = batch.plan(rows, files, out, "encode")
    assert [(r["file"], store.relative_to(out).as_posix()) for r, _, store in todo] == [
        ("A", "hg38/encode-cd4-positive-alpha-beta-t-cell-5kb"), ("B", "mm10/encode-es-e14-5kb")]
    assert batch.CHROMS["mm10"].split(",")[-2:] == ["chr19", "chrX"]


def test_two_rows_that_give_one_store_are_refused(tmp_path):
    rows = [dict(file="A", assembly="GRCh38", biosample="MCF-7"), dict(file="B", assembly="GRCh38", biosample="MCF 7")]
    with pytest.raises(ValueError):
        batch.plan(rows, tmp_path, tmp_path, "encode")
