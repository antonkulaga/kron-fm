"""The data layout, the downloader (over file:// URLs, so no network), the source registries and the CLIs."""
from __future__ import annotations

import gzip
import json

import pytest
from typer.testing import CliRunner

from kron_core import paths
from kron_core.download import RemoteFile, Source, fetch_source, link_folder, read_manifest


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.DATA_ENV, str(tmp_path / "data"))
    return tmp_path / "data"


def test_layout_follows_the_environment_variable(data_root):
    assert paths.data_root() == data_root
    assert paths.input_dir("cds", "x") == data_root / "input" / "cds" / "x"
    assert paths.interim_dir("cds") == data_root / "interim" / "cds"
    assert paths.runs_dir("kron-dna") == data_root / "output" / "runs" / "kron-dna"
    name = paths.run_name("ecoli k12", "mlm")
    assert name.startswith("ecoli_k12-mlm-") and name.endswith("Z")


def test_repo_default_is_the_checkout_data_folder(monkeypatch):
    monkeypatch.delenv(paths.DATA_ENV, raising=False)
    assert paths.data_root() == paths.repo_root() / "data"


def _csv_gz(path, rows=50):
    with gzip.open(path, "wt") as fh:
        fh.write("CDS,Species,Group,Set\n")
        for i in range(rows):
            fh.write(f"ATG{'GCC' * 40}TAA,species_{i},Bacteria,{'train' if i % 5 else 'test'}\n")
    return path


def test_fetch_writes_files_and_provenance_and_skips_what_is_present(tmp_path, capsys):
    remote = _csv_gz(tmp_path / "corpus.csv.gz")
    source = Source("demo", "cds", "a test corpus", (RemoteFile(remote.as_uri()),), license="CC0",
                    notes={"format": "csv"})
    manifest = fetch_source(source, tmp_path / "input")
    folder = tmp_path / "input" / "cds" / "demo"
    assert (folder / "corpus.csv.gz").read_bytes() == remote.read_bytes()
    entry = manifest["files"][0]
    assert entry["bytes"] == remote.stat().st_size and len(entry["sha256"]) == 64
    assert read_manifest(folder)["license"] == "CC0"
    assert not list(folder.glob("*.part"))
    fetch_source(source, tmp_path / "input")
    assert "skipped" in capsys.readouterr().out


def test_rows_takes_a_valid_gzip_of_the_first_rows(tmp_path):
    remote = _csv_gz(tmp_path / "corpus.csv.gz", rows=100)
    source = Source("demo", "cds", "a test corpus", (RemoteFile(remote.as_uri()),), notes={"format": "csv"})
    manifest = fetch_source(source, tmp_path / "input", rows=10)
    head = tmp_path / "input" / "cds" / "demo" / "corpus.head10.csv.gz"
    lines = gzip.open(head, "rt").read().splitlines()
    assert lines[0] == "CDS,Species,Group,Set" and len(lines) == 11
    assert manifest["files"][0]["subset"].startswith("first 10 lines")


def test_link_folder_points_at_another_checkout(tmp_path):
    target = tmp_path / "glucose" / "data" / "input" / "metabonet"
    target.mkdir(parents=True)
    folder = tmp_path / "input" / "timeseries" / "metabonet"
    record = link_folder(Source("metabonet", "timeseries", "x"), folder, target, "hint")
    assert folder.is_symlink() and folder.resolve() == target.resolve()
    assert record["linked_to"] == str(target.resolve())
    with pytest.raises(FileNotFoundError, match="hint"):
        link_folder(Source("m", "timeseries", "x"), tmp_path / "other", tmp_path / "missing", "hint")


def test_registries_have_distinct_names_and_known_formats():
    from kron_core.data_cli import all_sources

    sources = all_sources()
    assert {"ecoli-k12", "yeast-s288c", "human-grch38", "syncodonlm", "metabonet"} <= set(sources)
    for s in sources.values():
        assert s.notes.get("format"), f"{s.name} does not say its format"
        assert s.files or s.fetch is not None
    assert sources["syncodonlm"].files[0].size == 17_393_708_749


def test_metabonet_links_the_glucose_checkout(tmp_path, monkeypatch, data_root):
    from krontime.sources import GLUCOSE_ENV, SOURCES

    glucose = tmp_path / "glucose"
    (glucose / "data" / "input" / "metabonet").mkdir(parents=True)
    monkeypatch.setenv(GLUCOSE_ENV, str(glucose))
    fetch_source(SOURCES["metabonet"], paths.input_dir())
    assert (data_root / "input" / "timeseries" / "metabonet").is_symlink()


def test_kron_data_cli_lists_downloads_and_reports(tmp_path, data_root):
    from kron_core import download
    from kron_core.data_cli import app
    from kron_dna import sources

    remote = _csv_gz(tmp_path / "corpus.csv.gz")
    demo = Source("demo-cds", "cds", "a test corpus", (RemoteFile(remote.as_uri()),), license="CC0",
                  notes={"format": "csv", "sequence_column": "CDS"})
    runner = CliRunner()
    try:
        sources.SOURCES["demo-cds"] = demo
        result = runner.invoke(app, ["download", "demo-cds"])
        assert result.exit_code == 0, result.output
        listed = runner.invoke(app, ["list"]).output
        assert "demo-cds" in listed and "present" in listed
        status = runner.invoke(app, ["status"]).output
        assert "cds/demo-cds" in status and "CC0" in status
        assert json.loads(runner.invoke(app, ["list", "--json"]).output)["demo-cds"]["license"] == "CC0"
    finally:
        sources.SOURCES.pop("demo-cds")
    assert download.read_manifest(data_root / "input" / "cds" / "demo-cds") is not None


def test_kron_dna_prepare_and_train_use_the_layout(tmp_path, data_root, monkeypatch):
    from kron_core import blocks
    from kron_core.testing import CausalMixer
    from kron_dna import sources
    from kron_dna.cli import app

    monkeypatch.setattr(blocks, "kronlstm_mixer", CausalMixer)
    folder = data_root / "input" / "cds" / "demo-cds"
    folder.mkdir(parents=True)
    _csv_gz(folder / "corpus.csv.gz", rows=60)
    sources.SOURCES["demo-cds"] = Source("demo-cds", "cds", "x", notes={"format": "csv", "set_column": "Set"})
    runner = CliRunner()
    try:
        result = runner.invoke(app, ["prepare", "demo-cds", "--seq-len", "128", "--keep-set", "train"])
        assert result.exit_code == 0, result.output
        packed = data_root / "interim" / "cds" / "demo-cds" / "packed_128_train"
        meta = json.loads((packed / "meta.json").read_text())
        assert meta["sequences"] == 48 and meta["keep"] == {"Set": "train"}
        result = runner.invoke(app, ["train", str(packed), "--device", "cpu", "--steps", "2", "--eval-every", "2",
                                     "--batch-size", "2", "--d-model", "32", "--n-heads", "2", "--n-layers", "2",
                                     "--ff-units", "64", "--attention-every", "1", "--no-bf16", "--val-fraction", "0.2"])
        assert result.exit_code == 0, result.output
        runs = list((data_root / "output" / "runs" / "kron-dna").iterdir())
        assert len(runs) == 1 and runs[0].name.startswith("packed_128_train-mlm-")
        assert (runs[0] / "checkpoint.pt").is_file() and (runs[0] / "metrics.csv").is_file()
    finally:
        sources.SOURCES.pop("demo-cds")


def test_wrist_sources_are_registered_with_licence_and_rate():
    from kron_core.data_cli import all_sources
    from kron_ppg.watch_sources import SOURCES as WATCH

    sources = all_sources()
    assert set(WATCH) <= set(sources)
    for s in WATCH.values():
        assert s.domain == "ppg" and s.license and s.citation and s.homepage, s.name
        assert s.fetch is not None, f"{s.name} writes SOURCE.json file by file through fetch_tree"
    assert sum(f.size for f in WATCH["wildppg"].files) > 19e9
