"""prep_source.py: the manifest and the plan of a genome entry (no download, no preparation run here)."""
from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path

import pytest
from typer.testing import CliRunner

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("prep_source", ROOT / "scripts/kron_dna/prep_source.py")
PS = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PS)


def test_assembly_url():
    assert PS.assembly_url("GCF_052327205.1_FLI_Smic_1.0", "genomic.fna.gz") == (
        "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/052/327/205/GCF_052327205.1_FLI_Smic_1.0/"
        "GCF_052327205.1_FLI_Smic_1.0_genomic.fna.gz")
    assert "/GCA/056/099/535/" in PS.assembly_url("GCA_056099535.2_ASM5609953v2", "genomic.gtf.gz")
    with pytest.raises(ValueError):
        PS.assembly_url("hg38", "genomic.fna.gz")


def test_plan_follows_the_entry(tmp_path, monkeypatch):
    monkeypatch.setenv("KRON_FM_DATA", str(tmp_path))
    full = dict(accession="GCF_000000001.1_Toy", annotation=True, min_seq=5, held="auto", orthologs=True)
    steps = PS.plan("toy", full)
    assert [s["kind"] for s in steps] == ["fetch", "fetch", "run", "run", "fetch", "fetch", "run"]
    assert [s["script"] for s in steps if s["kind"] == "run"] == [
        "prep_species_genome.py", "prep_gene_structure_labels.py", "prep_ortholog_split.py"]
    genome = steps[2]["args"]
    assert genome[1:] == ["--name", "toy", "--min-seq", "5", "--held", "auto"] and genome[0].endswith("_genomic.fna.gz")
    assert all(str(s["done"]).startswith(str(tmp_path)) for s in steps)           # everything under $KRON_FM_DATA
    bare = PS.plan("toy", dict(accession="GCA_000000002.1_Toy", annotation=False, held=""), "other")
    assert [s["kind"] for s in bare] == ["fetch", "run"] and "--held" not in bare[1]["args"]
    assert bare[1]["args"][bare[1]["args"].index("--name") + 1] == "other"         # --as: another name, same source
    with pytest.raises(ValueError):                                                # a split needs the annotation
        PS.plan("toy", dict(accession="GCA_000000002.1_Toy", annotation=False, orthologs=True))


def test_the_manifest_is_complete_and_names_no_machine(tmp_path, monkeypatch):
    text = (ROOT / "scripts/kron_dna/sources.toml").read_text()
    genomes = tomllib.loads(text)["genome"]
    assert len(genomes) >= 22
    for n, e in genomes.items():
        assert {"species", "accession", "annotation", "min_seq", "held", "orthologs", "role", "verified"} <= set(e), n
        assert e["role"] in ("train", "benchmark") and e["held"] in ("", "auto")
        PS.plan(n, e)                                                              # every entry plans
    assert "/data/" not in text and "partition" not in text.lower().replace("no machine or cluster", "")
    monkeypatch.setenv("KRON_FM_DATA", str(tmp_path))
    res = CliRunner().invoke(PS.app, ["genome", "dsim_prin31", "--dry-run"])
    assert res.exit_code == 0 and "prep_ortholog_split.py" in res.output and not any(tmp_path.iterdir())
    assert CliRunner().invoke(PS.app, ["genome", "nope", "--dry-run"]).exit_code != 0
