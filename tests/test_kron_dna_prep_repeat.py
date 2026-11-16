"""scripts/kron_dna/prep_repeat.py on a small soft-masked genome: the layout of bases.u8, accession mapping through an
assembly report, refusal of a mismatching FASTA, of a FASTA without lower case, and of an existing repeat.u8."""
from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

pytest.importorskip("polars_bio")
pytest.importorskip("duckdb")

from kron_dna import genome as G  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "kron_dna" / "prep_repeat.py"


def make(tmp: Path, accession_names: bool = False, soft: bool = True):
    tmp.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    recs = {}
    for k, n in (("chr1", 30_000), ("chr2", 22_000)):
        letters = np.array(list("ACGT"))[rng.integers(0, 4, n)]
        if soft:
            low = rng.random(n) < 0.4
            letters = np.where(low, np.char.lower(letters), letters)
        s = "".join(letters)
        if k == "chr1":
            s = s[:12_000] + "N" * 80 + s[12_080:]                     # a gap: two contigs
        recs[k] = s
    fa = tmp / "g.fa"
    acc = {"chr1": "CM000001.1", "chr2": "CM000002.1"}
    with fa.open("w") as fh:
        for k, s in recs.items():
            fh.write(f">{acc[k] if accession_names else k} test\n")
            for i in range(0, len(s), 80):
                fh.write(s[i:i + 80] + "\n")
    gdir = tmp / "genome" / "tg"
    gdir.mkdir(parents=True)
    rows, off, parts = [], 0, []
    for k, s in recs.items():
        codes = G.encode_bases(s)
        for a, b in G.contigs_of(codes, 50, 1_000):
            parts.append(codes[a:b])
            rows.append((k, int(a), int(b), off))
            off += int(b - a)
    np.concatenate(parts).tofile(gdir / "bases.u8")
    pl.DataFrame(rows, schema=["chrom", "start", "end", "offset"], orient="row").write_parquet(gdir / "contigs.parquet")
    (gdir / "assembly.json").write_text(json.dumps({"held_out": ["chr2"]}))
    report = None
    if accession_names:
        report = tmp / "report.txt"
        report.write_text("# comment\n" + "".join(
            f"{k[3:]}\tassembled-molecule\t{k[3:]}\tChromosome\t{acc[k]}\t=\tna\tPrimary\t{len(recs[k])}\tna\n"
            for k in recs))
    return fa, gdir, report, recs, rows


def run_script(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["prep_repeat.py", *map(str, args)])
    runpy.run_path(str(SCRIPT), run_name="__main__")


@pytest.mark.parametrize("accession_names", [False, True])
def test_repeat_u8_matches_lower_case(tmp_path, monkeypatch, accession_names):
    fa, gdir, report, recs, rows = make(tmp_path, accession_names)
    args = ["--fasta", fa, "--genome-dir", gdir, "--stage-dir", tmp_path / "stage"]
    run_script(monkeypatch, *args, *(["--report", report] if report else []))
    rep = np.fromfile(gdir / "repeat.u8", dtype=np.uint8)
    assert len(rep) == (gdir / "bases.u8").stat().st_size
    for chrom, a, b, off in rows:
        want = np.array([c.islower() for c in recs[chrom][a:b]], dtype=np.uint8)
        assert np.array_equal(rep[off:off + b - a], want)
    summary = json.loads((tmp_path / "stage" / "tg.repeat.json").read_text())
    assert 0.35 < summary["lower_fraction"] < 0.45 and summary["split"]["held_out"]["bases"] == 22_000
    assert sorted(p.name for p in gdir.iterdir()) == ["assembly.json", "bases.u8", "contigs.parquet", "repeat.u8"]
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        run_script(monkeypatch, *args, *(["--report", report] if report else []))


def test_check_only_reports_every_contig(tmp_path, monkeypatch):
    fa, gdir, _, _, rows = make(tmp_path)
    with pytest.raises(SystemExit) as ok:
        run_script(monkeypatch, "--fasta", fa, "--genome-dir", gdir, "--stage-dir", tmp_path / "s", "--check-only",
                   "--held", "chr2")
    assert ok.value.code == 0 and not (gdir / "repeat.u8").exists()
    res = json.loads((tmp_path / "s" / "tg.check.json").read_text())
    assert res["contigs"] == res["contigs_matching"] == len(rows) and res["held_out"] == ["chr2"]
    b = np.fromfile(gdir / "bases.u8", dtype=np.uint8)
    for at in (100, 101, rows[-1][3] + 5):                         # two bases in the first contig, one in the last
        b[at] = (b[at] + 1) % 4
    b.tofile(gdir / "bases.u8")
    with pytest.raises(SystemExit) as bad:
        run_script(monkeypatch, "--fasta", fa, "--genome-dir", gdir, "--stage-dir", tmp_path / "s", "--check-only")
    res = json.loads((tmp_path / "s" / "tg.check.json").read_text())
    assert bad.value.code == 2 and res["contigs_matching"] == len(rows) - 2
    assert [(m["chrom"], m["n_diff"]) for m in res["mismatches"]] == [(rows[0][0], 2), (rows[-1][0], 1)]
    assert not (gdir / "repeat.u8").exists()


def test_refusals(tmp_path, monkeypatch):
    fa, gdir, _, _, _ = make(tmp_path / "a", soft=False)
    with pytest.raises(SystemExit, match="not soft-masked"):
        run_script(monkeypatch, "--fasta", fa, "--genome-dir", gdir, "--stage-dir", tmp_path / "s1")
    assert not (gdir / "repeat.u8").exists() and not list((tmp_path / "s1").glob("*.part"))
    fa2, gdir2, _, _, _ = make(tmp_path / "b")
    b = np.fromfile(gdir2 / "bases.u8", dtype=np.uint8)
    b[100] = (b[100] + 1) % 4
    b.tofile(gdir2 / "bases.u8")
    with pytest.raises(SystemExit, match="differs from bases.u8"):
        run_script(monkeypatch, "--fasta", fa2, "--genome-dir", gdir2, "--stage-dir", tmp_path / "s2")
    assert not (gdir2 / "repeat.u8").exists()
