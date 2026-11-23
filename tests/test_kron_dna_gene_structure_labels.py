"""prep_gene_structure_labels.py on a synthetic genome: two coding genes (one per strand, two exons each, GT..AG
introns, ATG starts) and a non-coding transcript; every channel lands on the right bases of the layout (across a
gap between two contigs), a window read through LabelStore on the other strand gives the partner channels, and a
GTF with shifted coordinates is refused by the motif check."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import typer

from kron_dna import xheads as X

spec = importlib.util.spec_from_file_location(
    "prep_gene_structure_labels", Path(__file__).resolve().parents[1] / "scripts" / "kron_dna" / "prep_gene_structure_labels.py")
GS = importlib.util.module_from_spec(spec)
spec.loader.exec_module(GS)

CODE = {b: i for i, b in enumerate("ACGT")}
RC = str.maketrans("ACGT", "TGCA")


def build(tmp_path, shift=0):
    tmp_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    chrom = list(rng.choice(list("ACGT"), 3000))
    # + gene: tx 100..400 (0-based, end excl.); exon1 100-200, intron 200-300, exon2 300-400; CDS 150-350 incl. stop
    chrom[150:153] = "ATG"
    chrom[200:202] = "GT"
    chrom[298:300] = "AG"
    # - gene: tx 1000..1300; exons 1000-1100 and 1200-1300 (genomic); transcript 5' end at 1299
    # on the minus strand the intron 1100..1200 starts (5') at 1199 and must read GT: genomic 1198-1199 = "AC"
    chrom[1198:1200] = "AC"
    chrom[1100:1102] = "CT"                                   # its 3' end AG on minus: genomic 1100-1101 = "CT"
    chrom[1247:1250] = list("CAT")                             # start codon on minus: genomic 1247-1249 = CAT
    seq = np.array([CODE[c] for c in chrom], dtype=np.uint8)
    g = tmp_path / "genome" / "toy"
    g.mkdir(parents=True)
    # two contigs: [0, 1500) and [1600, 3000) with a gap (no layout) in between
    pieces = [(0, 1500, 0), (1600, 3000, 1500)]
    np.concatenate([seq[a:b] for a, b, _ in pieces]).tofile(g / "bases.u8")
    pl.DataFrame(dict(chrom=["chr1", "chr1"], start=[0, 1600], end=[1500, 3000], offset=[0, 1500])).write_parquet(
        g / "contigs.parquet")
    s = shift
    rows = [
        ("chr1", "exon", 101 + s, 200 + s, "+", "t1"), ("chr1", "exon", 301 + s, 400 + s, "+", "t1"),
        ("chr1", "CDS", 151 + s, 200 + s, "+", "t1"), ("chr1", "CDS", 301 + s, 347 + s, "+", "t1"),
        ("chr1", "stop_codon", 348 + s, 350 + s, "+", "t1"), ("chr1", "start_codon", 151 + s, 153 + s, "+", "t1"),
        ("chr1", "exon", 1001 + s, 1100 + s, "-", "t2"), ("chr1", "exon", 1201 + s, 1300 + s, "-", "t2"),
        ("chr1", "CDS", 1051 + s, 1100 + s, "-", "t2"), ("chr1", "CDS", 1201 + s, 1250 + s, "-", "t2"),
        ("chr1", "start_codon", 1248 + s, 1250 + s, "-", "t2"),
        ("chr1", "exon", 2001 + s, 2100 + s, "+", "t3"),
        ("chrM", "exon", 11, 60, "+", "t4"), ("chrM", "exon", 71, 90, "+", "t4"),     # a chromosome the layout lacks
        ("chrM", "start_codon", 21, 23, "+", "t4"),
    ]
    gtf = tmp_path / "toy.gtf"
    gtf.write_text("".join(f"{c}\tX\t{t}\t{a}\t{b}\t.\t{st}\t.\tgene_id \"g\"; transcript_id \"{tid}\";\n"
                           for c, t, a, b, st, tid in rows))
    return g, gtf


def run(tmp_path, monkeypatch, shift=0):
    g, gtf = build(tmp_path, shift)
    monkeypatch.setattr(GS, "interim_dir", lambda *a: g)
    out = tmp_path / "labels"
    GS.main(out, name="toy", gtf=gtf, basic_only=False, chrom_prefix="")
    return out


def test_channels_land_on_the_right_bases(tmp_path, monkeypatch):
    out = run(tmp_path, monkeypatch)
    st = X.open_store(out)
    assert st.kind == "binary" and len(st.ids) == 14 and st.labels.dtype == np.uint16
    lab = st.window(0, 3000 - 100, 2900)                     # layout bases 0..2900
    col = {i: lab[:, k] for k, i in enumerate(st.ids)}
    on = lambda name: set(np.flatnonzero(col[name]).tolist())
    assert on("tss_plus") == {100, 1900} and on("polya_plus") == {399, 1999}      # t3 (2000-2100) sits at 1900 in layout
    assert on("utr5_plus") == set(range(100, 150)) and on("utr3_plus") == set(range(350, 400))
    assert on("cds_plus") == set(range(150, 200)) | set(range(300, 350))
    assert on("intron_plus") == set(range(200, 300)) and on("ncexon_plus") == set(range(1900, 2000))
    assert on("tss_minus") == {1299} and on("polya_minus") == {1000}
    assert on("cds_minus") == set(range(1050, 1100)) | set(range(1200, 1250))
    assert on("utr5_minus") == set(range(1250, 1300)) and on("utr3_minus") == set(range(1000, 1050))
    assert on("intron_minus") == set(range(1100, 1200)) and not on("ncexon_minus")
    # the other strand: reversed, each _plus channel read as its _minus partner
    flip = st.window(0, 2900, 2900, flip=True)
    k_p, k_m = st.ids.index("cds_plus"), st.ids.index("cds_minus")
    assert np.array_equal(flip[:, k_m], lab[::-1, k_p]) and np.array_equal(flip[:, k_p], lab[::-1, k_m])
    import json
    rec = json.loads((out / "labels.json").read_text())
    m = rec["check"]["motifs"]
    assert m["donor_GT"] == 1.0 and m["acceptor_AG"] == 1.0 and m["start_ATG"] == 1.0
    assert rec["check"]["cds_plus"]["sites"] == 100


def test_shifted_coordinates_are_refused(tmp_path, monkeypatch):
    with pytest.raises(typer.Exit):
        run(tmp_path, monkeypatch, shift=1)
    assert not (tmp_path / "labels" / "labels.u16").exists()


def test_a_gff3_gives_the_same_labels(tmp_path, monkeypatch):
    """The same annotation as GFF3 (transcripts by Parent=) gives the same label array as the GTF."""
    out = run(tmp_path / "gtf", monkeypatch)
    g, gtf = build(tmp_path / "gff")
    lines = []
    for line in gtf.read_text().splitlines():
        c = line.split("\t")
        tid = c[8].split('transcript_id "')[1].split('"')[0]
        lines.append("\t".join(c[:8] + [f"ID={c[2]}{len(lines)};Parent=transcript:{tid}"]))
    gff = tmp_path / "gff" / "toy.gff3"
    gff.write_text("##gff-version 3\n" + "\n".join(lines) + "\n")
    monkeypatch.setattr(GS, "interim_dir", lambda *a: g)
    GS.main(tmp_path / "gff" / "labels", name="toy", gtf=gff, basic_only=False, chrom_prefix="")
    a = np.fromfile(out / "labels.u16", dtype=np.uint16)
    b = np.fromfile(tmp_path / "gff" / "labels" / "labels.u16", dtype=np.uint16)
    assert np.array_equal(a, b) and a.any()


def test_chromosome_names_that_match_nothing_are_refused(tmp_path, monkeypatch):
    """A GTF named 'chr1' against a layout named '1' places nothing: refused, no empty store left behind."""
    g, gtf = build(tmp_path)
    gtf.write_text(gtf.read_text().replace("chr1\t", "1\t").replace("chrM\t", "MT\t"))
    pl.read_parquet(g / "contigs.parquet").with_columns(chrom=pl.lit("chr1")).write_parquet(g / "contigs.parquet")
    monkeypatch.setattr(GS, "interim_dir", lambda *a: g)
    with pytest.raises(typer.Exit):
        GS.main(tmp_path / "labels", name="toy", gtf=gtf, basic_only=False, chrom_prefix="")
    assert not (tmp_path / "labels" / "labels.u16").exists() and not (tmp_path / "labels" / "labels.json").exists()
