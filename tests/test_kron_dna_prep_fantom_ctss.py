"""prep_fantom_ctss.py: the biosample of a FANTOM5 sample name, the choice among a listing's files, and the reading
of a CTSS table into per-strand positions."""
from __future__ import annotations

import gzip
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "prep_fantom_ctss", Path(__file__).parents[1] / "scripts" / "kron_dna" / "prep_fantom_ctss.py")
fantom = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fantom)


def test_the_biosample_is_the_name_without_donor_and_replicate_marks():
    b = fantom.biosample_of
    assert b("B lymphoblastoid cell line: GM12878 ENCODE, biol_rep1") == "B lymphoblastoid cell line: GM12878 ENCODE"
    assert b("Hepatocyte, donor2") == "Hepatocyte"
    assert b("Smooth Muscle Cells - Aortic, donor0, tech_rep1") == "Smooth Muscle Cells - Aortic"
    assert b("heart, adult, pool1") == "heart, adult"
    assert b("brain, adult, donor1") == "brain, adult"
    assert b("acute myeloid leukemia (FAB M5) cell line:THP-1 (fresh)") == "acute myeloid leukemia (FAB M5) cell line:THP-1 (fresh)"


def test_a_listing_gives_one_file_per_biosample_the_largest():
    row = ('<tr><td><a href="{h}">x</a></td><td align="right">2017-01-20 10:{m}  </td>'
           '<td align="right">{s}</td></tr>\n')
    html = (row.format(h="Hepatocyte%252c%2520donor1.CNhs12340.11523-119G6.hg38.nobarcode.ctss.bed.gz", m="01", s="4.0M")
            + row.format(h="Hepatocyte%252c%2520donor2.CNhs12349.11603-120G5.hg38.nobarcode.ctss.bed.gz", m="02", s="9.5M")
            + row.format(h="Hepatocyte%252c%2520donor2.CNhs12349.11603-120G5.hg38.nobarcode.bam", m="03", s="1.1G")
            + row.format(h="heart%252c%2520adult%252c%2520pool1.CNhs10621.10016-101C7.hg38.nobarcode.ctss.bed.gz", m="04", s="812K")
            + row.format(h="other.CNhs1.1-1.hg19.ctss.bed.gz", m="05", s="1.0M"))
    files = fantom.parse_listing(html, "hg38")
    assert [(f["sample"], f["cnhs"]) for f in files] == [
        ("Hepatocyte, donor1", "CNhs12340"), ("Hepatocyte, donor2", "CNhs12349"), ("heart, adult, pool1", "CNhs10621")]
    assert files[1]["size"] == int(9.5 * (1 << 20)) and files[2]["size"] == 812 << 10
    chosen = fantom.choose(files)
    assert [(c["biosample"], c["cnhs"], c["reps"]) for c in chosen] == [
        ("Hepatocyte", "CNhs12349", 2), ("heart, adult", "CNhs10621", 1)]


def test_a_ctss_table_is_read_per_strand_and_other_sequences_are_left_out(tmp_path):
    path = tmp_path / "s.ctss.bed.gz"
    lines = ["chr1\t10\t11\tchr1:10..11,+\t3\t+", "chr1\t5\t6\tchr1:5..6,+\t1\t+", "chr1\t10\t11\tx\t2\t+",
             "chr1\t7\t8\tchr1:7..8,-\t4\t-", "chr2\t99\t100\tx\t9\t-", "chr2\t100\t101\tx\t1\t-",
             "chrUn_KI270742v1\t3\t4\tx\t5\t+"]
    with gzip.open(path, "wt") as f:
        f.write("\n".join(lines) + "\n")
    got, skipped = fantom.strands(path, {"chr1": 50, "chr2": 100})
    assert skipped == 2                                               # the other sequence, and the base past chr2's end
    assert set(got) == {("chr1", "+"), ("chr1", "-"), ("chr2", "-")}
    s, c = got[("chr1", "+")]
    assert s.tolist() == [5, 10] and c.tolist() == [1.0, 5.0]         # sorted; the base listed twice adds
    assert got[("chr1", "-")][0].tolist() == [7] and np.allclose(got[("chr2", "-")][1], [9.0])
