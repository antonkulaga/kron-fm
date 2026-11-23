"""prep_ortholog_split.py on a toy genome: a gene NCBI tags as the ortholog of a human chr8 gene, one paired in
gene_orthologs, one matched by a teleost paralog's symbol, and the genome's own held-out contig are code 4; a gene
whose symbol is a human gene elsewhere stays 1."""
from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1] / "scripts" / "kron_dna"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gz(path: Path, text: str) -> Path:
    with gzip.open(path, "wt") as f:
        f.write(text)
    return path


def test_ortholog_split(tmp_path, monkeypatch):
    OS = load("prep_ortholog_split")
    g = tmp_path / "genome"
    g.mkdir()
    pl.DataFrame(dict(chrom=["c1", "c1", "c2"], start=[0, 6000, 0], end=[5000, 10000, 1000],
                      offset=[0, 5000, 9000], held_out=[False, False, True])).write_parquet(g / "contigs.parquet")
    monkeypatch.setattr(OS, "interim_dir", lambda *a: g)
    info = gz(tmp_path / "info.gz", "#tax_id\tGeneID\tSymbol\tx\tx\tx\tchromosome\tx\tx\ttype\n"
              "9606\t1\tSOX9\t-\t-\t-\t8\t-\t-\tprotein-coding\n"
              "9606\t2\tMYC\t-\t-\t-\t8\t-\t-\tprotein-coding\n"
              "9606\t3\tGATA4\t-\t-\t-\t8\t-\t-\tprotein-coding\n"
              "9606\t4\tTP53\t-\t-\t-\t17\t-\t-\tprotein-coding\n")
    orth = gz(tmp_path / "orth.gz", "#tax_id\tGeneID\trelationship\tOther_tax_id\tOther_GeneID\n9606\t2\tOrtholog\t7955\t502\n")
    row = 'c1\tX\tgene\t{}\t{}\t.\t+\t.\tgene_id "{}"; db_xref "GeneID:{}";{} gene "{}";\n'
    gtf = gz(tmp_path / "a.gtf.gz",
             row.format(101, 200, "foo", 501, ' db_xref "NCBIOrtholog:3";', "foo")    # tagged ortholog of GATA4
             + row.format(1001, 1100, "bar", 502, "", "bar")                           # paired with MYC
             + row.format(2001, 2100, "sox9a", 503, "", "sox9a")                       # teleost paralog of SOX9
             + row.format(4951, 6100, "gata4", 504, "", "gata4")                       # spans the gap
             + row.format(3001, 3100, "tp53", 505, "", "tp53"))                        # human chr17: kept
    out = tmp_path / "split"
    OS.main(gtf, out, name="toy", gene_info=info, orthologs=orth, held="8+9", flank=0, chrom_prefix="")
    s = np.fromfile(out / "split.u8", dtype=np.uint8)
    assert len(s) == 10000
    assert (s[100:200] == 4).all() and s[99] == 1 and s[200] == 1
    assert (s[1000:1100] == 4).all() and (s[2000:2100] == 4).all()
    assert (s[3000:3100] == 1).all()
    assert (s[4950:5000] == 4).all() and (s[5000:5100] == 4).all() and s[5100] == 1   # both sides of the gap
    assert (s[9000:] == 4).all()
    m = json.loads((out / "split.json").read_text())
    assert m["by_reason"] == dict(ncbi_ortholog_tag=1, ncbi_ortholog_pair=1, symbol=2) and m["held_genes"] == 4
