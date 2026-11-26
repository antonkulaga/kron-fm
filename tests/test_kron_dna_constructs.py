"""prep_constructs.py: GenBank feature tables to per-base part labels."""
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import polars as pl

spec = importlib.util.spec_from_file_location(
    "prep_constructs", Path(__file__).parents[1] / "scripts" / "kron_dna" / "prep_constructs.py")
pc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pc)


def origin(seq: str) -> str:
    lines = []
    for i in range(0, len(seq), 60):
        chunk = seq[i:i + 60].lower()
        lines.append(f"{i + 1:>9} " + " ".join(chunk[j:j + 10] for j in range(0, len(chunk), 10)))
    return "\n".join(lines)


def record(acc: str, seq: str, features: str, circular: bool = True) -> str:
    shape = "circular" if circular else "linear  "
    return (f"LOCUS       {acc}        {len(seq)} bp    DNA     {shape} SYN 01-JAN-2020\n"
            f"DEFINITION  Cloning vector {acc},\n            complete sequence.\n"
            f"ACCESSION   {acc}\nVERSION     {acc}.1\n"
            "SOURCE      Cloning vector\n  ORGANISM  Cloning vector\n            other sequences; artificial sequences; vectors.\n"
            f"FEATURES             Location/Qualifiers\n{features}ORIGIN\n{origin(seq)}\n//\n")


FEATURES = (
    "     source          1..300\n"
    "                     /organism=\"Cloning vector\"\n"
    "     regulatory      11..20\n"
    "                     /regulatory_class=\"promoter\"\n"
    "                     /note=\"CMV\"\n"
    "     CDS             31..60\n"
    "                     /product=\"x\"\n"
    "     CDS             complement(join(71..80,\n"
    "                     91..100))\n"
    "                     /product=\"y\"\n"
    "     polyA_signal    101..106\n"
    "     rep_origin      join(291..300,1..5)\n"
    "     enhancer        <111..>120\n"
    "     misc_feature    121..130\n"
    "     CDS             J00194.1:1..9\n"
)


def test_locations() -> None:
    assert pc.spans("complement(join(71..80,91..100))") == ([(70, 80), (90, 100)], True)
    assert pc.spans("<111..>120") == ([(110, 120)], False)
    assert pc.spans("42") == ([(41, 42)], False)
    assert pc.spans("J00194.1:1..9") == ([], False)
    assert pc.spans("123^124") == ([], False)


def test_store(tmp_path: Path) -> None:
    rng = random.Random(0)
    seq = "".join(rng.choice("ACGT") for _ in range(300))
    other = "".join(rng.choice("ACGT") for _ in range(300))
    gb = tmp_path / "x.gb"
    gb.write_text(record("AB000001", seq, FEATURES) + record("AB000002", seq, FEATURES)          # an exact duplicate
                  + record("AB000003", other, "     source          1..300\n", circular=False)   # no parts
                  + record("AB000004", other, "     intron          1..10\n", circular=False)
                  + record("AB000005", "ACGT" * 5, FEATURES))                                    # too short
    out = tmp_path / "out"
    pc.main(out, [gb], min_len=100, max_len=1000, max_n=0.01, held_out_mod=10)
    info = json.loads((out / "constructs.json").read_text())
    assert info["records"] == 2 and info["left_out"]["duplicate"] == 1 and info["left_out"]["no_parts"] == 1
    assert info["left_out"]["short"] == 1
    table = pl.read_parquet(out / "contigs.parquet")
    assert table["chrom"].to_list() == ["AB000001.1", "AB000004.1"] and table["offset"].to_list() == [0, 300]
    assert table["circular"].to_list() == [True, False]
    assert table["definition"][0] == "Cloning vector AB000001, complete sequence."
    parts = np.fromfile(out / "parts.u8", dtype=np.uint8)
    bases = np.fromfile(out / "bases.u8", dtype=np.uint8)
    assert len(parts) == len(bases) == 600

    def where(name: str) -> list[int]:
        return np.flatnonzero((parts[:300] >> pc.PART_BITS[name]) & 1).tolist()

    assert where("promoter") == list(range(10, 20))
    assert where("CDS_plus") == list(range(30, 60))
    assert where("CDS_minus") == list(range(70, 80)) + list(range(90, 100))
    assert where("polyA_signal") == list(range(100, 106))
    assert where("rep_origin") == list(range(0, 5)) + list(range(290, 300))
    assert where("enhancer") == list(range(110, 120))
    assert parts[120:130].sum() == 0                                    # misc_feature is not a part
    assert np.flatnonzero((parts[300:] >> pc.PART_BITS["intron"]) & 1).tolist() == list(range(0, 10))
    labels = np.fromfile(out / "labels.u8", dtype=np.uint8)
    cds = np.flatnonzero((labels[:300] >> pc.LABEL_BITS["CDS"]) & 1).tolist()
    assert cds == list(range(30, 60)) + list(range(70, 80)) + list(range(90, 100))
    assert not ((labels >> pc.LABEL_BITS["exon"]) & 1).any()
