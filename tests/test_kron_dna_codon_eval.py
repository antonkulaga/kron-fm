"""codon_eval.py: the codon table by strand and the per-amino-acid mean baseline."""
import importlib.util
from pathlib import Path

import numpy as np
import torch

spec = importlib.util.spec_from_file_location(
    "codon_eval", Path(__file__).parents[1] / "scripts" / "kron_dna" / "codon_eval.py")
ce = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ce)


def test_the_table_is_the_standard_code():
    order = "ACGT"
    table = {a + b + c: ce.AA[16 * i + 4 * j + k] for i, a in enumerate(order) for j, b in enumerate(order)
             for k, c in enumerate(order)}
    assert table["ATG"] == "M" and table["TGG"] == "W" and table["TAA"] == table["TAG"] == table["TGA"] == "*"
    assert table["GCT"] == "A" and table["AAA"] == "K" and table["TTT"] == "F" and table["CAT"] == "H"
    assert sorted(set(ce.AA)) == sorted(set("ACDEFGHIKLMNPQRSTVWY*")) and ce.AA.count("L") == 6 and ce.AA.count("*") == 3


def test_amino_acids_by_strand():
    bases = np.array([0, 3, 2, 1, 0, 3, 4, 0, 0], dtype=np.uint8)       # A T G C A T N A A
    pos = np.array([[0, 1, 2],                                          # ATG, plus
                    [5, 4, 3],                                          # minus: complement of T A C = A T G
                    [6, 7, 8]])                                         # has an N
    assert [chr(x) if x >= 0 else "-" for x in ce.amino_acids(bases, pos)] == ["M", "M", "-"]


def test_class_means_are_unit_vectors_per_amino_acid():
    t = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [1.0, 0.2], [0.0, 1.0]]), dim=-1)
    means, overall = ce.class_means(t, np.array([77, 77, 87]))
    assert set(means) == {77, 87} and torch.allclose(means[87], torch.tensor([0.0, 1.0]))
    assert abs(float(means[77].norm()) - 1) < 1e-6 and abs(float(overall.norm()) - 1) < 1e-6
