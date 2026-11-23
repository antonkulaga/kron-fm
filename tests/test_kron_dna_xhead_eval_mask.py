"""Held-out head scores count only the positions given (review 23 Nov: padding entered the head Pearson and AP)."""
import numpy as np
import torch

from kron_dna import xheads as XH


def test_positions_at_reduces_per_base_mask_to_cells():
    ok = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.float32)
    assert XH.positions_at(ok, 8).tolist() == [[True] * 5 + [False] * 3]
    assert XH.positions_at(ok, 2).tolist() == [[True, False]]       # a cell counts only when all its bases do
    assert XH.positions_at(ok, 4).tolist() == [[True, True, False, False]]


def test_pearson_ignores_masked_bins():
    torch.manual_seed(0)
    x = torch.randn(1, 8, 2)
    y = x.clone()
    y[:, 4:] = -5 * x[:, 4:]                       # the masked half anti-correlates
    pm = torch.tensor([[True] * 4 + [False] * 4])
    full, masked = XH.Pearson(2, bin=1), XH.Pearson(2, bin=1)
    full.add(x, y)
    masked.add(x, y, pmask=pm)
    assert np.allclose(masked.value(), 1.0)
    assert (full.value() < 0.9).all()
    assert float(masked.n.max()) == 4


def test_pearson_bin_counts_only_whole_unmasked_bins():
    x = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1)
    pm = torch.tensor([[True] * 6 + [False] * 2])
    sc = XH.Pearson(1, bin=2)
    sc.add(x, x, pmask=pm)
    assert float(sc.n[0]) == 3                     # bins [0,1], [2,3], [4,5]; [6,7] is masked


def test_average_precision_ignores_masked_positions():
    pred = torch.tensor([[[0.1], [0.9], [0.95]]])
    want = torch.tensor([[[0], [1], [0]]])
    pm = torch.tensor([[True, True, False]])       # the high-scoring negative is padding
    a, b = XH.AveragePrecision(1, keep=1.0), XH.AveragePrecision(1, keep=1.0)
    a.add(pred, want)
    b.add(pred, want, pmask=pm)
    assert a.value()[0] < 1.0
    assert b.value()[0] == 1.0


def test_a_pack_bucket_can_be_empty():
    """The case cli.short() now guards (D crashed on gen.choice(0), 23 Nov): a length whose rung no contig of the
    table has gives a bucket without members."""
    from kron_dna import packing as PK
    ladder = PK.row_ladder(1 << 20, 128)
    b = PK.Buckets(np.array([1 << 20, (1 << 20) - 5]), None, ladder)
    short_len = ladder[-1][0] // 2                  # shorter than any contig of the table
    assert len(b.cell(short_len).members) == 0


def test_average_precision_counts_ties_against_the_positive():
    """A constant classifier is not perfect (review 23 Nov: it scored 1.0)."""
    pred = torch.zeros(1, 100, 1)
    want = torch.zeros(1, 100, 1, dtype=torch.long)
    want[0, 7, 0] = 1
    ap = XH.AveragePrecision(1, keep=1.0)
    ap.add(pred, want)
    assert abs(ap.value()[0] - 0.01) < 1e-9


def test_average_precision_matches_sklearn_on_distinct_and_tied_scores():
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(3)
    s = np.round(rng.normal(size=400), 1)          # many ties
    y = (rng.random(400) < 0.1).astype(int)
    ap = XH.AveragePrecision(1, keep=1.0)
    ap.add(torch.tensor(s, dtype=torch.float32).reshape(1, -1, 1), torch.tensor(y).reshape(1, -1, 1))
    assert abs(ap.value()[0] - average_precision_score(y, s)) < 1e-6
