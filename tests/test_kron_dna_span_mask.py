"""genome-train --span-mask: contiguous all-MASK spans, every known base inside scored, N never scored."""
import pytest
import torch

from kron_dna import cli
from kron_dna import genome as G


def test_parse():
    assert cli.parse_span_mask("32:0.05, 512:0.05") == [(32, 0.05), (512, 0.05)]
    assert cli.parse_span_mask("") == []
    with pytest.raises(Exception):
        cli.parse_span_mask("32")


def test_spans_mask_and_score_known_bases_only():
    g = torch.Generator().manual_seed(0)
    seq = torch.randint(0, 4, (2, 20000), generator=g)
    seq[:, 5000:6000] = 4                                       # an assembly gap
    inp, tgt = G.mask_bases(seq, 0.0, generator=g)              # no scattered masks
    inp, tgt = cli.add_span_masks(seq, inp, tgt, [(512, 0.1)], g)
    hidden = inp == G.MASK_CHANNEL
    assert 0.05 < float(hidden.float().mean()) <= 0.1           # overlaps only lower the share
    assert torch.equal(tgt[hidden], seq[hidden])                 # every hidden base scored with its own letter
    assert not hidden[:, 5000:6000].any() and (tgt[:, 5000:6000] == -100).all()
    assert (tgt[~hidden] == -100).all()
    runs = torch.diff(hidden[0].int()).ne(0).sum()               # contiguous: few edges for this many bases
    assert int(runs) <= 2 * int(round(0.1 * 20000 / 512)) + 4
