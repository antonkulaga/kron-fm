"""genome-train --grow-core: new core layers start as the identity (zero) or as copies of the old ones (copy)."""
import pytest
import torch

from kron_dna import cli
from kron_dna import genome as G


def _model(n):
    torch.manual_seed(0)
    return G.KronGenome(G.KronGenomeConfig(d_model=96, n_layers=n, stem_channels=16, n_down=3, arch="v2"))


def test_zero_makes_new_layers_identity_projections():
    m = _model(3)
    cli.grow_core_fn(m, 1, "zero")
    for i in (1, 2):
        b = m.core[i]
        for lin in (b.forward_mixer.o_proj, b.mlp.down) + ((b.backward_mixer.o_proj,) if b.backward_mixer else ()):
            assert float(lin.weight.abs().sum()) == 0.0
    assert float(m.core[0].mlp.down.weight.abs().sum()) > 0          # the old layer untouched


def test_copy_stacks_old_layers():
    m = _model(4)
    cli.grow_core_fn(m, 2, "copy")
    for i, j in ((2, 0), (3, 1)):
        for (ka, a), (kb, b) in zip(m.core[i].state_dict().items(), m.core[j].state_dict().items()):
            assert ka == kb and torch.equal(a, b)


def test_bad_mode():
    with pytest.raises(ValueError):
        cli.grow_core_fn(_model(2), 1, "nope")
