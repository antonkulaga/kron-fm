"""genome-train --lean-heads: the per-base heads run only where a training loss reads them, so training is unchanged:
the same weights after the same steps as without it (real KronGenome, CPU core)."""
from __future__ import annotations

import torch

import kron_core.blocks
from kron_core.testing import CausalMixer
from kron_dna import genome as G

import test_kron_dna_jepa_block as JB
from test_kron_dna_jepa_block import jgenomes  # noqa: F401  (fixture)
from test_kron_dna_multigenome import cpu_only, run


def test_lean_heads_train_the_same_weights(jgenomes, tmp_path, monkeypatch):
    cpu_only(monkeypatch)
    monkeypatch.setattr(G, "KronGenome", JB.CpuCoreGenome)
    monkeypatch.setattr(kron_core.blocks, "kronlstm_mixer", CausalMixer)
    from kron_dna import cli as new
    small = [a if a != "16" else "96" for a in JB.SMALLJ]
    args = small + JB.RECIPE + ["--steps", "3", "--eval-every", "3", "--batch-size", "2"]
    out = {}
    for name, extra in (("stock", []), ("lean", ["--lean-heads"])):
        res, _ = run(new, monkeypatch, jgenomes["root"], tmp_path / name, args + extra)
        assert res.exit_code == 0, res.output + repr(res.exception)
        out[name] = torch.load(tmp_path / name / "checkpoint.pt", map_location="cpu", weights_only=False)["model"]
    assert out["stock"].keys() == out["lean"].keys()
    bad = {k: float((out["stock"][k].float() - out["lean"][k].float()).abs().max()) for k in out["stock"]
           if not torch.allclose(out["stock"][k].float(), out["lean"][k].float(), rtol=1e-4, atol=1e-5)}  # summation order: a few weights differ by ~4e-6 after Adam
    assert not bad, sorted(bad.items(), key=lambda kv: -kv[1])[:8]
