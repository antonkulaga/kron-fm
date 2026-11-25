"""JEPA step 3 options: --jepa-skip-cut (the latent loss without the identity skip's derivative), --jepa-gate
(the predictor's gradient into the core scaled, a handover), --freeze-core-gain (the core's final gain kept)."""
from __future__ import annotations

import pytest
import torch

import kron_core.blocks
from kron_core.testing import CausalMixer
from kron_dna import genome as G

import test_kron_dna_jepa_block as JB
from test_kron_dna_jepa_block import jgenomes  # noqa: F401  (fixture)
from test_kron_dna_multigenome import run


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setattr(kron_core.blocks, "kronlstm_mixer", CausalMixer)
    torch.manual_seed(0)
    cfg = G.KronGenomeConfig(d_model=32, n_heads=2, n_layers=1, ff_units=64, stem_channels=8, n_down=3, arch="v2",
                             jepa=True, jepa_pred_kind="context", jepa_pred_layers=1)
    m = JB.CpuCoreGenome(cfg).train()
    torch.nn.init.normal_(m.jepa_pred.out.weight, std=0.3)            # past the zero start, so the layers matter
    m.jepa_hidden = torch.zeros(1, 12, dtype=torch.bool)
    m.jepa_hidden[:, 3:7] = True
    return m


def grad_of(m, use_loss_version):
    core = torch.randn(1, 12, 32, requires_grad=True)
    p = m.predict_jepa(core)
    out = m.last_jepa_loss if use_loss_version else p
    out.pow(2).sum().backward()
    return core.grad.clone(), p.detach(), None if m.last_jepa_loss is None else m.last_jepa_loss.detach()


def test_gate_keeps_values_and_scales_the_gradient(model):
    torch.manual_seed(1)
    g1, p1, _ = grad_of(model, False)
    model.jepa_gate = 0.0
    torch.manual_seed(1)
    g0, p0, _ = grad_of(model, False)
    model.jepa_gate = 0.25
    torch.manual_seed(1)
    gq, pq, _ = grad_of(model, False)
    assert torch.allclose(p0, p1) and torch.allclose(pq, p1)       # the same prediction
    assert torch.count_nonzero(g0) == 0                            # no gradient into the core at 0
    assert torch.allclose(gq, 0.25 * g1, atol=1e-6)


def test_skip_cut_drops_only_the_identity_derivative(model):
    model.jepa_skip_cut = True
    torch.manual_seed(2)
    core = torch.randn(1, 12, 32, requires_grad=True)
    p = model.predict_jepa(core)
    assert torch.allclose(model.last_jepa_loss, p)                 # same values
    v = torch.randn_like(p)
    (g_cut,) = torch.autograd.grad((model.last_jepa_loss * v).sum(), core, retain_graph=True)
    (g_full,) = torch.autograd.grad((p * v).sum(), core)
    assert torch.allclose(g_full - g_cut, v, atol=1e-5)            # the identity skip's share, exactly
    model.jepa_skip_cut = False
    model.predict_jepa(core)
    assert model.last_jepa_loss is None


@pytest.mark.parametrize("more,words", [
    (["--jepa-skip-cut"], "need --jepa-weight"),
    (JB.JEPA + ["--jepa-gate", "5"], "ZERO:RAMP"),
    (JB.JEPA + ["--jepa-gate", "5:-1"], "ZERO:RAMP"),
])
def test_option_checks(jgenomes, tmp_path, monkeypatch, more, words):
    JB.stub_jepa(monkeypatch)
    from kron_dna import cli as new
    res, _ = run(new, monkeypatch, jgenomes["root"], tmp_path / "bad", JB.SMALLJ + ["--steps", "2", "--batch-size", "2"]
                 + more)
    assert res.exit_code != 0 and words in " ".join(res.output.replace("│", " ").split())


def test_frozen_gain_stays_and_the_options_run(jgenomes, tmp_path, monkeypatch):
    from test_kron_dna_multigenome import cpu_only
    cpu_only(monkeypatch)                                  # the real model, its CUDA-only core mixed on the CPU
    monkeypatch.setattr(G, "KronGenome", JB.CpuCoreGenome)
    monkeypatch.setattr(kron_core.blocks, "kronlstm_mixer", CausalMixer)
    from kron_dna import cli as new
    small = [a if a != "16" else "96" for a in JB.SMALLJ]  # d_model 96: KronLSTM heads of 16 or more
    args = small + JB.JEPA + ["--steps", "4", "--eval-every", "4", "--batch-size", "2", "--jepa-pred", "context",
                              "--jepa-pred-layers", "1"]
    out = {}
    for name, extra in (("free", []), ("frozen", ["--freeze-core-gain", "--jepa-skip-cut", "--jepa-gate", "1:2"])):
        res, _ = run(new, monkeypatch, jgenomes["root"], tmp_path / name, args + extra)
        assert res.exit_code == 0, res.output + repr(res.exception)
        out[name] = torch.load(tmp_path / name / "checkpoint.pt", map_location="cpu", weights_only=False)["model"]
    assert torch.all(out["frozen"]["core_norm.weight"] == 1.0)       # RMSNorm starts at 1 and was never moved
    assert not torch.all(out["free"]["core_norm.weight"] == 1.0)
