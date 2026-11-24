"""JEPA step 2 (genome-train --jepa-pred context, --jepa-loss zcos|zcos-mse, --jepa-decode-pred,
--freeze-trunk-until): the context predictor starts as the masked core, the decoder reads the predictor at hidden
bins in training only, the zcos frame is carried in the checkpoint, and the warm-up trains the predictor alone.

The CLI tests reuse test_kron_dna_jepa_block.py's real model with the CUDA-only core replaced by a CPU mixing, and
kron_core.testing.CausalMixer standing in for the predictor's KronLSTM layers.
"""
from __future__ import annotations

import math

import pytest
import torch

import kron_core.blocks
from kron_core.testing import CausalMixer
from kron_dna import genome as G

import test_kron_dna_jepa_block as JB
from test_kron_dna_multigenome import cpu_only, make_genome, make_ntv3, metric_rows, run


@pytest.fixture
def cpu_mixer(monkeypatch):
    monkeypatch.setattr(kron_core.blocks, "kronlstm_mixer", CausalMixer)


def small_config(**kw) -> G.KronGenomeConfig:
    return G.KronGenomeConfig(**dict(dict(d_model=32, n_heads=2, n_layers=1, ff_units=64, stem_channels=8, n_down=3,
                                          arch="v2", jepa=True, jepa_pred_kind="context", jepa_pred_layers=1), **kw))


def test_context_predictor_starts_as_the_masked_core(cpu_mixer):
    torch.manual_seed(0)
    p = G.JepaContextPredictor(small_config())
    core = torch.randn(2, 16, 32)
    cu = torch.tensor([0, 16, 32], dtype=torch.int32)
    hidden = torch.zeros(2, 16, dtype=torch.bool)
    hidden[:, 4:9] = True
    assert torch.equal(p(core, cu, hidden), core)          # zero readout: exactly the input
    torch.nn.init.normal_(p.out.weight, std=0.1)
    torch.nn.init.normal_(p.mask, std=1.0)
    with_mask, without = p(core, cu, hidden), p(core, cu, None)
    assert not torch.allclose(with_mask, without)          # the mask embedding reaches the output
    assert with_mask.shape == core.shape


def test_mlp_stays_the_default_and_unknown_kinds_are_refused():
    m = G.make_jepa_predictor(small_config(jepa_pred_kind="mlp"))
    assert isinstance(m, torch.nn.Sequential)
    assert [k for k in m.state_dict()] == ["0.weight", "1.weight", "1.bias", "3.weight", "3.bias"]
    assert G.make_jepa_predictor(small_config(jepa=False)) is None
    with pytest.raises(ValueError, match="mlp or context"):
        G.make_jepa_predictor(small_config(jepa_pred_kind="attn"))


def test_core_to_decode_only_in_training_with_hidden_bins(cpu_mixer):
    m = JB.CpuCoreGenome(small_config())
    core = torch.randn(1, 8, 32)
    pred = torch.randn(1, 8, 32)
    hid = torch.zeros(1, 8, dtype=torch.bool)
    hid[0, 2:5] = True
    m.last_jepa, m.jepa_hidden = pred, hid
    m.train()
    assert torch.equal(m.core_to_decode(core), core)                    # flag off
    m.jepa_decode_pred = True
    got = m.core_to_decode(core)
    assert torch.equal(got[0, 2:5], pred[0, 2:5]) and torch.equal(got[0, :2], core[0, :2])
    m.eval()
    assert torch.equal(m.core_to_decode(core), core)                    # inference never mixes


@pytest.mark.parametrize("more,words", [
    (["--jepa-pred", "context"], "need --jepa-weight"),
    (JB.JEPA + ["--jepa-loss", "l2"], "--jepa-loss cos, zcos or zcos-mse"),
    (JB.JEPA + ["--jepa-decode-pred"], "--jepa-decode-pred needs"),
    (JB.JEPA + ["--jepa-decode-pred", "--jepa-block-share", "0.2"], "--jepa-decode-pred needs"),
])
def test_option_checks(jgenomes, tmp_path, monkeypatch, more, words):
    JB.stub_jepa(monkeypatch)
    from kron_dna import cli as new
    res, _ = run(new, monkeypatch, jgenomes["root"], tmp_path / "bad", JB.SMALLJ + ["--steps", "2", "--batch-size", "2"]
                 + more)
    assert res.exit_code != 0 and words in " ".join(res.output.replace("│", " ").split())


jgenomes = JB.jgenomes


def test_step2_through_the_cli_warmup_then_joint_and_resume(jgenomes, tmp_path, monkeypatch, cpu_mixer):
    """The real model, every step-2 option on: during the warm-up only jepa_pred.* parameters reach the update; after
    it the trunk does too; jepa.csv is finite; the checkpoint carries the zcos frame and a resumption restores it."""
    root = jgenomes["root"]
    cpu_only(monkeypatch)
    monkeypatch.setattr(kron_core.blocks, "kronlstm_mixer", CausalMixer)   # cpu_only may reset it
    monkeypatch.setattr(G, "KronGenome", JB.CpuCoreGenome)
    from kron_dna import cli as new
    grads = []
    real_clip = torch.nn.utils.clip_grad_norm_

    def clip(params, max_norm, *a, **k):
        params = list(params)
        grads.append([None if p.grad is None else float(p.grad.norm()) for p in params])
        return real_clip(params, max_norm, *a, **k)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    names = []
    real_init = JB.CpuCoreGenome.__init__

    def init(self, config):
        real_init(self, config)
        names[:] = [n for n, _ in self.named_parameters()]

    monkeypatch.setattr(JB.CpuCoreGenome, "__init__", init)
    small = [a if a != "16" else "96" for a in JB.SMALLJ]
    args = small + JB.JEPA + ["--bert-mask", "--batch-size", "4", "--eval-every", "2", "--curriculum", "4096:6",
                              "--jepa-block-share", "0.25", "--jepa-block-len", "256:1024", "--jepa-block-heads",
                              "include", "--jepa-log-every", "1", "--jepa-pred", "context", "--jepa-pred-layers", "1",
                              "--jepa-loss", "zcos-mse", "--jepa-decode-pred", "--freeze-trunk-until", "3"]
    out = tmp_path / "s2"
    res, _ = run(new, monkeypatch, root, out, args + ["--stop-step", "4"])
    assert res.exit_code == 0, res.output + repr(res.exception)
    assert "jepa step 2: predictor context (1 KronLSTM layers), loss zcos-mse" in " ".join(res.output.split())
    assert len(grads) == 4
    for k, g_ in enumerate(grads):
        named = dict(zip(names, g_))
        pred_ = [v for n, v in named.items() if n.startswith("jepa_pred.")]
        other = [v for n, v in named.items() if not n.startswith("jepa_pred.")]
        assert any(v is not None and v > 0 for v in pred_), k
        assert all(v is None or math.isfinite(v) for v in g_)
        if k < 3:
            assert all(v is None for v in other), k                   # the warm-up: the predictor alone
        else:
            assert any(v is not None and v > 0 for n, v in named.items() if n.startswith(("stem.", "down.", "up."))), k
    ck = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert ck["step"] == 4 and set(ck["jepa_norm"]) == {"mu", "var"} and ck["jepa_norm"]["mu"].shape == (96,)
    assert any(k.startswith("jepa_pred.layers.") for k in ck["model"]) and "jepa_pred.mask" in ck["model"]
    rows = (out / "jepa.csv").read_text().strip().split("\n")
    head = rows[0].split(",")
    for line in rows[1:]:
        r = dict(zip(head, map(float, line.split(","))))
        assert math.isfinite(r["jepa_hidden"]) and math.isfinite(r["copy_hidden"])
    grads.clear()
    res, _ = run(new, monkeypatch, root, out, args)
    assert res.exit_code == 0, res.output + repr(res.exception)
    assert "resumed at step 4" in res.output and "target statistics restored" in res.output
    assert len(grads) == 2
