"""diag/slow_band.py: one band of half-lives made short, the rest untouched."""
import importlib.util
import math
from pathlib import Path

import torch
import torch.nn.functional as F

spec = importlib.util.spec_from_file_location(
    "slow_band", Path(__file__).parents[1] / "scripts" / "kron_dna" / "diag" / "slow_band.py")
sb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sb)


def half(state: dict, p: str) -> torch.Tensor:
    a, dt = state[p + ".A_log"].double(), state[p + ".dt_bias"].double()
    return math.log(2.0) / (a.exp().repeat_interleave(dt.numel() // a.numel()) * F.softplus(dt))


def test_one_band_is_shortened_and_nothing_else(tmp_path: Path) -> None:
    want = torch.logspace(0, 4, 12, dtype=torch.float64)              # 1 to 10,000 steps over 3 heads of 4 channels
    a = torch.tensor([0.0, 1.0, -0.5])
    dt = torch.log(torch.expm1(math.log(2.0) / (want * a.double().exp().repeat_interleave(4)))).float()
    state = {"core.0.forward_mixer.A_log": a.clone(), "core.0.forward_mixer.dt_bias": dt.clone(),
             "core.0.backward_mixer.A_log": a.clone(), "core.0.backward_mixer.dt_bias": dt.clone(),
             "core.0.forward_mixer.f_proj.1.weight": torch.ones(12, 3), "head.A_log": a.clone()}
    done = sb.shorten(state, 100.0, 3000.0, 10.0)
    band = (want >= 100) & (want < 3000)
    assert done == {"core.0.forward_mixer": int(band.sum()), "core.0.backward_mixer": int(band.sum())} and band.sum() > 2
    for p in done:
        now = half(state, p)
        assert torch.allclose(now[band], torch.full_like(now[band], 10.0), rtol=1e-4)
        assert torch.allclose(now[~band], want[~band], rtol=1e-4)
        assert torch.equal(state[p + ".A_log"], a)
    assert torch.equal(state["core.0.forward_mixer.f_proj.1.weight"], torch.ones(12, 3))
    torch.save({"model": state, "step": 7}, tmp_path / "in.pt")
    sb.main(tmp_path / "in.pt", tmp_path / "out.pt", lo_kb=1.1, hi_kb=5.0, to_kb=0.5, bin=100)        # 11 to 50 steps
    s = torch.load(tmp_path / "out.pt", weights_only=False)
    assert s["step"] == 7 and s["slow_band"]["channels"] == 2 * int(((want >= 11) & (want < 50)).sum())
