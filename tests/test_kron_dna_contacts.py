"""contacts.py: contact-map stores read per window, the head that predicts them from the core's states, its loss."""
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

import kron_core.blocks
from kron_core.testing import CausalMixer
from kron_dna import contacts as C
from kron_dna import genome as G

CELL, DIAGS = 5000, 6


def store(path: Path, scale: float, cell: int = CELL) -> Path:
    """chrA with 40 cells, chrB with 10: the value of (cell i, distance d) is ``scale * (i + d / 10)``, counted
    over both chromosomes; cell 7 of chrA is masked."""
    path.mkdir()
    i, d = np.indices((50, DIAGS))
    m = (scale * (i + d / 10)).astype(np.float16)
    m[7] = np.nan
    m[:7][i[:7] + d[:7] == 7] = np.nan
    m.tofile(path / "contacts.f16")
    pl.DataFrame(dict(chrom=["chrA", "chrB"], length=[200_000, 50_000], cells=[40, 10], offset=[0, 40],
                      masked=[1, 0])).write_parquet(path / "chroms.parquet")
    (path / "hic.json").write_text(json.dumps(dict(bin=cell, diagonals=DIAGS)))
    return path


CONTIGS = pl.DataFrame(dict(chrom=["chrA", "chrA", "chrB", "chrC"], start=[1000, 120_000, 0, 0],
                            end=[100_000, 200_000, 50_000, 30_000], offset=[0, 99_000, 179_000, 229_000]))


def test_a_window_of_the_stores(tmp_path: Path) -> None:
    s = C.ContactStores([store(tmp_path / "a", 1.0), store(tmp_path / "b", 2.0)], CONTIGS)
    assert (s.cell, s.diagonals, s.names) == (CELL, DIAGS, ["a", "b"])
    cells, target = s.window(99_000 + 4096, 32768, 128)                # chrA from 124,096: cells 24 .. 31
    assert cells.shape == (256,) and cells[0] == 0 and cells[-1] == 7 and (np.diff(cells) >= 0).all()
    assert int(cells[7]) == 1                                          # the bin at 124,992 .. 125,120: its middle is in cell 25
    assert target.shape == (8, DIAGS, 2)
    i, d = np.indices((8, DIAGS))
    inside = i + d < 8
    assert np.isnan(target[~inside]).all() and not np.isnan(target[inside]).any()
    assert np.allclose(target[..., 0][inside], (24 + i + d / 10)[inside], atol=0.02)
    assert np.allclose(target[..., 1][inside], 2 * target[..., 0][inside], rtol=1e-3)
    back, flipped = s.window(99_000 + 4096, 32768, 128, flip=True)
    assert np.array_equal(back, (7 - cells)[::-1])
    for a in range(8):
        for k in range(DIAGS):
            if a + k < 8:
                assert np.array_equal(flipped[a, k], target[7 - a - k, k])
            else:
                assert np.isnan(flipped[a, k]).all()
    masked = s.window(0, 65536, 128)[1]                                # chrA 1,000 .. 66,536 holds the masked cell 7
    assert np.isnan(masked[7]).all() and np.isnan(masked[5, 2]).all() and not np.isnan(masked[5, 1]).any()
    assert s.window(90_000, 16384, 128) is None                        # runs over the gap between two contigs
    assert s.window(229_000, 8192, 128) is None                        # a chromosome the stores do not hold
    assert s.window(179_000, 8192, 128)[1][0, 0, 0] == pytest.approx(40.0)     # chrB starts at its own rows
    with pytest.raises(ValueError):
        C.ContactStores([tmp_path / "a", store(tmp_path / "c", 1.0, cell=10_000)], CONTIGS)


def test_the_head_reads_a_pair_from_both_ends_and_either_way() -> None:
    torch.manual_seed(0)
    head = C.ContactHead(16, 8, 5, 3).eval()
    core = torch.randn(2, 40, 16)
    cells = (torch.arange(40) // 4).expand(2, 40)                      # 10 cells of 4 bins
    with torch.no_grad():
        out = head(core, cells)
        assert out.shape == (2, 10, 5, 3)
        moved = core.clone()
        moved[0, 12:16] += 1.0                                         # cell 3 of row 0
        diff = (head(moved, cells) - out).abs().amax(-1)               # [2, 10, 5]
        touched = torch.zeros(10, 5, dtype=torch.bool)
        for i in range(10):
            for d in range(5):
                touched[i, d] = 3 in (i, i + d)
        assert float(diff[0][touched].min()) > 1e-6 and float(diff[0][~touched].max()) < 1e-6
        assert float(diff[1].max()) == 0.0
        # read as the reverse complement (bins and cells the other way) a pair has the same value
        back = head(core.flip(1), (9 - cells).flip(1))
        for i in range(10):
            for d in range(5):
                if i + d < 10:
                    assert torch.allclose(back[:, i, d], out[:, 9 - i - d, d], atol=1e-5)
        some = cells.clone()
        some[0, :4] = -1                                               # bins of no cell are left out
        assert torch.allclose(head(core, some, 10)[0, 1:], head(torch.cat([torch.randn(1, 4, 16), core[:1, 4:]], 1),
                                                                 some[:1], 10)[0, 1:], atol=1e-6)


def test_the_loss_skips_what_the_target_lacks_and_can_be_learned() -> None:
    torch.manual_seed(0)
    head = C.ContactHead(16, 16, 5, 1)
    core = torch.randn(1, 40, 16)
    cells = (torch.arange(40) // 4).unsqueeze(0)
    target = torch.rand(1, 10, 5, 1) * 3
    target[0, 2] = float("nan")
    i, d = np.indices((10, 5))
    target[0][torch.from_numpy(i + d >= 10)] = float("nan")
    assert float(C.contact_loss(head(core, cells), torch.full_like(target, float("nan")))) == 0.0
    filled = torch.nan_to_num(target, nan=1e4)
    pred = head(core, cells).detach()
    assert float(C.contact_loss(pred, target)) < float(C.contact_loss(pred, filled)) / 10
    opt = torch.optim.Adam(head.parameters(), 1e-2)
    first = None
    for _ in range(300):
        loss = C.contact_loss(head(core, cells), target)
        first = first if first is not None else float(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss) < first / 20
    r = C.contact_correlation(head(core, cells).detach(), target)
    assert r.shape == (1,) and float(r) > 0.9
    assert float(C.contact_correlation(C.contact_target(target).nan_to_num(), target)) > 0.999


def test_the_model_carries_the_head_as_new_parameters_only(monkeypatch) -> None:
    monkeypatch.setattr(kron_core.blocks, "kronlstm_mixer", CausalMixer)
    kw = dict(d_model=32, n_heads=2, n_layers=1, ff_units=64, stem_channels=8, n_down=3, arch="v2")
    torch.manual_seed(0)
    old = G.KronGenome(G.KronGenomeConfig(**kw)).eval()
    new = G.KronGenome(G.KronGenomeConfig(**kw, contact_tracks=2, contact_width=8, contact_reach=4)).eval()
    assert old.contact_head is None
    missing = new.load_state_dict(old.state_dict(), strict=False)
    assert missing.unexpected_keys == [] and all(k.startswith("contact_head.") for k in missing.missing_keys)
    seq = torch.randint(0, 4, (2, 512))
    with torch.no_grad():
        a, b = old(seq), new(seq)
        assert torch.allclose(a[0], b[0], atol=1e-6)                   # the model's own outputs are as before
    core = new(seq)[1]                                                 # [B, bins, d_model]
    cells = (torch.arange(64) // 8).expand(2, 64)
    pred = new.contact_head(core, cells)
    assert pred.shape == (2, 8, 4, 2)
    C.contact_loss(pred, torch.rand(2, 8, 4, 2)).backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in new.core.parameters())     # it trains the core
