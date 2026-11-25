"""Block JEPA (``genome-train --jepa-block-share``): block placement, the online input inside blocks, the losses
that leave block positions out, jepa.csv, and the default path against the code at 1c8bad9.

The CLI tests run the real ``genome-train`` command on the CPU as ``test_kron_dna_multigenome.py`` does (its helpers
are reused): with a stub model that records what it is fed, and, for the test with every head on, with the real
KronGenome whose KronLSTM core (CUDA only) is replaced by a small mixing of neighbouring bins.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from kron_dna import genome as G

import test_kron_dna_multigenome as MGT
from test_kron_dna_multigenome import cpu_only, make_genome, make_ntv3, metric_rows, run

BASE = "1c8bad9"        # the commit this branch starts from: without the new options nothing may differ from it
UNIT = 128
ROWS = {8192: 64, 131072: 1024, 524288: 4096}      # row length in bases -> bins of 128


# ------------------------------------------------------------------------------------------------- placement

def test_parse_block_len():
    assert G.parse_block_len("1024:8192", 128) == (8, 64)
    assert G.parse_block_len("2048", 128) == (16, 16)
    assert G.parse_block_len("16:64", 16) == (1, 4)
    for bad in ["", "a:b", "1000:8192", "1024:8000", "8192:1024", "0:1024", "1024:2048:4096", "-128:128"]:
        with pytest.raises(ValueError):
            G.parse_block_len(bad, 128)


def log_uniform_p(lo: int, hi: int) -> np.ndarray:
    """P(l) for l = lo..hi of the discrete log-uniform draw."""
    l_ = np.arange(lo, hi + 1)
    return np.log((l_ + 1) / l_) / math.log((hi + 1) / lo)


@pytest.mark.parametrize("bases", [8192, 131072, 524288])
@pytest.mark.parametrize("share", [0.2, 0.35])
def test_place_blocks_share_overlap_and_lengths(bases, share):
    """For 8,192 / 131,072 / 524,288-base rows (64 / 1,024 / 4,096 bins), blocks of 1,024-8,192 bases: the hidden
    bins are exactly round(share * n); blocks lie inside the row, never overlap or touch; every length is in
    8..64 bins (or the row's whole quota when that is smaller); the lengths follow the log-uniform law."""
    n, lo, hi = ROWS[bases], 8, 64
    rng = np.random.default_rng(11)
    k = int(math.floor(share * n + 0.5))
    lens, cover = [], np.zeros(n)
    reps = {64: 4000, 1024: 1500, 4096: 500}[n]
    for _ in range(reps):
        blocks = G.place_blocks(n, share, lo, hi, rng)
        assert sum(length for _, length in blocks) == k                       # the share, to half a bin
        assert abs(k / n - share) <= 0.5 / n
        prev_end = -1
        for start, length in blocks:
            assert start > prev_end and start >= 0 and start + length <= n    # sorted, a visible bin between
            assert min(lo, k) <= length <= min(hi, k)
            cover[start:start + length] += 1
            prev_end = start + length
        lens += [length for _, length in blocks]
    lens = np.array(lens)
    if k <= hi:                              # 64 bins: the quota (13 or 22 bins) in one block, or in two of >= 8
        assert set(lens.tolist()) == {13} if k == 13 else lens.min() >= 8 and lens.max() == k
    else:                                    # octave shares against the log-uniform law
        want = log_uniform_p(lo, hi)
        edges = [8, 16, 32, 65]
        got = [float(((lens >= a) & (lens < b)).mean()) for a, b in zip(edges, edges[1:])]
        exp = [float(want[a - lo:b - lo].sum()) for a, b in zip(edges, edges[1:])]
        # each row cuts its last draw to the quota and may adjust the block before it, so the fewer blocks a row
        # has, the further from the pure law: 4,096 bins ~30 blocks, 1,024 bins ~8
        tol = 0.03 if n == 4096 else 0.09
        assert np.abs(np.array(got) - np.array(exp)).max() < tol, (got, exp)
        assert abs(float(np.log(lens).mean()) - float((want * np.log(np.arange(lo, hi + 1))).sum())) < (
            0.05 if n == 4096 else 0.2)
    # blocks lie wholly inside the row, so its ends are hidden less often and the middle a little more: with a
    # single block in 64 bins the middle half is hidden 0.25 / 0.43 of the time for a share of 0.2 / 0.35
    mid = float((cover[n // 4: 3 * n // 4] / reps).mean())
    assert -0.005 <= mid - share <= (0.1 if n == 64 else 0.02), mid
    assert float(cover.mean() / reps) == pytest.approx(k / n)


def test_place_blocks_small_and_extreme_rows():
    rng = np.random.default_rng(3)
    assert G.place_blocks(1, 0.2, 8, 64, rng) == [] and G.place_blocks(2, 0.2, 8, 64, rng) == []
    assert G.place_blocks(64, 0.0, 8, 64, rng) == []
    for n in range(3, 200):                                   # any row length: the quota, inside the row, disjoint
        for share in (0.2, 0.5, 0.9):
            for lo, hi in ((8, 64), (1, 1), (4, 6), (16, 16)):
                blocks = G.place_blocks(n, share, lo, hi, rng)
                k = min(int(math.floor(share * n + 0.5)), n - 1)
                hid = np.zeros(n, dtype=int)
                for start, length in blocks:
                    assert 0 <= start and start + length <= n and length >= 1
                    hid[start:start + length] += 1
                assert hid.max(initial=0) <= 1 and hid.sum() == k
                if hi >= 2 * lo - 1 and k >= lo:
                    assert all(lo <= length <= hi for _, length in blocks)
    assert G.place_blocks(64, 0.2, 8, 64, np.random.default_rng(5)) == G.place_blocks(64, 0.2, 8, 64,
                                                                                      np.random.default_rng(5))


def test_block_mask_rows_padding_and_alignment():
    """Rows of different content: full rows, a padded row whose last sequence bin carries padding, a row that is
    all padding, a row of several sequence runs. Padding is never hidden, each run gets its own quota, and the base
    mask of hide_blocks is whole bins."""
    rng = np.random.default_rng(7)
    bins, n_rows = 64, 6
    seq = rng.integers(0, 4, (n_rows, bins * UNIT))
    seg = np.zeros((n_rows, bins), dtype=np.int64)
    take = 3000                                               # row 1: 3,000 bases = 23 full bins + 56 bases
    seq[1, take:] = 4
    seg[1, (take + UNIT - 1) // UNIT:] = 1
    seg[2, :] = 1                                             # row 2: padding only
    seq[2, :] = 4
    seg[3, 20:30] = 1                                         # row 3: two runs, 20 and 34 bins
    seq[3, 20 * UNIT:30 * UNIT] = 4
    seq[4, 40 * UNIT:] = 4                                    # row 4: 40 full bins, then padding
    seg[4, 40:] = 1
    seq[5, -1] = 4                                            # row 5: full, but its last base is N
    counts = np.zeros((n_rows, bins))
    for _ in range(400):
        m = G.block_mask(seg, seq, UNIT, 0.25, 4, 16, rng)
        assert m.shape == seg.shape and m.dtype == bool
        assert not m[seg != 0].any()                          # never padding
        assert not m[1, 23].any() and not m[5, 63]            # nor a sequence bin that carries padding / ends in N
        assert m[0].sum() == 16 and m[1].sum() == 6 and m[2].sum() == 0
        assert m[3, :20].sum() == 5 and m[3, 30:].sum() == 9 and m[4].sum() == 10      # 8.5 rounds up
        assert m[5].sum() == 16
        counts += m
    assert (counts[0] > 0).all()                              # every bin of a full row can be hidden
    blocks = torch.from_numpy(G.block_mask(seg, seq, UNIT, 0.25, 4, 16, rng))
    inputs = torch.from_numpy(seq)
    out, tg, extra, at = G.hide_blocks(inputs, torch.full_like(inputs, 2), blocks, UNIT,
                                       extra=torch.ones(n_rows, 3, bins * UNIT))
    assert torch.equal(at.reshape(n_rows, bins, UNIT).all(-1), blocks)        # whole bins,
    assert torch.equal(at.reshape(n_rows, bins, UNIT).any(-1), blocks)        # and nothing else
    assert (out[at] == G.MASK_CHANNEL).all() and torch.equal(out[~at], inputs[~at])
    assert (tg[at] == -100).all() and (tg[~at] == 2).all()
    assert (extra.transpose(1, 2)[at] == 0).all() and (extra.transpose(1, 2)[~at] == 1).all()
    _, tg_in, _, _ = G.hide_blocks(inputs, torch.full_like(inputs, 2), blocks, UNIT, exclude=False)
    assert (tg_in == 2).all()
    no_seq = G.block_mask(seg, None, UNIT, 0.25, 4, 16, rng)                  # without the bases: segments alone
    assert not no_seq[seg != 0].any() and no_seq[0].sum() == 16


# ------------------------------------------------------------------------------------------------- no leak

def online_input(seq, blocks, seed, rep, soft=0.3, bert=True):
    """The online input as genome-train's loss_on builds it: masks, blocks, soft bases, the repeat channel."""
    torch.manual_seed(seed)
    inputs, targets = G.mask_bases(seq, 0.15, bert=bert)
    inputs, targets, _, at = G.hide_blocks(inputs, targets, blocks, UNIT)
    x = G.soft_base_channels(inputs, targets, soft)
    extra = torch.zeros(seq.shape[0], 2, seq.shape[1])
    extra[:, G.EXTRA_INPUT_CHANNELS["repeat"]] = rep.float()
    return x, targets, extra.masked_fill(at.unsqueeze(1), 0.0), at


def test_online_input_inside_blocks_does_not_depend_on_the_hidden_bases():
    """Change every hidden base (to other bases, to N) and the hidden repeat flags: the online model's input (base
    channels and extra channels) and the masked-LM targets are identical, everywhere."""
    rng = np.random.default_rng(21)
    bins = 64
    seq = torch.from_numpy(rng.integers(0, 5, (4, bins * UNIT)))
    rep = torch.from_numpy(rng.integers(0, 2, (4, bins * UNIT)))
    seg = np.zeros((4, bins), dtype=np.int64)
    blocks = torch.from_numpy(G.block_mask(seg, None, UNIT, 0.3, 4, 16, rng))
    assert int(blocks.sum()) == 4 * 19
    x1, t1, e1, at = online_input(seq, blocks, 9, rep)
    seq2, rep2 = seq.clone(), rep.clone()
    seq2[at] = torch.from_numpy(rng.integers(0, 5, int(at.sum())))
    rep2[at] = 1 - rep2[at]
    assert (seq2[at] != seq[at]).float().mean() > 0.7
    x2, t2, e2, _ = online_input(seq2, blocks, 9, rep2)
    assert torch.equal(x1, x2) and torch.equal(e1, e2) and torch.equal(t1, t2)
    inside = x1.transpose(1, 2)[at]                           # [n, 6]: MASK and nothing else
    assert (inside[:, G.MASK_CHANNEL] == 1).all() and (inside.sum(-1) == 1).all()
    assert (e1.transpose(1, 2)[at] == 0).all() and (t1[at] == -100).all()
    # outside the blocks the scattered masking goes on: ~15% of the known bases are targets
    known = (seq < 4) & ~at
    assert abs(float((t1[known] != -100).float().mean()) - 0.15) < 0.02
    # a different visible base does change the input (the test can fail)
    seq3 = seq.clone()
    seq3[~at] = (seq3[~at] + 1) % 4
    x3, _, _, _ = online_input(seq3, blocks, 9, rep)
    assert not torch.equal(x1, x3)


# ------------------------------------------------------------------------------- losses that leave blocks out

def test_excluding_block_positions_equals_deleting_them():
    """Each head's loss with block positions excluded is the same loss on the row with those positions removed: a
    mean over what is left, not the old sum over fewer terms."""
    torch.manual_seed(4)
    b, bins, t = 2, 16, 5
    length = bins * 8
    blocks = torch.zeros(b, bins, dtype=torch.bool)
    blocks[:, 3:7] = True
    blocks[:, 11:13] = True
    at = blocks.repeat_interleave(8, dim=1)
    # masked-base cross-entropy
    logits = torch.randn(b, length, 4)
    targets = torch.randint(0, 4, (b, length)).masked_fill(torch.rand(b, length) > 0.3, -100)
    _, tg, _, _ = G.hide_blocks(torch.zeros(b, length, dtype=torch.long), targets, blocks, 8)
    got = F.cross_entropy(logits.reshape(-1, 4), tg.reshape(-1), ignore_index=-100)
    want = F.cross_entropy(logits[~at], targets[~at], ignore_index=-100)
    assert got.item() == pytest.approx(want.item(), rel=1e-6)
    # Poisson-multinomial track loss: the mask without the block positions == the tensors without them
    pred, target = torch.rand(b, length, t) + 0.1, torch.rand(b, length, t) * 3
    ok = torch.ones(b, length)
    ok[:, :10] = 0                                            # some positions unsupervised anyway
    cols = torch.ones(b, t)
    cols[1, 3:] = 0
    got = G.poisson_multinomial_loss(pred, target, (ok * (~at).float()).unsqueeze(-1) * cols.unsqueeze(1))
    keep = ~at[0]                                             # the same columns in both rows here
    want = G.poisson_multinomial_loss(pred[:, keep], target[:, keep], ok[:, keep].unsqueeze(-1) * cols.unsqueeze(1))
    assert got.item() == pytest.approx(want.item(), rel=1e-5)
    assert got.item() != pytest.approx(G.poisson_multinomial_loss(
        pred, target, ok.unsqueeze(-1) * cols.unsqueeze(1)).item(), rel=1e-3)
    # epigenome loss: a weighted mean over the visible bins
    per = torch.rand(b, bins, 3)
    w = torch.rand(b, bins)
    w_x = w * (~blocks).float()
    got = (per * w_x.unsqueeze(-1)).sum() / (w_x.sum().clamp_min(1) * 3)
    want = (per[~blocks] * w[~blocks].unsqueeze(-1)).sum() / (w[~blocks].sum().clamp_min(1) * 3)
    assert got.item() == pytest.approx(want.item(), rel=1e-6)


# --------------------------------------------------------------------------------------------- genome_train on CPU

class JepaStub(torch.nn.Module):
    """A stub model with core states and a JEPA predictor that records what it is fed (bases, extra channels)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.unit = 2 ** config.n_down
        self.emb = torch.nn.Embedding(8, 4)
        self.lin = torch.nn.Linear(G.N_IN, 4)
        self.to_core = torch.nn.Linear(4, 8)
        self.jepa_pred = torch.nn.Linear(8, 8) if config.jepa else None
        self.last_ratio_loss = self.last_epi = self.last_ntv3 = self.last_jepa = self.last_chunks = None

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, seq, segments=None, context=None, extra=None):
        MGT.EVENTS.append(("forward", self.training, None if context is None else tuple(context.tolist()),
                           seq.detach().clone(), None if extra is None else extra.detach().clone()))
        x = self.lin(seq.transpose(1, 2)) if seq.is_floating_point() else self.emb(seq.long())
        b, length = x.shape[0], x.shape[1]
        core = self.to_core(x.reshape(b, length // self.unit, self.unit, 4).mean(2))
        core = core + 0.5 * (core.roll(1, 1) + core.roll(-1, 1))            # a bin's state sees its neighbours
        if extra is not None:
            core = core + extra.sum(1).reshape(b, length // self.unit, self.unit).mean(2).unsqueeze(-1)
        self.last_jepa = self.jepa_pred(core) if self.jepa_pred is not None and self.training else None
        return x, core, None


def stub_jepa(monkeypatch):
    cpu_only(monkeypatch)
    monkeypatch.setattr(G, "KronGenome", JepaStub)


def cli_at_base(monkeypatch):
    """cli.py at BASE from ``git show`` (or from the file KRON_DNA_BASE_CLI_1C8BAD9 names)."""
    repo = Path(__file__).resolve().parents[1]
    if os.environ.get("KRON_DNA_BASE_CLI_1C8BAD9"):
        src = Path(os.environ["KRON_DNA_BASE_CLI_1C8BAD9"]).read_text()
    else:
        try:
            src = subprocess.run(["git", "-C", str(repo), "show", f"{BASE}:src/kron_dna/cli.py"], capture_output=True,
                                 check=True, text=True).stdout
        except (OSError, subprocess.CalledProcessError):
            pytest.skip(f"commit {BASE} is not reachable from {repo} (set KRON_DNA_BASE_CLI_1C8BAD9 to its cli.py)")
    mod = types.ModuleType("kron_dna_cli_1c8bad9")
    mod.__file__ = f"{BASE}:src/kron_dna/cli.py"
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    from test_kron_dna_v23pre import with_audit_rng      # the audit's RNG changes (2026-11-18)
    src = with_audit_rng(src)
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod


@pytest.fixture
def jgenomes(tmp_path):
    """hgj: the primary, long enough for 4,096-base windows (32 bins), chr3 shorter than a window (a padded row with
    a partly padded last bin); mt: an organelle set of short sequences; both with a repeat.u8 (1 where the base is
    A)."""
    root = tmp_path / "interim"
    d = dict(root=root)
    d["hgj"] = make_genome(root, "hgj", {"chr1": 40000, "chr2": 30000, "chr3": 3000, "chr8": 20000, "chr9": 12000}, 1)
    d["mt"] = make_genome(root, "mt", {f"m{i}": 900 + 37 * i for i in range(10)}, 5, held={"m0", "m3"},
                          mito_codes=[2, 5])
    for k in ("hgj", "mt"):
        b = np.fromfile(d[k] / "bases.u8", dtype=np.uint8)
        (b == 0).astype(np.uint8).tofile(d[k] / "repeat.u8")
    return d


SMALLJ = ["--name", "hgj", "--arch", "v2", "--window", "4096", "--warmup", "1", "--d-model", "16", "--n-layers", "1",
          "--stem-channels", "8", "--seed", "5"]
JEPA = ["--jepa-weight", "1.0", "--jepa-ema", "0.0", "--jepa-reg", "sigreg"]
RECIPE = ["--bert-mask", "--mito", "mt", "--mito-share", "0.2", "--n-contexts", "64", "--n-in-extra", "2",
          "--soft-base-rate", "0.05", "--repeat-channel", "--repeat-channel-drop", "0.0"]


def same_events(a, b):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if len(x) != len(y) or x[:3] != y[:3]:
            return False
        for u, v in zip(x[3:], y[3:]):
            if isinstance(u, torch.Tensor) or isinstance(v, torch.Tensor):
                if not (isinstance(u, torch.Tensor) and isinstance(v, torch.Tensor) and u.shape == v.shape
                        and torch.equal(u, v)):
                    return False
            elif u != v:
                return False
    return True


def jepa_notes(text: str) -> list[str]:
    return re.findall(r"jepa [0-9.]+ \(core std [0-9.]+\)", " ".join(text.split()))


@pytest.mark.parametrize("case", ["jepa-sigreg", "jepa-recipe-rc", "jepa-vicreg-ema"])
def test_options_off_is_the_code_at_1c8bad9(jgenomes, tmp_path, monkeypatch, case):
    """Without --jepa-block-share (and --jepa-log-every) a JEPA run is the run of 1c8bad9 for a fixed seed: every
    window read, every model input (so every sampled mask, soft base and repeat channel), every context id, every
    metrics row (train loss with its JEPA and SIGReg terms, held-out bits) and the logged JEPA loss and core std;
    and no jepa.csv is written."""
    root = jgenomes["root"]
    args = SMALLJ + ["--steps", "6", "--eval-every", "3", "--batch-size", "3"]
    if case == "jepa-sigreg":
        args += JEPA
    elif case == "jepa-recipe-rc":
        args += JEPA + RECIPE + ["--rc", "--grad-accum", "2"]
    else:
        args += ["--jepa-weight", "0.5", "--jepa-ema", "0.9", "--jepa-reg", "vicreg", "--bert-mask"]
    stub_jepa(monkeypatch)
    from kron_dna import cli as new
    old = cli_at_base(monkeypatch)
    res_old, ev_old = run(old, monkeypatch, root, tmp_path / "old", args)
    assert res_old.exit_code == 0, res_old.output + repr(res_old.exception)
    res_new, ev_new = run(new, monkeypatch, root, tmp_path / "new", args)
    assert res_new.exit_code == 0, res_new.output + repr(res_new.exception)
    n_train = sum(e[0] == "forward" and e[1] for e in ev_new)
    assert n_train == (12 if case == "jepa-recipe-rc" else 6)
    assert same_events(ev_new, ev_old)
    assert metric_rows(tmp_path / "new" / "metrics.csv") == metric_rows(tmp_path / "old" / "metrics.csv")
    assert len(metric_rows(tmp_path / "new" / "metrics.csv")) == 3
    notes = jepa_notes(res_new.output)
    assert len(notes) == 2 and notes == jepa_notes(res_old.output)
    assert not (tmp_path / "new" / "jepa.csv").exists()
    assert "block JEPA" not in res_new.output and "jepa step" not in res_new.output


def spy_on(monkeypatch, name, seen):
    real = getattr(G, name)

    def spy(*a, **k):
        out = real(*a, **k)
        seen.append((a, k, out))
        return out

    monkeypatch.setattr(G, name, spy)


def test_blocks_through_the_cli(jgenomes, tmp_path, monkeypatch):
    """With --jepa-block-share: the online model reads MASK and a zero repeat channel inside the blocks and nothing
    of the hidden bases; the target reads the clean row with its repeat channel; the scattered masks (and the
    windows) are those of the run without blocks; padding is never hidden; evaluation is untouched; jepa.csv gets
    its rows and metrics.csv keeps its header."""
    root = jgenomes["root"]
    args = SMALLJ + JEPA + RECIPE + ["--steps", "8", "--eval-every", "4", "--batch-size", "4"]
    stub_jepa(monkeypatch)
    from kron_dna import cli as new
    masks = {"off": [], "on": []}
    which = ["off"]
    real_mask = G.mask_bases

    def mask_spy(*a, **k):
        out = real_mask(*a, **k)
        masks[which[0]].append(out)
        return out

    monkeypatch.setattr(G, "mask_bases", mask_spy)
    blocks = []
    spy_on(monkeypatch, "block_mask", blocks)
    res0, ev0 = run(new, monkeypatch, root, tmp_path / "off", args)
    assert res0.exit_code == 0, res0.output + repr(res0.exception)
    assert not blocks
    which[0] = "on"
    res1, ev1 = run(new, monkeypatch, root, tmp_path / "on", args + [
        "--jepa-block-share", "0.25", "--jepa-block-len", "256:1024", "--jepa-log-every", "2"])
    assert res1.exit_code == 0, res1.output + repr(res1.exception)
    masks0, masks1 = masks["off"], masks["on"]
    assert "block JEPA: 0.25 of each row's bins hidden in blocks of 256-1,024 bases" in " ".join(res1.output.split())
    # the same windows, the same scattered masks: the blocks draw from their own stream
    rd = [e for e in ev1 if e[0] == "read"]
    assert rd == [e for e in ev0 if e[0] == "read"] and len(rd) > 30
    assert len(masks1) == len(masks0) > 8 and all(
        torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(masks1, masks0))
    fw1 = [e for e in ev1 if e[0] == "forward"]
    fw0 = [e for e in ev0 if e[0] == "forward"]
    assert len(fw1) == len(fw0) and [e[1:3] for e in fw1] == [e[1:3] for e in fw0]
    assert len(blocks) == 8                                                   # one draw per training micro-batch
    k = 0
    n_hidden = n_seq = padded_rows = 0
    for i, e in enumerate(fw1):
        if not e[1]:
            continue
        (seg, seq, unit, share, lo, hi, _), _, m = blocks[k]
        k += 1
        assert (unit, share, lo, hi) == (128, 0.25, 2, 8)
        at = torch.from_numpy(m).repeat_interleave(128, dim=1)
        assert not m[seg != 0].any()                                         # padding is never hidden
        for row in range(m.shape[0]):
            n_run = int((seg[row] == 0).sum())
            ends_in_pad = n_run < m.shape[1] or seq[row, -1] == 4
            n_ok = n_run - (1 if ends_in_pad and seq[row, n_run * 128 - 1] == 4 else 0)
            assert m[row].sum() == min(int(math.floor(0.25 * n_ok + 0.5)), max(n_ok - 1, 0))
            padded_rows += n_run < m.shape[1]
            n_hidden += int(m[row].sum())
            n_seq += n_run
        x, extra = e[3], e[4]                                                # the online model's input
        assert x.is_floating_point() and x.shape[1] == G.N_IN
        inside = x.transpose(1, 2)[at]
        assert (inside[:, G.MASK_CHANNEL] == 1).all() and (inside.sum(-1) == 1).all()
        assert (extra.transpose(1, 2)[at] == 0).all()
        tgt = fw1[i + 1]                                                     # then the target's forward: clean
        assert not tgt[1] and torch.equal(tgt[3], torch.from_numpy(np.asarray(seq)))
        rep_true = (torch.from_numpy(np.asarray(seq)) == 0).float()          # repeat.u8 is 1 where the base is A
        assert torch.equal(tgt[4][:, 0], rep_true) and (tgt[4][:, 1] == 0).all()
        assert torch.equal(extra[:, 0], rep_true.masked_fill(at, 0.0))       # outside blocks: the channel as it was
        # outside the blocks the online input is bit for bit what the run without blocks fed (same masks, same
        # soft-base draws: the blocks take no number from the streams the rest of the run draws from)
        assert torch.equal(x.transpose(1, 2)[~at], fw0[i][3].transpose(1, 2)[~at])
    assert k == 8 and padded_rows >= 2 and n_hidden > 100
    assert same_events([e for e in fw1 if not e[1] and e[4] is None], [e for e in fw0 if not e[1] and e[4] is None])
    # metrics.csv: the header of the run without blocks; jepa.csv: rows at steps 2, 4, 6, 8
    assert (tmp_path / "on" / "metrics.csv").read_text().split("\n")[0] == \
        (tmp_path / "off" / "metrics.csv").read_text().split("\n")[0]
    assert not (tmp_path / "off" / "jepa.csv").exists()
    lines = (tmp_path / "on" / "jepa.csv").read_text().strip().split("\n")
    head = lines[0].split(",")
    assert head == ["step", "jepa_hidden", "jepa_visible", "copy_hidden", "copy_visible", "hidden_share", "core_std",
                    "core_std_visible", "core_std_hidden", "train_mlm", "train_epigenome", "train_ntv3",
                    "train_total", "sec_per_step", "peak_gib", "window", "steps"]
    rows = [dict(zip(head, map(float, line.split(",")))) for line in lines[1:]]
    assert [r["step"] for r in rows] == [2, 4, 6, 8] and all(r["steps"] == 2 and r["window"] == 4096 for r in rows)
    for r in rows:
        assert all(math.isfinite(r[c]) for c in head if c not in ("train_epigenome", "train_ntv3"))
        assert r["copy_hidden"] > r["copy_visible"] > 0                      # a blanked bin copies worse
    assert abs(sum(r["hidden_share"] for r in rows) / 4 - n_hidden / n_seq) < 0.02
    assert len(re.findall(r"jepa step \d+: hidden bins", res1.output)) == 4


def test_jepa_on_all_heads_include_and_log_without_blocks(jgenomes, tmp_path, monkeypatch):
    """--jepa-on all, --jepa-block-heads include; and --jepa-log-every without blocks writes jepa.csv (hidden
    columns nan) while the run's metrics stay those of the run without it."""
    root = jgenomes["root"]
    args = SMALLJ + JEPA + ["--bert-mask", "--steps", "4", "--eval-every", "2", "--batch-size", "4"]
    stub_jepa(monkeypatch)
    from kron_dna import cli as new
    res0, ev0 = run(new, monkeypatch, root, tmp_path / "off", args)
    assert res0.exit_code == 0, res0.output + repr(res0.exception)
    res1, ev1 = run(new, monkeypatch, root, tmp_path / "log", args + ["--jepa-log-every", "1"])
    assert res1.exit_code == 0, res1.output + repr(res1.exception)
    assert same_events(ev0, ev1)
    assert metric_rows(tmp_path / "log" / "metrics.csv") == metric_rows(tmp_path / "off" / "metrics.csv")
    lines = (tmp_path / "log" / "jepa.csv").read_text().strip().split("\n")
    head = lines[0].split(",")
    rows = [dict(zip(head, line.split(","))) for line in lines[1:]]
    assert [r["step"] for r in rows] == ["1", "2", "3", "4"]
    assert all(r["jepa_hidden"] == "nan" and r["copy_hidden"] == "nan" and float(r["hidden_share"]) == 0 for r in rows)
    assert all(math.isfinite(float(r["jepa_visible"])) and math.isfinite(float(r["copy_visible"])) for r in rows)
    assert jepa_notes(res1.output) == jepa_notes(res0.output)
    seen = []
    spy_on(monkeypatch, "hide_blocks", seen)
    res2, _ = run(new, monkeypatch, root, tmp_path / "all", args + [
        "--jepa-block-share", "0.2", "--jepa-block-len", "512:1024", "--jepa-on", "all", "--jepa-block-heads",
        "include"])
    assert res2.exit_code == 0, res2.output + repr(res2.exception)
    assert len(seen) == 4 and all(k["exclude"] is False for _, k, _ in seen)
    assert all(((out[1] != -100) & out[3]).any() for _, _, out in seen)      # targets inside blocks are kept
    assert "JEPA loss on all bins; other heads include block positions" in " ".join(res2.output.split())


@pytest.mark.parametrize("more,words", [
    (["--jepa-block-share", "0.2", "--whole-contigs"], "does not take --whole-contigs"),
    (JEPA + ["--jepa-block-share", "0.95"], "must lie in [0, 0.9]"),
    (JEPA + ["--jepa-block-share", "0.2", "--jepa-block-len", "1000:8192"], "multiples of 128"),
    (JEPA + ["--jepa-block-share", "0.2", "--jepa-on", "some"], "--jepa-on must be"),
    (JEPA + ["--jepa-block-share", "0.2", "--jepa-block-heads", "maybe"], "--jepa-on must be"),
    (["--jepa-log-every", "5"], "needs --jepa-weight"),
])
def test_option_checks(jgenomes, tmp_path, monkeypatch, more, words):
    stub_jepa(monkeypatch)
    from kron_dna import cli as new
    res, _ = run(new, monkeypatch, jgenomes["root"], tmp_path / "bad", SMALLJ + ["--steps", "2", "--batch-size", "2"]
                 + more)
    assert res.exit_code != 0 and words in " ".join(res.output.replace("│", " ").split())


# ------------------------------------------------------------------------------ every head on, the real model

class CpuCoreGenome(G.KronGenome):
    """The real KronGenome (stem, encoder, decoder, every head, the JEPA predictor) with the CUDA-only KronLSTM core
    replaced by a mixing of neighbouring bins, so a tiny model runs on the CPU."""

    def run_core(self, h, cu, checkpoint=False):
        h = h + 0.5 * (h.roll(1, 1) + h.roll(-1, 1)) + 0.25 * (h.roll(3, 1) + h.roll(-3, 1))
        return self.core_norm(h)


def add_heads_data(d: Path, n_epi: int = 3) -> None:
    """regfeat.i32 / regfeat_active.npy (the epigenome head) and labels.u8 / activity.u8 (--labels) for a test
    genome."""
    rng = np.random.default_rng(13)
    n = int(np.fromfile(d / "bases.u8", dtype=np.uint8).size)
    feat = np.full(n, -1, dtype=np.int32)
    n_feat = 40
    for f_ in range(n_feat):
        a = int(rng.integers(0, n - 600))
        feat[a:a + int(rng.integers(100, 600))] = f_
    feat.tofile(d / "regfeat.i32")
    np.save(d / "regfeat_active.npy", rng.random((n_feat, n_epi)) < 0.5)
    rng.integers(0, 256, n, dtype=np.uint8).tofile(d / "labels.u8")
    rng.integers(0, 2, n, dtype=np.uint8).tofile(d / "activity.u8")


def add_dense(nd: Path, extra: Path) -> None:
    """--ntv3-target alphagenome reads the primary's tracks from a base-major dense.u8 with each track's non-zero
    mean; the further species' dense.json needs the means too."""
    cols = [np.fromfile(nd / "tracks" / f"{i}.u8", dtype=np.uint8) for i in ("T1", "T2")]
    np.stack(cols, 1).tofile(nd / "dense.u8")
    (nd / "dense.json").write_text(json.dumps({"ids": ["T1", "T2"], "total_bases": len(cols[0]), "n_tracks": 2,
                                               "nonzero_mean": [3.0, 5.0]}))
    xj = json.loads((extra / "ntv3" / "dense.json").read_text())
    (extra / "ntv3" / "dense.json").write_text(json.dumps(dict(xj, nonzero_mean=[2.0, 3.0, 4.0])))


@pytest.mark.parametrize("heads", ["exclude", "include"])
def test_every_head_on_losses_finite_and_gradients_flow(jgenomes, tmp_path, monkeypatch, heads):
    """The real model (CPU core stand-in) with the masked-base head, the NTv3 Poisson-multinomial head, the
    epigenome head, the --labels track head, contexts, soft bases, the repeat channel, --rc, SIGReg and block JEPA:
    every loss term is finite at every step, and the first backward reaches the stem, the encoder, each head and the
    JEPA predictor."""
    root = jgenomes["root"]
    add_heads_data(jgenomes["hgj"])
    gx = make_genome(root, "gX", {"x1": 30000, "x2": 20000}, 2)
    nd = make_ntv3(root, jgenomes["hgj"], gx)
    add_dense(nd, gx)
    cpu_only(monkeypatch)
    monkeypatch.setattr(G, "KronGenome", CpuCoreGenome)
    from kron_dna import cli as new
    grads = []
    real_clip = torch.nn.utils.clip_grad_norm_

    def clip(params, max_norm, *a, **k):
        params = list(params)
        grads.append([None if p.grad is None else float(p.grad.norm()) for p in params])
        return real_clip(params, max_norm, *a, **k)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    names = []
    real_init = CpuCoreGenome.__init__

    def init(self, config):
        real_init(self, config)
        names[:] = [n for n, _ in self.named_parameters()]

    monkeypatch.setattr(CpuCoreGenome, "__init__", init)
    small = [a if a != "16" else "96" for a in SMALLJ]        # the real model: 6 heads of at least 16 dimensions
    args = small + JEPA + RECIPE + ["--rc", "--steps", "6", "--eval-every", "3", "--batch-size", "4",
                                    "--epigenome-weight", "1.0", "--epigenome-focus", "0.05", "--labels",
                                     "--ntv3-dir", str(nd), "--ntv3-loss", "poisson", "--ntv3-target", "alphagenome", "--ntv3-center", "0.75",
                                     "--ntv3-weight", "0.5", "--ntv3-extra", "gX", "--ntv3-extra-share", "0.3",
                                     "--jepa-block-share", "0.25", "--jepa-block-len", "256:1024",
                                     "--jepa-block-heads", heads, "--jepa-log-every", "1"]
    res, _ = run(new, monkeypatch, root, tmp_path / "heads", args)
    assert res.exit_code == 0, res.output + repr(res.exception)
    assert len(grads) == 6 and len(names) == len(grads[0])
    first = dict(zip(names, grads[0]))
    for g_ in grads:
        assert all(v is None or math.isfinite(v) for v in g_)
    for part in ("stem.", "down.", "up.", "head.", "jepa_pred.", "epi_in.", "epi_out.", "ntv3.", "tracks.",
                 "context."):
        got = [v for n, v in first.items() if n.startswith(part)]
        assert got and any(v is not None and v > 0 for v in got), (part, got)
    lines = (tmp_path / "heads" / "jepa.csv").read_text().strip().split("\n")
    head = lines[0].split(",")
    rows = [dict(zip(head, map(float, line.split(",")))) for line in lines[1:]]
    assert len(rows) == 6
    for r in rows:
        assert all(math.isfinite(v) for v in r.values()), r
        assert 0.15 < r["hidden_share"] < 0.30 and r["jepa_hidden"] > 0 and r["train_mlm"] > 0
        assert r["train_ntv3"] != 0 and r["train_epigenome"] > 0
    m = metric_rows(tmp_path / "heads" / "metrics.csv")
    assert len(m) == 3 and all(math.isfinite(float(v)) for row in m[1:] for v in row)
    ck = torch.load(tmp_path / "heads" / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert ck["run_record"]["options"]["jepa_block_share"] == 0.25
    assert ck["run_record"]["options"]["jepa_block_heads"] == heads


def test_no_new_parameters_and_old_checkpoints_resume(jgenomes, tmp_path, monkeypatch):
    """Block JEPA adds no parameter: a JEPA run trained without blocks (as 1c8bad9 writes it) resumes with blocks
    on, model and optimizer state loaded strictly, and its state-dict keys are unchanged afterwards."""
    root = jgenomes["root"]
    stub_jepa(monkeypatch)
    from kron_dna import cli as new
    old = cli_at_base(monkeypatch)
    out = tmp_path / "run"
    args = SMALLJ + JEPA + ["--bert-mask", "--batch-size", "4", "--eval-every", "2"]
    res, _ = run(old, monkeypatch, root, out, args + ["--curriculum", "4096:2,4096:4", "--stop-step", "2"])
    assert res.exit_code == 0, res.output + repr(res.exception)
    before = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert before["step"] == 2
    res, _ = run(new, monkeypatch, root, out, args + ["--curriculum", "4096:2,4096:4", "--jepa-block-share", "0.2",
                                                      "--jepa-block-len", "512:1024"])
    assert res.exit_code == 0, res.output + repr(res.exception)
    assert "resumed at step 2" in res.output
    after = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert after["step"] == 6 and list(after["model"]) == list(before["model"])
    assert [r[0] for r in metric_rows(out / "metrics.csv")] == ["step", "2", "4", "6"]
    assert [line.split(",")[0] for line in (out / "jepa.csv").read_text().strip().split("\n")] == ["step", "4", "6"]


def test_blocks_at_jepa_weight_zero_are_the_jepa_off_twin(jgenomes, tmp_path, monkeypatch):
    """--jepa-block-share at --jepa-weight 0: the same blocks hidden from the same inputs as the run with JEPA on
    and --jepa-reg-own-stream (SIGReg's directions off the global stream), and no target, predictor or jepa.csv, so
    the pair differs in the latent loss alone."""
    root = jgenomes["root"]
    base = SMALLJ + RECIPE + ["--steps", "8", "--eval-every", "4", "--batch-size", "4",
                              "--jepa-block-share", "0.25", "--jepa-block-len", "256:1024"]
    stub_jepa(monkeypatch)
    from kron_dna import cli as new
    blocks = {"on": [], "off": []}
    which = ["on"]
    real = G.block_mask

    def spy(*a, **k):
        out = real(*a, **k)
        blocks[which[0]].append(out)
        return out

    monkeypatch.setattr(G, "block_mask", spy)
    res1, ev1 = run(new, monkeypatch, root, tmp_path / "on", base + JEPA + ["--jepa-reg-own-stream"])
    assert res1.exit_code == 0, res1.output + repr(res1.exception)
    which[0] = "off"
    res0, ev0 = run(new, monkeypatch, root, tmp_path / "off", base + ["--jepa-weight", "0"])
    assert res0.exit_code == 0, res0.output + repr(res0.exception)
    assert len(blocks["off"]) == len(blocks["on"]) == 8
    assert all(np.array_equal(a, b) for a, b in zip(blocks["off"], blocks["on"]))
    on_train = [e for e in ev1 if e[0] == "forward" and e[1]]
    off_train = [e for e in ev0 if e[0] == "forward" and e[1]]
    assert len(off_train) == 8                                              # online passes only: no target
    assert len(on_train) == 8 and len([e for e in ev1 if e[0] == "forward"]) > len([e for e in ev0 if e[0] == "forward"])
    assert all(torch.equal(a[3], b[3]) and torch.equal(a[4], b[4]) for a, b in zip(off_train, on_train))
    assert not (tmp_path / "off" / "jepa.csv").exists()
    assert "block JEPA: 0.25 of each row's bins hidden" in " ".join(res0.output.split())
