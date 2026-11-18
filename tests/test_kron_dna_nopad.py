"""genome-train --short-sources pack (packing.py): sources shorter than the window are not padded to it.

As in test_kron_dna_multigenome.py the sampler lives in genome_train's closure, so these tests run the real command
on the CPU. ``HeadsStub`` stands in for KronGenome with every head's output in the right shape, so loss_on's own
code runs for each head and the tensors the sampler made can be compared one by one; the gradient test and the
two-process test use the real KronGenome with the CPU stand-in for the KronLSTM layer.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
import torch.nn.functional as F

from kron_dna import genome as G
from kron_dna import packing as PK

import test_kron_dna_multigenome as MGT
from test_kron_dna_multigenome import flat, make_genome, metric_rows

BASE = "1c8bad9"                      # backbone-v2.1, which this branch starts from and the running arms use
REAL_KRONGENOME = G.KronGenome
RECORD: list = []                     # what the sampler handed the losses, in order
MODELS: list = []


# ------------------------------------------------------------------------------------------------ the row ladder

def test_row_ladder_rule():
    lad = PK.row_ladder(524288, 128)
    assert lad[0] == (524288, 1) and lad[-1] == (1024, 512)
    assert [r for r, _ in lad] == sorted({r for r, _ in lad}, reverse=True)
    for r, n in lad:
        assert r % 128 == 0 and n == 524288 // r and n * r >= 7 / 8 * 524288
    assert (16384, 32) in lad and (14336, 36) in lad and (12288, 42) in lad and (262144, 2) in lad
    assert (393216, 1) not in lad and (327680, 1) not in lad            # one row of 3/4 or 5/8 fills too little
    # a contig always fills more than half of the row it is read in, and a row is never longer than its contig
    b = PK.Buckets(np.arange(1024, 524288, 37), None, lad)
    rows = np.array([lad[k][0] for k in b.rung])
    assert (rows <= b.lengths).all() and (rows * 7 > b.lengths * 4).all()
    assert (np.median(rows / b.lengths[: len(rows)]) > 0.85)
    assert PK.row_ladder(1024, 128) == [(1024, 1)]                       # nothing shorter than 8 core steps
    assert PK.row_ladder(8192, 128)[:3] == [(8192, 1), (7168, 1), (4096, 2)]
    assert int(b.rung_of(500)) == len(lad) - 1                           # shorter than every row: the shortest
    assert all(r % 128 == 0 for r, _ in PK.row_ladder(131072 + 128, 128))    # a window that is not a power of two


def test_buckets_draw_like_the_table_and_expose_every_base_alike():
    """Rows of a pack are drawn by the table's own probabilities within the bucket; over many windows drawn by
    length, every base of a by-length source is then read equally often (padding used to starve short contigs)."""
    rng = np.random.default_rng(3)
    lens = np.concatenate([rng.integers(10_000, 500_000, 60), rng.integers(600_000, 900_000, 3)])
    p = lens / lens.sum()
    window = 524288
    b = PK.Buckets(lens, p, PK.row_ladder(window, 128))
    cell = b.cell(int(lens[0]))
    assert set(cell.members) == {i for i, n in enumerate(lens) if b.rung_of(n) == b.rung_of(lens[0])}
    assert cell.p.sum() == pytest.approx(1) and cell.row <= lens[cell.members].min()
    real = np.zeros(len(lens))
    for _ in range(40_000):
        i = int(rng.choice(len(lens), p=p))
        if lens[i] >= window:
            real[i] += window
            continue
        c = b.cell(int(lens[i]))
        real[i] += c.row
        for _ in range(c.n - 1):
            real[c.draw(rng)] += c.row
    per_base = real / lens
    short = lens < window
    by_rung = {int(k): per_base[(b.rung == k) & short].mean() for k in set(b.rung[short])}
    # each bucket's bases are read n * row / window of the full rate (>= 7/8), instead of once per padded window
    for k, v in by_rung.items():
        row, n = b.ladder[k]
        assert v / per_base[~short].mean() == pytest.approx(n * row / window, rel=0.15), (k, v)
    uni = PK.Buckets(lens, None, PK.row_ladder(window, 128)).cell(int(lens[0]))
    draws = np.bincount([uni.draw(rng) for _ in range(4000)], minlength=len(lens))[uni.members]
    assert draws.min() > 0 and draws.max() < 3 * 4000 / len(uni.members) + 30   # uniform within the bucket


# ------------------------------------------------------------------------------------------------ prepared genomes

def add_tracks(d: Path, nd: Path | None, n: int, seed: int) -> Path:
    """NTv3-style base-major tracks (dense.u8 with nonzero_mean, as --ntv3-target alphagenome needs): for the
    primary genome in ``nd`` with tracks.json, for a further species in ``d/ntv3``. Split 1 (train) but a stretch."""
    rng = np.random.default_rng(seed)
    n_b = int((d / "bases.u8").stat().st_size)
    out = nd if nd is not None else d / "ntv3"
    out.mkdir(parents=True)
    rng.integers(0, 40, (n_b, n), dtype=np.uint8).tofile(out / "dense.u8")
    split = np.ones(n_b, dtype=np.uint8)
    split[2048:4096] = 3
    split.tofile(out / "split.u8")
    ids = [f"T{seed}_{k}" for k in range(n)]
    (out / "dense.json").write_text(json.dumps({"ids": ids, "total_bases": n_b, "n_tracks": n, "scales": [5.0] * n,
                                                "assays": ["ATAC-seq"] * n, "nonzero_mean": [3.0] * n}))
    if nd is not None:
        (out / "tracks.json").write_text(json.dumps({"tracks": [
            {"file_id": i, "assay": "ATAC-seq", "log1p_scale": 5.0} for i in ids]}))
    return out


def add_regfeat(d: Path, n_epi: int = 3) -> None:
    """Regulatory Build stand-in: a feature id per base (-1: none) and each feature's activity per epigenome."""
    rng = np.random.default_rng(11)
    n_b = int((d / "bases.u8").stat().st_size)
    ids = np.full(n_b, -1, dtype=np.int32)
    for k, at in enumerate(range(500, n_b - 400, 1500)):
        ids[at:at + 300] = k % 50
    ids.tofile(d / "regfeat.i32")
    np.save(d / "regfeat_active.npy", rng.random((50, n_epi)) < 0.5)


def add_repeat(d: Path) -> None:
    b = np.fromfile(d / "bases.u8", dtype=np.uint8)
    (b == 0).astype(np.uint8).tofile(d / "repeat.u8")


def build_world(root: Path, k: int = 1, tiny: bool = True) -> dict:
    """Arm D's sources in miniature, lengths in units of ``k`` bases: the primary hgt (chr8/chr9 held out), two
    further genomes (gB with a context of its own), two further species with tracks (gX mostly short contigs, as
    tomato is), and organelle genomes mt of ~16.5k (two genetic codes, every second one circular, one long, one
    very short with ``tiny``). Every source has contigs shorter and longer than 32k."""
    d = dict(root=root)
    d["hgt"] = make_genome(root, "hgt", {"chr1": 150 * k, "chr2": 70 * k, "chr3": 20 * k, "chr4": 12 * k,
                                         "chr8": 40 * k, "chr9": 36 * k}, 1)
    d["gA"] = make_genome(root, "gA", {"a1": 90 * k, "a2": 30 * k, "a3": 11 * k, "a4": 40 * k}, 2)
    d["gB"] = make_genome(root, "gB", {"b1": 80 * k, "b2": 14 * k, "b3": 35 * k}, 3, held={"b3"})
    d["gX"] = make_genome(root, "gX", {"x1": 70 * k, "x2": 30 * k, "x3": 11 * k, "x4": 10 * k + k // 2,
                                       "x5": 19 * k, "x6": 13 * k}, 4)
    d["gY"] = make_genome(root, "gY", {"y1": 50 * k, "y2": 12 * k}, 6)
    mt = {f"m{i}": 15 * k + k // 4 * i for i in range(12)}             # 15.0k .. 17.75k
    mt.update(m12=40 * k)
    if tiny:
        mt.update(m13=3 * k)
    d["mt"] = make_genome(root, "mt", mt, 5, held={"m0", "m3"}, mito_codes=[2, 5])
    d["nd"] = add_tracks(d["hgt"], root / "hg_ntv3", 3, 21)
    add_tracks(d["gX"], None, 2, 22)
    add_tracks(d["gY"], None, 2, 23)
    add_regfeat(d["hgt"])
    for n in ("hgt", "gA", "gX", "mt"):                                 # gB and gY have no repeat.u8: zeros
        add_repeat(d[n])
    return d


@pytest.fixture
def world(tmp_path):
    return build_world(tmp_path / "interim", 1000)


def arm_d_args(w: dict, more: list[str] = ()) -> list[str]:
    """Arm D's option string (v21/d2card_524.sbatch) on the miniature sources, with a small model."""
    return ["--name", "hgt", "--val-chroms", "chr8,chr9", "--arch", "v2", "--n-layers", "1", "--d-model", "16",
            "--stem-channels", "8", "--lr", "4e-4", "--warmup", "2", "--bert-mask", "--rc",
            "--epigenome-weight", "1.0", "--epigenome-focus", "0.05", "--ntv3-dir", str(w["nd"]),
            "--ntv3-assays", "all", "--ntv3-target", "alphagenome", "--ntv3-loss", "poisson", "--ntv3-center", "0.375",
            "--ntv3-extra", "gX,gY", "--ntv3-extra-share", "0.2", "--mito", "mt", "--mito-share", "0.05",
            "--genome", "gA:0.05:0:a4", "--genome", "gB:0.10:64", "--n-contexts", "256", "--n-in-extra", "8",
            "--soft-base-rate", "0.01", "--repeat-channel", "--repeat-channel-drop", "0.5", "--ntv3-weight", "0.5",
            "--jepa-weight", "1.0", "--jepa-ema", "0.0", "--jepa-reg", "sigreg", "--checkpoint-core",
            "--seed", "5", *more]


# ------------------------------------------------------------------------------------------- genome_train on CPU

class HeadsStub(torch.nn.Module):
    """Stands in for KronGenome with every head: outputs of the model's shapes from an embedding of the input, and
    a record of each forward's inputs (sequence or soft-base channels, segments, contexts, extra channels)."""

    def __init__(self, config):
        super().__init__()
        self.config, d = config, config.d_model
        self.emb, self.chan = torch.nn.Embedding(8, 4), torch.nn.Linear(6, 4)
        self.context = torch.nn.Embedding(max(config.n_contexts, 1), 4)
        self.core = torch.nn.Linear(4, d)
        self.jepa_pred = torch.nn.Linear(d, d) if config.jepa else None
        self.epi = torch.nn.Linear(d, config.n_epigenomes) if config.n_epigenomes else None
        self.ntv3 = torch.nn.Linear(4, config.n_ntv3) if config.n_ntv3 else None
        self.checkpoint_core = False
        self.last_ratio_loss = self.last_epi = self.last_ntv3 = self.last_jepa = self.last_chunks = None

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, seq, segments=None, context=None, extra=None):
        RECORD.append(("forward", self.training, seq.detach().clone(), segments.clone(),
                       None if context is None else context.clone(), None if extra is None else extra.clone()))
        x = self.chan(seq.transpose(1, 2)) if seq.dim() == 3 else self.emb(seq.long())        # [B, L, 4]
        if context is not None:
            x = x + self.context(context.long()).unsqueeze(1)
        unit = 2 ** self.config.n_down
        core = self.core(x.reshape(x.shape[0], x.shape[1] // unit, unit, 4).mean(2))
        self.last_jepa = self.jepa_pred(core) if self.jepa_pred is not None and self.training else None
        self.last_epi = self.epi(core) if self.epi is not None else None
        self.last_ntv3 = self.ntv3(x) if self.ntv3 is not None else None
        return x, core, None


class SpyGenome(REAL_KRONGENOME):
    """The real KronGenome, remembered (for its gradients) and with each forward's inputs recorded."""

    def __init__(self, config):
        super().__init__(config)
        MODELS.append(self)
        for n, q in self.named_parameters():      # the CPU stand-in layer never uses its A_log, which data-parallel
            if n.endswith(".A_log"):              # training refuses (a parameter without gradient); the real
                q.requires_grad_(False)           # KronLSTM layer has no such parameter

    def forward(self, seq, segments=None, context=None, extra=None):
        RECORD.append(("forward", self.training, seq.detach().clone(), segments.clone(),
                       None if context is None else context.clone(), None if extra is None else extra.clone()))
        return super().forward(seq, segments, context, extra)


def record_losses(monkeypatch):
    """Record what each loss is given (the sampler's tensors, after the model) and what it returns."""
    real_mask, real_bce, real_pm, real_ce, real_cos = (G.mask_bases, F.binary_cross_entropy_with_logits,
                                                        G.poisson_multinomial_loss, F.cross_entropy,
                                                        F.cosine_similarity)

    def mask_bases(seq, *a, **k):
        RECORD.append(("seq", seq.detach().clone()))
        return real_mask(seq, *a, **k)

    def bce(inp, target, *a, **k):
        out = real_bce(inp, target, *a, **k)
        RECORD.append(("epi", target.detach().clone(), float(out.detach().mean())))
        return out

    def pm(pred, target, mask, *a, **k):
        out = real_pm(pred, target, mask, *a, **k)
        RECORD.append(("ntv3", target.detach().clone(), mask.detach().clone(), float(out.detach())))
        return out

    def ce(inp, target, *a, **k):
        out = real_ce(inp, target, *a, **k)
        RECORD.append(("mlm", float(out.detach())))
        return out

    def cos(a_, b_, *a, **k):
        out = real_cos(a_, b_, *a, **k)
        RECORD.append(("jepa", float(out.detach().mean())))
        return out

    monkeypatch.setattr(G, "mask_bases", mask_bases)
    monkeypatch.setattr(F, "binary_cross_entropy_with_logits", bce)
    monkeypatch.setattr(G, "poisson_multinomial_loss", pm)
    monkeypatch.setattr(F, "cross_entropy", ce)
    monkeypatch.setattr(F, "cosine_similarity", cos)
    real_clip = torch.nn.utils.clip_grad_norm_

    def clip(params, *a, **k):               # once per optimizer step, after backward: the gradients per module
        if MODELS:
            by = {}
            for n, q in MODELS[0].named_parameters():
                if q.grad is not None:
                    by[n.split(".")[0]] = by.get(n.split(".")[0], 0.0) + float(q.grad.double().pow(2).sum())
            RECORD.append(("grad", {n: v ** 0.5 for n, v in by.items()}, len(PK.TRACE)))
        return real_clip(params, *a, **k)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)


def cpu(monkeypatch, model="stub"):
    """genome_train on the CPU (as test_kron_dna_multigenome.cpu_only) with HeadsStub, or the real KronGenome with
    the CPU stand-in for the KronLSTM layer; the losses' inputs are recorded either way."""
    MGT.cpu_only(monkeypatch)
    if model == "stub":
        monkeypatch.setattr(G, "KronGenome", HeadsStub)
    else:
        import kron_core.blocks
        from kron_core.testing import CausalMixer
        monkeypatch.setattr(kron_core.blocks, "kronlstm_mixer", CausalMixer)
        monkeypatch.setattr(G, "KronGenome", SpyGenome)
    record_losses(monkeypatch)
    monkeypatch.setattr(PK, "TRACE", [])


def base_cli(monkeypatch):
    """cli.py at BASE from ``git show``, or the file KRON_DNA_NOPAD_BASE_CLI names (nodes without git)."""
    repo = Path(__file__).resolve().parents[1]
    if os.environ.get("KRON_DNA_NOPAD_BASE_CLI"):
        src = Path(os.environ["KRON_DNA_NOPAD_BASE_CLI"]).read_text()
    else:
        try:
            src = subprocess.run(["git", "-C", str(repo), "show", f"{BASE}:src/kron_dna/cli.py"], capture_output=True,
                                 check=True, text=True).stdout
        except (OSError, subprocess.CalledProcessError):
            pytest.skip(f"commit {BASE} is not reachable from {repo} (set KRON_DNA_NOPAD_BASE_CLI to its cli.py)")
    assert "short_sources" not in src
    mod = types.ModuleType("kron_dna_cli_1c8bad9")
    mod.__file__ = f"{BASE}:src/kron_dna/cli.py"
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    from test_kron_dna_v23pre import with_audit_rng      # the audit's RNG changes (2026-11-18)
    src = with_audit_rng(src)
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod


def run(mod, monkeypatch, root: Path, out: Path, args: list[str]):
    """(result, record, trace) of one genome-train run."""
    from typer.testing import CliRunner
    monkeypatch.setattr(mod, "interim_dir", lambda *parts: root.joinpath(*parts))
    RECORD.clear()
    MODELS.clear()
    PK.TRACE.clear()
    torch.manual_seed(0)                       # the training masks draw from torch's global generator
    res = CliRunner().invoke(mod.app, ["genome-train", str(out), *args])
    assert res.exit_code == 0, res.output + repr(res.exception)
    return res, list(RECORD), list(PK.TRACE)


def same(a, b) -> bool:
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, (tuple, list)):
        return isinstance(b, (tuple, list)) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b or (isinstance(a, float) and isinstance(b, float) and a != a and b != b)


def first_difference(ra, rb):
    for i, (x, y) in enumerate(zip(ra, rb)):
        if not same(x, y):
            return i, x[0], y[0]
    return None if len(ra) == len(rb) else (min(len(ra), len(rb)), "length", (len(ra), len(rb)))


CURRICULA = {"8192": ["--curriculum", "8192:5", "--tokens-per-step", "32768"],
             "32768": ["--curriculum", "32768:5", "--tokens-per-step", "65536"],
             "8192-then-32768-accum2": ["--curriculum", "8192:3,32768:3", "--tokens-per-step", "131072",
                                        "--grad-accum", "2"]}


@pytest.mark.parametrize("case", list(CURRICULA))
def test_option_off_draws_what_1c8bad9_draws(world, tmp_path, monkeypatch, case):
    """Without the option, and with --short-sources pad, every tensor the sampler makes (sequences before and after
    masking, segments, contexts, repeat channel, cell-type and track targets and masks), every loss value and every
    metrics row equal those of the code at 1c8bad9 with arm D's options, at 8,192 and at a window longer than the
    organelle genomes and several contigs, where rows are padded."""
    cpu(monkeypatch)
    from kron_dna import cli as new
    old = base_cli(monkeypatch)
    args = arm_d_args(world, CURRICULA[case] + ["--eval-every", "3", "--eval-window", "16384"])
    _, rec_old, _ = run(old, monkeypatch, world["root"], tmp_path / "old", args)
    _, rec_new, trace = run(new, monkeypatch, world["root"], tmp_path / "new", args)
    _, rec_pad, _ = run(new, monkeypatch, world["root"], tmp_path / "pad", args + ["--short-sources", "pad"])
    kinds = {r[0] for r in rec_old}
    assert kinds == {"seq", "forward", "epi", "ntv3", "mlm", "jepa"}
    assert sum(r[0] == "forward" and r[1] for r in rec_old) >= 5
    assert first_difference(rec_old, rec_new) is None and first_difference(rec_old, rec_pad) is None
    assert metric_rows(tmp_path / "new" / "metrics.csv") == metric_rows(tmp_path / "old" / "metrics.csv") \
        == metric_rows(tmp_path / "pad" / "metrics.csv")
    padded = [t for t in trace if t[3] < t[2]]
    assert all(t[2] == t[1] and t[4] == 1 and t[5] == 1 for t in trace)          # every row a full window,
    assert padded and {t[0] for t in padded} >= {"mito"}                          # padded where the source is short
    if case == "8192":
        assert {t[3] for t in padded} == {3000}                                   # only the 3 kb organelle genome


def micro_batches(trace, batch: int):
    """The trace cut into micro-batches: each holds ``batch`` windows (a pack's first row opens a window)."""
    out, n = [], 0
    for t in trace:
        if t[5]:
            if n % batch == 0:
                out.append([])
            n += 1
        out[-1].append(t)
    return out


def training_forwards(rec):
    """Per training forward of the model that is trained: (raw sequence, model input, segments, contexts, repeat
    channel). The raw sequence is the last one mask_bases was given before it; the JEPA target's forward (in eval
    mode, on the unmasked sequence) follows and is skipped."""
    out, seq = [], None
    for r in rec:
        if r[0] == "seq":
            seq = r[1]
        elif r[0] == "forward" and r[1]:
            out.append((seq, r[2], r[3], r[4], r[5]))
    return out


def test_pack_is_pad_when_nothing_is_shorter_than_the_window(tmp_path, monkeypatch):
    """At 8,192 with no source shorter than that, --short-sources pack draws and computes exactly what pad does:
    tensors, loss values and metrics rows (64-row micro-batches stay one rectangular batch, one forward)."""
    w = build_world(tmp_path / "interim", 1000, tiny=False)
    cpu(monkeypatch)
    from kron_dna import cli as new
    args = arm_d_args(w, ["--curriculum", "8192:6", "--tokens-per-step", "524288", "--eval-every", "3",
                          "--eval-window", "16384"])
    _, rec_pad, tr_pad = run(new, monkeypatch, w["root"], tmp_path / "pad", args)
    res, rec_pack, tr_pack = run(new, monkeypatch, w["root"], tmp_path / "pack", args + ["--short-sources", "pack"])
    assert first_difference(rec_pad, rec_pack) is None and tr_pad == tr_pack
    assert len(tr_pack) == 6 * 64 and len(training_forwards(rec_pack)) == 6
    assert {t[0].split(":")[0] for t in tr_pack} == {"primary", "genome", "ntv3_extra", "mito"}
    assert metric_rows(tmp_path / "pad" / "metrics.csv") == metric_rows(tmp_path / "pack" / "metrics.csv")
    assert "short sources: a training window whose contig is shorter" in flat(res.output)
    assert "training rows since the last report (--short-sources pack)" in flat(res.output)


def test_packed_micro_batches(world, tmp_path, monkeypatch):
    """Window 32,768, four windows per micro-batch, no --rc (so rows can be matched to their genomes): every window
    is drawn from the sources as before; a short contig's window becomes rows of its ladder length without
    padding, in a part of their own; organelle rows carry their genetic code and cross the origin of circular
    genomes; the parts of a micro-batch hold at most its base budget."""
    root = world["root"]
    cpu(monkeypatch)
    from kron_dna import cli as new
    args = [a for a in arm_d_args(world) if a != "--rc"] + [
        "--curriculum", "32768:60", "--tokens-per-step", "131072", "--eval-every", "1000", "--eval-window", "16384",
        "--short-sources", "pack"]
    args[args.index("--mito-share") + 1] = "0.3"            # more organelle windows than arm D's 0.05
    res, rec, trace = run(new, monkeypatch, root, tmp_path / "run", args)
    lad = dict(PK.row_ladder(32768, 128))
    mbs = micro_batches(trace, 4)
    fwd = training_forwards(rec)
    assert len(mbs) == 60
    mt = pl.read_parquet(root / "genome" / "mt" / "contigs.parquet")
    mt_bases = np.fromfile(root / "genome" / "mt" / "bases.u8", dtype=np.uint8)
    mt_seqs = [(bytes(mt_bases[o:o + e - s_]), bool(c), int(code), bool(h)) for s_, e, o, c, code, h in
               mt.select("start", "end", "offset", "circular", "code", "held_out").iter_rows()]
    at, crossed, kinds = 0, 0, set()
    for mb in mbs:
        assert sum(t[5] for t in mb) == 4                                    # four windows, drawn as before
        assert sum(1 / t[4] for t in mb) == pytest.approx(4)                 # the parts' weights add up to one
        shapes = {}
        for label, window, row, real, n, lead in mb:
            assert window == 32768 and lad[row] == n
            assert real == row                                               # no padding at all (3 kb > 1,024)
            shapes[row] = shapes.get(row, 0) + 1
            kinds.add((label.split(":")[0], row < 32768))
        assert 4 * 32768 * 7 / 8 <= sum(r * k for r, k in shapes.items()) <= 4 * 32768
        for row in sorted(shapes, reverse=True):                             # one forward per row length
            seq, inp, seg, ctx, rep = fwd[at]
            at += 1
            assert tuple(seq.shape) == (shapes[row], row) and tuple(seg.shape) == (shapes[row], row // 128)
            assert inp.shape[0] == shapes[row] and inp.shape[-1] == row and rep.shape == (shapes[row], 8, row)
            assert int((seq == 4).sum()) == 0 and int(seg.sum()) == 0        # no N, no padding segment
            for b, label in enumerate([t[0] for t in mb if t[2] == row]):
                assert int(ctx[b]) in {"genome:gA": {0}, "genome:gB": {64}, "mito": {2, 5}}.get(label, {0})
                if label == "mito":
                    row_b = bytes(seq[b].numpy().astype(np.uint8))
                    hits = [m for m in mt_seqs if len(m[0]) >= row and row_b in (m[0] + m[0][:row] if m[1] else m[0])]
                    assert len(hits) == 1 and hits[0][2] == int(ctx[b])      # its genome, its genetic code,
                    assert not hits[0][3]                                    # never a held-out genome
                    assert row == 32768 or 4 * len(hits[0][0]) < 7 * row     # the longest row the genome fills
                    crossed += row_b not in hits[0][0]
    assert at == len(fwd) and crossed > 0                                    # some rows cross a circular origin
    assert kinds >= {("primary", False), ("primary", True), ("mito", False), ("mito", True), ("ntv3_extra", True),
                     ("genome", False), ("genome", True)}
    assert json.loads((tmp_path / "run" / "run_record.json").read_text())["options"]["short_sources"] == "pack"


@pytest.mark.parametrize("extra, words", [(["--short-sources", "crop"], "must be 'pad' or 'pack'"),
                                          (["--short-sources", "pack", "--whole-contigs"], "--whole-contigs")])
def test_cli_rejects(world, tmp_path, monkeypatch, extra, words):
    cpu(monkeypatch)
    from typer.testing import CliRunner
    from kron_dna import cli as new
    monkeypatch.setattr(new, "interim_dir", lambda *parts: world["root"].joinpath(*parts))
    res = CliRunner().invoke(new.app, ["genome-train", str(tmp_path / "run"), "--name", "hgt", "--arch", "v2",
                                       "--window", "8192", *extra])
    assert res.exit_code != 0 and words in flat(res.output)


# ------------------------------------------------------------------------- every head on packed micro-batches

REAL_MODEL = ["--d-model", "96"]            # the real KronGenome: 6 heads of 16


def real_args(w, more):
    a = arm_d_args(w, more)
    a[a.index("--d-model") + 1] = "96"
    return a


def test_every_head_trains_on_packed_micro_batches(world, tmp_path, monkeypatch):
    """The real KronGenome (CPU stand-in for the KronLSTM layer) with arm D's options and --short-sources pack, one
    window per micro-batch at 32,768: on packed micro-batches of each source every loss is finite and each head
    that source trains receives gradient - masked bases and JEPA everywhere, the cell-type head and the human
    track channels on primary packs, a species' track channels on its packs; organelle packs carry several genetic
    codes in one micro-batch."""
    cpu(monkeypatch, model="real")
    from kron_dna import cli as new
    args = real_args(world, ["--curriculum", "32768:40", "--tokens-per-step", "32768", "--eval-every", "1000",
                             "--eval-window", "16384", "--short-sources", "pack"])
    args[args.index("--mito-share") + 1] = "0.25"
    res, rec, trace = run(new, monkeypatch, world["root"], tmp_path / "run", args)
    steps, losses, ctxs, at = [], {}, [], 0
    for r in rec:
        if r[0] in ("mlm", "epi", "ntv3", "jepa"):
            losses.setdefault(r[0], []).append(r[-1])
        elif r[0] == "forward" and r[1]:
            ctxs.append(r[4])
        elif r[0] == "grad":
            steps.append((trace[at:r[2]], r[1], losses, ctxs))
            losses, ctxs, at = {}, [], r[2]
    assert len(steps) == 40
    seen = set()
    for rows, grads, loss, ctx in steps:
        assert sum(t[5] for t in rows) == 1                                   # one window per micro-batch
        kind, packed = rows[0][0].split(":")[0], len(rows) > 1
        assert all(np.isfinite(v) for vs in loss.values() for v in vs) and set(loss) == {"mlm", "epi", "ntv3", "jepa"}
        assert all(np.isfinite(v) for v in grads.values())
        if not packed:
            continue
        seen.add(kind)
        assert all(t[3] == t[2] < 32768 for t in rows) and len(ctx) == 1 and ctx[0].shape[0] == len(rows)
        for part in ("head", "stem", "down", "core", "up", "jepa_pred", "context"):
            assert grads[part] > 0, (kind, part)
        assert (grads["ntv3"] > 0) == (kind in ("primary", "ntv3_extra")), (kind, grads["ntv3"])
        assert (grads["epi_out"] > 0) == (kind == "primary"), (kind, grads["epi_out"])
        if kind == "mito":
            assert set(ctx[0].tolist()) <= {2, 5}
            seen.add("mito codes %d" % len(set(ctx[0].tolist())))
    assert seen >= {"primary", "mito", "ntv3_extra", "genome", "mito codes 2"}
    row = metric_rows(tmp_path / "run" / "metrics.csv")[-1]
    assert row[0] == "40" and np.isfinite(float(row[2])) and np.isfinite(float(row[3]))


# ------------------------------------------------------------------------------ data-parallel cards (two ranks)

def ddp_worker(root: str, out: str, result: str) -> None:
    """One rank of a two-process run on the CPU (gloo instead of nccl): the real training loop, the real KronGenome
    with the stand-in layer inside the real DistributedDataParallel."""
    import torch.distributed as dist
    mp = pytest.MonkeyPatch()
    cpu(mp, model="real")
    real_init, real_ddp = dist.init_process_group, torch.nn.parallel.DistributedDataParallel
    mp.setattr(dist, "init_process_group", lambda backend, **k: real_init("gloo", **k))
    mp.setattr(torch.cuda, "set_device", lambda *a, **k: None)
    mp.setattr(torch.nn.parallel, "DistributedDataParallel", lambda m, device_ids=None: real_ddp(m))
    from kron_dna import cli as new
    w = dict(nd=Path(root) / "hg_ntv3")
    args = real_args(w, ["--curriculum", "32768:14", "--tokens-per-step", "131072", "--eval-every", "7",
                         "--eval-window", "16384", "--short-sources", "pack"])
    args[args.index("--mito-share") + 1] = "0.25"
    res, rec, trace = run(new, mp, Path(root), Path(out), args)
    steps, shapes = [], []
    for r in rec:
        if r[0] == "forward" and r[1]:
            shapes.append([int(r[2].shape[0]), int(r[2].shape[-1])])
        elif r[0] == "grad":
            steps.append(dict(shapes=shapes, grads=r[1]))
            shapes = []
    Path(result).write_text(json.dumps(dict(steps=steps, output=res.output[-3000:])))
    mp.undo()


def test_two_cards_with_different_shapes_stay_in_step(world, tmp_path):
    """Two data-parallel ranks draw independently, so with pack their micro-batches differ in row length, in rows
    and in the number of parts (forward passes); each rank still calls backward once per micro-batch, where the
    gradients are averaged. After every step both ranks hold the same gradients."""
    import socket
    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0))
        port = sk.getsockname()[1]
    procs = []
    for rank in (0, 1):
        env = dict(os.environ, RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2", MASTER_ADDR="127.0.0.1",
                   MASTER_PORT=str(port), OMP_NUM_THREADS="2",
                   PYTHONPATH=os.pathsep.join([str(Path(__file__).parent), str(Path(G.__file__).parents[1]),
                                               os.environ.get("PYTHONPATH", "")]))
        code = (f"import test_kron_dna_nopad as T; T.ddp_worker({str(world['root'])!r}, {str(tmp_path / 'run')!r}, "
                f"{str(tmp_path / f'rank{rank}.json')!r})")
        procs.append(subprocess.Popen([sys.executable, "-c", code], env=env, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True))
    outs = [q.communicate(timeout=900)[0] for q in procs]
    assert all(q.returncode == 0 for q in procs), "\n".join(o[-3000:] for o in outs)
    r0, r1 = (json.loads((tmp_path / f"rank{k}.json").read_text()) for k in (0, 1))
    assert len(r0["steps"]) == len(r1["steps"]) == 14
    assert "data-parallel over 2 cards" in flat(r0["output"])
    differ = sum(a["shapes"] != b["shapes"] for a, b in zip(r0["steps"], r1["steps"]))
    parts = sum(len(a["shapes"]) != len(b["shapes"]) for a, b in zip(r0["steps"], r1["steps"]))
    assert differ >= 5 and parts >= 2, (differ, parts)                    # other shapes, other numbers of parts
    assert max(len(a["shapes"]) for a in r0["steps"] + r1["steps"]) >= 2   # several forwards under one backward
    for a, b in zip(r0["steps"], r1["steps"]):
        assert a["grads"].keys() == b["grads"].keys()
        for k in a["grads"]:
            assert a["grads"][k] == pytest.approx(b["grads"][k], rel=1e-6, abs=1e-12), k
        assert a["grads"]["head"] > 0
    assert len(metric_rows(tmp_path / "run" / "metrics.csv")) == 3         # rank 0 evaluated at steps 7 and 14


# ---------------------------------------------------------------- padding and shares at arm D's window and shares

def big_world(root: Path) -> dict:
    """Sources with arm D's kinds of length distribution at full scale: a primary and a first further genome that
    are almost all long (hg38, CHM13), a second with a few short contigs (the mole-rat assembly), a species of
    mostly short contigs (tomato: 86 % of its draws are shorter than 524 kb) and one of a few (yeast), and
    organelle genomes around 16.5 kb with a tail of long ones, ~1 % longer than the window, a few under 8 kb."""
    rng = np.random.default_rng(17)
    d = dict(root=root)
    short = {f"u{i}": int(n) for i, n in enumerate(np.exp(rng.uniform(np.log(10_000), np.log(500_000), 12)))}
    d["hgt"] = make_genome(root, "hgt", {"chr1": 2_600_000, "chr2": 1_800_000, **short, "chr8": 600_000}, 1)
    d["gA"] = make_genome(root, "gA", {"a1": 1_500_000, "a4": 560_000}, 2)
    d["gB"] = make_genome(root, "gB", {"b1": 1_300_000, "b2": 105_000, "b3": 560_000}, 3, held={"b3"})
    tom = {f"x{i}": int(n) for i, n in enumerate(np.exp(rng.uniform(np.log(10_000), np.log(520_000), 40)))}
    d["gX"] = make_genome(root, "gX", {**tom, "xl": 700_000}, 4)
    d["gY"] = make_genome(root, "gY", {"y1": 900_000, "y2": 640_000, "y3": 440_000, "y4": 270_000}, 6)
    lens = np.concatenate([rng.integers(13_600, 16_384, 126), rng.integers(16_384, 17_600, 144),
                           np.exp(rng.uniform(np.log(18_000), np.log(300_000), 25)).astype(int),
                           [560_000, 610_000, 720_000], [4_800, 6_100]])
    rng.shuffle(lens)
    d["mt"] = make_genome(root, "mt", {f"m{i}": int(n) for i, n in enumerate(lens)}, 5,
                          held={f"m{i}" for i in range(0, 300, 30) if lens[i] < 20_000}, mito_codes=[2, 5])
    d["nd"] = add_tracks(d["hgt"], root / "hg_ntv3", 2, 21)
    add_tracks(d["gX"], None, 1, 22)
    add_tracks(d["gY"], None, 1, 23)
    return d


SHARES = {"genome:gA": 0.05, "genome:gB": 0.10, "mito": 0.05, "ntv3_extra": 0.8 * 0.2, "primary": 0.8 * 0.8}


def class_table(trace, shares: dict | None = None) -> list[dict]:
    """Per source class, over the training windows of a trace: windows, their share (and the configured one with
    ``shares``), rows per window, real bases per window (one micro-batch when a card holds one window), the
    padded share of the bases the model computes on, and the real bases inside the central 37.5 % of their row
    (where --ntv3-center 0.375 supervises the tracks: a padded row holds its contig at the start, so a contig
    shorter than 31 % of the window had none)."""
    by: dict = {}
    for label, window, row, real, n, lead in trace:
        k = label if label in (shares or {label: 0}) else label.split(":")[0]
        t = by.setdefault(k, dict(windows=0, rows=0, real=0, computed=0, budget=0, per_window=[], centre=0))
        t["centre"] += max(0.0, min(real, 0.6875 * row) - 0.3125 * row)
        if lead:
            t["windows"] += 1
            t["budget"] += window
            t["per_window"].append(0)
        t["rows"] += 1
        t["real"] += real
        t["computed"] += row
        t["per_window"][-1] += real
    total = sum(t["windows"] for t in by.values())
    out = []
    for k in sorted(by):
        t = by[k]
        pw = np.array(t["per_window"])
        out.append(dict(source=k, windows=t["windows"], share=t["windows"] / total,
                        configured=None if shares is None else shares.get(k),
                        rows_per_window=t["rows"] / t["windows"], real_bases_mean=float(pw.mean()),
                        real_bases_median=float(np.median(pw)), real_bases_min=int(pw.min()),
                        real_of_budget=t["real"] / t["budget"], padded=1 - t["real"] / t["computed"],
                        centre_real_per_window=t["centre"] / t["windows"],
                        windows_mostly_padding=float((pw < 0.5 * t["computed"] / t["windows"]).mean())))
    out.append(dict(source="all", windows=total, share=1.0, configured=None if shares is None else 1.0,
                    rows_per_window=sum(t["rows"] for t in by.values()) / total,
                    real_bases_mean=sum(t["real"] for t in by.values()) / total, real_bases_median=None,
                    real_bases_min=min(min(t["per_window"]) for t in by.values()),
                    real_of_budget=sum(t["real"] for t in by.values()) / sum(t["budget"] for t in by.values()),
                    padded=1 - sum(t["real"] for t in by.values()) / sum(t["computed"] for t in by.values()),
                    centre_real_per_window=sum(t["centre"] for t in by.values()) / total,
                    windows_mostly_padding=None))
    return out


def light_losses(monkeypatch):
    """The sampler is what a draw simulation measures: mask a few bases without drawing 524,288 random numbers."""
    def mask_bases(seq, *a, **k):
        labels = torch.full_like(seq, -100)
        labels[..., ::4096] = seq[..., ::4096].masked_fill(seq[..., ::4096] > 3, -100)
        return seq, labels
    monkeypatch.setattr(G, "mask_bases", mask_bases)


N_DRAWS = int(os.environ.get("KRON_NOPAD_DRAWS", "2000"))


def test_padding_and_shares_at_524k(tmp_path, monkeypatch):
    """N_DRAWS (2,000) training micro-batches of one 524,288 window, as each of arm D's cards draws them, with arm
    D's shares (CHM13 0.05, mole-rat 0.10, organelles 0.05, further species 0.2 of the rest), pad against pack:
    the realised source shares follow the configured ones in both; pad leaves organelle micro-batches ~97 % N and
    the short-contig species mostly N; pack leaves no padding and at least 7/8 of the base budget real."""
    w = big_world(tmp_path / "interim")
    MGT.cpu_only(monkeypatch)
    monkeypatch.setattr(PK, "TRACE", [])
    light_losses(monkeypatch)
    from kron_dna import cli as new
    args = ["--name", "hgt", "--val-chroms", "chr8", "--arch", "v2", "--n-layers", "1", "--d-model", "16",
            "--stem-channels", "8", "--warmup", "2", "--rc", "--ntv3-dir", str(w["nd"]), "--ntv3-assays", "all",
            "--ntv3-extra", "gX,gY", "--ntv3-extra-share", "0.2", "--mito", "mt", "--mito-share", "0.05",
            "--genome", "gA:0.05:0:a4", "--genome", "gB:0.10:64", "--n-contexts", "256", "--seed", "5",
            "--curriculum", f"524288:{N_DRAWS}", "--tokens-per-step", "524288", "--eval-every", "1000000",
            "--eval-window", "8192"]
    tables = {}
    for mode in ("pad", "pack"):
        _, _, trace = run(new, monkeypatch, w["root"], tmp_path / mode, args + ["--short-sources", mode])
        assert sum(t[5] for t in trace) == N_DRAWS
        tables[mode] = {r["source"]: r for r in class_table(trace, SHARES)}
    if os.environ.get("KRON_NOPAD_TABLE"):
        Path(os.environ["KRON_NOPAD_TABLE"]).write_text(json.dumps(tables, indent=1))
    for mode, tab in tables.items():
        for k, cfg in SHARES.items():                                         # shares: within 4 binomial sigmas
            assert abs(tab[k]["share"] - cfg) < 4 * (cfg * (1 - cfg) / N_DRAWS) ** 0.5, (mode, k, tab[k]["share"])
    pad, pack = tables["pad"], tables["pack"]
    assert pad["mito"]["padded"] > 0.85 and pad["mito"]["real_bases_median"] < 20_000
    assert pad["mito"]["windows_mostly_padding"] > 0.95 and pad["ntv3_extra"]["padded"] > 0.3
    assert all(r["rows_per_window"] == 1 for r in pad.values())
    for k, r in pack.items():
        assert r["padded"] == 0 and r["real_bases_min"] >= 7 / 8 * 524288, (k, r)
        assert r["real_of_budget"] >= pad[k]["real_of_budget"] - 1e-9
    assert pack["mito"]["real_of_budget"] > 0.97 and pack["mito"]["rows_per_window"] > 25
    assert pack["ntv3_extra"]["real_of_budget"] > 0.93 and pack["all"]["real_of_budget"] > 0.98
    assert pack["genome:gA"]["rows_per_window"] == 1                         # nothing short there: untouched
    # track supervision (central 37.5 % of each row): padded short contigs sat outside it, packed rows are inside
    assert pack["ntv3_extra"]["centre_real_per_window"] > 0.36 * 7 / 8 * 524288
    assert pad["ntv3_extra"]["centre_real_per_window"] < 0.75 * pack["ntv3_extra"]["centre_real_per_window"]
