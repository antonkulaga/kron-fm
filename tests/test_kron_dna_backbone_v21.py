"""backbone-v2.1 integration: --soft-base-rate, --repeat-channel, --new-contexts, context ids under --n-contexts,
and the run record (written at startup, stored in checkpoints, checked on resume).

The CLI tests run the real ``genome-train`` command on the CPU with the recording memmap and a stub model, as
``test_kron_dna_multigenome.py`` does (its helpers are reused). The default path is covered there: without any of the
new options every read, context id and metrics row equals 3c90c8c's, which also shows that no random number is drawn
for them.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from kron_dna import genome as G
from kron_dna import runrecord as RR

import test_kron_dna_multigenome as MGT
from test_kron_dna_multigenome import SMALL, cpu_only, flat, genomes, run  # noqa: F401  (genomes is a fixture)


class InputStub(torch.nn.Module):
    """A stub model that records what it is fed: integer or fractional bases, the extra channels, the contexts."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.emb = torch.nn.Embedding(8, 4)
        self.lin = torch.nn.Linear(G.N_IN, 4)
        self.last_ratio_loss = self.last_epi = self.last_ntv3 = self.last_jepa = self.last_chunks = None

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, seq, segments=None, context=None, extra=None):
        MGT.EVENTS.append(("forward", self.training, None if context is None else tuple(context.tolist()),
                           seq.detach().clone(), None if extra is None else extra.detach().clone()))
        if seq.is_floating_point():
            return self.lin(seq.transpose(1, 2)), None, None
        return self.emb(seq.long()), None, None


def stub_inputs(monkeypatch):
    cpu_only(monkeypatch)
    monkeypatch.setattr(G, "KronGenome", InputStub)


def forwards(ev, training=None):
    return [e for e in ev if e[0] == "forward" and (training is None or e[1] == training)]


def reads(ev, skip_repeat=False):
    return [e for e in ev if e[0] == "read" and not (skip_repeat and e[1].endswith("repeat.u8"))]


def add_repeat(d):
    """repeat.u8 for a test genome: 1 where the base is A, so the channel can be checked against the sequence."""
    b = np.fromfile(d / "bases.u8", dtype=np.uint8)
    (b == 0).astype(np.uint8).tofile(d / "repeat.u8")


# ------------------------------------------------------------------------------------------------ fractional bases

def test_soft_base_channels_mixes_only_free_bases():
    torch.manual_seed(0)
    seq = torch.randint(0, 5, (4, 20_000))
    inputs, targets = G.mask_bases(seq, prob=0.15, bert=True)
    rate = 0.2
    x = G.soft_base_channels(inputs, targets, rate, generator=torch.Generator().manual_seed(1))
    hard = G.one_hot(inputs)
    assert x.shape == (4, G.N_IN, 20_000) and x.dtype == torch.float32
    assert torch.allclose(x.sum(1), torch.ones(4, 20_000))
    changed = (x != hard).any(1)
    free = (inputs < 4) & (targets == -100)
    assert not changed[~free].any()                                   # targets, N and MASK never mixed
    assert (x[:, 4][inputs == 4] == 1).all()                          # N stays N
    frac = float(changed[free].float().mean())
    assert abs(frac - rate) < 0.01                                    # ~5.7 SD at n ~ 54k
    xm = x.permute(0, 2, 1)[changed]                                  # [n, 6]
    true = inputs[changed]
    assert (xm[torch.arange(len(true)), true] == 0.5).all()
    assert ((xm == 0.5).sum(1) == 2).all() and (xm[:, 4:] == 0).all()
    half = xm[:, :4] == 0.5
    half[torch.arange(len(true)), true] = False                       # what is left is the other base
    other = half.float().argmax(1)
    assert (half.sum(1) == 1).all() and (other != true).all()
    shift = (other - true) % 4                                        # 1, 2, 3 alike
    counts = torch.bincount(shift, minlength=4)[1:].float() / len(shift)
    assert (counts - 1 / 3).abs().max() < 0.02
    assert torch.equal(G.soft_base_channels(inputs, targets, 0.0), hard)
    every = (G.soft_base_channels(inputs, None, 1.0) != hard).any(1)
    assert every[inputs < 4].all() and not every[inputs >= 4].any()


def test_soft_base_rate_through_the_cli(genomes, tmp_path, monkeypatch):
    """R > 0: training forwards get [B, 6, L] float input with ~R of the free bases mixed and every masked-LM target,
    N and MASK hard; evaluation forwards get the integer codes. The windows drawn are those of a run without it."""
    stub_inputs(monkeypatch)
    from kron_dna import cli as new
    seen = []
    real = G.soft_base_channels

    def spy(inputs, targets, rate, generator=None):
        out = real(inputs, targets, rate, generator)
        seen.append((inputs.clone(), targets.clone(), out))
        return out

    monkeypatch.setattr(G, "soft_base_channels", spy)
    args = SMALL + ["--steps", "4", "--eval-every", "2", "--batch-size", "4", "--rc", "--bert-mask"]
    res, ev = run(new, monkeypatch, genomes["root"], tmp_path / "soft", args + ["--soft-base-rate", "0.3"])
    assert res.exit_code == 0, res.output + repr(res.exception)
    tr, evl = forwards(ev, True), forwards(ev, False)
    assert len(tr) == 4 and all(e[3].is_floating_point() and e[3].shape[1] == G.N_IN for e in tr)
    assert evl and all(not e[3].is_floating_point() for e in evl)
    assert len(seen) == 4
    mixed = free = 0
    for inputs, targets, out in seen:
        ch = (out != G.one_hot(inputs)).any(1)
        assert not ch[targets != -100].any() and not ch[inputs >= 4].any()
        f = (inputs < 4) & (targets == -100)
        mixed += int(ch[f].sum())
        free += int(f.sum())
    assert free > 5_000 and abs(mixed / free - 0.3) < 0.03            # SD ~0.006 at n ~ 6k
    _, ev0 = run(new, monkeypatch, genomes["root"], tmp_path / "hard", args)
    assert reads(ev) == reads(ev0)                                    # the same windows
    assert [e[2] for e in forwards(ev)] == [e[2] for e in forwards(ev0)]


def test_soft_base_rate_zero_feeds_the_integer_codes(genomes, tmp_path, monkeypatch):
    """R = 0: the helper is never called, so the input tensor (integer codes) and the torch RNG stream are those of a
    run without the option (the default-path tests against 3c90c8c cover the metrics rows)."""
    stub_inputs(monkeypatch)
    from kron_dna import cli as new
    args = SMALL + ["--steps", "2", "--eval-every", "2", "--batch-size", "2"]
    res0, ev0 = run(new, monkeypatch, genomes["root"], tmp_path / "b", args)
    assert res0.exit_code == 0, res0.output + repr(res0.exception)
    monkeypatch.setattr(G, "soft_base_channels", lambda *a, **k: pytest.fail("called at R = 0"))
    res, ev = run(new, monkeypatch, genomes["root"], tmp_path / "a", args + ["--soft-base-rate", "0"])
    assert res.exit_code == 0, res.output + repr(res.exception)
    assert all(not e[3].is_floating_point() for e in forwards(ev))
    assert len(forwards(ev)) == len(forwards(ev0))
    assert all(torch.equal(a[3], b[3]) for a, b in zip(forwards(ev), forwards(ev0)))
    assert MGT.metric_rows(tmp_path / "a" / "metrics.csv") == MGT.metric_rows(tmp_path / "b" / "metrics.csv")


# ------------------------------------------------------------------------------------------------ repeat channel

def test_repeat_channel_feeds_repeat_u8_in_training_only(genomes, tmp_path, monkeypatch):
    root = genomes["root"]
    add_repeat(genomes["hgt"])
    add_repeat(genomes["gA"])                                    # gB and mt have none: they feed zeros
    stub_inputs(monkeypatch)
    from kron_dna import cli as new
    common = SMALL + ["--steps", "6", "--eval-every", "3", "--batch-size", "12", "--rc", "--mito", "mt",
                      "--mito-share", "0.15", "--genome", "gA:0.25::a4", "--genome", "gB:0.25", "--n-in-extra", "3"]
    res, ev = run(new, monkeypatch, root, tmp_path / "rep", common + ["--repeat-channel", "--repeat-channel-drop", "0"])
    assert res.exit_code == 0, res.output + repr(res.exception)
    assert "gB has no repeat.u8" in flat(res.output) and "mt has no repeat.u8" in flat(res.output)
    tr, evl = forwards(ev, True), forwards(ev, False)
    assert all(e[4] is not None and e[4].shape == (12, 3, e[3].shape[1]) for e in tr)
    assert all(e[4] is None for e in evl)                        # evaluation: channel zeroed (not fed)
    rows = MGT.rows_by_forward([e[:3] for e in ev])
    n_checked = {"hgt": 0, "gA": 0, "zero": 0}
    for (training, _, row_reads), fwd in zip(rows, forwards(ev)):
        if not training:
            continue
        seq, extra = fwd[3], fwd[4]
        assert (extra[:, 1:] == 0).all()
        for b, rr in enumerate(row_reads):
            gname = rr[0][1].split("/")[0]
            chan, s_ = extra[b, 0], seq[b]
            vis = s_ < 4
            if gname in ("hgt", "gA"):
                assert any(r[1] == f"{gname}/repeat.u8" for r in rr)
                as_read = torch.equal(chan[vis], (s_[vis] == 0).float())
                reversed_ = torch.equal(chan[vis], (s_[vis] == 3).float())   # rc: bases complemented, channel not
                assert as_read or reversed_
                n_checked[gname] += 1
            else:
                assert (chan == 0).all() and not any(r[1].endswith("repeat.u8") for r in rr)
                n_checked["zero"] += 1
    assert min(n_checked.values()) > 0, n_checked
    # the same windows, contexts and held-out numbers as without the channel (the stub ignores it)
    res0, ev0 = run(new, monkeypatch, root, tmp_path / "norep", common)
    assert res0.exit_code == 0, res0.output + repr(res0.exception)
    assert reads(ev, skip_repeat=True) == reads(ev0)
    assert [e[2] for e in forwards(ev)] == [e[2] for e in forwards(ev0)]
    assert MGT.metric_rows(tmp_path / "rep" / "metrics.csv") == MGT.metric_rows(tmp_path / "norep" / "metrics.csv")


def test_repeat_channel_drop_and_requirements(genomes, tmp_path, monkeypatch):
    root = genomes["root"]
    add_repeat(genomes["hgt"])
    stub_inputs(monkeypatch)
    from kron_dna import cli as new
    base = SMALL + ["--steps", "8", "--eval-every", "100", "--batch-size", "40", "--n-in-extra", "1", "--repeat-channel"]
    res, ev = run(new, monkeypatch, root, tmp_path / "half", base + ["--repeat-channel-drop", "0.5"])
    assert res.exit_code == 0, res.output + repr(res.exception)
    fed = torch.cat([(e[4][:, 0] > 0).any(1) for e in forwards(ev, True)])
    assert len(fed) == 320 and 0.40 < float(fed.float().mean()) < 0.60      # ~0.5 of windows keep it (3.6 SD)
    res, ev = run(new, monkeypatch, root, tmp_path / "all", base + ["--repeat-channel-drop", "1"])
    assert res.exit_code == 0 and all((e[4] == 0).all() for e in forwards(ev, True))
    res, _ = run(new, monkeypatch, root, tmp_path / "bad", SMALL + ["--steps", "2", "--repeat-channel"])
    assert res.exit_code == 2 and "needs --n-in-extra >= 1" in flat(res.output)
    (genomes["hgt"] / "repeat.u8").write_bytes(b"\x00" * 10)
    res, _ = run(new, monkeypatch, root, tmp_path / "short", base)
    assert res.exit_code == 2 and "holds 10 values" in flat(res.output)


# ------------------------------------------------------------------------------------------------ contexts

@pytest.mark.parametrize("arch", ["v2", "v2l"])
def test_from_scratch_context_rows_start_as_row_zero(arch):
    """From scratch every context row is initialised as row 0 (nuclear) is: zeros."""
    torch.manual_seed(0)
    m = G.KronGenome(G.KronGenomeConfig(d_model=32, n_heads=2, n_layers=1, ff_units=64, stem_channels=16, n_down=3,
                                        arch=arch, n_contexts=256, n_in_extra=8))
    w = m.context.weight
    assert w.shape == (256, 32) and torch.equal(w, torch.zeros_like(w))
    assert torch.equal(m.stem.weight[:, G.N_IN:], torch.zeros_like(m.stem.weight[:, G.N_IN:]))


def test_n_contexts_alone_delivers_context_ids(genomes, tmp_path, monkeypatch):
    """--n-contexts without --mito: the model gets the ids (ctx_on), and a --genome id is checked against it."""
    stub_inputs(monkeypatch)
    from kron_dna import cli as new
    args = SMALL + ["--steps", "2", "--eval-every", "100", "--batch-size", "40", "--n-contexts", "256",
                    "--genome", "gB:0.5:64"]
    res, ev = run(new, monkeypatch, genomes["root"], tmp_path / "ok", args)
    assert res.exit_code == 0, res.output + repr(res.exception)
    assert {c for e in forwards(ev, True) for c in e[2]} == {0, 64}
    res, _ = run(new, monkeypatch, genomes["root"], tmp_path / "bad",
                 SMALL + ["--steps", "2", "--n-contexts", "64", "--genome", "gB:0.5:64"])
    assert res.exit_code == 2 and "beyond the model's context table (64 entries" in flat(res.output)
    res, _ = run(new, monkeypatch, genomes["root"], tmp_path / "bad2", SMALL + ["--steps", "2", "--genome", "gB:0.5:3"])
    assert res.exit_code == 2 and "needs a context table" in flat(res.output)
    res, ev = run(new, monkeypatch, genomes["root"], tmp_path / "none", SMALL + ["--steps", "2", "--eval-every", "100"])
    assert res.exit_code == 0 and all(e[2] is None for e in forwards(ev))       # no table: no ids, as before


def test_new_contexts_option(genomes, tmp_path, monkeypatch):
    stub_inputs(monkeypatch)
    from kron_dna import cli as new
    res, _ = run(new, monkeypatch, genomes["root"], tmp_path / "x", SMALL + ["--steps", "2", "--new-contexts", "nuclear"])
    assert res.exit_code == 2 and "applies to --init only" in flat(res.output)
    res, _ = run(new, monkeypatch, genomes["root"], tmp_path / "y", SMALL + ["--steps", "2", "--new-contexts", "ones"])
    assert res.exit_code == 2 and "must be 'zero' or 'nuclear'" in flat(res.output)
    res, _ = run(new, monkeypatch, genomes["root"], tmp_path / "src", SMALL + ["--steps", "2", "--eval-every", "2"])
    assert res.exit_code == 0, res.output + repr(res.exception)
    got = []

    def fake_expand(state, model, new_contexts="zero"):
        got.append(new_contexts)
        return state, []

    monkeypatch.setattr(G, "expand_state_dict", fake_expand)
    for how in ("zero", "nuclear"):
        res, _ = run(new, monkeypatch, genomes["root"], tmp_path / f"init-{how}",
                     SMALL + ["--steps", "4", "--eval-every", "4", "--init", str(tmp_path / "src" / "checkpoint.pt"),
                              "--start-step", "2", "--new-contexts", how])
        assert res.exit_code == 0, res.output + repr(res.exception)
        rec = json.loads((tmp_path / f"init-{how}" / "run_record.json").read_text())
        assert rec["init_from"]["new_contexts"] == how and rec["init_from"]["step"] == 2
    assert got == ["zero", "nuclear"]


def test_expand_nuclear_rows_equal_row_zero():
    torch.manual_seed(0)
    small = dict(d_model=32, n_heads=2, n_layers=1, ff_units=64, stem_channels=16, n_down=3, arch="v2")
    old = G.KronGenome(G.KronGenomeConfig(**small, n_contexts=34))
    with torch.no_grad():
        old.context.weight.normal_()
    new = G.KronGenome(G.KronGenomeConfig(**small, n_contexts=256))
    notes, _ = G.load_expanded(new, old.state_dict(), new_contexts="nuclear")
    w = new.context.weight
    assert torch.equal(w[:34], old.context.weight) and torch.equal(w[34:], w[:1].expand(222, -1))
    assert any("new rows nuclear" in n for n in notes)


# ------------------------------------------------------------------------------------------------ run record

def record_args(*more):
    return SMALL + ["--eval-every", "2", "--batch-size", "4", "--mito", "mt", "--mito-share", "0.1",
                    "--n-contexts", "256", "--n-in-extra", "2", "--repeat-channel", *more]


def test_run_record_written_and_in_checkpoint(genomes, tmp_path, monkeypatch):
    add_repeat(genomes["hgt"])
    stub_inputs(monkeypatch)
    from kron_dna import cli as new
    out = tmp_path / "run"
    res, _ = run(new, monkeypatch, genomes["root"], out, record_args("--steps", "2", "--genome", "gA:0.2:64:a4",
                                                                     "--genome", "gB:0.1:65"))
    assert res.exit_code == 0, res.output + repr(res.exception)
    rec = json.loads((out / "run_record.json").read_text())
    ck = torch.load(out / "checkpoint.pt", map_location="cpu")
    stored = dict(ck["run_record"])
    assert stored.pop("started_at_step") == 0 and rec.pop("started_at_step") == 0
    assert stored == rec
    assert [(g["role"], g["name"], g["context"]) for g in rec["genomes"]] == [
        ("primary", "hgt", 0), ("genome", "gA", 64), ("genome", "gB", 65),
        ("mito", "mt", "per sequence: its genetic code table")]
    by = {g["name"]: g for g in rec["genomes"]}
    assert by["hgt"]["held_out"] == ["chr8", "chr9"] and by["gA"]["held_out"] == ["a4"]
    assert by["gB"]["held_out"] == ["b3"]
    assert by["mt"]["held_out"] == ["m0", "m3"] and by["mt"]["codes"] == [2, 5]
    for g in rec["genomes"]:
        assert g["contigs_md5"] == RR.file_md5(genomes[g["name"]] / "contigs.parquet")
    assert by["hgt"]["share"] == pytest.approx(1 - 0.2 - 0.1 - 0.1) and by["gA"]["share"] == 0.2
    assert rec["input_channels"] == {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4, "MASK": 5, "repeat": 6, "reserved1": 7}
    assert rec["context_rows"] == {"nuclear": 0, "organelle:genetic_code_2": 2, "organelle:genetic_code_5": 5,
                                   "primary:hgt": 0, "genome:gA": 64, "genome:gB": 65}
    assert rec["options"]["genome"] == ["gA:0.2:64:a4", "gB:0.1:65"] and rec["options"]["repeat_channel"] is True
    assert rec["resolved"]["n_contexts"] == 256 and rec["resolved"]["ctx_on"] is True
    assert rec["source"]["module_md5"]["cli.py"] == RR.file_md5(new.__file__)


def test_resume_refuses_changed_genomes(genomes, tmp_path, monkeypatch):
    stub_inputs(monkeypatch)
    from kron_dna import cli as new
    out = tmp_path / "run"
    g = ["--genome", "gA:0.2:64:a4"]
    res, _ = run(new, monkeypatch, genomes["root"], out, record_args("--steps", "2", *g))
    assert res.exit_code == 0, res.output + repr(res.exception)
    res, _ = run(new, monkeypatch, genomes["root"], out, record_args("--steps", "4", *g))   # the same: resumes
    assert res.exit_code == 0 and "resumed at step 2" in flat(res.output), res.output
    assert (out / "run_record.json").exists() and (out / "run_record.step2.json").exists()
    for changed, words in [(["--genome", "gA:0.2:64:a3"], "gA: held_out"),
                           (["--genome", "gA:0.2:70:a4"], "gA: context"),
                           ([], "genome list"),
                           (g + ["--genome", "gB:0.1:65"], "genome list"),
                           (g + ["--val-chroms", "chr9"], "hgt: held_out")]:
        res, _ = run(new, monkeypatch, genomes["root"], out, record_args("--steps", "6", *changed))
        assert res.exit_code == 2, (changed, res.output)
        assert "refusing to resume" in flat(res.output) and words in flat(res.output), (changed, res.output)
    res, _ = run(new, monkeypatch, genomes["root"], out, record_args("--steps", "6", *g))      # still resumable
    assert res.exit_code == 0, res.output


def test_resume_of_a_checkpoint_without_record(genomes, tmp_path, monkeypatch):
    stub_inputs(monkeypatch)
    from kron_dna import cli as new
    out = tmp_path / "run"
    g = ["--genome", "gA:0.2:64:a4"]
    res, _ = run(new, monkeypatch, genomes["root"], out, record_args("--steps", "2", *g))
    assert res.exit_code == 0
    ck = torch.load(out / "checkpoint.pt", map_location="cpu")
    del ck["run_record"]                                          # as the multi-genome branch wrote it
    torch.save(ck, out / "checkpoint.pt")
    res, _ = run(new, monkeypatch, genomes["root"], out, record_args("--steps", "4", "--genome", "gA:0.2:64:a3"))
    assert res.exit_code == 2 and "--genome list" in flat(res.output)
    res, _ = run(new, monkeypatch, genomes["root"], out, record_args("--steps", "4", *g))
    assert res.exit_code == 0 and "no run record" in flat(res.output), res.output


def test_identity_differences_unit():
    a = dict(genomes=[dict(role="primary", name="h", held_out=["chr8"], context=0, contigs_md5="x")],
             context_rows={"nuclear": 0}, input_channels={"A": 0})
    b = json.loads(json.dumps(a))
    assert RR.identity_differences(a, b) == []
    b["genomes"][0]["contigs_md5"] = "y"
    assert RR.identity_differences(a, b) == ["primary h: contigs_md5 was x, now y"]


def test_source_info_without_git(tmp_path, monkeypatch):
    """Compute nodes have no git binary: the commit is read from the worktree's .git file."""
    common = tmp_path / "repo" / ".git"
    wt_git = common / "worktrees" / "wt"
    (common / "refs" / "heads").mkdir(parents=True)
    wt_git.mkdir(parents=True)
    (common / "refs" / "heads" / "feat").write_text("abc123\n")
    (wt_git / "HEAD").write_text("ref: refs/heads/feat\n")
    (wt_git / "commondir").write_text("../..\n")
    pkg = tmp_path / "wt" / "src" / "pkg"
    pkg.mkdir(parents=True)
    (tmp_path / "wt" / ".git").write_text(f"gitdir: {wt_git}\n")
    (pkg / "m.py").write_text("x = 1\n")
    import subprocess

    def no_git(*a, **k):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", no_git)
    info = RR.source_info(pkg)
    assert info["commit"] == "abc123" and info["branch"] == "feat" and info["dirty"] is None
    assert info["module_md5"] == {"m.py": RR.file_md5(pkg / "m.py")}
