"""Regression tests for the independent code audit of 18 Nov 2026 (reviews/codex-20261118 on the cluster).

Each test drives the failure path the audit reproduced and asserts the repaired behaviour."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import typer


# --------------------------------------------------------------------------------- D1: genome-prepare publication

def _fasta(path, seq):
    path.write_text(">chr1\n" + seq + "\n")
    return path


def _prepare(tmp_path, monkeypatch, fasta, **kw):
    from kron_dna import cli
    monkeypatch.setenv("KRON_FM_DATA", str(tmp_path / "data"))
    cli.genome_prepare(fasta=fasta, min_gap=50, min_len=10, name="g", **kw)
    return tmp_path / "data" / "interim" / "genome" / "g"


def test_d1_prepare_publishes_a_consistent_pair(tmp_path, monkeypatch):
    out = _prepare(tmp_path, monkeypatch, _fasta(tmp_path / "a.fa", "ACGT" * 50), overwrite=False)
    idx = pl.read_parquet(out / "contigs.parquet")
    size = (out / "bases.u8").stat().st_size
    assert size == 200 and int((idx["offset"] + idx["end"] - idx["start"]).max()) == size
    assert not any(p.name.startswith(".prepare-") for p in out.iterdir())


def test_d1_existing_genome_needs_overwrite(tmp_path, monkeypatch):
    out = _prepare(tmp_path, monkeypatch, _fasta(tmp_path / "a.fa", "ACGT" * 50), overwrite=False)
    before = (out / "bases.u8").read_bytes(), (out / "contigs.parquet").read_bytes()
    with pytest.raises(typer.BadParameter):
        _prepare(tmp_path, monkeypatch, _fasta(tmp_path / "b.fa", "GGCC" * 60), overwrite=False)
    assert ((out / "bases.u8").read_bytes(), (out / "contigs.parquet").read_bytes()) == before


@pytest.mark.parametrize("failure", ["missing_fasta", "fails_while_writing"])
def test_d1_failed_rebuild_leaves_the_old_pair_byte_identical(tmp_path, monkeypatch, failure):
    from kron_dna import genome as G
    out = _prepare(tmp_path, monkeypatch, _fasta(tmp_path / "a.fa", "ACGT" * 50), overwrite=False)
    before = (out / "bases.u8").read_bytes(), (out / "contigs.parquet").read_bytes()
    if failure == "missing_fasta":
        fasta = tmp_path / "does_not_exist.fa"
    else:
        fasta = _fasta(tmp_path / "b.fa", "GGCC" * 60)
        def boom(*a, **k):
            raise RuntimeError("injected failure after parsing")
        monkeypatch.setattr(G, "contigs_of", boom)
    with pytest.raises(Exception):
        _prepare(tmp_path, monkeypatch, fasta, overwrite=True)
    assert ((out / "bases.u8").read_bytes(), (out / "contigs.parquet").read_bytes()) == before
    assert not any(p.name.startswith(".prepare-") or p.name.endswith(".prev") for p in out.iterdir())


def test_d1_overwrite_keeps_the_previous_pair(tmp_path, monkeypatch):
    out = _prepare(tmp_path, monkeypatch, _fasta(tmp_path / "a.fa", "ACGT" * 50), overwrite=False)
    old = (out / "bases.u8").read_bytes()
    _prepare(tmp_path, monkeypatch, _fasta(tmp_path / "b.fa", "GGCC" * 60), overwrite=True)
    assert (out / "bases.u8.prev").read_bytes() == old and (out / "bases.u8").stat().st_size == 240


# ------------------------------------------------------------------------------ TR01-TR05: genome-train on the CPU
# The real genome-train command with a toy model (the audit's harness): the reads of bases.u8 show where every window
# was drawn, and the checkpoint shows what was saved.

import copy
import json
import torch
from typer.testing import CliRunner

import test_kron_dna_multigenome as MGT

CREATED: list[dict] = []


class ToyGenome(torch.nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.emb = torch.nn.Embedding(8, 4)
        self.core = torch.nn.Linear(4, c.d_model)
        self.jepa_pred = torch.nn.Linear(c.d_model, c.d_model) if c.jepa else None
        self.ntv3 = torch.nn.Linear(4, c.n_ntv3) if c.n_ntv3 else None
        self.last_ratio_loss = self.last_epi = self.last_ntv3 = self.last_jepa = self.last_chunks = None
        CREATED.append({k: v.detach().clone() for k, v in self.state_dict().items()})

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, seq, segments=None, context=None, extra=None):
        MGT.EVENTS.append(("forward", self.training, None if context is None else tuple(context.tolist())))
        x = self.emb(seq.long())
        u = 2 ** self.config.n_down
        core = self.core(x.reshape(x.shape[0], x.shape[1] // u, u, 4).mean(2))
        self.last_jepa = self.jepa_pred(core) if self.jepa_pred is not None and self.training else None
        self.last_ntv3 = self.ntv3(x) if self.ntv3 is not None else None
        return x, core, None


ARGS = ["--name", "hgt", "--arch", "v2", "--window", "1024", "--n-down", "4", "--d-model", "16", "--stem-channels",
        "8", "--n-layers", "1", "--warmup", "1", "--batch-size", "1", "--seed", "5", "--no-bf16"]


@pytest.fixture
def toy(tmp_path, monkeypatch):
    from kron_dna import cli, genome as G
    root = tmp_path / "interim"
    primary = MGT.make_genome(root, "hgt", {"chr1": 6000, "chr2": 4000, "chr8": 2600, "chr9": 1900}, 1)
    MGT.cpu_only(monkeypatch)
    monkeypatch.setattr(G, "KronGenome", ToyGenome)
    monkeypatch.setattr(cli, "interim_dir", lambda *parts: root.joinpath(*parts))
    torch.set_num_threads(2)

    def invoke(tag, extra, expect_ok=True):
        MGT.EVENTS.clear()
        res = CliRunner().invoke(cli.app, ["genome-train", str(tmp_path / tag), *ARGS, *extra])
        if expect_ok and res.exit_code:
            raise AssertionError((tag, res.exit_code, res.output, repr(res.exception)))
        return res, list(MGT.EVENTS)

    def state(tag):
        return torch.load(tmp_path / tag / "checkpoint.pt", map_location="cpu", weights_only=False)
    return dict(invoke=invoke, state=state, primary=primary, tmp=tmp_path)


def _training_reads(ev):
    out, reads = [], []
    for e in ev:
        if e[0] == "read":
            reads.append(e)
        elif e[0] == "forward":
            if e[1]:
                out.append(list(reads))
            reads = []
    return out


def _ntv3_dir(toy, tracks):
    nd = toy["tmp"] / "ntv3"
    (nd / "tracks").mkdir(parents=True, exist_ok=True)
    n = int(np.fromfile(toy["primary"] / "bases.u8", dtype=np.uint8).size)
    (nd / "tracks.json").write_text(json.dumps({"tracks": tracks}))
    for i, t in enumerate(tracks):
        np.full(n, 20 + 100 * i, dtype=np.uint8).tofile(nd / "tracks" / f"{t['file_id']}.u8")
    np.ones(n, dtype=np.uint8).tofile(nd / "split.u8")
    return nd


TRACKS = [{"file_id": "T1", "assay": "ATAC-seq", "log1p_scale": 5.0},
          {"file_id": "T2", "assay": "total RNA-seq", "log1p_scale": 6.0}]


def test_tr01_resume_refuses_reordered_ntv3_channels(toy):
    nd = _ntv3_dir(toy, TRACKS)
    toy["invoke"]("ch", ["--steps", "8", "--stop-step", "2", "--eval-every", "2", "--ntv3-dir", str(nd)])
    before = toy["state"]("ch")
    assert before["ntv3"]["ids"] == ["T1", "T2"]
    (nd / "tracks.json").write_text(json.dumps({"tracks": TRACKS[::-1]}))      # same width, other order
    res, _ = toy["invoke"]("ch", ["--steps", "8", "--stop-step", "4", "--eval-every", "2", "--ntv3-dir", str(nd)],
                           expect_ok=False)
    assert res.exit_code != 0 and "channel ids" in MGT.flat(res.output)
    after = toy["state"]("ch")
    assert after["step"] == 2 and after["ntv3"]["ids"] == ["T1", "T2"]     # nothing was trained or saved


def test_tr01_resume_with_the_same_channels_still_works(toy):
    nd = _ntv3_dir(toy, TRACKS)
    toy["invoke"]("ch", ["--steps", "8", "--stop-step", "2", "--eval-every", "2", "--ntv3-dir", str(nd)])
    toy["invoke"]("ch", ["--steps", "8", "--stop-step", "4", "--eval-every", "2", "--ntv3-dir", str(nd)])
    assert toy["state"]("ch")["step"] == 4


@pytest.mark.parametrize("stop,every", [(1, 4), (3, 2), (4, 2), (8, 3)])
def test_tr02_checkpoint_is_written_at_the_requested_stop(toy, stop, every):
    toy["invoke"]("stop", ["--steps", "8", "--stop-step", str(stop), "--eval-every", str(every)])
    assert toy["state"]("stop")["step"] == stop


def test_tr03_seed_fixes_initial_weights(toy):
    torch.manual_seed(111)
    toy["invoke"]("a", ["--steps", "1", "--eval-every", "1"])
    a = CREATED[-1]
    torch.manual_seed(222)
    toy["invoke"]("b", ["--steps", "1", "--eval-every", "1"])
    b = CREATED[-1]
    assert all(torch.equal(a[k], b[k]) for k in a)
    torch.manual_seed(111)
    toy["invoke"]("c", ["--steps", "1", "--eval-every", "1", "--seed", "6"])
    assert any(not torch.equal(a[k], CREATED[-1][k]) for k in a)


def test_tr03_training_masks_do_not_depend_on_the_ambient_stream(toy):
    torch.manual_seed(111)
    toy["invoke"]("a", ["--steps", "2", "--eval-every", "2"])
    torch.manual_seed(222)
    toy["invoke"]("b", ["--steps", "2", "--eval-every", "2"])
    sa, sb = toy["state"]("a")["model"], toy["state"]("b")["model"]
    assert all(torch.equal(sa[k], sb[k]) for k in sa)      # same init, same masks, same windows: same weights


def test_tr04_resume_does_not_replay_the_first_windows(toy):
    _, first = toy["invoke"]("r", ["--steps", "8", "--stop-step", "2", "--eval-every", "2"])
    _, second = toy["invoke"]("r", ["--steps", "8", "--stop-step", "4", "--eval-every", "2"])
    assert _training_reads(first) != _training_reads(second)
    _, fresh = toy["invoke"]("f", ["--steps", "8", "--stop-step", "2", "--eval-every", "2"])
    assert _training_reads(fresh) == _training_reads(first)         # a fresh run keeps its stream


def test_tr05_ema_teacher_survives_a_resume(toy, monkeypatch):
    targets = []
    real_copy = copy.deepcopy

    def capture(x, *a, **k):
        y = real_copy(x, *a, **k)
        if isinstance(x, ToyGenome):
            targets.append(y)
        return y
    monkeypatch.setattr(copy, "deepcopy", capture)
    jepa = ["--jepa-weight", "0.2", "--jepa-ema", "0.996"]
    toy["invoke"]("ema", ["--steps", "8", "--stop-step", "2", "--eval-every", "2", *jepa])
    old_teacher = {k: v.detach().clone() for k, v in targets[-1].state_dict().items()}
    st = toy["state"]("ema")
    assert "jepa_target" in st and all(torch.equal(st["jepa_target"][k], old_teacher[k]) for k in old_teacher)
    assert any(not torch.equal(old_teacher[k], st["model"][k]) for k in old_teacher)
    # resume to the same stop step: no training step runs, so the teacher must be exactly the saved one,
    # not a copy of the online weights (the audit's failure)
    toy["invoke"]("ema", ["--steps", "8", "--stop-step", "2", "--eval-every", "2", *jepa])
    restored = targets[-1].state_dict()
    assert all(torch.equal(restored[k], old_teacher[k]) for k in old_teacher)
    toy["invoke"]("ema", ["--steps", "8", "--stop-step", "4", "--eval-every", "2", *jepa])
    assert toy["state"]("ema")["step"] == 4 and "jepa_target" in toy["state"]("ema")


def test_tr05_ema_zero_saves_no_teacher(toy):
    toy["invoke"]("e0", ["--steps", "8", "--stop-step", "2", "--eval-every", "2", "--jepa-weight", "0.2",
                         "--jepa-ema", "0.0"])
    assert "jepa_target" not in toy["state"]("e0")


# ------------------------------------------------------------------------------- TR06: local track row isolation

@pytest.mark.parametrize("length,span", [(96, 64), (128, 64), (100, 33)])
def test_tr06_local_track_keeps_batch_rows_apart(monkeypatch, length, span):
    from kron_core import blocks
    from kron_core.testing import CausalMixer
    from kron_dna import genome as G
    monkeypatch.setattr(blocks, "kronlstm_mixer", CausalMixer)
    torch.manual_seed(1)
    c = G.KronGenomeConfig(d_model=32, n_heads=2, n_layers=1, ff_units=64, stem_channels=8, n_down=3, arch="v2",
                           dropout=0.0, local_width=32, local_layers=1, local_span=span)
    m = G.KronGenome(c).eval()
    with torch.no_grad():
        m.local_gate.fill_(1.0)
        x = torch.zeros(3, 8, length)
        stem = torch.randn(3, 8, length)
        a = m.add_local(x, [stem])
        changed = stem.clone()
        changed[1] += 3.0                                   # only the middle row changes
        b = m.add_local(x, [changed])
        assert torch.allclose(a[0], b[0], atol=1e-6) and torch.allclose(a[2], b[2], atol=1e-6)
        for r in range(3):                                  # batched == each row on its own
            assert torch.allclose(a[r], m.add_local(x[r:r + 1], [stem[r:r + 1]])[0], atol=1e-5)


# --------------------------------------------------------------------------- D2: monolithic vs streamed datasets

def _corpus(tmp_path):
    import csv
    import gzip
    import test_kron_dna_stream as ST
    rng = np.random.default_rng(1)
    path = tmp_path / "corpus.csv.gz"
    with gzip.open(path, "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["CDS", "Species", "Group", "Set"])
        w.writerows([(ST._random_cds(rng), f"sp{i % 7}", "plant", "Train") for i in range(200)])
    return path


def test_d2_stream_rebuild_refuses_an_old_tokens_npy(tmp_path):
    from kron_dna.stream import prepare_stream
    out = tmp_path / "packed"
    (out / "train").mkdir(parents=True)
    old = np.zeros((1, 64), dtype=np.uint8)
    np.save(out / "train" / "tokens.npy", old)
    with pytest.raises(FileExistsError):
        prepare_stream(_corpus(tmp_path), out, seq_len=64, shard_rows=16)
    assert np.array_equal(np.load(out / "train" / "tokens.npy"), old)


def test_d2_loader_rejects_a_mixed_directory(tmp_path):
    import json
    from kron_dna.data import shard_files
    d = tmp_path / "train"
    d.mkdir()
    np.save(d / "tokens.npy", np.zeros((1, 64), dtype=np.uint8))
    np.save(d / "tokens_00000.npy", np.zeros((3, 64), dtype=np.uint8))
    (d / "meta.json").write_text(json.dumps({"shards": [{"file": "tokens_00000.npy", "rows": 3}]}))
    with pytest.raises(ValueError):
        shard_files(d)
    (d / "tokens.npy").unlink()
    assert shard_files(d) == [d / "tokens_00000.npy"]


def test_d2_stream_build_into_a_fresh_directory_still_loads(tmp_path):
    from kron_dna.data import shard_files
    from kron_dna.stream import prepare_stream
    out = tmp_path / "packed"
    prepare_stream(_corpus(tmp_path), out, seq_len=64, shard_rows=16)
    files = shard_files(out / "train")
    assert files and all(f.name.startswith("tokens_") for f in files)


# ------------------------------------------------------------------------------- EV-02: AlphaGenome cache identity

def _ag_fake(calls, effects):
    from types import SimpleNamespace

    class Fake:
        def score_variant(self, interval, variant, variant_scorers, condition=None):
            calls.append((variant.position, variant.alternate_bases, condition))
            e = effects(variant, condition)
            return [SimpleNamespace(X=np.array([[e]])) for _ in variant_scorers]
    return Fake()


def test_ev02_changed_conditions_are_scored_not_reused(tmp_path, monkeypatch):
    from kron_dna import variants as V
    monkeypatch.setattr(V.time, "sleep", lambda s: None)
    calls = []
    fake = _ag_fake(calls, lambda v, c: {"A": 0.2, "B": 0.7}[c])
    table = pl.DataFrame({"id": ["x"]})
    pos = pl.DataFrame({"id": ["x"], "chrom": ["1"], "pos": [10], "ref": ["A"], "alt": ["G"]})
    kw = dict(scorers=("SPLICE_SITES",), workers=1, client=fake, version=2)
    cache = tmp_path / "ag2.parquet"
    assert V.alphagenome_scores(table, pos, "", cache, conditions=("A",), **kw)["alphagenome2:splice_sites"][0] == -0.2
    assert V.alphagenome_scores(table, pos, "", cache, conditions=("B",), **kw)["alphagenome2:splice_sites"][0] == -0.7
    assert [c[2] for c in calls] == ["A", "B"]
    calls.clear()
    assert V.alphagenome_scores(table, pos, "", cache, conditions=("A", "B"), **kw)["alphagenome2:splice_sites"][0] == -0.7
    assert calls == []                                          # both conditions cached: no call
    assert V.alphagenome_scores(table, pos, "", cache, conditions=("A",), **kw)["alphagenome2:splice_sites"][0] == -0.2


def test_ev02_changed_allele_and_new_scorer_are_recomputed(tmp_path, monkeypatch):
    from kron_dna import variants as V
    monkeypatch.setattr(V.time, "sleep", lambda s: None)
    calls = []
    fake = _ag_fake(calls, lambda v, c: {"G": 0.3, "T": 0.9}[v.alternate_bases])
    table = pl.DataFrame({"id": ["x"]})
    cache = tmp_path / "ag.parquet"
    kw = dict(workers=1, client=fake, version=1)
    p1 = pl.DataFrame({"id": ["x"], "chrom": ["1"], "pos": [10], "ref": ["A"], "alt": ["G"]})
    p2 = p1.with_columns(pl.lit("T").alias("alt"))                     # same id, another allele
    assert V.alphagenome_scores(table, p1, "", cache, scorers=("SPLICE_SITES",), **kw)["alphagenome:splice_sites"][0] == -0.3
    assert V.alphagenome_scores(table, p2, "", cache, scorers=("SPLICE_SITES",), **kw)["alphagenome:splice_sites"][0] == -0.9
    calls.clear()
    out = V.alphagenome_scores(table, p2, "", cache, scorers=("SPLICE_SITES", "RNA_SEQ"), **kw)
    assert out["alphagenome:rna_seq"][0] == -0.9 and len(calls) == 1   # the missing scorer was computed


# ------------------------------------------------------------------------- EV-01 / EV-03: ClinVar cache and names

def test_ev01_clinvar_cache_follows_the_gene_split_and_inputs(tmp_path, monkeypatch):
    from kron_dna import cli, variants as V
    tx = {"G": {"name": "tx", "gene": "G", "chrom": "1", "strand": "+", "exons": [(1, 12)], "seq": "ATGGCTAAATAA"}}
    variants = pl.DataFrame([("1", 6, "T", "C", "synthetic-id", ["G"], "Benign", 0)],
                            schema=["chrom", "pos", "ref", "alt", "id", "genes", "clnsig", "label"], orient="row")
    reads = []
    monkeypatch.setattr(V, "read_transcripts", lambda _: tx)
    monkeypatch.setattr(V, "read_clinvar", lambda p: (reads.append(str(p)), variants)[1])
    monkeypatch.setattr(cli, "interim_dir", lambda *a: tmp_path / "cache")
    vcf, fa = tmp_path / "in.vcf", tmp_path / "in.fa"
    vcf.write_text("x")
    fa.write_text(">x\nATG\n")
    kw = dict(vcf=vcf, cds=str(fa), runs=[], window=16, holdout_genes=1.0, max_per_class=0, batch_size=2,
              device="cpu", seed=0, alphamissense=None, alphagenome_key=None, alphagenome_version=1, ag_conditions=[],
              esm_models=[], out=tmp_path / "first.json")
    cli.clinvar(**kw)
    assert pl.read_parquet(tmp_path / "first.parquet")["in_training"].to_list() == [False]
    cli.clinvar(**{**kw, "holdout_genes": 0.0, "out": tmp_path / "second.json"})
    assert pl.read_parquet(tmp_path / "second.parquet")["in_training"].to_list() == [True]
    n = len(reads)
    cli.clinvar(**{**kw, "out": tmp_path / "third.json"})                 # the first split again: its cache
    assert len(reads) == n
    vcf.write_text("a newer release")                                     # another VCF: mapped afresh
    cli.clinvar(**{**kw, "out": tmp_path / "fourth.json"})
    assert len(reads) == n + 1


def test_ev03_checkpoints_of_one_folder_get_distinct_names(tmp_path):
    from kron_dna.cli import run_keys
    snap = tmp_path / "run" / "snapshots"
    snap.mkdir(parents=True)
    a, b = snap / "step100.pt", snap / "step200.pt"
    a.write_bytes(b"")
    b.write_bytes(b"")
    assert run_keys([a, b]) == ["snapshots/step100", "snapshots/step200"]
    assert run_keys([tmp_path / "run"]) == ["run"]                        # a lone run keeps its usual name
    with pytest.raises(typer.BadParameter):
        run_keys([a, a])


def test_ev03_genome_clinvar_uses_the_checked_names():
    import inspect
    from kron_dna import cli
    src = inspect.getsource(cli.genome_clinvar)
    assert "run_keys(runs)" in src and "run.parent.name" not in src


# --------------------------------------------------------------------------------- OP-03: GPU lease in job.sbatch

def _fake_node(tmp_path, busy=""):
    import os
    import stat
    from pathlib import Path
    repo = tmp_path / "repo"
    (repo / "slurm").mkdir(parents=True)
    (repo / "slurm" / "env.sh").write_text("")
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "nvidia-smi").write_text("#!/bin/bash\ncase \"$*\" in\n"
                                     f"  *query-compute-apps*) printf '{busy}' ;;\n"
                                     "  *query-gpu=index,uuid*) printf '0, GPU-aaa\\n1, GPU-bbb\\n' ;;\n"
                                     "  *) : ;;\nesac\n")
    (bin_ / "uv").write_text("#!/bin/bash\necho \"card=$CUDA_VISIBLE_DEVICES\"\nsleep 2\n")
    for f in ("nvidia-smi", "uv"):
        (bin_ / f).chmod((bin_ / f).stat().st_mode | stat.S_IEXEC)
    env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "REPO": str(repo), "KRON_JOB_KIND": "gpu",
           "KRON_GPU_LOCK_DIR": str(tmp_path / "locks"), "CUDA_DEVICE": ""}
    script = Path(__file__).resolve().parents[1] / "slurm" / "job.sbatch"
    return script, env


def test_op03_concurrent_gpu_jobs_get_distinct_cards(tmp_path):
    import subprocess
    script, env = _fake_node(tmp_path)
    procs = [subprocess.Popen(["bash", str(script), "run"], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True) for _ in range(3)]
    outs = [p.communicate(timeout=30) for p in procs]
    cards = sorted(o.strip().splitlines()[-1] for o, _ in outs if p_ok(o))
    codes = sorted(p.returncode for p in procs)
    assert cards == ["card=0", "card=1"] and codes == [0, 0, 3]     # the third finds both cards leased


def p_ok(out):
    return out.strip().splitlines()[-1].startswith("card=") if out.strip() else False


def test_op03_lease_is_released_when_the_job_ends(tmp_path):
    import subprocess
    script, env = _fake_node(tmp_path)
    for _ in range(2):                                          # sequential jobs reuse card 0
        r = subprocess.run(["bash", str(script), "run"], env=env, capture_output=True, text=True, timeout=30)
        assert r.returncode == 0 and r.stdout.strip().splitlines()[-1] == "card=0"
