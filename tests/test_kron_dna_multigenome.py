"""genome-train --genome (multigenome.py): option parsing and validation, and the window sampler's bookkeeping.

The sampler lives in genome_train's closure, so the sampler tests run the real command on the CPU: a stub model
(an embedding, no KronLSTM layer) records the context ids it is given, and a recording np.memmap logs every read of
bases.u8 (and of the track files), which is exactly where each window was placed. The default-path test runs the
same commands through the cli.py of commit 3c90c8c (read with ``git show``) and requires identical reads, contexts
and metrics rows, i.e. an identical RNG stream.
"""
from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from kron_dna import genome as G
from kron_dna import multigenome as MG

BASE = "3c90c8c"                      # tip of ntv3-track-objective, which this branch builds on
REAL_MEMMAP = np.memmap
EVENTS: list[tuple] = []


# --------------------------------------------------------------------------------------------- parsing (no files)

def test_parse_forms():
    assert MG.parse_genome_spec("chm13v2:0.3") == MG.GenomeSpec("chm13v2", 0.3, 0, None)
    assert MG.parse_genome_spec("nmr:0.2:35") == MG.GenomeSpec("nmr", 0.2, 35, None)
    assert MG.parse_genome_spec("nmr:0.2:35:chr8+chr9") == MG.GenomeSpec("nmr", 0.2, 35, ("chr8", "chr9"))
    assert MG.parse_genome_spec("y:0.05::chrIV").held_out == ("chrIV",) and MG.parse_genome_spec("y:0.05::x").context == 0
    assert MG.parse_genome_spec("y:0.05:0:none").held_out == ()
    assert MG.parse_genome_spec("y:0.05:0:NONE").held_out == ()


@pytest.mark.parametrize("text, words", [
    ("hg", "expected NAME:SHARE"), ("a:0.1:0:x:y", "expected NAME:SHARE"), (":0.1", "NAME must be"),
    ("../x:0.1", "NAME must be"), ("a:abc", "not a number"), ("a:0", "SHARE must be in (0, 1)"),
    ("a:1", "SHARE must be in (0, 1)"), ("a:-0.1", "SHARE must be in (0, 1)"), ("a:nan", "SHARE must be"),
    ("a:0.1:x", "not an integer"), ("a:0.1:-1", "CONTEXT must be >= 0"), ("a:0.1:0:chr1++chr2", "empty name"),
    ("a:0.1:0:chr1+chr1", "twice"),
])
def test_parse_rejects(text, words):
    with pytest.raises(MG.GenomeSpecError, match=None) as e:
        MG.parse_genome_spec(text)
    assert words in str(e.value)


def test_parse_specs_together():
    assert [s.name for s in MG.parse_genome_specs(["a:0.3", "b:0.2"], primary="hg38")] == ["a", "b"]
    assert MG.parse_genome_specs([], primary="hg38", other_share=0.5) == []
    for texts, other, words in [(["a:0.3", "a:0.2"], 0.0, "more than once"), (["hg38:0.1"], 0.0, "primary genome"),
                                (["a:0.6", "b:0.4"], 0.0, "sum to 1"), (["a:0.5", "b:0.45"], 0.05, "--mito-share")]:
        with pytest.raises(MG.GenomeSpecError) as e:
            MG.parse_genome_specs(texts, primary="hg38", other_share=other)
        assert words in str(e.value)


def test_pick_genome_respects_shares_with_fixed_rng():
    cum = np.cumsum([0.2, 0.05, 0.3])
    rng = np.random.default_rng(1234)
    picks = [MG.pick_genome(rng.random(), cum) for _ in range(200_000)]
    counts = {k: sum(p == k for p in picks) / len(picks) for k in (0, 1, 2, None)}
    for k, want in [(0, 0.2), (1, 0.05), (2, 0.3), (None, 0.45)]:
        assert counts[k] == pytest.approx(want, abs=0.005)
    assert MG.pick_genome(0.0, cum) == 0 and MG.pick_genome(0.2, cum) == 1 and MG.pick_genome(0.55, cum) is None
    assert MG.pick_genome(0.3, np.zeros(0)) is None
    assert MG.window_shares([0.2, 0.1], mito_share=0.1, extra_share=0.5, extra_p=[0.25, 0.75]) == pytest.approx(
        {"genome0": 0.2, "genome1": 0.1, "mito": 0.1, "extra0": 0.075, "extra1": 0.225, "primary": 0.3})


# --------------------------------------------------------------------------------------------- prepared genomes

def make_genome(root: Path, name: str, contigs: dict[str, int], seed: int, held: set[str] | None = None,
                mito_codes: list[int] | None = None) -> Path:
    """A prepared genome in the hg38 layout (random bases); ``held`` adds a held_out column; ``mito_codes`` makes it
    an organelle set like genome-prepare-mito (code, circular, held_out)."""
    d = root / "genome" / name
    d.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    rows, off, parts = [], 0, []
    for chrom, n in contigs.items():
        parts.append(rng.integers(0, 4, n, dtype=np.uint8))
        rows.append((chrom, 0, n, off))
        off += n
    np.concatenate(parts).tofile(d / "bases.u8")
    t = pl.DataFrame(rows, schema=["chrom", "start", "end", "offset"], orient="row")
    if held is not None:
        t = t.with_columns(pl.Series("held_out", [c in held for c in contigs]))
    if mito_codes is not None:
        t = t.with_columns(pl.Series("code", [mito_codes[i % len(mito_codes)] for i in range(t.height)]),
                           pl.Series("circular", [i % 2 == 0 for i in range(t.height)]))
    t.write_parquet(d / "contigs.parquet")
    return d


def make_ntv3(root: Path, primary: Path, extra: Path) -> Path:
    """NTv3-style tracks: per-track files for the primary genome (in ``root/hg_ntv3``) and a base-major ntv3/ for a
    further species, with a test split (code 3) the eval windows are placed on."""
    rng = np.random.default_rng(7)
    n_h = int(np.fromfile(primary / "bases.u8", dtype=np.uint8).size)
    nd = root / "hg_ntv3"
    (nd / "tracks").mkdir(parents=True)
    tracks = [{"file_id": "T1", "assay": "ATAC-seq", "log1p_scale": 5.0},
              {"file_id": "T2", "assay": "total RNA-seq", "log1p_scale": 6.0}]
    (nd / "tracks.json").write_text(json.dumps({"tracks": tracks}))
    for t in tracks:
        rng.integers(0, 255, n_h, dtype=np.uint8).tofile(nd / "tracks" / f"{t['file_id']}.u8")
    split = np.ones(n_h, dtype=np.uint8)
    split[1000:3000] = 3
    split.tofile(nd / "split.u8")
    n_x = int(np.fromfile(extra / "bases.u8", dtype=np.uint8).size)
    xd = extra / "ntv3"
    xd.mkdir()
    rng.integers(0, 255, (n_x, 3), dtype=np.uint8).tofile(xd / "dense.u8")
    xs = np.ones(n_x, dtype=np.uint8)
    xs[2048:4096] = 3
    xs.tofile(xd / "split.u8")
    (xd / "dense.json").write_text(json.dumps({"ids": ["X1", "X2", "X3"], "total_bases": n_x, "n_tracks": 3,
                                               "scales": [5.0, 5.0, 5.0], "assays": ["ATAC-seq"] * 3}))
    return nd


@pytest.fixture
def genomes(tmp_path):
    """hgt: the primary (chr8/chr9 held out, chr3 shorter than a window); gA: no held_out column; gB: a held_out
    column; gS: every contig shorter than a window; mt: an organelle set (codes 2 and 5)."""
    root = tmp_path / "interim"
    d = dict(root=root)
    d["hgt"] = make_genome(root, "hgt", {"chr1": 6000, "chr2": 4000, "chr3": 700, "chr8": 2600, "chr9": 1900}, 1)
    d["gA"] = make_genome(root, "gA", {"a1": 5000, "a2": 3000, "a3": 2000, "a4": 2500}, 2)
    d["gB"] = make_genome(root, "gB", {"b1": 4000, "b2": 3500, "b3": 1500}, 3, held={"b3"})
    d["gS"] = make_genome(root, "gS", {"s1": 300, "s2": 500}, 4)
    d["mt"] = make_genome(root, "mt", {f"m{i}": 900 + 37 * i for i in range(10)}, 5, held={"m0", "m3"},
                          mito_codes=[2, 5])
    return d


def test_load_and_validate(genomes):
    root = genomes["root"]

    def load(*texts, other=0.0):
        return MG.load_further_genomes(MG.parse_genome_specs(list(texts), "hgt", other), lambda n: root / "genome" / n)

    a, b = load("gA:0.2:0:a4", "gB:0.1:7")
    assert a.val["chrom"].to_list() == ["a4"] and sorted(a.train["chrom"].to_list()) == ["a1", "a2", "a3"]
    assert b.val["chrom"].to_list() == ["b3"] and b.context == 7               # its held_out column
    assert a.train_p.sum() == pytest.approx(1) and a.train_p[0] == pytest.approx(5000 / 10000)
    (s,) = load("gS:0.1:0:none")                                                # short contigs are fine
    assert s.val.height == 0 and s.train.height == 2
    bad = [(("nope:0.1",), "no prepared genome"), (("gA:0.1",), "no 'held_out' column"),
           (("gA:0.1:0:zz+a4",), "match no contig"), (("gA:0.1:0:a1+a2+a3+a4",), "every contig is held out")]
    for texts, words in bad:
        with pytest.raises(MG.GenomeSpecError) as e:
            load(*texts)
        assert words in str(e.value), (texts, str(e.value))
    (root / "genome" / "gTrunc").mkdir()                                        # being written by another job
    np.zeros(100, dtype=np.uint8).tofile(root / "genome" / "gTrunc" / "bases.u8")
    pl.read_parquet(root / "genome" / "gA" / "contigs.parquet").write_parquet(root / "genome" / "gTrunc" / "contigs.parquet")
    with pytest.raises(MG.GenomeSpecError, match="still being written"):
        load("gTrunc:0.1:0:a4")
    (root / "genome" / "gNoBases").mkdir()
    with pytest.raises(MG.GenomeSpecError, match="bases.u8 is missing"):
        load("gNoBases:0.1:0:none")
    pl.DataFrame({"chrom": ["x"], "held_out": [1]}).write_parquet(root / "genome" / "gTrunc" / "contigs.parquet")
    with pytest.raises(MG.GenomeSpecError, match="lacks columns"):
        load("gTrunc:0.1")


def test_check_contexts_against_the_table_the_model_has():
    """Against a config built directly with a larger table (as --n-contexts will make one), and today's sizes."""
    cfg = G.KronGenomeConfig(d_model=32, n_heads=2, n_layers=1, ff_units=64, stem_channels=16, n_down=3,
                             dropout=0.0, arch="v2", n_contexts=64)
    assert torch.nn.Embedding(cfg.n_contexts, cfg.d_model).num_embeddings == 64

    def fake(ctx):
        return types.SimpleNamespace(name="g", context=ctx)

    MG.check_contexts([fake(0), fake(40), fake(63)], cfg.n_contexts, {2, 5})
    MG.check_contexts([fake(0)], 0)                                             # nuclear needs no table
    for gs, n, codes, words in [([fake(64)], 64, None, "beyond the model's context table"),
                                ([fake(3)], 0, None, "needs a context table"),
                                ([fake(5)], 34, {2, 5}, "genetic code table")]:
        with pytest.raises(MG.GenomeSpecError) as e:
            MG.check_contexts(gs, n, codes)
        assert words in str(e.value)


# --------------------------------------------------------------------------------------------- genome_train on CPU

class RecordingMemmap:
    """np.memmap that logs (genome dir/file, start, stop) for every read."""

    def __init__(self, filename, dtype=np.uint8, mode="r", shape=None, **kw):
        self._a = REAL_MEMMAP(filename, dtype=dtype, mode=mode, shape=shape, **kw)
        p = Path(filename)
        self._tag = f"{p.parent.name}/{p.name}"

    def __getitem__(self, key):
        if isinstance(key, slice):
            EVENTS.append(("read", self._tag, key.start, key.stop, key.step))
        elif isinstance(key, np.ndarray):
            EVENTS.append(("read", self._tag, int(key.min()), int(key.max()) + 1, len(key)))
        else:
            EVENTS.append(("read", self._tag, repr(key), None, None))
        return self._a[key]

    def __len__(self):
        return len(self._a)

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self._a, dtype=dtype)

    def __getattr__(self, k):
        return getattr(self._a, k)


class StubGenome(torch.nn.Module):
    """Stands in for KronGenome: logits from an embedding of the (masked) input; logs the context ids it is given."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.emb = torch.nn.Embedding(8, 4)
        self.last_ratio_loss = self.last_epi = self.last_ntv3 = self.last_jepa = self.last_chunks = None

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, seq, segments=None, context=None):
        EVENTS.append(("forward", self.training, None if context is None else tuple(context.tolist())))
        return self.emb(seq.long()), None, None


def cpu_only(monkeypatch):
    """genome_train asks for a CUDA device; give it the CPU and the stub model instead."""
    real = torch.device

    class _Meta(type):
        def __instancecheck__(cls, obj):
            return isinstance(obj, real)

    class _Device(metaclass=_Meta):
        def __new__(cls, *a, **k):
            return real("cpu") if a and str(a[0]).startswith("cuda") else real(*a, **k)

    monkeypatch.setattr(torch, "device", _Device)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: 0)
    monkeypatch.setattr(G, "KronGenome", StubGenome)
    monkeypatch.setattr(np, "memmap", RecordingMemmap)


def base_cli(monkeypatch):
    """cli.py at BASE, from ``git show``, or from the file KRON_DNA_BASE_CLI names (a ``git archive`` export, for
    nodes without git)."""
    import os
    repo = Path(__file__).resolve().parents[1]
    if os.environ.get("KRON_DNA_BASE_CLI"):
        src = Path(os.environ["KRON_DNA_BASE_CLI"]).read_text()
    else:
        try:
            src = subprocess.run(["git", "-C", str(repo), "show", f"{BASE}:src/kron_dna/cli.py"], capture_output=True,
                                 check=True, text=True).stdout
        except (OSError, subprocess.CalledProcessError):
            pytest.skip(f"commit {BASE} is not reachable from {repo} (set KRON_DNA_BASE_CLI to its cli.py)")
    mod = types.ModuleType("kron_dna_cli_base")
    mod.__file__ = f"{BASE}:src/kron_dna/cli.py"
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    from test_kron_dna_v23pre import with_audit_rng      # the audit's RNG changes (2026-11-18)
    src = with_audit_rng(src)
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod


def run(mod, monkeypatch, root: Path, out: Path, args: list[str]):
    from typer.testing import CliRunner
    monkeypatch.setattr(mod, "interim_dir", lambda *parts: root.joinpath(*parts))
    EVENTS.clear()
    torch.manual_seed(0)                       # the training masks draw from torch's global generator
    res = CliRunner().invoke(mod.app, ["genome-train", str(out), *args])
    return res, list(EVENTS)


def flat(text: str) -> str:
    return " ".join(text.replace("│", " ").split())


SMALL = ["--name", "hgt", "--arch", "v2", "--window", "1024", "--warmup", "1", "--d-model", "16", "--n-layers", "1",
         "--stem-channels", "8", "--seed", "5"]


def rows_by_forward(events):
    """[(training, contexts, [per-row reads])]: each forward with the reads its batch made, cut into rows at each
    bases.u8 read (every row reads its sequence first)."""
    out, buf = [], []
    for e in events:
        if e[0] == "read":
            buf.append(e)
        else:
            rows = []
            for r in buf:
                if r[1].endswith("/bases.u8"):
                    rows.append([r])
                elif rows:
                    rows[-1].append(r)
            out.append((e[1], e[2], rows))
            buf = []
    return out


def metric_rows(path: Path, drop_hours: bool = True):
    lines = path.read_text().strip().split("\n")
    head = lines[0].split(",")
    keep = [i for i, h in enumerate(head) if not (drop_hours and h == "hours")]
    return [[line.split(",")[i] for i in keep] for line in lines]


@pytest.mark.parametrize("case", ["plain", "mito-rc-bert", "ntv3-extra"])
def test_default_path_draws_what_3c90c8c_draws(genomes, tmp_path, monkeypatch, case):
    """Without --genome, every read (window placement), every context id and every metrics row (train loss, held-out
    bits/base) is identical to the code at 3c90c8c, over training steps and two evals."""
    root = genomes["root"]
    args = SMALL + ["--steps", "6", "--eval-every", "3", "--batch-size", "3"]
    if case == "mito-rc-bert":
        args += ["--mito", "mt", "--mito-share", "0.2", "--rc", "--bert-mask"]
    if case == "ntv3-extra":
        nd = make_ntv3(root, genomes["hgt"], genomes["gA"])
        args += ["--ntv3-dir", str(nd), "--ntv3-extra", "gA", "--ntv3-extra-share", "0.3", "--rc"]
    cpu_only(monkeypatch)
    from kron_dna import cli as new
    old = base_cli(monkeypatch)
    res_old, ev_old = run(old, monkeypatch, root, tmp_path / "old", args)
    assert res_old.exit_code == 0, res_old.output + repr(res_old.exception)
    res_new, ev_new = run(new, monkeypatch, root, tmp_path / "new", args)
    assert res_new.exit_code == 0, res_new.output + repr(res_new.exception)
    assert sum(e[0] == "forward" for e in ev_new) > 6 and sum(e[0] == "read" for e in ev_new) > 18
    assert ev_new == ev_old
    assert metric_rows(tmp_path / "new" / "metrics.csv") == metric_rows(tmp_path / "old" / "metrics.csv")
    assert len(metric_rows(tmp_path / "new" / "metrics.csv")) == 3            # header + evals at steps 3 and 6


def contig_ranges(d: Path) -> dict[str, tuple[int, int]]:
    t = pl.read_parquet(d / "contigs.parquet")
    return {c: (o, o + e - s) for c, s, e, o in t.select("chrom", "start", "end", "offset").iter_rows()}


def inside(read, ranges: dict[str, tuple[int, int]], names) -> bool:
    return any(ranges[n][0] <= read[2] and read[3] <= ranges[n][1] for n in names)


def test_shares_held_out_and_contexts(genomes, tmp_path, monkeypatch):
    """Two further genomes beside --mito: shares respected over 2,000 training windows, held-out contigs never in a
    training window and the only place of their eval windows, context ids attached per row, metrics columns added."""
    root = genomes["root"]
    cpu_only(monkeypatch)
    from kron_dna import cli as new
    args = SMALL + ["--steps", "10", "--eval-every", "100", "--batch-size", "200", "--rc", "--mito", "mt",
                    "--mito-share", "0.1", "--genome", "gA:0.25:0:a4", "--genome", "gB:0.15:7"]
    res, ev = run(new, monkeypatch, root, tmp_path / "run", args)
    assert res.exit_code == 0, res.output + repr(res.exception)
    rng_a, rng_b, rng_h = (contig_ranges(genomes[k]) for k in ("gA", "gB", "hgt"))
    mt = pl.read_parquet(genomes["mt"] / "contigs.parquet")
    code_at = {o: c for o, c in mt.select("offset", "code").iter_rows()}
    counts = {"hgt": 0, "gA": 0, "gB": 0, "mt": 0}
    n_train = 0
    evals = {"hgt": [], "gA": [], "gB": [], "mt": []}
    for training, ctxs, rows in rows_by_forward(ev):
        assert ctxs is not None and len(ctxs) == len(rows)                     # --mito: the model reads contexts
        for ctx, row in zip(ctxs, rows):
            src = row[0][1].split("/")[0]
            if not training:
                evals[src].append((row[0], ctx))
                continue
            n_train += 1
            counts[src] += 1
            if src == "gA":
                assert ctx == 0 and inside(row[0], rng_a, ["a1", "a2", "a3"]) and not inside(row[0], rng_a, ["a4"])
            elif src == "gB":
                assert ctx == 7 and inside(row[0], rng_b, ["b1", "b2"])
            elif src == "mt":
                off = max(o for o in code_at if o <= row[0][2])
                assert ctx == code_at[off] and ctx in (2, 5)
            else:
                assert ctx == 0 and inside(row[0], rng_h, ["chr1", "chr2", "chr3"])
    assert n_train == 2000
    for src, want in [("gA", 0.25), ("gB", 0.15), ("mt", 0.10), ("hgt", 0.50)]:
        assert counts[src] / n_train == pytest.approx(want, abs=0.04), counts
    # the eval at the last step: 8 primary windows on chr8/chr9, 8 per further genome on its held-out contigs
    assert len(evals["gA"]) == 8 and all(inside(r, rng_a, ["a4"]) and c == 0 for r, c in evals["gA"])
    assert len(evals["gB"]) == 8 and all(inside(r, rng_b, ["b3"]) and c == 7 for r, c in evals["gB"])
    assert len(evals["hgt"]) == 8 and all(inside(r, rng_h, ["chr8", "chr9"]) for r, _ in evals["hgt"])
    head, *rows = metric_rows(tmp_path / "run" / "metrics.csv", drop_hours=False)
    assert head == ["step", "lr", "train_loss", "val_bits_per_base", "hours", "val_mito_bits_per_base",
                    "val_bits_per_base_gA", "val_bits_per_base_gB"]
    assert len(rows) == 1 and all(np.isfinite(float(v)) for v in rows[0])
    assert ", gA " in res.output and ", gB " in res.output                     # printed in the step line
    ck = torch.load(tmp_path / "run" / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert ck["genomes"] == [dict(name="gA", share=0.25, context=0, held_out=["a4"]),
                             dict(name="gB", share=0.15, context=7, held_out=["b3"])]


def test_primary_paired_evals_unchanged_by_further_genomes(genomes, tmp_path, monkeypatch):
    """The primary genome's eight held-out windows (and so its bits/base given the same weights) are the same with
    and without --genome: the further genomes' evals draw from streams of their own."""
    root = genomes["root"]
    cpu_only(monkeypatch)
    from kron_dna import cli as new
    common = SMALL + ["--steps", "4", "--eval-every", "2", "--batch-size", "2", "--lr", "0"]
    _, ev0 = run(new, monkeypatch, root, tmp_path / "a", common)
    _, ev1 = run(new, monkeypatch, root, tmp_path / "b", common + ["--genome", "gA:0.3:0:a4"])

    def primary_evals(ev):
        return [row[0] for training, _, rows in rows_by_forward(ev) if not training for row in rows
                if row[0][1].startswith("hgt/")]

    assert len(primary_evals(ev0)) == 16 and primary_evals(ev0) == primary_evals(ev1)
    a = metric_rows(tmp_path / "a" / "metrics.csv")
    b = metric_rows(tmp_path / "b" / "metrics.csv")
    assert [r[3] for r in a[1:]] == [r[3] for r in b[1:]]                       # --lr 0: identical weights
    assert b[0][-1] == "val_bits_per_base_gA" and a[0] == b[0][:-1]


def test_genome_also_in_ntv3_extra(genomes, tmp_path, monkeypatch):
    """A genome in both --genome and --ntv3-extra: shares add, both kinds of window carry its --genome context, and
    neither trains on its held-out contigs; only the --ntv3-extra windows read its tracks."""
    root = genomes["root"]
    nd = make_ntv3(root, genomes["hgt"], genomes["gA"])
    cpu_only(monkeypatch)
    from kron_dna import cli as new
    args = SMALL + ["--steps", "10", "--eval-every", "100", "--batch-size", "200", "--mito", "mt", "--mito-share",
                    "0.1", "--ntv3-dir", str(nd), "--ntv3-extra", "gA", "--ntv3-extra-share", "0.3",
                    "--genome", "gA:0.2:9:a4"]
    res, ev = run(new, monkeypatch, root, tmp_path / "run", args)
    assert res.exit_code == 0, res.output + repr(res.exception)
    assert "also an --ntv3-extra species" in flat(res.output)
    rng_a = contig_ranges(genomes["gA"])
    n, mlm_only, with_tracks = 0, 0, 0
    for training, ctxs, rows in rows_by_forward(ev):
        if not training:
            continue
        for ctx, row in zip(ctxs, rows):
            n += 1
            if not row[0][1].startswith("gA/"):
                continue
            assert ctx == 9 and not inside(row[0], rng_a, ["a4"])
            if any(r[1] == "ntv3/dense.u8" for r in row):
                with_tracks += 1
            else:
                mlm_only += 1
    want = MG.window_shares([0.2], mito_share=0.1, extra_share=0.3, extra_p=[1.0])
    assert mlm_only / n == pytest.approx(want["genome0"], abs=0.04)
    assert with_tracks / n == pytest.approx(want["extra0"], abs=0.04)          # (1 - 0.2 - 0.1) x 0.3 = 0.21


def test_short_genome_is_taken_whole_and_padded(genomes, tmp_path, monkeypatch):
    root = genomes["root"]
    cpu_only(monkeypatch)
    from kron_dna import cli as new
    res, ev = run(new, monkeypatch, root, tmp_path / "run",
                  SMALL + ["--steps", "3", "--eval-every", "100", "--batch-size", "50", "--genome", "gS:0.5:0:none"])
    assert res.exit_code == 0, res.output + repr(res.exception)
    assert "taken whole and padded" in flat(res.output) and "no held-out set" in flat(res.output)
    reads = [r for e in rows_by_forward(ev) if e[0] for row in e[2] for r in row if r[1] == "gS/bases.u8"]
    assert reads and {(r[2], r[3]) for r in reads} <= {(0, 300), (300, 800)}
    assert metric_rows(tmp_path / "run" / "metrics.csv")[0] == ["step", "lr", "train_loss", "val_bits_per_base"]


@pytest.mark.parametrize("extra, words", [
    (["--genome", "nope:0.1"], "no prepared genome"),
    (["--genome", "gA:0.6:0:a4", "--genome", "gB:0.5"], "sum to 1"),
    (["--genome", "gA:0.85:0:a4", "--mito", "mt", "--mito-share", "0.2"], "--mito-share"),
    (["--genome", "gA:0.1:3:a4"], "needs a context table"),
    (["--genome", "gA:0.1:34:a4", "--mito", "mt"], "beyond the model's context table"),
    (["--genome", "gA:0.1:5:a4", "--mito", "mt"], "genetic code table"),
    (["--genome", "gA:0.1:0:zz"], "match no contig"),
    (["--genome", "gA:0.1"], "no 'held_out' column"),
    (["--genome", "hgt:0.1"], "primary genome"),
    (["--genome", "gA:0.1:0:a4", "--whole-contigs"], "--whole-contigs does not take --genome"),
])
def test_cli_rejects_at_startup(genomes, tmp_path, monkeypatch, extra, words):
    cpu_only(monkeypatch)
    from kron_dna import cli as new
    res, ev = run(new, monkeypatch, genomes["root"], tmp_path / "run", SMALL + ["--steps", "2"] + extra)
    assert res.exit_code == 2, res.output + repr(res.exception)
    assert words in flat(res.output), flat(res.output)
    assert not any(e[0] == "forward" for e in ev)                              # nothing trained


def test_larger_context_table_through_the_cli(genomes, tmp_path, monkeypatch):
    """With a model whose table has 64 entries (as --n-contexts will build), context 40 is accepted and attached and
    64 is refused; here the config is built with n_contexts=64 directly."""
    cpu_only(monkeypatch)
    from kron_dna import cli as new
    real = G.KronGenomeConfig
    monkeypatch.setattr(G, "KronGenomeConfig", lambda **kw: real(**{**kw, "n_contexts": 64}))
    res, ev = run(new, monkeypatch, genomes["root"], tmp_path / "ok",
                  SMALL + ["--steps", "2", "--eval-every", "100", "--batch-size", "40", "--genome", "gB:0.5:40"])
    assert res.exit_code == 0, res.output + repr(res.exception)
    got = {(row[0][1].split("/")[0], c) for tr, ctxs, rows in rows_by_forward(ev) if tr for c, row in zip(ctxs, rows)}
    assert got == {("hgt", 0), ("gB", 40)}
    res, _ = run(new, monkeypatch, genomes["root"], tmp_path / "bad", SMALL + ["--steps", "2", "--genome", "gB:0.5:64"])
    assert res.exit_code == 2 and "beyond the model's context table (64 entries" in flat(res.output)
