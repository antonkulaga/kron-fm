"""v2.4 = v2.3 + genome-train --mito-slice BASES: every training step also trains on one packed window of BASES
bases of mitochondrial genomes (the packing ladder: rows of 4 x 1,024 / 896 here, 4 x 16,384 at 65,536 for a
~16.5 kb mitogenome), each row with its genetic-code context, and --mito leaves the main window draw.

Tested on the CPU (the real KronGenome with the KronLSTM stand-in, two logical pieces on one device for
--split-cards, as test_kron_dna_v23):

- option off: v2.4 trains bit-identically to v2.3 (8795a54): same weights, optimizer state, metrics and jepa.csv,
  split and unsplit, pack and pad;
- option on: one slice per step, drawn from the mito training table only (never a held-out mitogenome), rows from
  the ladder, contexts = the drawn genomes' genetic codes; the main draw never holds a mito row; the main draw's
  realised shares are the configured ones without mito, rescaled; the run record holds them;
- loss weights: the main draw's loss x main / (main + slice) bases, the slice's x slice / (main + slice), each
  part as loss_parts weights it;
- split mode: the slice runs unsplit (no forward_split call for it), and the first step's losses and gradients
  equal those of the same run unsplit within split-window's tolerances;
- resume: a v2.3 checkpoint (run record without the option) is accepted and the run goes on with the slice;
- refusals at startup.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from kron_dna import genome as G
from kron_dna import packing as PK

import test_kron_dna_jepa_guard as GT
from test_kron_dna_multigenome import genomes, run  # noqa: F401  (genomes is a fixture)
from test_kron_dna_split import _restore_precision, make_recipe_data, real_model_cpu, recipe, smooth_routing  # noqa: F401
from test_kron_dna_v23pre import OFF_ARGS, assert_identical, parent_cli, trained, with_option

V23_PARENT = "8795a54"
BLOCK = ["--jepa-block-share", "0.25", "--jepa-block-len", "64:256"]            # n-down 4: 16-base bins, 4-16 bins
GUARD = ["--jepa-guard-windows", "2", "--jepa-guard-window", "1024"]
QUIET = ["--jepa-guard-rank", "0.001", "--jepa-guard-std", "0.0001"]            # a guard that never triggers
TIMING = ("sec_per_step", "peak_gib")
MITO_CODES = {2, 5}                                    # the fixture's organelle set (test_kron_dna_multigenome)
MAIN_CONTEXTS = {0, 64}                                # hgt and gA (ntv3 extra): 0, gB: 64


def base_args(nd) -> list[str]:
    """The long runs' recipe (tracks, extra species, --genome, mito, soft bases, repeat channel, JEPA) with half of
    the windows mito, so a main draw that still held mito would show it at once."""
    return with_option(recipe(nd, "v2"), "--mito-share", "0.5")


def jepa_csv_rows(path: Path) -> list[dict]:
    return [{k: v for k, v in r.items() if k not in TIMING} for r in GT.jepa_rows(path)]


def mito_tables(genomes):
    import polars as pl
    t = pl.read_parquet(genomes["mt"] / "contigs.parquet").with_columns((pl.col("end") - pl.col("start")).alias("len"))
    return t.filter(~pl.col("held_out")), t.filter(pl.col("held_out"))


def spy_forwards(monkeypatch, sink: list) -> None:
    """Record every unsplit forward ((training, contexts, row length, rows)) and every forward_split call
    (("split", training, core_only, contexts))."""
    real_fwd, real_split = G.KronGenome.forward, G.KronGenome.forward_split

    def fwd(self, seq, segments=None, context=None, *a, **k):
        sink.append(("fwd", self.training, None if context is None else tuple(int(c) for c in context.tolist()),
                     int(seq.shape[-1]), int(seq.shape[0])))
        return real_fwd(self, seq, segments, context, *a, **k)

    def split(self, seq, segments=None, context=None, extra=None, **kw):
        sink.append(("split", self.training, bool(kw.get("core_only")),
                     None if context is None else tuple(int(c) for c in context.tolist())))
        return real_split(self, seq, segments, context, extra, **kw)

    monkeypatch.setattr(G.KronGenome, "forward", fwd)
    monkeypatch.setattr(G.KronGenome, "forward_split", split)


# ------------------------------------------------------------------------------------ (a) option off = v2.3
@pytest.mark.parametrize("short,split", [("pack", []), ("pack", ["--split-cards", "2"]), ("pad", [])])
def test_option_off_trains_as_v23(genomes, tmp_path, monkeypatch, short, split):
    nd = make_recipe_data(genomes)
    real_model_cpu(monkeypatch)
    from kron_dna import cli
    base = parent_cli(monkeypatch, V23_PARENT, "KRON_V24_V23_CLI")
    assert "mito_slice" not in base.genome_train.__code__.co_varnames
    assert "mito_slice" in cli.genome_train.__code__.co_varnames
    args = (base_args(nd) + OFF_ARGS + ["--short-sources", short] + split + BLOCK
            + ["--jepa-log-every", "1", "--jepa-guard", "on"] + GUARD + QUIET)
    traces = {}
    for tag, mod, extra in (("parent", base, []), ("v24", cli, ["--mito-slice", "0"])):
        monkeypatch.setattr(PK, "TRACE", [])
        traces[tag] = trained(mod, monkeypatch, genomes, tmp_path / tag, args + extra)
        traces[tag + "_rows"] = list(PK.TRACE)
    assert_identical(*traces["parent"], *traces["v24"])
    assert traces["parent_rows"] == traces["v24_rows"]                       # the same windows and rows drawn
    if short == "pack":
        assert any(r[0] == "mito" for r in traces["v24_rows"]), "no mito window drawn: the test would be weak"
    a, b = jepa_csv_rows(tmp_path / "parent" / "jepa.csv"), jepa_csv_rows(tmp_path / "v24" / "jepa.csv")
    assert a == b and len(a) == 4
    rec = json.loads((tmp_path / "v24" / "run_record.json").read_text())
    assert "mito_slice" not in rec["resolved"] and rec["options"]["mito_slice"] == 0


# ------------------------------------------------------------- (b) one slice per step, never mito in the main draw
def slice_args(nd, steps: int, more: list[str] = ()) -> list[str]:
    return base_args(nd) + ["--curriculum", f"2048:{steps}", "--tokens-per-step", "4096", "--eval-every",
                            str(steps), "--checkpoint-core", "--short-sources", "pack", "--jepa-log-every", "5",
                            *more] + BLOCK


@pytest.mark.parametrize("split", [[], ["--split-cards", "2"]])
def test_slice_every_step_and_main_draw_without_mito(genomes, tmp_path, monkeypatch, split):
    nd = make_recipe_data(genomes)
    real_model_cpu(monkeypatch)
    monkeypatch.setattr(PK, "TRACE", [])
    calls: list = []
    spy_forwards(monkeypatch, calls)
    from kron_dna import cli
    steps = 12
    res, _ = run(cli, monkeypatch, genomes["root"], tmp_path / "on", slice_args(nd, steps, ["--mito-slice", "4096"])
                 + split)
    assert res.exit_code == 0, res.output + repr(res.exception)
    text = " ".join(res.output.replace("│", " ").split())
    rows = list(PK.TRACE)
    main = [r for r in rows if r[1] == 2048]
    sl = [r for r in rows if r[1] == 4096]
    assert len(main) + len(sl) == len(rows)
    # the main draw: 2 windows per step, never a mito row
    assert sum(r[5] for r in main) == 2 * steps
    assert not any(r[0] == "mito" for r in main)
    # the slice: one packed window per step, mito only, rows of the 4,096-base ladder (4 x 1,024 or 4 x 896)
    assert sum(r[5] for r in sl) == steps and all(r[0] == "mito" for r in sl)
    ladder = dict(PK.row_ladder(4096, 16))
    assert all(r[2] in (1024, 896) and r[4] == 4 == ladder[r[2]] and r[3] == r[2] for r in sl)
    assert len(sl) == 4 * steps
    # forwards: the slice is an unsplit training forward of 4 rows with mito genetic codes, once per step; the
    # main draw's forwards carry nuclear / species contexts only
    train_fwd = [c for c in calls if c[0] == "fwd" and c[1]]
    slice_fwd = [c for c in train_fwd if c[3] in (1024, 896) and c[4] == 4 and set(c[2]) <= MITO_CODES]
    assert len(slice_fwd) == steps
    if split:
        assert len(train_fwd) == steps                                       # the slice is the only unsplit one
        main_ctx = {x for c in calls if c[0] == "split" and c[1] and not c[2] for x in c[3]}
        assert main_ctx and main_ctx <= MAIN_CONTEXTS
        assert "split windows over 2 cards" in text
    else:
        main_ctx = {x for c in train_fwd if c not in slice_fwd for x in c[2]}
        assert main_ctx and main_ctx <= MAIN_CONTEXTS
    assert {x for c in slice_fwd for x in c[2]} == MITO_CODES                # both codes drawn in 12 slices
    # the log: the slice's tally line at the evaluation, and the start-up line with the shares
    assert f"mito slice rows since the last report (--mito-slice 4,096, one packed window per step): {steps} " \
           f"windows: mito 100.0% (4.0 rows/window" in text
    assert "mito slice: every step also 4,096 bases of packed mitogenomes" in text
    assert not any(r_[0] == "mito" for r_ in rows if r_[1] == 2048)
    # jepa.csv keeps the main draw's rows only: one per 5 steps and the evaluation's
    assert [r["steps"] for r in GT.jepa_rows(tmp_path / "on" / "jepa.csv")] == ["5", "5", "2"]


def test_slice_reads_training_mitogenomes_only_with_their_codes(genomes, tmp_path, monkeypatch):
    """Every slice row is a stretch of one training mitogenome (never a held-out one), read whole across the
    origin for a circular one, with that genome's genetic code as context."""
    nd = make_recipe_data(genomes)
    real_model_cpu(monkeypatch)
    train, val = mito_tables(genomes)
    mt = np.fromfile(genomes["mt"] / "bases.u8", dtype=np.uint8)
    seen: list = []
    real_fwd = G.KronGenome.forward

    def fwd(self, seq, segments=None, context=None, *a, **k):
        if self.training and context is not None and set(int(c) for c in context.tolist()) <= MITO_CODES \
                and seq.shape[0] == 4:
            seen.append((seq.detach().cpu().clone(), context.detach().cpu().clone()))
        return real_fwd(self, seq, segments, context, *a, **k)

    monkeypatch.setattr(G.KronGenome, "forward", fwd)
    from kron_dna import cli
    res, _ = run(cli, monkeypatch, genomes["root"], tmp_path / "on",
                 [a_ for a_ in slice_args(nd, 6, ["--mito-slice", "4096"]) if a_ not in ("--rc", "--bert-mask")]
                 + ["--soft-base-rate", "0"])
    assert res.exit_code == 0, res.output + repr(res.exception)
    assert len(seen) == 6
    genomes_ = [(np.concatenate([mt[r["offset"]:r["offset"] + r["len"]]] * 2), int(r["code"]), r["len"])
                for r in train.iter_rows(named=True)]
    held_ = [mt[r["offset"]:r["offset"] + r["len"]] for r in val.iter_rows(named=True)]
    for x_, ctx_ in seen:
        x_ = x_.numpy() if x_.dim() == 2 else x_.argmax(1).numpy()
        for row, c in zip(x_, ctx_.tolist()):
            known = row < 4                                                  # MASK and N are not the genome's
            hits = []
            for doubled, code, n in genomes_:
                if n < len(row):                                             # a row never outgrows its genome
                    continue
                for s0 in range(n):
                    if np.array_equal(doubled[s0:s0 + len(row)][known], row[known]):
                        hits.append(code)
                        break
            assert hits, "a slice row is not a stretch of a training mitogenome"
            assert c in hits
            assert not any(len(h) >= len(row) and any(np.array_equal(np.concatenate([h, h])[s0:s0 + len(row)][known],
                                                                     row[known]) for s0 in range(len(h)))
                           for h in held_)


@pytest.mark.parametrize("bases,row_lengths,n", [(1024, {1024, 896}, 1), (4096, {1024, 896}, 4)])
def test_slice_rows_follow_the_ladder(genomes, tmp_path, monkeypatch, bases, row_lengths, n):
    """A mitogenome at least as long as the slice is one full row; a shorter one gives the ladder's rows."""
    nd = make_recipe_data(genomes)
    real_model_cpu(monkeypatch)
    monkeypatch.setattr(PK, "TRACE", [])
    from kron_dna import cli
    res, _ = run(cli, monkeypatch, genomes["root"], tmp_path / "on", slice_args(nd, 10, ["--mito-slice", str(bases)]))
    assert res.exit_code == 0, res.output + repr(res.exception)
    sl = [r for r in PK.TRACE if r[1] == bases]
    assert sum(r[5] for r in sl) == 10
    assert {r[2] for r in sl} <= row_lengths and all(r[4] == n for r in sl)
    if bases == 1024:
        assert any(r[2] == 1024 for r in sl), "no mitogenome as long as the slice drawn: one full row expected"


# ------------------------------------------------------------------------------ (c) shares and the run record
def test_main_draw_shares_without_mito_and_run_record(genomes, tmp_path, monkeypatch):
    nd = make_recipe_data(genomes)
    real_model_cpu(monkeypatch)
    monkeypatch.setattr(PK, "TRACE", [])
    from kron_dna import cli
    steps = 150
    args = slice_args(nd, steps, ["--mito-slice", "1024"])
    res, _ = run(cli, monkeypatch, genomes["root"], tmp_path / "on", args)
    assert res.exit_code == 0, res.output + repr(res.exception)
    rec = json.loads((tmp_path / "on" / "run_record.json").read_text())
    # configured (absolute window shares): gB 0.1, mito 0.5, gA (ntv3 extra) 0.3 of the rest, hg 0.7 of the rest
    want = {"genome0": 0.1 / 0.5, "mito": 0.0, "extra0": 0.4 * 0.3 / 0.5, "primary": 0.4 * 0.7 / 0.5}
    got = rec["resolved"]["window_shares"]
    assert got.keys() == want.keys() and all(got[k] == pytest.approx(v) for k, v in want.items())
    ms = rec["resolved"]["mito_slice"]
    assert ms["bases"] == 1024 and ms["configured_mito_share"] == 0.5
    ph = ms["per_phase"][0]
    assert ph["main_bases"] == 4096 and ph["mito_base_share"] == pytest.approx(1024 / 5120)
    assert sum(ph["base_shares"].values()) == pytest.approx(1.0)
    assert ph["base_shares"]["primary"] == pytest.approx(want["primary"] * 4096 / 5120)
    by = {(g["role"], g["name"]): g for g in rec["genomes"]}
    assert by[("mito", "mt")]["main_draw_share"] == 0.0 and by[("mito", "mt")]["mito_slice"] == 1024
    assert by[("genome", "gB")]["share"] == pytest.approx(0.2)
    # realised: the main draw's windows by source (lead rows), 2 x 150 = 300 windows
    lead = [r[0].split(":")[0] for r in PK.TRACE if r[1] == 2048 and r[5]]
    assert len(lead) == 2 * steps and "mito" not in lead
    frac = {k: lead.count(k) / len(lead) for k in ("genome", "ntv3_extra", "primary")}
    se = {k: math.sqrt(v * (1 - v) / len(lead)) for k, v in
          (("genome", want["genome0"]), ("ntv3_extra", want["extra0"]), ("primary", want["primary"]))}
    assert abs(frac["genome"] - want["genome0"]) < 4 * se["genome"]
    assert abs(frac["ntv3_extra"] - want["extra0"]) < 4 * se["ntv3_extra"]
    assert abs(frac["primary"] - want["primary"]) < 4 * se["primary"]


# ------------------------------------------------------------------------------------------ (d) loss weights
def test_loss_weights_proportional_to_bases(genomes, tmp_path, monkeypatch):
    """--bench --bench-grads: the first step's losses are [main draw x main/(main+slice), slice x slice/(main+
    slice)], each the parts' losses (JEPA_TRACE totals) with loss_parts' weights (rows / pack size / batch)."""
    nd = make_recipe_data(genomes)
    real_model_cpu(monkeypatch)
    monkeypatch.setattr(PK, "TRACE", [])
    from kron_dna import cli
    monkeypatch.setattr(cli, "JEPA_TRACE", [])
    args = base_args(nd) + ["--curriculum", "2048:10", "--tokens-per-step", "4096", "--bench", "--checkpoint-core",
                            "--short-sources", "pack", "--jepa-log-every", "1", "--mito-slice", "4096",
                            "--bench-grads", str(tmp_path / "g.pt")] + BLOCK
    res, _ = run(cli, monkeypatch, genomes["root"], tmp_path / "b", args)
    assert res.exit_code == 0, res.output + repr(res.exception)
    losses = torch.load(tmp_path / "g.pt")["losses"]
    assert len(losses) == 2
    # step 0's rows and terms: TRACE and JEPA_TRACE in call order (main parts by row length, longest first, then
    # the slice's part); step 0 = everything before the second slice
    rows, terms = list(PK.TRACE), list(cli.JEPA_TRACE)
    first_slice = next(i for i, r in enumerate(rows) if r[1] == 4096)
    main0 = rows[:first_slice]
    slice0 = [r for r in rows[first_slice:first_slice + 4]]
    assert all(r[1] == 2048 for r in main0) and sum(r[5] for r in main0) == 2
    lengths = sorted({r[2] for r in main0}, reverse=True)
    part_w = [sum(1.0 / r[4] for r in main0 if r[2] == L) / 2 for L in lengths]
    assert sum(part_w) == pytest.approx(1.0)
    main_terms, slice_terms = terms[:len(lengths)], terms[len(lengths):len(lengths) + 1]
    w_main, w_slice = 4096 / (4096 + 4096), 4096 / (4096 + 4096)
    want_main = w_main * sum(w * t["total"] for w, t in zip(part_w, main_terms))
    want_slice = w_slice * sum(1.0 * t["total"] for t in slice_terms)            # one part, 4 rows of 1/4
    assert all(r[4] == 4 for r in slice0)
    assert losses[0] == pytest.approx(want_main, rel=1e-5)
    assert losses[1] == pytest.approx(want_slice, rel=1e-5)
    assert all(math.isfinite(x) for x in losses)


def test_loss_weights_unequal_bases(genomes, tmp_path, monkeypatch):
    """Main 4,096 bases (2 x 2,048) and a 1,024-base slice: weights 0.8 and 0.2."""
    nd = make_recipe_data(genomes)
    real_model_cpu(monkeypatch)
    monkeypatch.setattr(PK, "TRACE", [])
    from kron_dna import cli
    monkeypatch.setattr(cli, "JEPA_TRACE", [])
    args = base_args(nd) + ["--curriculum", "2048:10", "--tokens-per-step", "4096", "--bench", "--checkpoint-core",
                            "--short-sources", "pack", "--jepa-log-every", "1", "--mito-slice", "1024",
                            "--bench-grads", str(tmp_path / "g.pt")]
    res, _ = run(cli, monkeypatch, genomes["root"], tmp_path / "b", args)
    assert res.exit_code == 0, res.output + repr(res.exception)
    losses = torch.load(tmp_path / "g.pt")["losses"]
    rows, terms = list(PK.TRACE), list(cli.JEPA_TRACE)
    first_slice = next(i for i, r in enumerate(rows) if r[1] == 1024)
    main0 = rows[:first_slice]
    lengths = sorted({r[2] for r in main0}, reverse=True)
    part_w = [sum(1.0 / r[4] for r in main0 if r[2] == L) / 2 for L in lengths]
    want_main = 0.8 * sum(w * t["total"] for w, t in zip(part_w, terms[:len(lengths)]))
    want_slice = 0.2 * terms[len(lengths)]["total"]
    assert losses == pytest.approx([want_main, want_slice], rel=1e-5)


# ------------------------------------------------------------------- (e) split: slice unsplit, step = unsplit run
def test_split_first_step_equals_unsplit_with_slice(genomes, tmp_path, monkeypatch):
    nd = make_recipe_data(genomes)
    real_model_cpu(monkeypatch)
    smooth_routing(monkeypatch)
    from kron_dna import cli
    args = base_args(nd) + ["--curriculum", "2048:10", "--tokens-per-step", "4096", "--bench", "--checkpoint-core",
                            "--short-sources", "pack", "--jepa-log-every", "1", "--mito-slice", "4096"] + BLOCK
    out, traces = {}, {}
    for tag, extra in (("one", []), ("split", ["--split-cards", "2"])):
        monkeypatch.setattr(PK, "TRACE", [])
        res, _ = run(cli, monkeypatch, genomes["root"], tmp_path / tag,
                     args + extra + ["--bench-grads", str(tmp_path / f"{tag}.pt")])
        assert res.exit_code == 0, res.output + repr(res.exception)
        out[tag], traces[tag] = torch.load(tmp_path / f"{tag}.pt"), list(PK.TRACE)
    assert traces["one"] == traces["split"]
    a, b = out["one"], out["split"]
    assert len(a["losses"]) == 2 and all(np.isfinite(a["losses"])) and all(np.isfinite(b["losses"]))
    assert b["losses"] == pytest.approx(a["losses"], rel=1e-5)
    n_grad = 0
    for n_, g in a["grads"].items():
        assert (g is None) == (b["grads"][n_] is None), n_
        if g is not None:
            n_grad += 1
            assert float((g - b["grads"][n_]).abs().max()) <= 1e-4 * (float(g.abs().max()) + 1e-12) + 1e-9, n_
    assert n_grad > 50


# ------------------------------------------------------------------------------- (f) resume a v2.3 checkpoint
def test_resume_v23_checkpoint_with_slice(genomes, tmp_path, monkeypatch):
    nd = make_recipe_data(genomes)
    real_model_cpu(monkeypatch)
    from kron_dna import cli
    base = parent_cli(monkeypatch, V23_PARENT, "KRON_V24_V23_CLI")
    run_dir = tmp_path / "F"
    common = base_args(nd) + ["--tokens-per-step", "4096", "--eval-every", "3", "--checkpoint-core",
                              "--short-sources", "pack", "--split-cards", "2", "--jepa-log-every", "1",
                              "--jepa-guard", "on"] + BLOCK + GUARD + QUIET
    res, _ = run(base, monkeypatch, genomes["root"], run_dir, common + ["--curriculum", "2048:3"])
    assert res.exit_code == 0, res.output + repr(res.exception)
    ck0 = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert ck0["step"] == 3 and "mito_slice" not in ck0["run_record"]["options"]
    head0 = (run_dir / "jepa.csv").read_text().split("\n", 1)[0]
    res, _ = run(cli, monkeypatch, genomes["root"], run_dir, common + ["--curriculum", "2048:6", "--mito-slice", "4096"])
    assert res.exit_code == 0, res.output + repr(res.exception)
    text = " ".join(res.output.replace("│", " ").split())
    assert "resumed at step 3" in text and "refusing" not in text
    assert "mito slice rows since the last report" in text
    ck = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert ck["step"] == 6 and ck["run_record"]["resolved"]["mito_slice"]["bases"] == 4096
    rec3 = json.loads((run_dir / "run_record.step3.json").read_text())
    assert rec3["started_at_step"] == 3 and rec3["options"]["mito_slice"] == 4096
    assert (run_dir / "jepa.csv").read_text().split("\n", 1)[0] == head0           # one header, rows appended
    m = (run_dir / "metrics.csv").read_text().strip().split("\n")
    assert [r.split(",")[0] for r in m[1:]] == ["3", "6"]
    assert all(math.isfinite(float(x)) for x in m[-1].split(",")[1:])
    # and back: the v2.4 checkpoint resumes on v2.3 (the fallback the launch script uses)
    res, _ = run(base, monkeypatch, genomes["root"], run_dir, common + ["--curriculum", "2048:9"])
    assert res.exit_code == 0, res.output + repr(res.exception)
    assert torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)["step"] == 9


# ------------------------------------------------------------------------------------------------ (g) refusals
@pytest.mark.parametrize("extra,words", [
    (["--mito-slice", "4096", "--short-sources", "pad"], "--mito-slice needs --mito and --short-sources pack"),
    (["--mito-slice", "4008", "--short-sources", "pack"], "--mito-slice must be a multiple of 16"),
    (["--mito-slice", "64", "--short-sources", "pack"], "at least 128 bases"),
    (["--mito-slice", "-1", "--short-sources", "pack"], "--mito-slice must be >= 0"),
])
def test_refusals(genomes, tmp_path, monkeypatch, extra, words):
    nd = make_recipe_data(genomes)
    real_model_cpu(monkeypatch)
    from kron_dna import cli
    res, _ = run(cli, monkeypatch, genomes["root"], tmp_path / "x",
                 base_args(nd) + ["--curriculum", "2048:3", "--tokens-per-step", "4096"] + extra)
    assert res.exit_code != 0
    assert words in " ".join(res.output.replace("│", " ").split())


def test_refuses_without_mito(genomes, tmp_path, monkeypatch):
    nd = make_recipe_data(genomes)
    real_model_cpu(monkeypatch)
    from kron_dna import cli
    args = base_args(nd)
    i = args.index("--mito")
    args = args[:i] + args[i + 2:]
    res, _ = run(cli, monkeypatch, genomes["root"], tmp_path / "x",
                 args + ["--curriculum", "2048:3", "--tokens-per-step", "4096", "--short-sources", "pack",
                         "--mito-slice", "4096"])
    assert res.exit_code != 0 and "--mito-slice needs --mito" in " ".join(res.output.replace("│", " ").split())
