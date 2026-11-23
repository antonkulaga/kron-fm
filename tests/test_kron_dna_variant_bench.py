"""scripts/kron_dna/variant_bench.py: reference against alternative allele through a track head, on a toy genome."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from test_kron_dna_multigenome import genomes  # noqa: F401  (a fixture)

spec = importlib.util.spec_from_file_location(
    "variant_bench", Path(__file__).resolve().parents[1] / "scripts" / "kron_dna" / "variant_bench.py")
VB = importlib.util.module_from_spec(spec)
spec.loader.exec_module(VB)


def test_alleles_and_ranks():
    seq = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    assert VB.with_alt(seq, 2, "G", "T").tolist() == [0, 1, 3, 3, 0, 1, 2, 3]
    assert VB.with_alt(seq, 2, "GT", "G").tolist() == [0, 1, 2, 0, 1, 2, 3, 4]      # a deletion: N-padded at the end
    assert VB.with_alt(seq, 2, "G", "GAA").tolist() == [0, 1, 2, 0, 0, 3, 0, 1]     # an insertion: cut at the end
    assert VB.with_alt(seq, 2, "A", "T") is None and VB.with_alt(seq, 7, "TT", "T") is None
    y = np.array([1, 1, 0, 0, 1])
    assert VB.auroc(np.array([3.0, 2.0, 1.0, 0.0, 0.5]), y) == pytest.approx(5 / 6)
    assert VB.auroc(np.zeros(5), y) == pytest.approx(0.5)                           # ties count half


def test_score_merge_and_asb_report(genomes, tmp_path, monkeypatch, device):
    import torch
    from typer.testing import CliRunner

    from kron_dna import genome as G
    from test_kron_dna_split import small
    from test_kron_dna_xheads_cli import make_store

    if device.type != "cpu":
        pytest.skip("the end-to-end run is the CPU one, with the stand-in for the KronLSTM layer")
    root, hgt = genomes["root"], genomes["hgt"]
    monkeypatch.setattr(VB, "interim_dir", lambda *parts: root.joinpath(*parts))
    store = make_store(tmp_path, hgt, n=3)
    bases = np.fromfile(hgt / "bases.u8", dtype=np.uint8)
    c = pl.read_parquet(hgt / "contigs.parquet")
    letters = "ACGT"
    rows = []
    for chrom, pos in [("chr1", 1500), ("chr1", 3100), ("chr8", 1200)]:
        r = c.filter(pl.col("chrom") == chrom).row(0, named=True)
        b = int(bases[r["offset"] + pos - r["start"]])
        if b > 3:
            continue
        for alt in (letters[(b + 1) % 4], letters[(b + 2) % 4]):
            rows.append(dict(chrom=chrom, pos=pos, ref=letters[b], alt=alt))
    rows.append(dict(chrom="chr1", pos=2500, ref="N", alt="A"))                     # not the assembly's base
    r1 = c.filter(pl.col("chrom") == "chr1").row(0, named=True)
    two = bases[r1["offset"] + 2800:r1["offset"] + 2802]
    if (two < 4).all():
        rows.append(dict(chrom="chr1", pos=2800, ref=letters[two[0]] + letters[two[1]], alt=letters[two[0]]))
    rng = np.random.default_rng(0)
    n = len(rows)
    t = pl.DataFrame(rows).with_columns(
        side=pl.Series(rng.choice(["alt", "ref"], n)), effect=pl.Series(rng.normal(size=n)),
        motif_fc=pl.Series(rng.normal(size=n)), motif_conc=pl.Series(rng.choice(["No Hit", "Concordant"], n)),
        match=pl.lit("tf_cell"), channels=pl.lit(["F1"]), held_out=pl.col("chrom") == "chr8")
    t.write_parquet(tmp_path / "v.parquet")
    cfg = G.KronGenomeConfig(**{**small("v2").config.to_dict(), "xheads": [["dnase", 3]], "xhead_kind": "kron",
                                "xhead_bin": 4, "xhead_width": 64})
    torch.manual_seed(0)
    torch.save({"model": G.KronGenome(cfg).state_dict(), "config": cfg.to_dict(), "step": 7}, tmp_path / "ck.pt")
    out = tmp_path / "bench"
    base = ["--variants", tmp_path / "v.parquet", "--head", "dnase", "--store", store, "--name", "hgt", "--context", 768,
            "--edge", 64, "--flank", 32, "--device", "cpu", "--dtype", "fp32"]
    run = lambda *a: CliRunner().invoke(VB.app, [str(x) for x in a])
    res = run("score", out, f"m={tmp_path / 'ck.pt'}", *base)
    assert res.exit_code == 0, res.output + repr(res.exception)
    d = out / "pred" / "m"
    info = json.loads((d / "index.json").read_text())
    sites = pl.read_parquet(d / "sites.parquet")
    ref, alt = np.load(d / "ref.npy"), np.load(d / "alt.npy")
    assert info["ref_mismatch"] == 1 and info["ids"] == ["F0", "F1", "F2"] and sites.height == n - 1
    assert ref.shape == alt.shape == (n - 1, 3) and (ref > 0).all()
    a, b = (sites.with_row_index("k").filter((pl.col("chrom") == "chr1") & (pl.col("pos") == 1500))["k"]).to_list()
    assert np.array_equal(ref[a], ref[b]) and not np.allclose(alt[a], alt[b])      # one reference, two alternatives
    assert not np.allclose(ref, alt)
    for i in (1, 0):                                     # two shards, merged: the unsharded arrays
        res = run("score", out, f"s={tmp_path / 'ck.pt'}", *base, "--shard", f"{i}/2")
        assert res.exit_code == 0, res.output + repr(res.exception)
    assert not (out / "pred" / "s").exists()
    res = run("merge", out, "s", "--shards", 2)
    assert res.exit_code == 0, res.output + repr(res.exception)
    m = out / "pred" / "s"
    assert pl.read_parquet(m / "sites.parquet").equals(sites)
    assert np.array_equal(np.load(m / "ref.npy"), ref) and np.array_equal(np.load(m / "alt.npy"), alt)
    for label, extra in (("b", ["--batch", 4]), ("c", ["--chunk", 256])):          # batched; memory-capped
        res = run("score", out, f"{label}={tmp_path / 'ck.pt'}", *base, *extra)
        assert res.exit_code == 0, res.output + repr(res.exception)
        assert np.allclose(np.load(out / "pred" / label / "alt.npy"), alt, rtol=1e-4, atol=1e-4)
        assert np.allclose(np.load(out / "pred" / label / "ref.npy"), ref, rtol=1e-4, atol=1e-4)
    res = run("report-asb", out, "--n-boot", 5, "--min-rows", 3)
    assert res.exit_code == 0, res.output + repr(res.exception)
    rep = json.loads((out / "report_asb.json").read_text())
    top = next(r for r in rep if r["label"] == "m" and r["subset"] == "tf_cell")
    assert top["rows"] == n - 1 and 0 <= top["model_auroc"] <= 1 and "motif_auroc" in top
    rows_ = VB.asb_rows(d, t)
    want = np.log2((alt[:, 1] + 1) / (ref[:, 1] + 1))                               # the row's own channel, F1
    assert np.allclose(np.sort(rows_["pred"].to_numpy()), np.sort(want), atol=1e-6)
    # the MPRA report on the same predictions: a table read at each variant's own position
    mp = t.select("chrom", "pos", "ref", "alt", "effect", "held_out", centre="pos").with_columns(
        experiment=pl.lit("E1"), element=pl.lit("E1"), kind=pl.lit("enhancer"), cell=pl.lit("K562"), primary=pl.lit(True),
        confident=pl.lit(True))
    mp.write_parquet(tmp_path / "mpra.parquet")
    res = run("report-mpra", out, "--variants", tmp_path / "mpra.parquet", "--min-rows", 3)
    assert res.exit_code == 0, res.output + repr(res.exception)
    rm = next(r for r in json.loads((out / "report_mpra.json").read_text()) if r["label"] == "m")
    assert rm["variants"] == n - 1 and -1 <= rm["spearman_all"] <= 1 and rm["own_channels"] == 0
    # the eQTL report: the same sites as a positive/negative table read at "the TSS" (here each variant's position)
    eq = t.select("chrom", "ref", "alt", "held_out", position="pos", tss="pos").with_row_index("k").with_columns(
        label=(pl.col("k") % 2).cast(pl.Int32), pair=(pl.col("k") // 2).cast(pl.Int64), strand=pl.lit("+"),
        distance=pl.col("k").cast(pl.Int64) * 100).drop("k")
    eq.write_parquet(tmp_path / "eqtl.parquet")
    res = run("report-eqtl", out, "--variants", tmp_path / "eqtl.parquet", "--n-boot", 5)
    assert res.exit_code == 0, res.output + repr(res.exception)
    assert (out / "report_eqtl.json").exists()
    # elements read once (lentiMPRA's shape): the reference readout of the same places, and the probe report
    el = pl.DataFrame(dict(chrom=["chr1"] * 4 + ["chr8"] * 2, pos=[1500, 3100, 2000, 2600, 1200, 1500])).with_columns(
        ref=pl.lit(""), alt=pl.lit(""), cell=pl.lit("K562"), activity=pl.Series(rng.normal(size=6)),
        fold=pl.Series([1, 2, 1, 2, 1, 2]), held_out=pl.col("chrom") == "chr8")
    el.write_parquet(tmp_path / "el.parquet")
    out2 = tmp_path / "bench2"
    res = run("score", out2, f"m={tmp_path / 'ck.pt'}", *[tmp_path / "el.parquet" if x == tmp_path / "v.parquet" else x for x in base],
              "--ref-only", "--batch", 3)
    assert res.exit_code == 0, res.output + repr(res.exception)
    r2, s2 = np.load(out2 / "pred" / "m" / "ref.npy"), pl.read_parquet(out2 / "pred" / "m" / "sites.parquet")
    assert np.array_equal(r2, np.load(out2 / "pred" / "m" / "alt.npy")) and s2.height == 6
    k2 = s2.with_row_index("k").filter((pl.col("chrom") == "chr1") & (pl.col("pos") == 1500))["k"][0]
    assert np.allclose(r2[k2], ref[a], rtol=1e-4, atol=1e-4)                        # the variant run's reference
    res = run("report-lentimpra", out2)
    assert res.exit_code == 0, res.output + repr(res.exception)
    assert json.loads((out2 / "report_lentimpra.json").read_text())[0]["elements"] == 6


def test_label_head_and_mfass_report(genomes, tmp_path, monkeypatch, device):
    """The splice head: the highest logit per channel within each row's own flank; the MFASS report on it."""
    import torch
    from typer.testing import CliRunner

    from kron_dna import genome as G
    from test_kron_dna_split import small

    if device.type != "cpu":
        pytest.skip("the end-to-end run is the CPU one, with the stand-in for the KronLSTM layer")
    root, hgt = genomes["root"], genomes["hgt"]
    monkeypatch.setattr(VB, "interim_dir", lambda *parts: root.joinpath(*parts))
    bases = np.fromfile(hgt / "bases.u8", dtype=np.uint8)
    lab_dir = tmp_path / "splice"
    lab_dir.mkdir()
    bits = {"donor_plus": 0, "acceptor_plus": 1, "donor_minus": 2, "acceptor_minus": 3}
    np.zeros(bases.size, np.uint8).tofile(lab_dir / "splice.u8")
    (lab_dir / "splice.json").write_text(json.dumps(dict(
        genome="hgt", total_bases=int(bases.size), bits=bits, definition="one base per site, per strand", splice_sha256="x",
        check={k: dict(sites=0) for k in bits})))
    c = pl.read_parquet(hgt / "contigs.parquet").filter(pl.col("chrom") == "chr1").row(0, named=True)
    rows, letters = [], "ACGT"
    for k, pos in enumerate(range(1400, 4400, 150)):
        b = int(bases[c["offset"] + pos - c["start"]])
        if b < 4:
            rows.append(dict(chrom="chr1", pos=pos, ref=letters[b], alt=letters[(b + 1) % 4], strand="+-"[k % 2],
                             exon_mid=pos + 20, half=40 + k, region=["exon", "upstr_intron"][k % 2], sdv=k % 3 == 0,
                             dpsi=-0.1 * k, held_out=False))
    t = pl.DataFrame(rows)
    t.write_parquet(tmp_path / "mfass.parquet")
    cfg = G.KronGenomeConfig(**{**small("v2").config.to_dict(), "xheads": [["splice", 4]], "xhead_kind": "kron",
                                "xhead_bin": 4, "xhead_width": 64})
    torch.manual_seed(0)
    torch.save({"model": G.KronGenome(cfg).state_dict(), "config": cfg.to_dict(), "step": 7}, tmp_path / "ck.pt")
    out = tmp_path / "bench"
    run = lambda *a: CliRunner().invoke(VB.app, [str(x) for x in a])
    res = run("score", out, f"m={tmp_path / 'ck.pt'}", "--variants", tmp_path / "mfass.parquet", "--head", "splice",
              "--store", lab_dir, "--name", "hgt", "--context", 768, "--edge", 64, "--read-col", "exon_mid",
              "--flank-col", "half", "--device", "cpu", "--dtype", "fp32", "--batch", 2)
    assert res.exit_code == 0, res.output + repr(res.exception)
    d = out / "pred" / "m"
    info = json.loads((d / "index.json").read_text())
    ref, alt = np.load(d / "ref.npy"), np.load(d / "alt.npy")
    assert info["kind"] == "binary" and info["ids"] == list(bits) and ref.shape == (t.height, 4)
    assert (ref < 0).any() and not np.allclose(ref, alt)            # logits, not signal
    res = run("report-mfass", out)
    assert res.exit_code == 0, res.output + repr(res.exception)
    rep = json.loads((out / "report_mfass.json").read_text())
    assert rep[0]["variants"] == t.height and 0 <= rep[0]["ap"] <= 1
