"""genes_eval.py on a toy genome with a small model (the CPU stand-in for the KronLSTM layer): it scores the genes head
against two annotations, and where they disagree it reports the head's agreement with each, summing to one."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

import kron_core.blocks
from kron_dna import genome as G

ROOT = Path(__file__).resolve().parents[1] / "scripts" / "kron_dna"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_genes_eval_scores_two_annotations(tmp_path, monkeypatch):
    from kron_core.testing import CausalMixer
    from test_kron_dna_split import small
    monkeypatch.setattr(kron_core.blocks, "kronlstm_mixer", CausalMixer)
    GS, GE, SZ = load("prep_gene_structure_labels"), load("genes_eval"), load("splice_zeroshot")
    rng = np.random.default_rng(0)
    n = 800_000
    seq = rng.integers(0, 4, n).astype(np.uint8)
    for s0 in range(1000, n - 5000, 20_000):                    # genes: exon, GT..AG intron, exon; ATG at the start
        seq[s0:s0 + 3] = [0, 3, 2]
        seq[s0 + 300:s0 + 302] = [2, 3]
        seq[s0 + 798:s0 + 800] = [0, 2]
    g = tmp_path / "genome" / "toy"
    g.mkdir(parents=True)
    seq.tofile(g / "bases.u8")
    pl.DataFrame(dict(chrom=["chr1"], start=[0], end=[n], offset=[0])).write_parquet(g / "contigs.parquet")
    for tag, extra in (("old", 0), ("new", 1)):                 # the new annotation has one more gene's 3' exon longer
        rows = []
        for k, s0 in enumerate(range(1000, n - 5000, 20_000)):
            e2 = 1100 + (200 if extra and k % 2 else 0)
            rows += [f"chr1\tX\texon\t{s0 + 1}\t{s0 + 300}\t.\t+\t.\ttranscript_id \"t{k}\";",
                     f"chr1\tX\texon\t{s0 + 801}\t{s0 + e2}\t.\t+\t.\ttranscript_id \"t{k}\";",
                     f"chr1\tX\tCDS\t{s0 + 1}\t{s0 + 300}\t.\t+\t.\ttranscript_id \"t{k}\";",
                     f"chr1\tX\tstart_codon\t{s0 + 1}\t{s0 + 3}\t.\t+\t.\ttranscript_id \"t{k}\";"]
        (tmp_path / f"{tag}.gtf").write_text("\n".join(rows) + "\n")
        monkeypatch.setattr(GS, "interim_dir", lambda *a: g)
        GS.main(tmp_path / tag, name="toy", gtf=tmp_path / f"{tag}.gtf", basic_only=False, chrom_prefix="")
    cfg = G.KronGenomeConfig(**{**small("v2").config.to_dict(), "xheads": [["genes", 14]], "xhead_kind": "kron",
                                "xhead_bin": 4, "xhead_width": 64})
    torch.manual_seed(0)
    model = G.KronGenome(cfg)
    torch.save(dict(config=cfg.to_dict(), model=model.state_dict(), step=7), tmp_path / "ck.pt")
    for mod in (GE, SZ):
        monkeypatch.setattr(mod, "interim_dir", lambda *a: g)
    out = tmp_path / "eval"
    SZ.windows(out, name="toy", n=2, seed=0, context=131072, chroms="", keep=None)
    GE.score(out, [f"m={tmp_path / 'ck.pt'}"], name="toy", labels=[f"old={tmp_path / 'old'}", f"new={tmp_path / 'new'}"],
             windows_csv=None, head="genes", context=131072, device="cpu")
    r = json.loads((out / "m.genes.json").read_text())
    assert set(r["annotations"]) == {"old", "new"} and r["n_windows"] == 2 and r["step"] == 7
    assert 0 <= r["annotations"]["new"]["genic_f1"] <= 1
    d = r["disagreement"]
    assert set(d) == {"utr3_plus", "polya_plus"} or "utr3_plus" in d     # the longer 3' exons are the difference
    for v in d.values():
        assert v["agrees_with"]["old"] + v["agrees_with"]["new"] == pytest.approx(1.0)
