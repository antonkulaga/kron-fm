"""kron_dna.recipes: every arm's arguments parse as genome-train options, paths sit under the given data root, and
the recipes carry what each arm was trained with (checked once against the real runs' records on 24 Nov: D, F,
F-step2, A3 and G-a matched option for option)."""
from __future__ import annotations

import shlex

import pytest
import typer.main
from typer.testing import CliRunner

from kron_dna import cli, recipes


def parse(args, tmp_path):
    cmd = typer.main.get_command(cli.app).commands["genome-train"]
    return cmd.make_context("genome-train", [str(tmp_path / "run")] + args).params


@pytest.mark.parametrize("arm", recipes.ARMS)
@pytest.mark.parametrize("genomes", ["reference", "all"])
def test_every_arm_parses(arm, genomes, tmp_path):
    args = recipes.recipe(arm, genomes=genomes, cards=2, root=tmp_path / "data")
    p = parse(args, tmp_path)
    assert p["split_cards"] == 2 and p["arch"] == "v2" and p["val_chroms"] == "chr8,chr9"
    for v in p["xhead"]:
        assert str(tmp_path / "data") in v                         # every store under the given root
    assert all("/data/uq2347" not in a for a in args)              # nothing of our cluster


def test_arm_specifics(tmp_path):
    f = parse(recipes.recipe("F", root=tmp_path), tmp_path)
    assert f["n_layers"] == 10 and f["curriculum"] == "1048576:380000" and f["mito_slice"] == 65536
    assert f["jepa_weight"] == 1.0 and f["jepa_pred"] == "mlp" and f["stem_channels"] == 256
    d = parse(recipes.recipe("D", root=tmp_path), tmp_path)
    assert d["n_layers"] == 8 and d["curriculum"] == "8192:46000,524288:354000" and d["mito_slice"] == 0
    a = parse(recipes.recipe("A", root=tmp_path), tmp_path)
    assert a["jepa_weight"] == 0.0 and a["tokens_per_step"] == 524288
    g = parse(recipes.recipe("G", edits=True, pairs=True, root=tmp_path), tmp_path)
    assert g["stem_channels"] == 64 and g["local_track"] == "64,2,8192" and g["jepa_pred"] == "context"
    assert g["jepa_loss"] == "zcos-mse" and g["jepa_decode_pred"] and g["jepa_edits"] == 1.0
    assert len(g["cf_pairs"]) == 4 and g["cf_batch"] == 1
    a3 = recipes.recipe("A3", genomes="all", root=tmp_path)
    assert sum(1 for x in a3 if x.startswith("ape_")) == 5          # the apes once, at A3's share
    fa = recipes.recipe("F", genomes="all", root=tmp_path)
    gen = [fa[i + 1] for i, x in enumerate(fa) if x == "--genome"]
    assert gen.index("mm10:0.10:70:chr8+chr9") < gen.index("ape_chimp_t2t:0.01:65")   # F's order


def test_refusals(tmp_path):
    with pytest.raises(ValueError, match="step2"):
        recipes.recipe("F", edits=True, root=tmp_path)
    with pytest.raises(ValueError, match="unknown arm"):
        recipes.recipe("Z", root=tmp_path)


def test_cli_prints_quoted_args(tmp_path, monkeypatch):
    monkeypatch.setenv("KRON_FM_DATA", str(tmp_path))
    res = CliRunner().invoke(recipes.app, ["G", "--cards", "2", "--edits", "--pairs"])
    assert res.exit_code == 0, res.output
    args = shlex.split(res.output)
    assert args == recipes.recipe("G", cards=2, edits=True, pairs=True, root=tmp_path)


def test_genome_sh_submits_the_resolved_recipe(tmp_path):
    """slurm/genome.sh with a stand-in sbatch: the scheduler flags come from env.local-style variables, and the
    arguments file holds exactly the recipe's arguments plus the extra genome-train options after --."""
    import os
    import subprocess
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "sbatch").write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$SBATCH_LOG"\necho "Submitted batch job 1"\n')
    (bin_ / "sbatch").chmod(0o755)
    (bin_ / "uv").write_text('#!/bin/bash\nshift 3\nexec "' + __import__("sys").executable + '" "$@"\n')   # uv run --no-sync python X
    (bin_ / "uv").chmod(0o755)
    env = dict(os.environ, PATH=f"{bin_}:{os.environ['PATH']}", KRON_GPU_PARTITION="gpus", KRON_ACCOUNT="acct",
               KRON_FM_DATA=str(tmp_path / "data"), LOG_DIR=str(tmp_path / "logs"), SBATCH_LOG=str(tmp_path / "sb"),
               COMMENT="seat=x; what=test", NODE="n1", KRON_NO_ENV_LOCAL="1", PYTHONPATH=str(repo / "src"),
               HOME=str(tmp_path))      # env.sh puts ~/.local/bin (a real uv) first
    res = subprocess.run(["bash", str(repo / "slurm" / "genome.sh"), "G", str(tmp_path / "runs" / "G"), "--cards", "2",
                          "--edits", "--", "--stop-step", "10"], env=env, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    flags = (tmp_path / "sb").read_text().split("\n")
    assert "--partition=gpus" in flags and "--account=acct" in flags and "--nodelist=n1" in flags
    assert "--comment=seat=x; what=test" in flags
    export = next(f for f in flags if f.startswith("--export="))
    assert "KRON_NCARDS=2" in export and "KRON_ARM=G" in export
    args_file = export.split("KRON_ARGS_FILE=")[1].split(",")[0]
    lines = Path(args_file).read_text().split("\n")
    assert shlex.split(lines[0]) == recipes.recipe("G", cards=2, edits=True, root=tmp_path / "data")
    assert shlex.split(lines[1]) == ["--stop-step", "10"]
