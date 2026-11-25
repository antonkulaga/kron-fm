"""The training recipes of the KronGenome arms as code: ``genome-train`` arguments for each arm, with every path
under the data root (``$KRON_FM_DATA``) and nothing that names a cluster. ``slurm/launch_genome.sh`` turns a recipe
into a job with the cluster settings of ``slurm/env.local.sh``; the run's own record keeps the exact options.

    python -m kron_dna.recipes F --genomes all --jepa step2 --edits --pairs --cards 2      # prints the arguments

Arms (as trained in November 2026):

- ``A``  : v2, 8 layers, no JEPA; 8 kb then 512 kb windows over 400k steps (A1m/A3 continued it at 1 Mb).
- ``A1m``: A continued at 1 Mb, a fresh warmup and cosine to 60k steps at 1e-4.
- ``A3`` : A1m's recipe with the great apes and a new cosine at 2e-4 over 100k steps (``--genomes`` adds the rest).
- ``D``  : v2, 8 layers, block JEPA; 8 kb then 512 kb windows (``--curriculum`` to change), the human reference set.
- ``F``  : v2, 10 layers, block JEPA, 1 Mb windows from step 0 over 380k steps, a 64 kb mito slice each step.
- ``G``  : F's recipe from scratch with narrow single-base layers (``--stem``) and a single-base KronLSTM track
  (``--local-track``): the memory-lean arm of 24 Nov.

Options common to all: ``--genomes`` (``reference``: CHM13, naked mole-rat, mito and the NTv3 species; ``all``: plus
mouse with its tracks, HPRC, the great apes, the vertebrates with gene labels and ortholog splits, the honeybee as
sequence), ``--heads`` (``eight``: the eight further heads; ``two``: the first two), ``--jepa`` (``none``, ``v1``:
block JEPA as D/F trained it; ``step2``: the context predictor, centred loss, decoding through it, heads inside
blocks), ``--edits``/``--pairs`` (step 2b/2c), ``--cards`` (split each window over that many cards).
"""
from __future__ import annotations

import shlex
from pathlib import Path

import typer

from kron_core.paths import data_root

NTV3_EXTRA = ("ntv3_cattle", "ntv3_rat", "ntv3_maize", "ntv3_tomato", "ntv3_rice", "ntv3_arabidopsis", "ntv3_yeast",
              "ntv3_fission_yeast", "ntv3_malaria_parasite", "ntv3_soil_amoeba")
APES = ("chimp", "bonobo", "gorilla", "sorangutan", "borangutan")
VERTEBRATES = ("tgut_btaegut7", "apla_pekin_t2t", "cmyd_rchemyd1", "amis_rallmis1", "acar_ranocar3", "drer_grcz13ab",
               "olat_hdrr31", "xtro_xtro10")
MOUSE_HEADS = {"dnase196b": "dnase-v1-bin128", "rna462b": "rna-v1-bin128", "chip492b": "histone-core-v1-bin128",
               "chiphr": "histone-rest-v1-bin128", "chiptf": "tf-v1-bin128", "splice": "splice-sites-v1",
               "genes": "gene-structure-v1"}
CF_TABLES = ("asb", "eqtl", "crispri", "hsub2")


def _paths(root: Path) -> dict[str, Path]:
    return dict(genome=root / "interim" / "genome", hg38=root / "interim" / "tracks" / "hg38",
                mm10=root / "interim" / "tracks" / "mm10", tracks=root / "interim" / "tracks",
                pairs=root / "interim" / "train" / "cf-pairs-v1")


def base_args(stem: int = 256) -> list[str]:
    return ["--name", "hg38", "--val-chroms", "chr8,chr9", "--arch", "v2", "--eval-window", "1048576",
            "--warmup", "2000", "--bert-mask", "--rc", "--stem-channels", str(stem), "--n-contexts", "256",
            "--n-in-extra", "8", "--epigenome-weight", "1.0", "--lean-convs"]


def ntv3_args(p: dict) -> list[str]:
    return ["--ntv3-dir", str(p["genome"] / "hg38" / "ntv3"), "--ntv3-assays", "all", "--ntv3-target", "alphagenome",
            "--ntv3-loss", "poisson", "--ntv3-center", "0.375", "--ntv3-extra", ",".join(NTV3_EXTRA),
            "--ntv3-extra-share", "0.2", "--ntv3-weight", "0.5"]


def reference_genome_args() -> list[str]:
    return ["--mito", "mito", "--mito-share", "0.05", "--genome", "chm13v2:0.05:0:chr8+chr9",
            "--genome", "nmr_t2t_hap1:0.10:64:chr8+chr9", "--soft-base-rate", "0.01", "--repeat-channel",
            "--repeat-channel-drop", "0.5"]


def ape_args(share: float) -> list[str]:
    out = []
    for k, a in enumerate(APES):
        out += ["--genome", f"ape_{a}_t2t:{share:g}:{65 + k}"]
    return out


def all_genome_args(p: dict) -> list[str]:
    """Mouse (its tracks and split), HPRC, the vertebrates (gene labels, ortholog splits), the honeybee (sequence)."""
    out = ["--genome", "mm10:0.10:70:chr8+chr9"]
    for h, store in MOUSE_HEADS.items():
        out += ["--xhead-species", f"{h}=mm10:{p['mm10'] / store}"]
    out += ["--xhead-genome-split", f"mm10={p['mm10'] / 'split-v2' / 'split.u8'}", "--genome", "hprc_r2:0.03:0"]
    for k, g in enumerate(VERTEBRATES):
        out += ["--genome", f"{g}:0.005:{71 + k}",
                "--xhead-species", f"genes={g}:{p['tracks'] / g / 'gene-structure-refseq'}",
                "--xhead-genome-split", f"{g}={p['tracks'] / g / 'split-orthologs-v1' / 'split.u8'}"]
    return out + ["--genome", "amel_hav31:0.005:79"]


def head_args(p: dict, lr: str, heads: str, xonly: int) -> list[str]:
    out = ["--xhead", f"dnase:{p['hg38'] / 'dnase-pilot-v1'}", "--xhead", f"splice:{p['hg38'] / 'splice-sites-v1'}",
           "--xhead-hidden", "128", "--xhead-weight", "0.5", "--xhead-lr", lr, "--xhead-warmup", "200",
           "--xhead-only", str(xonly), "--inherit-lr", lr, "--inherit-warmup", "500"]
    if heads == "eight":
        out += ["--xhead-kind", "kron", "--xhead-layers", "1",
                "--xhead", f"dnase196b:{p['hg38'] / 'dnase-biosamples-v1-bin128'}",
                "--xhead", f"rna462b:{p['hg38'] / 'rnaseq-stranded-v1-bin128'}",
                "--xhead", f"chip492b:{p['hg38'] / 'chip-histone-core-v1-bin128'}",
                "--xhead", f"genes:{p['hg38'] / 'gene-structure-v1'}",
                "--xhead", f"chiptf:{p['hg38'] / 'chip-tf-v1-bin128'}",
                "--xhead", f"chiphr:{p['hg38'] / 'chip-histone-rest-v1-bin128'}",
                "--xhead-recompute", "--xhead-cycle", "dnase,splice,genes;dnase196b,rna462b;chip492b,chiphr;chiptf"]
    elif heads != "two":
        raise ValueError(f"--heads must be eight or two, got {heads!r}")
    return out


def jepa_args(kind: str, freeze_until: int = 0) -> list[str]:
    if kind == "none":
        return []
    v1 = ["--jepa-weight", "1.0", "--jepa-ema", "0.0", "--jepa-reg", "sigreg", "--jepa-block-share", "0.2",
          "--jepa-log-every", "100", "--jepa-guard", "on", "--jepa-guard-from", "train"]
    if kind == "v1":
        return v1
    if kind != "step2":
        raise ValueError(f"--jepa must be none, v1 or step2, got {kind!r}")
    out = v1 + ["--jepa-block-heads", "include", "--jepa-pred", "context", "--jepa-pred-layers", "2",
                "--jepa-loss", "zcos-mse", "--jepa-decode-pred"]
    return out + (["--freeze-trunk-until", str(freeze_until)] if freeze_until else [])


ARMS = ("A", "A1m", "A3", "D", "F", "G")


def recipe(arm: str, *, genomes: str = "reference", heads: str = "eight", jepa: str | None = None,
           edits: bool = False, pairs: bool = False, cards: int = 1, stem: int | None = None,
           local_track: str = "", curriculum: str = "", xonly: int = 1000, freeze_until: int = 0,
           root: Path | None = None) -> list[str]:
    """``genome-train`` arguments (without the run directory) for ``arm``."""
    p = _paths(root or data_root())
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; one of {', '.join(ARMS)}")
    if genomes not in ("reference", "all"):
        raise ValueError(f"--genomes must be reference or all, got {genomes!r}")
    jepa = jepa or {"A": "none", "A1m": "none", "A3": "none", "D": "v1", "F": "v1", "G": "step2"}[arm]
    if (edits or pairs) and jepa != "step2":
        raise ValueError("--edits and --pairs belong to --jepa step2")
    lr = {"A1m": "1e-4", "A3": "2e-4"}.get(arm, "4e-4")
    stem = stem or (64 if arm == "G" else 256)
    out = base_args(stem) + ["--n-layers", "10" if arm in ("F", "G") else "8", "--lr", lr, "--epigenome-focus", "0.05"]
    out += ntv3_args(p) + reference_genome_args() + jepa_args(jepa, freeze_until)
    if arm == "A":
        out += ["--eval-every", "2000", "--curriculum", curriculum or "8192:134000,524288:266000",
                "--tokens-per-step", "524288"]
    elif arm == "A1m":
        out += ["--eval-every", "2000", "--curriculum", curriculum or "1048576:60000", "--tokens-per-step", "1048576"]
    elif arm == "A3":
        out += ape_args(0.02) + ["--eval-every", "2000", "--curriculum", curriculum or "1048576:100000",
                                 "--tokens-per-step", "1048576"]
    elif arm == "D":
        out += ["--eval-every", "1000", "--curriculum", curriculum or "8192:46000,524288:354000",
                "--tokens-per-step", "1048576"]
    else:                                            # F and G: 1 Mb from step 0, the mito slice, 380k steps
        out += ["--eval-every", "1000", "--mito-slice", "65536", "--curriculum", curriculum or "1048576:380000",
                "--steps", "380000", "--tokens-per-step", "1048576"]
    out += ["--grad-accum", "1", "--checkpoint-core", "--short-sources", "pack"]
    if genomes == "all":
        out += all_genome_args(p) + ([] if arm == "A3" else ape_args(0.01))      # this order, as F was trained
    out += head_args(p, lr, heads, xonly)
    if arm == "G" or local_track:
        out += ["--local-track", local_track or "64,2,8192"]
    if edits:
        out += ["--jepa-edits", "1.0"]
    if pairs:
        for t in CF_TABLES:
            out += ["--cf-pairs", str(p["pairs"] / f"{t}.parquet")]
        out += ["--cf-weight", "0.5", "--cf-batch", "1"]
    if cards > 1:
        out += ["--split-cards", str(cards)]
    return out


app = typer.Typer(add_completion=False)


@app.command()
def main(arm: str = typer.Argument(..., help=f"one of {', '.join(ARMS)}"), genomes: str = "reference",
         heads: str = "eight", jepa: str = typer.Option("", help="none | v1 | step2 (default: the arm's own)"),
         edits: bool = False, pairs: bool = False, cards: int = 1, stem: int = 0, local_track: str = "",
         curriculum: str = "", xonly: int = 1000, freeze_until: int = 0) -> None:
    """Print the arm's genome-train arguments, shell-quoted on one line."""
    try:
        args = recipe(arm, genomes=genomes, heads=heads, jepa=jepa or None, edits=edits, pairs=pairs, cards=cards,
                      stem=stem or None, local_track=local_track, curriculum=curriculum, xonly=xonly,
                      freeze_until=freeze_until)
    except ValueError as err:
        raise typer.BadParameter(str(err)) from None
    typer.echo(shlex.join(args))


if __name__ == "__main__":
    app()
