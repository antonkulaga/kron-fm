"""One entry point for every data-preparation step (B23): a subcommand per adapter, each running the existing
script unchanged on the shared ``kron_dna.prep`` base.

    uv run python scripts/kron_dna/prep.py genome species FASTA --name dmel_r6 --keep 2L+2R+...
    uv run python scripts/kron_dna/prep.py genome pangenome part|assemble INDEX WORK ...
    uv run python scripts/kron_dna/prep.py encode select OUT.tsv --assembly mm10
    uv run python scripts/kron_dna/prep.py encode fetch SELECT.tsv OUT_DIR
    uv run --with pyBigWig python scripts/kron_dna/prep.py tracks binned SELECT.tsv BIGWIGS OUT [...]
    uv run --with pyBigWig python scripts/kron_dna/prep.py tracks perbase SELECT.tsv BIGWIGS OUT [...]
    uv run python scripts/kron_dna/prep.py tracks bin STORE OUT [--bin 128]        (a per-base store to cells)
    uv run python scripts/kron_dna/prep.py labels splice OUT --name hg38 --gtf GTF
    uv run python scripts/kron_dna/prep.py labels genes OUT --name hg38 --gtf GTF [--basic-only] [--source S]
    uv run python scripts/kron_dna/prep.py split syntenic CHAIN OUT --name mm10
    uv run python scripts/kron_dna/prep.py split orthologs GTF OUT --name tgut_btaegut7 --gene-info INFO --orthologs ORTH
    uv run python scripts/kron_dna/prep.py eval crispri|asb|mpra-satmut|lentimpra|mfass|vexseq|hsub-mpra|exp-starr OUT.parquet INPUTS...   (benchmark tables)
    uv run python scripts/kron_dna/prep.py check histone|tf|accessibility|rna STORE GENOME [--gtf GTF]

Every subcommand takes exactly the arguments of the script it names (``--help`` after it shows them).
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUTES = {
    ("genome", "species"): ("prep_species_genome.py", None),
    ("genome", "pangenome"): ("prep_pangenome.py", None),
    ("encode", "select"): ("encode_select.py", None),
    ("encode", "fetch"): ("encode_fetch.py", None),
    ("tracks", "binned"): ("prep_bigwig_binned.py", None),
    ("tracks", "perbase"): ("prep_bigwig_tracks.py", None),
    ("tracks", "bin"): ("bin_track_store.py", None),
    ("labels", "splice"): ("prep_splice_labels.py", None),
    ("labels", "genes"): ("prep_gene_structure_labels.py", None),
    ("split", "syntenic"): ("prep_syntenic_split.py", None),
    ("split", "orthologs"): ("prep_ortholog_split.py", None),
    ("eval", "crispri"): ("prep_crispri_pairs.py", None),
    ("eval", "asb"): ("prep_asb.py", None),
    ("eval", "mpra-satmut"): ("prep_mpra_satmut.py", None),
    ("eval", "lentimpra"): ("prep_lentimpra.py", None),
    ("eval", "mfass"): ("prep_mfass.py", None),
    ("eval", "vexseq"): ("prep_vexseq.py", None),        # uv run --with pyliftover
    ("eval", "hsub-mpra"): ("prep_hsub_mpra.py", None),  # uv run --with pyliftover
    ("eval", "exp-starr"): ("prep_exp_starr.py", None),  # uv run --with pyliftover
    ("check", "histone"): ("check_chip_store.py", "keep"),
    ("check", "tf"): ("check_chip_store.py", "keep"),
    ("check", "accessibility"): ("check_chip_store.py", "keep"),
    ("check", "rna"): ("check_chip_store.py", "keep"),
}


def main(argv: list[str]) -> int:
    if len(argv) < 2 or (argv[0], argv[1]) not in ROUTES:
        print(__doc__)
        print("steps: " + ", ".join(" ".join(k) for k in ROUTES))
        return 2
    script, sub = ROUTES[(argv[0], argv[1])]
    # a script with subcommands of its own (check, pangenome) keeps the second word; the others take the rest
    sys.argv = [str(HERE / script)] + (argv[1:] if sub == "keep" else argv[2:])
    sys.path.insert(0, str(HERE))
    runpy.run_path(str(HERE / script), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
