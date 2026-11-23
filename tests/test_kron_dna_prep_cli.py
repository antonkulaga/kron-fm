"""scripts/kron_dna/prep.py routes each step to its script with the script's own arguments."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PREP = Path(__file__).resolve().parents[1] / "scripts" / "kron_dna" / "prep.py"


def run(*args):
    return subprocess.run([sys.executable, str(PREP), *args], capture_output=True, text=True)


def test_every_step_reaches_its_script():
    for args, script in ((("labels", "genes", "--help"), "prep_gene_structure_labels.py"),
                         (("check", "tf", "--help"), "check_chip_store.py tf"),
                         (("genome", "pangenome", "assemble", "--help"), "prep_pangenome.py assemble"),
                         (("split", "syntenic", "--help"), "prep_syntenic_split.py")):
        r = run(*args)
        assert r.returncode == 0 and script in r.stdout, (args, r.stdout[-400:], r.stderr[-400:])
    r = run("nope")
    assert r.returncode == 2 and "labels genes" in r.stdout
