"""hic_select.py: untreated experiments only, one map per biosample."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "hic_select", Path(__file__).parents[1] / "scripts" / "kron_dna" / "hic_select.py")
hs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hs)


def exp(acc: str, name: str, **biosample) -> dict:
    return {"@id": f"/experiments/{acc}/", "accession": acc,
            "biosample_ontology": {"term_name": name, "classification": "cell line"},
            "replicates": [{"library": {"biosample": biosample}}]}


def hic(acc: str, e: str, size: int, reps: list[int]) -> dict:
    return {"accession": acc, "dataset": f"/experiments/{e}/", "file_size": size, "biological_replicates": reps,
            "href": f"/files/{acc}/@@download/{acc}.hic"}


def test_choose() -> None:
    exps = [exp("E1", "K562"), exp("E2", "K562"), exp("E3", "K562", treatments=[{"treatment_term_name": "x"}]),
            exp("E4", "HCT116", genetic_modifications=["/gm/1/"]), exp("E5", "HepG2")]
    files = [hic("F1", "E1", 10, [1]), hic("F2", "E1", 30, [1, 2]), hic("F3", "E1", 50, [1]),   # pooled wins in E1
             hic("F4", "E2", 20, [1]), hic("F5", "E3", 900, [1, 2]), hic("F6", "E4", 900, [1]), hic("F7", "E5", 5, [1])]
    rows = hs.choose(exps, files)
    assert [(r["biosample"], r["file"], r["experiment"]) for r in rows] == [("HepG2", "F7", "E5"), ("K562", "F2", "E1")]
    assert rows[1]["url"].endswith("/files/F2/@@download/F2.hic") and rows[1]["bio_reps"] == "1,2"
    assert [r["i"] for r in rows] == [0, 1]
