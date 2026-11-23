"""GTEx v8 fine-mapped eQTLs on hg38 for the variant-effect benchmark, from the eQTL Catalogue's SuSiE fine-mapping
of GTEx (release 7, study QTS000015, 49 tissues, gene expression), as Enformer and Borzoi score them: causal
variants (PIP >= 0.9) against matched non-causal ones (PIP < 0.01 in every tissue).

    uv run python scripts/kron_dna/prep_eqtl_finemap.py tissue METADATA.tsv WORK [--part K --parts N]
    uv run python scripts/kron_dna/prep_eqtl_finemap.py assemble METADATA.tsv WORK OUT.parquet --gtf GTF --genome hg38

``tissue``: for every Nth GTEx ``ge`` dataset of the catalogue's ``dataset_metadata_r7.tsv``, downloads its
``lbf_variable.txt.gz`` (log Bayes factors of each SuSiE effect for every tested variant of every gene), computes each
variant's PIP per gene, ``1 - prod_l (1 - alpha_l)`` with ``alpha_l`` the softmax of effect l's log Bayes factors over
the gene's variants (effects whose factors are all zero are unused and skipped), writes ``WORK/pip/<dataset>.parquet``
(gene, variant, pip; float32) and deletes the download. A finished tissue is skipped on a rerun.

``assemble``: a positive is a (variant, gene) pair with PIP >= ``--pos`` in some tissue (the tissues listed); a
negative is a variant tested for the same gene in the positive's first tissue that never reaches ``--neg`` for any
gene in any tissue. Each positive gets ``--per`` negatives, those whose distance to the gene's TSS (GENCODE gene
start, strand-aware) is closest to the positive's, each negative used once. SNVs only, chr1-22 and X; the
reference allele is checked against the prepared genome. ``held_out`` marks hg38 chr8/chr9. ``pair`` ties each
negative to its positive.
"""
from __future__ import annotations

import gzip
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import polars as pl
import typer

from kron_core.paths import interim_dir
from kron_dna import genome as G

app = typer.Typer(add_completion=False)
FTP = "https://ftp.ebi.ac.uk/pub/databases/spot/eQTL/susie"
HELD = ["chr8", "chr9"]


def datasets(meta: Path) -> pl.DataFrame:
    m = pl.read_csv(meta, separator="\t", infer_schema_length=0)
    return m.filter((pl.col("study_label") == "GTEx") & (pl.col("quant_method") == "ge")).sort("dataset_id")


def pips(lbf: pl.DataFrame) -> pl.DataFrame:
    """gene, variant, pip from a catalogue lbf_variable table (one gene's variants share a molecular_trait_id)."""
    cols = [c for c in lbf.columns if c.startswith("lbf_variable")]
    used = lbf.group_by("molecular_trait_id").agg([(pl.col(c) != 0).any().alias(c) for c in cols])
    out = lbf.select("molecular_trait_id", "variant")
    log_keep = pl.lit(0.0)
    for c in cols:
        a = (pl.col(c) - pl.col(c).max().over("molecular_trait_id")).exp()
        alpha = a / a.sum().over("molecular_trait_id")
        lbf = lbf.with_columns(**{f"a_{c}": alpha})
    lbf = lbf.join(used.rename({c: f"u_{c}" for c in cols}), on="molecular_trait_id")
    for c in cols:
        log_keep = log_keep + pl.when(pl.col(f"u_{c}")).then((1 - pl.col(f"a_{c}")).clip(1e-300, 1).log()).otherwise(0.0)
    return lbf.select(gene="molecular_trait_id", variant="variant",
                      pip=(1 - log_keep.exp()).cast(pl.Float32))


@app.command()
def tissue(meta: Path, work: Path, part: int = 0, parts: int = 1) -> None:
    ds = datasets(meta)
    (work / "pip").mkdir(parents=True, exist_ok=True)
    for k, r in enumerate(ds.iter_rows(named=True)):
        if k % parts != part:
            continue
        q = r["dataset_id"]
        dest = work / "pip" / f"{q}.parquet"
        if dest.exists():
            continue
        t0 = time.time()
        tmp = work / "tmp" / f"{q}.lbf.txt.gz"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        url = f"{FTP}/{r['study_id']}/{q}/{q}.lbf_variable.txt.gz"
        subprocess.run(["curl", "-sSL", "--fail", "--retry", "5", "--retry-all-errors", "-C", "-", "-o", str(tmp), url],
                       check=True)
        lbf = pl.read_csv(tmp, separator="\t", schema_overrides={"chromosome": pl.String, "position": pl.Int64,
                          **{f"lbf_variable{i}": pl.Float64 for i in range(1, 11)}})
        p = pips(lbf.drop("region", "chromosome", "position"))
        p.write_parquet(str(dest) + ".part")
        os.replace(str(dest) + ".part", dest)
        tmp.unlink()
        typer.echo(f"{q} {r['sample_group']}: {p.height:,} tests, {int((p['pip'] >= 0.9).sum()):,} with PIP >= 0.9, "
                   f"{time.time() - t0:.0f} s")


def tss_of(gtf: Path) -> pl.DataFrame:
    rows = []
    with gzip.open(gtf, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split("\t", 9)
            if p[2] != "gene":
                continue
            gid = p[8].split('gene_id "', 1)[1].split('"', 1)[0].split(".")[0]
            name = p[8].split('gene_name "', 1)[1].split('"', 1)[0] if 'gene_name "' in p[8] else ""
            rows.append((gid, name, p[0], int(p[3]) - 1 if p[6] == "+" else int(p[4]) - 1, p[6]))
    return pl.DataFrame(rows, schema=["gene", "gene_name", "chrom_tss", "tss", "strand"], orient="row").unique("gene")


@app.command()
def assemble(meta: Path, work: Path, out: Path, gtf: Path = typer.Option(...), genome: str = "hg38",
             pos: float = 0.9, neg: float = 0.01, per: int = 1) -> None:
    ds = datasets(meta)
    names = dict(zip(ds["dataset_id"], ds["sample_group"]))
    missing = [q for q in names if not (work / "pip" / f"{q}.parquet").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(names)} tissues not done: {missing[:5]}")
    scan = lambda q: pl.scan_parquet(work / "pip" / f"{q}.parquet")                       # noqa: E731
    # pass 1: the positives, and every variant that reaches --neg for any gene in any tissue (never a negative)
    P = pl.concat([scan(q).filter(pl.col("pip") >= pos).with_columns(dataset=pl.lit(q)) for q in names]).collect()
    B = pl.concat([scan(q).filter(pl.col("pip") >= neg).select("variant") for q in names]).unique().collect(
        engine="streaming")
    P = P.group_by("gene", "variant").agg(max_pip=pl.col("pip").max(), datasets=pl.col("dataset").sort())
    P = P.with_columns(dataset=pl.col("datasets").list.first(),
                       tissues=pl.col("datasets").list.eval(pl.element().replace_strict(names)))
    keep = [f"chr{c}" for c in [*map(str, range(1, 23)), "X"]]

    def parsed(df: pl.DataFrame) -> pl.DataFrame:
        sp = pl.col("variant").str.split("_")
        return df.with_columns(chrom=sp.list.get(0), position=sp.list.get(1).cast(pl.Int64) - 1, ref=sp.list.get(2),
                               alt=sp.list.get(3)).filter(pl.col("chrom").is_in(keep) & (pl.col("ref").str.len_chars() == 1)
                                                          & (pl.col("alt").str.len_chars() == 1))
    tss = tss_of(gtf)
    sign = pl.when(pl.col("strand") == "+").then(1).otherwise(-1)
    P = parsed(P).join(tss, on="gene", how="inner").filter(pl.col("chrom") == pl.col("chrom_tss"))
    P = P.with_columns(distance=(pl.col("position") - pl.col("tss")) * sign).sort("gene", "variant")
    P = P.with_row_index("pair")
    # pass 2, per tissue: negatives for the positives first found there, tested for the same gene in that tissue
    negs = []
    for q, Pq in P.group_by("dataset"):
        genes = Pq["gene"].unique()
        C = parsed(scan(q[0]).filter(pl.col("gene").is_in(genes.implode()) & (pl.col("pip") < neg)).collect())
        C = C.join(B, on="variant", how="anti").join(tss, on="gene").with_columns(
            distance=(pl.col("position") - pl.col("tss")) * sign).with_columns(ad=pl.col("distance").abs())
        cand = {g[0]: d.sort("ad") for g, d in C.group_by("gene")}
        for r in Pq.sort("pair").iter_rows(named=True):
            d = cand.get(r["gene"])
            if d is None or not d.height:
                continue
            ad = d["ad"].to_numpy()
            used = np.zeros(len(ad), bool) if "used" not in d.columns else d["used"].to_numpy()
            order = np.argsort(np.abs(ad - abs(r["distance"])), kind="stable")
            pick = [i for i in order if not used[i]][:per]
            used[pick] = True
            cand[r["gene"]] = d.with_columns(used=pl.Series(used))
            for i in pick:
                negs.append(dict(d.row(int(i), named=True), pair=r["pair"], max_pip=None, tissues=[names[q[0]]]))
    cols = ["pair", "gene", "gene_name", "variant", "chrom", "position", "ref", "alt", "tss", "strand", "distance",
            "max_pip", "tissues"]
    N = pl.DataFrame(negs, infer_schema_length=None).select([c for c in cols if c != "max_pip"]).with_columns(
        max_pip=pl.lit(None, pl.Float32), label=pl.lit(0)) if negs else None
    t = pl.concat([P.select(cols).with_columns(pl.col("max_pip").cast(pl.Float32), label=pl.lit(1)),
                   *([N.select([*cols, "label"])] if N is not None else [])]).sort("pair", "label", descending=[False, True])
    src = interim_dir("genome", genome)
    idx = pl.read_parquet(src / "contigs.parquet")
    bases = np.memmap(src / "bases.u8", dtype=np.uint8, mode="r")
    code = {c: i for i, c in enumerate("ACGT")}
    ok = np.zeros(t.height, bool)
    for c, d in t.with_row_index("i").group_by("chrom"):
        ci = idx.filter(pl.col("chrom") == c[0]).sort("start")
        starts, ends, offs = ci["start"].to_numpy(), ci["end"].to_numpy(), ci["offset"].to_numpy()
        pos_ = d["position"].to_numpy()
        k = np.searchsorted(starts, pos_, side="right") - 1
        inside = (k >= 0) & (pos_ < ends[np.clip(k, 0, None)])
        got = np.full(len(pos_), 255)
        got[inside] = bases[offs[k[inside]] + pos_[inside] - starts[k[inside]]]
        ok[d["i"].to_numpy()] = got == np.array([code.get(x, -1) for x in d["ref"].to_list()])
    t = t.with_columns(ref_ok=pl.Series(ok), held_out=pl.col("chrom").is_in(HELD))
    out.parent.mkdir(parents=True, exist_ok=True)
    t.write_parquet(out)
    pos_n = int((t["label"] == 1).sum())
    summ = dict(positives=pos_n, negatives=int((t["label"] == 0).sum()),
                positives_without_negative=pos_n - int(t.filter(pl.col("label") == 0)["pair"].n_unique()),
                ref_mismatch=int((~t["ref_ok"]).sum()),
                held_out_positives=int(t.filter((pl.col("label") == 1) & pl.col("held_out")).height),
                genes=int(t.filter(pl.col("label") == 1)["gene"].n_unique()), pos=pos, neg=neg, per=per,
                tissues=len(names), excluded_variants_reaching_neg=B.height,
                release="eQTL Catalogue r7, QTS000015 (GTEx v8), SuSiE")
    out.with_suffix(".json").write_text(json.dumps(summ, indent=1))
    typer.echo(json.dumps(summ))


if __name__ == "__main__":
    app()
