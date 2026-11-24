"""Measured counterfactual pairs for training (the JEPA counterfactual arm; Anton, 24 Nov: pairs may train): one
table per source, from the benchmark tables under ``interim/eval`` and kept apart from what the benchmarks quote
by chromosome.

    uv run [--with pyliftover] python scripts/kron_dna/prep_cf_pairs.py OUT_DIR [--eval-dir interim/eval]
        [--credible DIR] [--mpra2 GSE140983_MPRAcounts.tsv.gz --mpra1 GSE140983_COUNTS_ANNOTATED.txt.gz --chain CHAIN]

The split (``split`` column; agreed with the trainer, 24 Nov): hg38 chr8/chr9, the models' held-out chromosomes,
are **not written**; chr3, chr10 and chr16 are ``pair_test`` (their sequence is trained on, no pair from them may
be); every other chromosome is ``train``. A distance rule was not possible: 96 % of the genome lies within 1 Mb
of a row of some scored set. The benchmarks of an arm trained on these pairs are quoted on chr8/chr9 and on the
``pair_test`` chromosomes only.

Columns of every table: ``chrom``, ``pos`` (0-based), ``ref``, ``alt``, ``read_pos`` (where the effect is read),
``head``, ``channel_ids`` (the head's channels the measurement maps to; empty = the mean over the head, for
``rna462b`` over the channels of ``strand``), ``effect`` (alt minus ref, see ``effect_kind``), ``effect_kind``,
``weight`` (0..1), ``source``, ``split``; then the source's own columns.

- ``asb.parquet`` (ADASTRA allele-specific binding, ``adastra-asb-v1``): one row per (SNV, TF, cell type) whose
  factor is in the chiptf store; ``effect`` is the log2 allelic effect size (``log2``), read at the variant;
  ``weight`` = min(1, -log10(FDR) / 5), halved when only other cell types of the factor are channels (``match``).
- ``eqtl.parquet`` (GTEx fine-mapped eQTLs, the positives of ``gtex-finemap-v1``): read at the gene's TSS on the
  RNA head; the scored table has no direction, so ``effect`` is the sign (``sign``) of the mean z-score of the
  variant for the gene over the GTEx tissues' credible sets (eQTL Catalogue ``credible_sets.tsv.gz`` in
  ``--credible``; beta is the alternative allele's), ``beta`` and ``z`` their means, ``sign_share`` the share of
  tissues with that sign; ``weight`` = max PIP x (2 x sign_share - 1). Pairs without a credible-set row are left out.
- ``crispri.parquet`` (CRISPRi enhancer-gene pairs, ``crispri-eg-v1``, its valid rows): not a substitution: the
  element ``del_start``..``del_end`` is silenced (``ref`` = ``alt`` = "", ``pos`` = ``del_start``), the effect read
  at the gene's TSS; ``effect`` is the fractional change of the gene's expression (``fraction``), ``regulated``
  the paper's call; ``weight`` 1 for regulated pairs, 0.25 for the others (a measured absence of effect);
  ``channel_ids`` the RNA head's channels of the row's cell type, where the store has it.
- ``hsub2.parquet`` (the second MPRA of Uebbing et al. 2021, human neural stem cells; not a scored set, but its
  fragments are those of ``hsub-mpra-v1``): each fragment's substitutions in combinations; one row per allele
  with >= 10 barcodes against the fragment's all-human allele, ``ref`` the human span and ``alt`` the chosen
  substitutions as chimp bases, read at the fragment's middle (``half`` its half-length) on the DNase head;
  ``effect`` the log2 difference of activity (``log2``), ``se``/``t``/``p_value`` Welch's test between the
  barcodes, ``weight`` = 1 / (1 + (se / 0.1)^2). The allele string is taken in the order the first MPRA's file
  lists the substitutions; the files carry no sequences to check that, so a partial allele of a fragment with
  several substitutions has ``order_assumed`` true (single-substitution fragments and all-chimp alleles do not
  depend on it).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import polars as pl
import typer

from kron_core.paths import interim_dir

HELD = ["chr8", "chr9"]
PAIR_TEST = ["chr3", "chr10", "chr16"]
COLS = ["chrom", "pos", "ref", "alt", "read_pos", "head", "channel_ids", "effect", "effect_kind", "weight", "source", "split"]


def split_of(t: pl.DataFrame) -> pl.DataFrame:
    """Drops the held-out chromosomes and marks the rest ``train`` or ``pair_test``."""
    return t.filter(~pl.col("chrom").is_in(HELD)).with_columns(
        split=pl.when(pl.col("chrom").is_in(PAIR_TEST)).then(pl.lit("pair_test")).otherwise(pl.lit("train")))


def check(t: pl.DataFrame) -> None:
    """The exclusion every table must pass (also the test's): no held-out chromosome, splits by chromosome."""
    assert t.filter(pl.col("chrom").is_in(HELD)).height == 0, "a row on a held-out chromosome"
    assert t.filter((pl.col("split") == "train") & pl.col("chrom").is_in(PAIR_TEST)).height == 0, "a train row on a pair-test chromosome"
    assert t.filter((pl.col("split") == "pair_test") & ~pl.col("chrom").is_in(PAIR_TEST)).height == 0
    assert set(t["split"].unique()) <= {"train", "pair_test"} and t.select(COLS).null_count().sum_horizontal()[0] == 0


def asb(table: Path) -> pl.DataFrame:
    t = pl.read_parquet(table).filter(pl.col("match").is_in(["tf_cell", "tf"]) & pl.col("ref_ok") & pl.col("effect").is_finite())
    fdr = pl.min_horizontal("fdr_ref", "fdr_alt").clip(1e-300, 1.0)
    t = t.with_columns(read_pos=pl.col("pos"), head=pl.lit("chiptf"), channel_ids=pl.col("channels"), effect_kind=pl.lit("log2"),
                       weight=(-fdr.log10() / 5).clip(0.0, 1.0) * pl.when(pl.col("match") == "tf_cell").then(1.0).otherwise(0.5),
                       source=pl.lit("adastra-asb-v1"))
    return split_of(t).select(*COLS, "tf", "cell", "match", "rsid", "motif_fc")


def eqtl(table: Path, credible: Path) -> pl.DataFrame:
    t = pl.read_parquet(table).filter((pl.col("label") == 1) & pl.col("ref_ok"))
    cs = pl.concat([pl.read_csv(f, separator="\t", columns=["gene_id", "variant", "beta", "z"],
                                schema_overrides={"beta": pl.Float64, "z": pl.Float64})
                    for f in sorted(credible.glob("*.credible_sets.tsv.gz"))])
    cs = cs.join(t.select(gene_id="gene", variant="variant").unique(), on=["gene_id", "variant"]).group_by("gene_id", "variant").agg(
        beta=pl.col("beta").mean(), z=pl.col("z").mean(), tissues_with_sign=pl.len(), pos_share=(pl.col("z") > 0).mean())
    t = t.join(cs, left_on=["gene", "variant"], right_on=["gene_id", "variant"]).filter(pl.col("z") != 0)
    share = pl.when(pl.col("z") > 0).then(pl.col("pos_share")).otherwise(1 - pl.col("pos_share"))
    t = t.with_columns(pos=pl.col("position"), read_pos=pl.col("tss"), head=pl.lit("rna462b"),
                       channel_ids=pl.lit([], dtype=pl.List(pl.String)), effect=pl.col("z").sign(), effect_kind=pl.lit("sign"),
                       sign_share=share, source=pl.lit("gtex-finemap-v1")).with_columns(
        weight=(pl.col("max_pip").cast(pl.Float64) * (2 * pl.col("sign_share") - 1)).clip(0.0, 1.0))
    return split_of(t).select(*COLS, "gene", "gene_name", "strand", "distance", "max_pip", "beta", "z", "sign_share", "tissues_with_sign")


def crispri(table: Path, rna_store: Path) -> pl.DataFrame:
    t = pl.read_parquet(table).filter(pl.col("valid") & pl.col("effect").is_finite() & (pl.col("chrom") == pl.col("chrom_tss")))
    meta = rna_store / "binned.json"
    tracks = json.loads(meta.read_text())["tracks"] if meta.exists() else []
    norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())
    by_cell: dict = {}
    for tr in tracks:
        by_cell.setdefault(norm(tr.get("biosample", "")), []).append(tr["id"])
    cells = {c: by_cell.get(norm(c), []) for c in t["cell_type"].unique().to_list()}
    t = t.with_columns(pos=pl.col("start"), ref=pl.lit(""), alt=pl.lit(""), read_pos=pl.col("tss"), head=pl.lit("rna462b"),
                       channel_ids=pl.col("cell_type").replace_strict(cells, return_dtype=pl.List(pl.String)),
                       effect_kind=pl.lit("fraction"), weight=pl.when(pl.col("regulated")).then(1.0).otherwise(0.25),
                       source=pl.lit("crispri-eg-v1"), del_start=pl.col("start"), del_end=pl.col("end"))
    return split_of(t).select(*COLS, "del_start", "del_end", "gene", "cell_type", "regulated", "significant", "distance",
                              bench_set="set")


def hsub2(counts2: Path, counts1: Path, chain: Path, name: str, min_plasmid: int = 20, min_tags: int = 10) -> pl.DataFrame:
    """The second MPRA of Uebbing et al. 2021 (``GSE140983_MPRAcounts.tsv.gz``): fragments of the first MPRA again,
    with their substitutions in combinations (an allele string, H or C per substitution). One row per allele
    against the fragment's all-human allele; the substitutions' places come from the first MPRA's file."""
    import numpy as np
    from pyliftover import LiftOver
    from scipy import stats
    frag = pl.read_csv(counts1, separator="\t", infer_schema_length=100000, null_values=["NA"],
                       columns=["Alignment", "Chr", "Start", "Stop", "Species", "eVar_BPChange", "eVar_Pos"]).filter(
        (pl.col("Species") == "hg19") & pl.col("eVar_Pos").is_not_null()).unique("Alignment").with_columns(
        Pos=pl.col("Alignment").str.replace("_hsSUB$", ""))
    t = pl.read_csv(counts2, separator="\t", infer_schema_length=100000, null_values=["NA"]).filter(pl.col("Type") == "hsSUB")
    t = t.with_columns(c=pl.col("cDNA1") + pl.col("cDNA2"), p=pl.col("pDNA1") + pl.col("pDNA2")).filter(pl.col("p") >= min_plasmid)
    scale = float(np.log2(t["c"].sum() / t["p"].sum()))
    a = t.with_columns(a=((pl.col("c") + 1) / (pl.col("p") + 1)).log(2) - scale).group_by("Alignment", "Pos", "Allele").agg(
        tags=pl.len(), act=pl.col("a").mean(), var=pl.col("a").var()).filter(pl.col("tags") >= min_tags)
    ref = a.filter(pl.col("Allele").str.contains("^H+$")).select("Pos", tags_h="tags", act_h="act", var_h="var")
    j = a.filter(pl.col("Allele").str.contains("C")).join(ref, on="Pos").join(frag.drop("Alignment", "Species"), on="Pos")
    eh, ec = pl.col("var_h") / pl.col("tags_h"), pl.col("var") / pl.col("tags")
    j = j.with_columns(effect=pl.col("act") - pl.col("act_h"), se=(eh + ec).sqrt(),
                       dof=(eh + ec) ** 2 / (eh ** 2 / (pl.col("tags_h") - 1) + ec ** 2 / (pl.col("tags") - 1)))
    j = j.with_columns(t=pl.col("effect") / pl.col("se")).filter(pl.col("t").is_finite())
    j = j.with_columns(p_value=pl.Series(2 * stats.t.sf(np.abs(j["t"].to_numpy()), j["dof"].to_numpy())))
    lift = LiftOver(str(chain))
    src = interim_dir("genome", name)
    bases = np.memmap(src / "bases.u8", dtype=np.uint8, mode="r")
    segs = pl.read_parquet(src / "contigs.parquet").to_dicts()

    def at(chrom: str, x: int) -> int | None:
        return next((s["offset"] + x - s["start"] for s in segs if s["chrom"] == chrom and s["start"] <= x < s["end"]), None)

    rows = []
    for r in j.to_dicts():
        offs = [int(x) - 1 for x in str(r["eVar_Pos"]).split(";")]                 # in the file's order: the allele string's
        subs = [x.split("->") for x in r["eVar_BPChange"].split(";")]              # chimp -> human
        if len(r["Allele"]) != len(offs) or len(set(offs)) < len(offs):
            continue
        x, y = lift.convert_coordinate(r["Chr"], r["Start"]), lift.convert_coordinate(r["Chr"], r["Stop"] - 1)
        if not x or not y or x[0][0] != y[0][0] or x[0][2] != "+" or y[0][1] - x[0][1] != r["Stop"] - 1 - r["Start"]:
            continue
        chrom, start = x[0][0], x[0][1]
        on = [(o, chimp, hum) for o, (chimp, hum), state in zip(offs, subs, r["Allele"]) if state == "C"]
        lo, hi = min(o for o, _, _ in on), max(o for o, _, _ in on)
        g = at(chrom, start + lo)
        if g is None or at(chrom, start + hi) != g + hi - lo:
            continue
        ref_s = "".join("ACGTN"[v] for v in bases[g:g + hi - lo + 1])
        alt_s = list(ref_s)
        for o, chimp, _ in on:
            alt_s[o - lo] = chimp
        if not all(ref_s[o - lo] == hum for o, _, hum in on):
            continue
        rows.append(dict(chrom=chrom, pos=start + lo, ref=ref_s, alt="".join(alt_s),
                         read_pos=start + (r["Stop"] - r["Start"]) // 2, half=(r["Stop"] - r["Start"] + 1) // 2,
                         effect=r["effect"], se=r["se"], t=r["t"], p_value=r["p_value"], tags=min(r["tags"], r["tags_h"]),
                         fragment=r["Pos"], allele=r["Allele"], n_subs=len(on), fragment_subs=len(offs),
                         order_assumed=0 < len(on) < len(offs), order_ascending=offs == sorted(offs)))
    v = pl.DataFrame(rows).with_columns(
        head=pl.lit("dnase196b"), channel_ids=pl.lit([], dtype=pl.List(pl.String)), effect_kind=pl.lit("log2"),
        weight=1 / (1 + (pl.col("se") / 0.1) ** 2), source=pl.lit("uebbing2021-mpra2"))
    return split_of(v).select(*COLS, "half", "se", "t", "p_value", "tags", "fragment", "allele", "n_subs", "fragment_subs",
                              "order_assumed", "order_ascending")


def main(out: Path, eval_dir: Path = None, credible: Path = None, name: str = "hg38", mpra2: Path = None,
         mpra1: Path = None, chain: Path = None) -> None:
    root = interim_dir("genome", name).parent.parent
    eval_dir = eval_dir or root / "eval"
    out.mkdir(parents=True, exist_ok=True)
    tables = {"asb": asb(eval_dir / "adastra-asb-v1" / "variants.parquet"),
              "crispri": crispri(eval_dir / "crispri-eg-v1" / "pairs.parquet", root / "tracks" / name / "rnaseq-stranded-v1-bin128")}
    if credible:
        tables["eqtl"] = eqtl(eval_dir / "gtex-finemap-v1" / "pairs.parquet", credible)
    if mpra2:
        tables["hsub2"] = hsub2(mpra2, mpra1, chain, name)
    counts = dict(held_out_chroms=HELD, pair_test_chroms=PAIR_TEST, tables={})
    for k, t in tables.items():
        check(t)
        t.write_parquet(out / f"{k}.parquet")
        c = dict(rows=t.height, by_split=dict(sorted(t["split"].value_counts().iter_rows())),
                 sites=t.select(pl.struct("chrom", "pos", "ref", "alt").n_unique()).item(), head=t["head"][0],
                 weight_sum_train=round(float(t.filter(pl.col("split") == "train")["weight"].sum()), 1),
                 with_channels=int((t["channel_ids"].list.len() > 0).sum()),
                 effect=dict(kind=t["effect_kind"][0], positive_share=round(float((t["effect"] > 0).mean()), 3),
                             abs_median=round(float(t["effect"].abs().median()), 3)))
        counts["tables"][k] = c
        typer.echo(f"{k}: {c}")
    (out / "counts.json").write_text(json.dumps(counts, indent=1))


if __name__ == "__main__":
    typer.run(main)
