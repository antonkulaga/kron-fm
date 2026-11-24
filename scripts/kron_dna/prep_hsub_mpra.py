"""Human-versus-chimpanzee enhancer fragments on hg38 for the variant benchmark: the first MPRA of Uebbing et al.
2021 (PNAS 118:e2007049118; GEO GSE140983, ``GSE140983_COUNTS_ANNOTATED.txt.gz``): 137-bp fragments of human
accelerated regions and human-gain enhancers around human-specific substitutions, each as the human and as the
chimpanzee sequence, assayed in human neural stem cells (H9-derived). The file is per barcode; this script sums it.

    uv run --with pyliftover python scripts/kron_dna/prep_hsub_mpra.py OUT.parquet COUNTS.txt.gz
        --chain hg19ToHg38.over.chain.gz [--enhancers DatasetS1.tsv] [--name hg38]

One row per fragment whose two alleles were both measured: ``chrom``, ``pos`` (0-based), ``ref`` (the human
sequence from the first to the last substituted base, as hg38 has it), ``alt`` (the same span with the chimpanzee
bases; a fragment carries 1 to 11 substitutions, ``n_subs``; ``sub_pos`` and ``sub_alt`` list them), ``centre`` and
``half`` (the fragment, lifted from hg19: where a track head is read, ``variant_bench.py score --read-col centre
--flank-col half``), and

- ``act_human``, ``act_chimp``: the mean over the allele's barcodes of log2 of cDNA over plasmid counts (summed
  over the four replicates, scaled by the library totals); ``effect`` = ``act_chimp - act_human`` (alt over ref,
  as the other MPRA table), ``se``, ``t`` and ``p_value`` (Welch's test between the two alleles' barcodes),
  ``fdr`` (Benjamini-Hochberg over the confident rows), ``differential`` (fdr < 0.05). This is this script's
  statistic, not the paper's;
- ``tags`` (barcodes with >= ``--min-plasmid`` plasmid reads, the smaller of the two alleles), ``confident`` (>= 10);
- ``experiment``, ``cell``, ``kind``, ``primary``: constants, the columns ``prep_mpra_satmut.py`` has;
- ``ref_ok`` (every human base of the file is hg38's base; the file writes a substitution as chimp->human) and
  ``held_out`` (hg38 chr8/chr9).

The file's control fragments without a substitution (the same sequence synthesised twice) are not variants; the
spread of their "effect" is the assay's noise and goes to the summary (``null_sd``).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
import typer

from kron_core.paths import interim_dir

HELD = ["chr8", "chr9"]
REPS = ["Rep_1_2", "Rep_1_3", "Rep_2_2", "Rep_2_3"]


def activities(t: pl.DataFrame, min_plasmid: int) -> pl.DataFrame:
    """Per ``Alignment``: its barcodes with at least ``min_plasmid`` plasmid reads over the replicates (``tags``),
    and the mean and variance over them of log2 of cDNA over plasmid counts (summed over the replicates, one
    pseudocount each, scaled by the library totals)."""
    c, p = [f"{r}_cDNA_Tag_Counts" for r in REPS], [f"{r}_pDNA_Tag_Counts" for r in REPS]
    t = t.with_columns(c=pl.sum_horizontal(c), p=pl.sum_horizontal(p)).filter(pl.col("p") >= min_plasmid)
    scale = float(np.log2(t["c"].sum() / t["p"].sum()))
    return t.with_columns(a=((pl.col("c") + 1) / (pl.col("p") + 1)).log(2) - scale).group_by("Alignment").agg(
        tags=pl.len(), act=pl.col("a").mean(), var=pl.col("a").var())


def main(out: Path, counts: Path, chain: Path = typer.Option(...), name: str = "hg38", min_plasmid: int = 20) -> None:
    from pyliftover import LiftOver
    from scipy import stats
    t = pl.read_csv(counts, separator="\t", infer_schema_length=100000, null_values=["NA"])
    human = t.filter(pl.col("Species") == "hg19").unique("Alignment").select(
        "Alignment", "Ortholog_Seq", "Chr", "Start", "Stop", "eVar_BPChange", "eVar_Pos")
    act = activities(t, min_plasmid)
    j = human.join(act.rename({"act": "act_human", "var": "var_h", "tags": "tags_h"}), on="Alignment").join(
        act.rename({"Alignment": "Ortholog_Seq", "act": "act_chimp", "var": "var_c", "tags": "tags_c"}), on="Ortholog_Seq")
    j = j.filter((pl.col("tags_h") >= 2) & (pl.col("tags_c") >= 2))
    eh, ec = pl.col("var_h") / pl.col("tags_h"), pl.col("var_c") / pl.col("tags_c")
    j = j.with_columns(effect=pl.col("act_chimp") - pl.col("act_human"), se=(eh + ec).sqrt(),      # Welch's t-test
                       dof=(eh + ec) ** 2 / (eh ** 2 / (pl.col("tags_h") - 1) + ec ** 2 / (pl.col("tags_c") - 1)),
                       tags=pl.min_horizontal("tags_h", "tags_c"))
    j = j.with_columns(t=pl.col("effect") / pl.col("se")).filter(pl.col("t").is_finite())
    j = j.with_columns(p_value=pl.Series(2 * stats.t.sf(np.abs(j["t"].to_numpy()), j["dof"].to_numpy())))
    control = j.filter(pl.col("eVar_Pos").is_null())
    j = j.filter(pl.col("eVar_Pos").is_not_null())

    lift = LiftOver(str(chain))
    src = interim_dir("genome", name)
    bases = np.memmap(src / "bases.u8", dtype=np.uint8, mode="r")
    segs = pl.read_parquet(src / "contigs.parquet").to_dicts()

    def at(chrom: str, x: int) -> int | None:
        return next((s["offset"] + x - s["start"] for s in segs if s["chrom"] == chrom and s["start"] <= x < s["end"]), None)

    rows, dropped = [], Counter()
    for r in j.to_dicts():
        offs = [int(x) - 1 for x in str(r["eVar_Pos"]).split(";")]                 # 0-based from the fragment's start
        subs = [x.split("->") for x in r["eVar_BPChange"].split(";")]              # chimp -> human
        order = np.argsort(offs)
        offs, subs = [offs[k] for k in order], [subs[k] for k in order]
        a, b = lift.convert_coordinate(r["Chr"], r["Start"]), lift.convert_coordinate(r["Chr"], r["Stop"] - 1)
        if not a or not b or a[0][0] != b[0][0] or a[0][2] != "+" or b[0][1] - a[0][1] != r["Stop"] - 1 - r["Start"]:
            dropped["not lifted in one piece"] += 1
            continue
        chrom, start = a[0][0], a[0][1]
        g = at(chrom, start + offs[0])
        if g is None or at(chrom, start + offs[-1]) != g + offs[-1] - offs[0] or len(set(offs)) < len(offs):
            dropped["in an assembly gap or repeated position"] += 1
            continue
        ref = "".join("ACGTN"[x] for x in bases[g:g + offs[-1] - offs[0] + 1])
        alt = list(ref)
        for o, (chimp, _) in zip(offs, subs):
            alt[o - offs[0]] = chimp
        ok = all(ref[o - offs[0]] == hum for o, (_, hum) in zip(offs, subs))
        rows.append(dict(id=r["Alignment"], chrom=chrom, pos=start + offs[0], ref=ref, alt="".join(alt), n_subs=len(offs),
                         sub_pos=[start + o for o in offs], sub_alt=[s[0] for s in subs],
                         centre=start + (r["Stop"] - r["Start"]) // 2, half=(r["Stop"] - r["Start"] + 1) // 2,
                         act_human=r["act_human"], act_chimp=r["act_chimp"], effect=r["effect"], se=r["se"], t=r["t"],
                         p_value=r["p_value"], tags=r["tags"],
                         ref_ok=ok))
    v = pl.DataFrame(rows).with_columns(
        confident=pl.col("tags") >= 10, held_out=pl.col("chrom").is_in(HELD),
        experiment=pl.lit("uebbing2021-mpra1"), element=pl.col("id"), cell=pl.lit("neural stem cell"),
        kind=pl.lit("enhancer"), primary=pl.lit(True))
    p = v["p_value"].to_numpy()
    fdr = np.ones(len(p))
    keep = np.flatnonzero(v["confident"].to_numpy())
    o = keep[np.argsort(p[keep])]
    fdr[o] = np.minimum.accumulate((p[o] * len(o) / np.arange(1, len(o) + 1))[::-1])[::-1].clip(max=1)
    v = v.with_columns(fdr=pl.Series(fdr)).with_columns(differential=pl.col("fdr") < 0.05)
    out.parent.mkdir(parents=True, exist_ok=True)
    v.write_parquet(out)
    conf = v.filter("confident")
    meta = dict(rows=v.height, source="Uebbing et al. 2021, GEO GSE140983 COUNTS_ANNOTATED (MPRA 1), hg19 lifted to hg38",
                held_out_chroms=HELD, dropped=dict(dropped), confident=conf.height, differential=int(v["differential"].sum()),
                held_out=int(v["held_out"].sum()), held_out_differential=int(v.filter("held_out")["differential"].sum()),
                ref_ok=round(float(v["ref_ok"].mean()), 4), effect_sd=round(float(conf["effect"].std()), 4),
                null_sd=round(float(control.filter(pl.col("tags") >= 10)["effect"].std()), 4), controls=control.height,
                controls_p_below_0_05=round(float((control.filter(pl.col("tags") >= 10)["p_value"] < 0.05).mean()), 4),
                confident_p_below_0_05=round(float((conf["p_value"] < 0.05).mean()), 4),
                n_subs=dict(sorted(v["n_subs"].value_counts().iter_rows())))
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    typer.run(main)
