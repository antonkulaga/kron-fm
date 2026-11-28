"""FANTOM5 CAGE as stranded tracks: choose samples from the reprocessed listings, and turn each sample's CTSS table
into a plus and a minus bigWig that ``prep.py tracks binned`` reads like ENCODE's stranded pairs.

    uv run python scripts/kron_dna/prep_fantom_ctss.py select OUT.tsv --assembly hg38 [--groups cell_line+tissue+primary_cell]
    uv run --with pyBigWig python scripts/kron_dna/prep_fantom_ctss.py convert SELECT.tsv CTSS_DIR OUT_DIR --name hg38

FANTOM5 (fantom.gsc.riken.jp/5, CC BY 4.0) gives, per sample, ``<sample name>.<CNhs id>.<library>.<assembly>.
nobarcode.ctss.bed.gz``: one line per base and strand with the number of capped 5' ends there (chrom, start, end,
name, count, strand). ``select`` reads the directory listings of ``reprocessed/<assembly>_latest/basic/
<species>.<group>.hCAGE/`` and keeps one sample per biosample: the sample name with its donor, replicate and pool
suffix taken off is the biosample, and the largest file among its samples is kept (file size follows depth).
Columns as the ENCODE selections: i, experiment (the CNhs id), file (``<CNhs>_P`` / ``_M``), assay ``CAGE``,
biosample, target, output_type, size, bio_reps, url, channel; a sample is two rows with one url. ``encode_fetch.py``
is not used for these (no ENCODE record): ``fetch`` downloads each url once, checked against the listing's size.

``convert`` writes ``OUT_DIR/<CNhs>_P.bigWig`` and ``_M.bigWig``: the counts as they are, per base, on the
chromosomes of the prepared genome ``--name`` (lines on other sequences are counted and left out). A finished
pair is skipped.
"""
from __future__ import annotations

import csv
import gzip
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import typer

BASE = "https://fantom.gsc.riken.jp/5/datafiles/reprocessed"
SPECIES = {"hg38": "human", "mm10": "mouse"}
SUFFIX = re.compile(r"(,\s*|\s+)(donor\s*\d+(\s*\([^)]*\))?|biol_?\s?rep\s*\d+(\s*\([^)]*\))?|tech_?\s?rep\s*\d+|"
                    r"pool\s*\d+|rep\s*\d+|replicate\s*\d+)\s*$", re.IGNORECASE)
app = typer.Typer(add_completion=False)


def biosample_of(sample: str) -> str:
    """The sample's name without its trailing donor, replicate and pool marks (``..., donor2``, ``biol_rep1``,
    ``tech_rep2``, ``pool1``), taken off repeatedly."""
    name = sample.strip()
    while True:
        cut = SUFFIX.sub("", name).strip().rstrip(",").strip()
        if cut == name:
            return name
        name = cut


def parse_listing(html: str, assembly: str) -> list[dict]:
    """The CTSS files of one directory listing: name (decoded), CNhs id, the href as it stands, size in bytes."""
    out = []
    for m in re.finditer(r'href="([^"]+?\.ctss\.bed\.gz)"[^\n]*?(\d{2}:\d{2})\s*</td><td[^>]*>\s*([\d.]+)([KMG]?)', html):
        href, size, unit = m.group(1), float(m.group(3)), m.group(4)
        name = unquote(unquote(href))
        parts = name.split(".")
        k = next((i for i, p in enumerate(parts) if re.fullmatch(r"CNhs\d+", p)), None)
        if k is None or assembly not in parts:
            continue
        out.append(dict(sample=".".join(parts[:k]), cnhs=parts[k], href=href,
                        size=int(size * {"": 1, "K": 1 << 10, "M": 1 << 20, "G": 1 << 30}[unit])))
    return out


def choose(files: list[dict]) -> list[dict]:
    """One file per biosample: the largest, ties to the lowest CNhs id; ``reps`` counts the samples it stands for."""
    by = defaultdict(list)
    for f in files:
        by[biosample_of(f["sample"])].append(f)
    out = []
    for bio in sorted(by):
        best = sorted(by[bio], key=lambda f: (-f["size"], f["cnhs"]))[0]
        out.append(dict(best, biosample=bio, reps=len(by[bio])))
    return out


@app.command()
def select(out: Path, assembly: str = "hg38", groups: str = "cell_line+tissue+primary_cell") -> None:
    species, files = SPECIES[assembly], []
    for g in groups.split("+"):
        url = f"{BASE}/{assembly}_latest/basic/{species}.{g}.hCAGE/"
        html = subprocess.run(["curl", "-sS", "--fail", "--retry", "5", "--max-time", "300", url],
                              capture_output=True, text=True, check=True).stdout
        got = parse_listing(html, assembly)
        typer.echo(f"{g}: {len(got)} samples")
        files += [dict(f, url=url + f["href"], group=g) for f in got]
    chosen = choose(files)
    ids = [c["cnhs"] for c in chosen]
    assert len(set(ids)) == len(ids), "a CNhs id chosen twice"
    with out.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["i", "experiment", "file", "assay", "biosample", "target", "output_type", "size", "bio_reps",
                    "url", "channel"])
        for i, c in enumerate(chosen):
            for s, what in (("P", "plus strand 5' end counts"), ("M", "minus strand 5' end counts")):
                w.writerow([2 * i + (s == "M"), c["cnhs"], f"{c['cnhs']}_{s}", "CAGE", c["biosample"], "", what,
                            c["size"], c["reps"], c["url"], f"{c['cnhs']}_{s}"])
    typer.echo(f"{len(files)} samples, {len(chosen)} biosamples, {sum(c['size'] for c in chosen) / 1e9:.1f} GB -> {out}")


@app.command()
def fetch(select: Path, out: Path) -> None:
    """Each sample's table once, to ``OUT/<CNhs>.ctss.bed.gz``; a file of the listed size (within 1 %, the listing
    rounds) that gunzips to its end is kept, anything else is downloaded again."""
    out.mkdir(parents=True, exist_ok=True)
    rows = {r["experiment"]: r for r in csv.DictReader(select.open(), delimiter="\t")}
    bad = 0
    for cnhs, r in rows.items():
        dst = out / f"{cnhs}.ctss.bed.gz"
        if not dst.exists():
            tmp = Path(str(dst) + ".part")
            subprocess.run(["curl", "-sS", "--fail", "--retry", "5", "--retry-all-errors", "-o", str(tmp), r["url"]],
                           check=True)
            ok = abs(tmp.stat().st_size - int(r["size"])) <= max(2048, 0.01 * int(r["size"]))
            ok = ok and subprocess.run(["gzip", "-t", str(tmp)]).returncode == 0
            if not ok:
                bad += 1
                typer.echo(f"{cnhs}: size or gzip check failed, left as {tmp.name}")
                continue
            tmp.rename(dst)
    typer.echo(f"{len(rows) - bad} of {len(rows)} tables in {out}")
    if bad:
        raise typer.Exit(1)


def strands(path: Path, sizes: dict[str, int]) -> tuple[dict, int]:
    """``{(chrom, strand): (starts, counts)}`` sorted by start, for the chromosomes in ``sizes``, and the number of
    lines left out (another sequence, or a position past the chromosome's end)."""
    got, skipped = defaultdict(lambda: ([], [])), 0
    with gzip.open(path, "rt") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 6 or p[0] not in sizes or int(p[2]) > sizes[p[0]] or p[5] not in "+-":
                skipped += 1
                continue
            s, c = got[(p[0], p[5])]
            s.append(int(p[1]))
            c.append(float(p[4]))
    out = {}
    for key, (s, c) in got.items():
        s, c = np.asarray(s, dtype=np.int64), np.asarray(c, dtype=np.float64)
        order = np.argsort(s, kind="stable")
        s, c = s[order], c[order]
        if len(s) > 1 and bool((np.diff(s) == 0).any()):                   # a base listed twice: its counts add
            first = np.r_[True, np.diff(s) != 0]
            c = np.add.reduceat(c, np.flatnonzero(first))
            s = s[first]
        out[key] = (s, c)
    return out, skipped


@app.command()
def convert(select: Path, ctss: Path, out: Path, name: str = "hg38") -> None:
    import polars as pl
    import pyBigWig

    from kron_core.paths import interim_dir

    idx = pl.read_parquet(interim_dir("genome", name) / "contigs.parquet")
    sizes = {r["chrom"]: int(r["end"]) for r in idx.group_by("chrom").agg(pl.col("end").max()).iter_rows(named=True)}
    order = sorted(sizes)
    out.mkdir(parents=True, exist_ok=True)
    ids = sorted({r["experiment"] for r in csv.DictReader(select.open(), delimiter="\t")})
    for cnhs in ids:
        dst = {s: out / f"{cnhs}_{s}.bigWig" for s in "PM"}
        if all(d.exists() for d in dst.values()):
            continue
        got, skipped = strands(ctss / f"{cnhs}.ctss.bed.gz", sizes)
        for s, sign in (("P", "+"), ("M", "-")):
            tmp = str(dst[s]) + ".part"
            bw = pyBigWig.open(tmp, "w")
            bw.addHeader([(c, sizes[c]) for c in order])
            n = 0
            for c in order:
                if (c, sign) in got:
                    st, ct = got[(c, sign)]
                    bw.addEntries([c] * len(st), st.tolist(), ends=(st + 1).tolist(), values=ct.tolist())
                    n += len(st)
            bw.close()
            Path(tmp).rename(dst[s])
        typer.echo(f"{cnhs}: {sum(len(v[0]) for v in got.values()):,} positions, {skipped:,} lines left out")


if __name__ == "__main__":
    app()
