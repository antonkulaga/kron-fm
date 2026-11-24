"""One source of ``sources.toml`` end to end: download, prepare, labels, split. The orchestration that used to be
shell scripts per cluster; this names no machine (data under ``$KRON_FM_DATA``, jobs through ``slurm/submit.sh``).

    uv run python scripts/kron_dna/prep_source.py genome NAME [--manifest sources.toml] [--as OTHER_NAME] [--dry-run]
    uv run python scripts/kron_dna/prep_source.py verify NAME        (rebuild under NAME.verify and compare with NAME)
    uv run python scripts/kron_dna/prep_source.py list [--manifest sources.toml]
    slurm/submit.sh cpu genome-NAME uv run python scripts/kron_dna/prep_source.py genome NAME      (as a job)

``genome``: fetches the assembly's FASTA (and GTF, with ``annotation``) from NCBI into ``input/genome/NAME``, then
runs ``prep_species_genome.py`` (``min_seq``, ``held``), ``prep_gene_structure_labels.py`` and, with ``orthologs``,
``prep_ortholog_split.py`` against NCBI Gene's human table and ortholog pairs (fetched once into
``input/genome/hg38/ncbi``). A step whose output exists is skipped, so a rerun continues. ``--as`` prepares the
entry under another name (a rebuild to compare with the existing one). ``--dry-run`` prints the steps only.
"""
from __future__ import annotations

import subprocess
import sys
import tomllib
import urllib.request
from pathlib import Path

import typer

from kron_core.paths import input_dir, interim_dir

app = typer.Typer(add_completion=False)
HERE = Path(__file__).resolve().parent
NCBI = "https://ftp.ncbi.nlm.nih.gov"
GENE = {"Homo_sapiens.gene_info.gz": f"{NCBI}/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz",
        "gene_orthologs.gz": f"{NCBI}/gene/DATA/gene_orthologs.gz"}


def assembly_url(accession: str, suffix: str, base: str = NCBI) -> str:
    """``GCF_052327205.1_FLI_Smic_1.0`` -> its file on NCBI's genomes FTP (``.../GCF/052/327/205/<accession>/``)."""
    if accession[:4] not in ("GCF_", "GCA_") or not accession[4:13].isdigit():
        raise ValueError(f"not an NCBI assembly folder name: {accession!r}")
    d = accession[4:13]
    return f"{base}/genomes/all/{accession[:3]}/{d[:3]}/{d[3:6]}/{d[6:9]}/{accession}/{accession}_{suffix}"


def plan(name: str, entry: dict, out_name: str | None = None) -> list[dict]:
    """The steps of a genome entry: ``fetch`` (url -> path) and ``run`` (a script's arguments), each with the path
    whose existence means done."""
    n = out_name or name
    acc, src = entry["accession"], input_dir("genome", name)       # downloads belong to the entry, whatever it is built as
    fasta, gtf = src / f"{acc}_genomic.fna.gz", src / f"{acc}_genomic.gtf.gz"
    steps = [dict(kind="fetch", url=assembly_url(acc, "genomic.fna.gz"), path=fasta, done=fasta)]
    if entry.get("annotation"):
        steps.append(dict(kind="fetch", url=assembly_url(acc, "genomic.gtf.gz"), path=gtf, done=gtf))
    genome = interim_dir("genome", n)
    args = [str(fasta), "--name", n, "--min-seq", str(entry.get("min_seq", 1_000_000))]
    if entry.get("held"):
        args += ["--held", entry["held"]]
    steps.append(dict(kind="run", script="prep_species_genome.py", args=args, done=genome / "bases.u8"))
    tracks = genome.parent.parent / "tracks" / n
    if entry.get("annotation"):
        lab = tracks / "gene-structure-refseq"
        steps.append(dict(kind="run", script="prep_gene_structure_labels.py",
                          args=[str(lab), "--name", n, "--gtf", str(gtf)], done=lab / "labels.json"))
    if entry.get("orthologs"):
        if not entry.get("annotation"):
            raise ValueError(f"{name}: an ortholog split needs the annotation")
        ncbi, sp = input_dir("genome", "hg38", "ncbi"), tracks / "split-orthologs-v1"
        steps += [dict(kind="fetch", url=u, path=ncbi / f, done=ncbi / f) for f, u in GENE.items()]
        steps.append(dict(kind="run", script="prep_ortholog_split.py",
                          args=[str(gtf), str(sp), "--name", n, "--gene-info", str(ncbi / "Homo_sapiens.gene_info.gz"),
                                "--orthologs", str(ncbi / "gene_orthologs.gz")], done=sp / "split.json"))
    return steps


def fetch(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:       # honours http(s)_proxy
        while chunk := r.read(1 << 22):
            f.write(chunk)
    tmp.rename(path)


def manifest_of(path: Path | None) -> dict:
    return tomllib.loads((path or HERE / "sources.toml").read_text())


@app.command()
def genome(name: str, manifest: Path = None, as_name: str = typer.Option("", "--as"), dry_run: bool = False) -> None:
    entries = manifest_of(manifest).get("genome", {})
    if name not in entries:
        raise typer.BadParameter(f"no genome {name!r} in the manifest ({', '.join(sorted(entries))})")
    for s in plan(name, entries[name], as_name or None):
        what = f"fetch {s['url']} -> {s['path']}" if s["kind"] == "fetch" else f"{s['script']} {' '.join(s['args'])}"
        if Path(s["done"]).exists() and (s["kind"] != "fetch" or Path(s["done"]).stat().st_size > 0):
            typer.echo(f"done already: {what}")
            continue
        typer.echo(what)
        if dry_run:
            continue
        if s["kind"] == "fetch":
            fetch(s["url"], s["path"])
        else:
            subprocess.run([sys.executable, str(HERE / s["script"]), *s["args"]], check=True)


def digest(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


@app.command()
def verify(name: str, manifest: Path = None, keep: bool = False) -> None:
    """Rebuilds the entry under ``NAME.verify`` from its downloads and compares it with the prepared ``NAME``: the
    bases, the contig table, the gene labels and the split. Exit 1 on any difference (the rebuild is then kept);
    the rebuild is removed when all agree, unless ``--keep``."""
    import shutil

    import polars as pl
    other = f"{name}.verify"
    genome(name, manifest, other, False)
    a, b = interim_dir("genome", name), interim_dir("genome", other)
    ta, tb = a.parent.parent / "tracks" / name, a.parent.parent / "tracks" / other
    pairs = [(a / "bases.u8", b / "bases.u8")] + [(p, tb / p.relative_to(ta)) for p in sorted(ta.rglob("*"))
                                                    if p.suffix in (".u8", ".u16") and (tb / p.relative_to(ta)).exists()]
    same = pl.read_parquet(a / "contigs.parquet").equals(pl.read_parquet(b / "contigs.parquet"))
    typer.echo(f"{name}: contigs.parquet {'same' if same else 'DIFFERENT'}")
    for x, y in pairs:
        ok = digest(x) == digest(y)
        same &= ok
        typer.echo(f"{name}: {x.relative_to(a.parent.parent)} {'same' if ok else 'DIFFERENT'}")
    typer.echo(f"{name}: {'VERIFIED' if same else 'NOT REPRODUCED'} from the manifest entry")
    if same and not keep:
        shutil.rmtree(b)
        shutil.rmtree(tb, ignore_errors=True)
    raise typer.Exit(0 if same else 1)


@app.command("list")
def list_(manifest: Path = None) -> None:
    for n, e in sorted(manifest_of(manifest).get("genome", {}).items()):
        state = "prepared" if (interim_dir("genome", n) / "bases.u8").exists() else "not prepared"
        typer.echo(f"{n:20s} {e['role']:9s} {e['accession']:42s} {state:13s} {e['species']}")


if __name__ == "__main__":
    app()
