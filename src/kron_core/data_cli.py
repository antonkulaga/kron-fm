"""``kron-data``: list, download and inspect kron-fm's datasets.

    uv run kron-data paths                      # where input, interim and output live
    uv run kron-data list                       # every dataset kron-fm knows
    uv run kron-data download ecoli-k12         # into data/input/cds/ecoli-k12, with SOURCE.json
    uv run kron-data download syncodonlm --rows 200000   # the first 200,000 rows only
    uv run kron-data download metabonet         # link the glucose repo's MetaboNet download
    uv run kron-data status                     # what is on disk, and how large
"""
from __future__ import annotations

import json

import typer

from kron_core.download import Source, describe, fetch_source, folder_size, read_manifest
from kron_core.paths import DATA_ENV, data_root, input_dir, interim_dir, output_dir

app = typer.Typer(add_completion=False, no_args_is_help=True)


def all_sources() -> dict[str, Source]:
    """The datasets of every kron-fm model, by name."""
    from kron_dna.sources import SOURCES as CDS
    from kron_health.sources import SOURCES as HEALTH
    from kron_ppg.sources import SOURCES as PPG
    from krontime.sources import SOURCES as TIMESERIES

    merged: dict[str, Source] = {}
    for registry in (CDS, TIMESERIES, PPG, HEALTH):
        clash = set(merged) & set(registry)
        if clash:
            raise RuntimeError(f"two sources share a name: {sorted(clash)}")
        merged.update(registry)
    return merged


def _source(name: str) -> Source:
    sources = all_sources()
    if name not in sources:
        raise typer.BadParameter(f"unknown dataset {name!r}; one of: {', '.join(sorted(sources))}")
    return sources[name]


@app.command()
def paths() -> None:
    """Print the data folders (set $KRON_FM_DATA to move the whole tree)."""
    typer.echo(f"root     {data_root()}  (${DATA_ENV} overrides)")
    typer.echo(f"input    {input_dir()}    downloads as published, with SOURCE.json")
    typer.echo(f"interim  {interim_dir()}  derived and rebuildable: packed rows, splits")
    typer.echo(f"output   {output_dir()}   runs, checkpoints, ONNX graphs, evaluations")


@app.command("list")
def list_sources(as_json: bool = typer.Option(False, "--json", help="full records as JSON")) -> None:
    """Every dataset kron-fm can fetch, and whether it is on disk."""
    sources = all_sources()
    if as_json:
        typer.echo(json.dumps({n: describe(s) for n, s in sources.items()}, indent=2))
        return
    for name, s in sources.items():
        folder = s.folder(input_dir())
        state = "linked" if folder.is_symlink() else ("present" if folder.exists() else "-")
        size = s.notes.get("approx_mb")
        size_text = f"{size:,.0f} MB" if size and size >= 10 else (f"{size} MB" if size else "")
        typer.echo(f"{name:14s} {s.domain:11s} {state:8s} {size_text:>10s}  {s.description}")


@app.command()
def download(names: list[str] = typer.Argument(..., help="dataset names (see `kron-data list`)"),
             rows: int | None = typer.Option(None, help="take only the first ROWS lines of a gzipped text file"),
             force: bool = typer.Option(False, help="download again even when the files are present")) -> None:
    """Download datasets into data/input/<domain>/<name>."""
    for name in names:
        source = _source(name)
        typer.echo(f"{name} -> {source.folder(input_dir())}")
        record = fetch_source(source, input_dir(), force=force, rows=rows)
        typer.echo(json.dumps({k: v for k, v in record.items() if k in ("files", "linked_to")}, indent=2))


@app.command()
def status() -> None:
    """Sizes of everything under input, interim and output, with each download's provenance."""
    for stage, root in (("input", input_dir()), ("interim", interim_dir()), ("output", output_dir())):
        typer.echo(f"{stage}: {root}")
        if not root.exists():
            typer.echo("  (empty)")
            continue
        found = False
        for domain in sorted(p for p in root.iterdir() if p.is_dir()):
            for item in sorted(domain.iterdir()):
                found = True
                link = f" -> {item.resolve()}" if item.is_symlink() else ""
                manifest = read_manifest(item) if item.is_dir() and not item.is_symlink() else None
                origin = f"  ({len(manifest['files'])} files, {manifest['license']})" if manifest else ""
                typer.echo(f"  {domain.name}/{item.name}{link}: {folder_size(item) / 1e6:,.1f} MB{origin}")
        if not found:
            typer.echo("  (empty)")


if __name__ == "__main__":
    app()
