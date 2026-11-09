"""Downloading a dataset into ``data/input`` with a record of where every file came from.

A :class:`Source` names a dataset and the files it is made of. :func:`fetch_source` streams each file to a
``.part`` file, renames it when complete, and writes ``SOURCE.json`` beside the files: the URL, the
size, the SHA-256, the time, the licence and the citation. A file already present with the size the
manifest recorded is skipped. ``rows`` takes only the first rows of a large gzipped text file (a CSV or
a FASTA), streamed and recompressed, so a development subset needs neither the whole download nor a
truncated archive.

Only the standard library is used: every source here is public over plain HTTPS.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
import time
import urllib.request
import zlib
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

USER_AGENT = "kron-fm-downloader/0.1"
CHUNK = 1 << 20


@dataclass(frozen=True)
class RemoteFile:
    url: str
    #: The file name on disk; defaults to the URL's last segment.
    name: str = ""
    #: The published size in bytes, when known, to check the download against.
    size: int | None = None

    @property
    def filename(self) -> str:
        return self.name or self.url.rstrip("/").rsplit("/", 1)[-1]


@dataclass(frozen=True)
class Source:
    """A dataset kron-fm can download. ``domain`` and ``name`` give its folder, ``data/input/<domain>/<name>``."""

    name: str
    domain: str
    description: str
    files: tuple[RemoteFile, ...] = ()
    license: str = ""
    citation: str = ""
    homepage: str = ""
    #: Free-form facts a consumer needs, e.g. ``{"format": "csv", "sequence_column": "CDS"}``.
    notes: dict = field(default_factory=dict)
    #: For a source that is not downloaded but taken from elsewhere (a link), the function that does it.
    fetch: Callable[["Source", Path, bool], dict] | None = None

    def folder(self, root: Path) -> Path:
        return root / self.domain / self.name


def hf_dataset_url(repo: str, path: str, revision: str = "main") -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"


def _open(url: str, start: int = 0):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    if start:
        request.add_header("Range", f"bytes={start}-")
    return urllib.request.urlopen(request, timeout=60)


def _progress(name: str, done: int, total: int | None, started: float) -> None:
    rate = done / max(time.time() - started, 1e-6) / 1e6
    of = f" of {total / 1e6:,.1f}" if total else ""
    sys.stderr.write(f"\r  {name}: {done / 1e6:,.1f}{of} MB, {rate:,.1f} MB/s")
    sys.stderr.flush()


def download_file(url: str, dest: Path, expected_size: int | None = None) -> dict:
    """Stream ``url`` to ``dest`` through ``dest.part``, resuming a partial file; return its manifest entry."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    start = part.stat().st_size if part.exists() else 0
    started = time.time()
    with _open(url, start) as response:
        if start and response.status != 206:  # the server ignored the range: start over
            start = 0
        total = response.headers.get("Content-Length")
        total = int(total) + start if total is not None else expected_size
        done = start
        with part.open("ab" if start else "wb") as out:
            while chunk := response.read(CHUNK):
                out.write(chunk)
                done += len(chunk)
                _progress(dest.name, done, total, started)
    sys.stderr.write("\n")
    if expected_size is not None and done != expected_size:
        raise OSError(f"{dest.name}: got {done} bytes, expected {expected_size}; the .part file is kept")
    part.rename(dest)
    return {"file": dest.name, "url": url, "bytes": done, "sha256": sha256(dest),
            "downloaded_at": _now()}


def _gunzip_lines(response, rows: int, keep_header: bool) -> Iterator[bytes]:
    """The first ``rows`` lines (plus a header line when asked) of a gzipped HTTP stream."""
    inflater = zlib.decompressobj(16 + zlib.MAX_WBITS)
    pending = b""
    produced = 0
    limit = rows + (1 if keep_header else 0)
    while produced < limit and (chunk := response.read(CHUNK)):
        pending += inflater.decompress(chunk)
        *lines, pending = pending.split(b"\n")
        for line in lines:
            yield line + b"\n"
            produced += 1
            if produced >= limit:
                return


def download_head(url: str, dest: Path, rows: int, keep_header: bool = True) -> dict:
    """The first ``rows`` lines of a gzipped text file at ``url``, written as a valid gzip at ``dest``.

    For a FASTA, ``rows`` counts lines, not records; the last record may be cut and should be dropped by
    the reader (``check_cds`` refuses a cut coding sequence because it lacks its stop codon).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    started, n = time.time(), 0
    with _open(url) as response, gzip.open(part, "wb") as out:
        for line in _gunzip_lines(response, rows, keep_header):
            out.write(line)
            n += 1
            if n % 100_000 == 0:
                sys.stderr.write(f"\r  {dest.name}: {n:,} lines")
    sys.stderr.write(f"\r  {dest.name}: {n:,} lines in {time.time() - started:,.0f} s\n")
    part.rename(dest)
    return {"file": dest.name, "url": url, "bytes": dest.stat().st_size, "sha256": sha256(dest),
            "downloaded_at": _now(), "subset": f"first {rows} lines" + (" after the header" if keep_header else ""),
            "lines": n}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_manifest(folder: Path) -> dict | None:
    path = folder / "SOURCE.json"
    return json.loads(path.read_text()) if path.is_file() else None


def write_manifest(folder: Path, source: Source, entries: list[dict], extra: dict | None = None) -> dict:
    manifest = {"source": source.name, "domain": source.domain, "description": source.description,
                "license": source.license, "citation": source.citation, "homepage": source.homepage,
                "notes": source.notes, "files": entries, "updated_at": _now(), **(extra or {})}
    (folder / "SOURCE.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def fetch_source(source: Source, root: Path, force: bool = False, rows: int | None = None) -> dict:
    """Download every file of ``source`` into ``root/<domain>/<name>`` and write its ``SOURCE.json``."""
    folder = source.folder(root)
    if source.fetch is not None:
        return source.fetch(source, folder, force)
    folder.mkdir(parents=True, exist_ok=True)
    old = {e["file"]: e for e in (read_manifest(folder) or {}).get("files", [])}
    entries = []
    for remote in source.files:
        name = remote.filename
        if rows is not None:
            stem = name[:-3] if name.endswith(".gz") else name
            base, _, ext = stem.partition(".")
            name = f"{base}.head{rows}.{ext}.gz" if ext else f"{base}.head{rows}.gz"
        dest = folder / name
        previous = old.get(name)
        if dest.exists() and not force and previous and previous.get("bytes") == dest.stat().st_size:
            print(f"  {name}: present ({dest.stat().st_size / 1e6:,.1f} MB), skipped")
            entries.append(previous)
            continue
        if rows is not None:
            entries.append(download_head(remote.url, dest, rows, keep_header=source.notes.get("format") == "csv"))
        else:
            entries.append(download_file(remote.url, dest, remote.size))
    return write_manifest(folder, source, entries)


def link_folder(source: Source, folder: Path, target: Path, hint: str) -> dict:
    """Make ``folder`` a symbolic link to ``target`` (data another checkout already downloaded)."""
    if not target.is_dir():
        raise FileNotFoundError(f"{target} does not exist. {hint}")
    folder.parent.mkdir(parents=True, exist_ok=True)
    if folder.is_symlink():
        folder.unlink()
    elif folder.exists():
        raise FileExistsError(f"{folder} exists and is not a link; move it away first")
    os.symlink(target.resolve(), folder, target_is_directory=True)
    return {"source": source.name, "linked_to": str(target.resolve()), "updated_at": _now()}


def folder_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def describe(source: Source) -> dict:
    d = asdict(source)
    d.pop("fetch")
    return d


__all__ = ["RemoteFile", "Source", "describe", "download_file", "download_head", "fetch_source", "folder_size",
           "hf_dataset_url", "link_folder", "read_manifest", "sha256", "write_manifest"]
