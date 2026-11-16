"""The run record of ``genome-train``: what a run was trained on and how, written at startup and stored in every
checkpoint, so that a held-out set or a context id cannot change silently between a run and its resumption.

A record holds the source commit (and the md5 of every ``kron_dna`` module, since compute nodes have no ``git``), the
full resolved options, one entry per genome (role, name, share of training windows, context id(s), held-out contigs,
md5 of its ``contigs.parquet``), and two name-to-index registries: the stem's input channels and the context rows.
:func:`identity_differences` lists what a resumption may not change. Pure bookkeeping, no torch.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import time
from pathlib import Path

RECORD_VERSION = 1


def file_md5(path: Path) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _gitdir_commit(start: Path) -> dict:
    """HEAD of the work tree that holds ``start``, read from the ``.git`` file/directory without the ``git`` binary
    (compute nodes have none): follows a worktree's ``gitdir:`` and ``commondir``, loose refs and packed-refs."""
    for d in [start, *start.parents]:
        dot = d / ".git"
        if dot.is_dir():
            gitdir = dot
        elif dot.is_file() and dot.read_text().startswith("gitdir:"):
            gitdir = (d / dot.read_text().split(":", 1)[1].strip()).resolve()
        else:
            continue
        common = gitdir
        if (gitdir / "commondir").is_file():
            common = (gitdir / (gitdir / "commondir").read_text().strip()).resolve()
        head = (gitdir / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return dict(commit=head, branch=None, worktree=str(d), via="gitdir")
        ref = head.split(":", 1)[1].strip()
        for base in (gitdir, common):
            if (base / ref).is_file():
                return dict(commit=(base / ref).read_text().strip(), branch=ref.removeprefix("refs/heads/"),
                            worktree=str(d), via="gitdir")
        packed = common / "packed-refs"
        if packed.is_file():
            for line in packed.read_text().splitlines():
                if line.endswith(" " + ref):
                    return dict(commit=line.split()[0], branch=ref.removeprefix("refs/heads/"), worktree=str(d),
                                via="gitdir")
        return dict(commit=None, branch=ref.removeprefix("refs/heads/"), worktree=str(d), via="gitdir")
    return dict(commit=None, branch=None, worktree=None, via=None)


def source_info(package_dir: Path) -> dict:
    """Commit, branch and work tree of the code that runs, plus the md5 of each module of the package. With ``git``
    available, also whether tracked files differ from HEAD (``dirty``); without it ``dirty`` is None and the module
    md5s are what pins the code."""
    package_dir = Path(package_dir).resolve()
    info = _gitdir_commit(package_dir)
    info["dirty"] = None
    try:
        out = subprocess.run(["git", "-C", str(package_dir), "status", "--porcelain", "--untracked-files=no"],
                             capture_output=True, text=True, timeout=20)
        if out.returncode == 0:
            info["dirty"] = bool(out.stdout.strip())
            info["dirty_files"] = [ln[3:] for ln in out.stdout.splitlines()][:50]
    except (OSError, subprocess.SubprocessError):
        pass
    info["module_md5"] = {p.name: file_md5(p) for p in sorted(package_dir.glob("*.py"))}
    return info


def genome_entry(role: str, name: str, share: float | None, context, held_out: list[str],
                 contigs_parquet: Path, **more) -> dict:
    """One genome of a run. ``context``: an id, or a description for --mito (one id per genetic code)."""
    return dict(role=role, name=name, share=None if share is None else round(float(share), 10), context=context,
                held_out=sorted(str(h) for h in held_out), n_held_out=len(held_out),
                contigs_md5=file_md5(contigs_parquet), contigs_parquet=str(contigs_parquet), **more)


def jsonable(v):
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    # plain builtins only: a subclass (torch.__version__ is a TorchVersion, a str) would make the checkpoint
    # unloadable by torch.load's weights-only unpickler, which genome-train uses to resume
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, str):
        return str(v)
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float):
        return float(v)
    if hasattr(v, "item"):                    # numpy scalars
        return v.item()
    return repr(v)


def make_record(options: dict, resolved: dict, genomes: list[dict], input_channels: dict[str, int],
                context_rows: dict[str, int], source: dict, **more) -> dict:
    return jsonable(dict(record_version=RECORD_VERSION, written=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                         host=socket.gethostname(), slurm_job=os.environ.get("SLURM_JOB_ID"), source=source,
                         options=options, resolved=resolved, genomes=genomes, input_channels=input_channels,
                         context_rows=context_rows, **more))


def _identity(rec: dict) -> list[tuple]:
    return [(g["role"], g["name"]) for g in rec.get("genomes", [])]


def identity_differences(old: dict, new: dict) -> list[str]:
    """What a resumed run may not change, as readable lines (empty: the same): the genome list (role and name, in
    order), each genome's held-out contigs, its context id(s) and its contigs.parquet md5 (the contig layout the
    held-out names refer to), and the context-row and input-channel registries."""
    out = []
    a, b = _identity(old), _identity(new)
    if a != b:
        out.append(f"genome list: checkpoint {a}, now {b}")
    olds = {(g["role"], g["name"]): g for g in old.get("genomes", [])}
    for g in new.get("genomes", []):
        o = olds.get((g["role"], g["name"]))
        if o is None:
            continue
        for key in ("held_out", "context", "contigs_md5"):
            if o.get(key) != g.get(key):
                ov, nv = o.get(key), g.get(key)
                if key == "held_out":
                    ov = f"{len(ov)} contigs {ov[:6]}{'...' if len(ov) > 6 else ''}"
                    nv = f"{len(nv)} contigs {nv[:6]}{'...' if len(nv) > 6 else ''}"
                out.append(f"{g['role']} {g['name']}: {key} was {ov}, now {nv}")
    for key in ("context_rows", "input_channels"):
        if old.get(key) != new.get(key):
            out.append(f"{key}: checkpoint {old.get(key)}, now {new.get(key)}")
    return out


def legacy_differences(old_genomes: list[dict], new: dict) -> list[str]:
    """Against a checkpoint of the multi-genome branch (a ``genomes`` entry, no record): further genomes' names,
    contexts and held-out contigs."""
    a = [(g["name"], int(g["context"]), sorted(g["held_out"])) for g in old_genomes]
    b = [(g["name"], int(g["context"]), list(g["held_out"])) for g in new.get("genomes", []) if g["role"] == "genome"]
    return [] if a == b else [f"--genome list: checkpoint {a}, now {b}"]


def write_json(path: Path, rec: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(rec, indent=1) + "\n")
    os.replace(tmp, path)
