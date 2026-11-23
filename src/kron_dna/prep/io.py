"""File writing and digests every preparation step shares: a file appears whole or not at all."""
from __future__ import annotations

import hashlib
import json
import os
import socket
from contextlib import contextmanager
from pathlib import Path

PART = f".part.{socket.gethostname()}.{os.getpid()}"


def digest(path: Path, kind: str = "sha256", block: int = 1 << 24) -> str:
    """The file's sha256 (or md5) hex digest, read in blocks."""
    h = hashlib.new(kind)
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


@contextmanager
def atomic(path: Path):
    """Yield a temporary path beside ``path``; on a clean exit it is renamed over ``path``, on an error removed. A
    reader never sees a half-written file, and a killed job leaves only a ``.part`` file behind."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + PART)
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_json(path: Path, obj) -> None:
    with atomic(path) as tmp:
        tmp.write_text(json.dumps(obj, indent=1))
