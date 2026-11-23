"""A prepared genome's layout (``bases.u8`` + ``contigs.parquet``): its contigs, one ``locate`` from chromosome
coordinates to global offsets, its training split, and chromosome-name matching that refuses a mismatch."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from kron_core.paths import interim_dir
from kron_dna import genome_variants as GV
from kron_dna.prep.io import digest


@dataclass
class Layout:
    name: str
    path: Path
    contigs: pl.DataFrame                              # chrom, start, end, offset (+ held_out where present)
    total: int
    _split: np.ndarray | None = field(default=None, repr=False)

    @classmethod
    def open(cls, name: str, path: Path | None = None) -> "Layout":
        p = Path(path) if path is not None else interim_dir("genome", name)
        c = pl.read_parquet(p / "contigs.parquet").sort("offset")
        return cls(name=name, path=p, contigs=c, total=int((c["end"] - c["start"]).sum()))

    @property
    def bases(self) -> np.ndarray:
        return np.memmap(self.path / "bases.u8", dtype=np.uint8, mode="r")

    def rows(self) -> list[tuple]:
        """(chrom, start, end, offset) per contig, in layout order."""
        return list(self.contigs.select("chrom", "start", "end", "offset").iter_rows())

    def chroms(self) -> set[str]:
        return set(self.contigs["chrom"].to_list())

    def locate(self, chrom: np.ndarray, pos0: np.ndarray) -> np.ndarray:
        """Global offsets of 0-based chromosome positions; -1 outside the gap-free contigs."""
        return GV.locate(self.contigs, np.asarray(chrom).astype(str), np.asarray(pos0, dtype=np.int64))[0]

    def split(self, path: Path | None = None) -> np.ndarray | None:
        """The training split (uint8 per base, 1 = training): ``path``, else the genome's ``ntv3/split.u8``; None
        when neither exists."""
        p = Path(path) if path is not None else self.path / "ntv3" / "split.u8"
        if not p.exists():
            return None
        s = np.memmap(p, dtype=np.uint8, mode="r")
        if len(s) != self.total:
            raise ValueError(f"{p}: {len(s):,} bases, the layout {self.name} has {self.total:,}")
        return s

    def identity(self) -> dict:
        return dict(genome=self.name, total_bases=self.total, contigs_sha256=digest(self.path / "contigs.parquet"))


def match_names(names, layout: Layout, prefix: str = "", min_share: float = 0.5) -> dict[str, str]:
    """A source's chromosome names onto the layout's: as they are, with ``prefix`` added, with a ``chr`` prefix added or
    removed, and MT/chrM. Returns {source name: layout name} for those that match; raises ValueError when the
    matched names carry under ``min_share`` of the source's names (counted with their multiplicity, e.g. one per
    feature), so a source on another naming never yields an empty or partial store unnoticed."""
    have = layout.chroms()
    names = list(names)
    out = {}
    for n in set(names):
        for cand in (prefix + n, n, "chr" + n, n.removeprefix("chr"), {"MT": "chrM", "chrM": "MT", "M": "chrM"}.get(n)):
            if cand and cand in have:
                out[n] = cand
                break
    share = sum(n in out for n in names) / max(len(names), 1)
    if share < min_share:
        raise ValueError(f"only {share:.1%} of the source's chromosome names match the layout {layout.name} "
                         f"(source e.g. {sorted(set(names))[:4]}, layout e.g. {sorted(have)[:4]})")
    return out
