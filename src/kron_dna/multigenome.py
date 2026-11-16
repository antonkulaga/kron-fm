"""Further genomes for KronGenome's masked-base pre-training (``genome-train --genome``).

Each further genome is a directory prepared like the primary one (``bases.u8`` and ``contigs.parquet`` with columns
chrom, start, end, offset). On the command line it is one repeatable option::

    --genome NAME:SHARE[:CONTEXT[:HELD]]

``SHARE`` is the fraction of ALL training windows drawn from that genome (the primary ``--name`` genome keeps what
the further genomes, ``--mito`` and ``--ntv3-extra`` leave). ``CONTEXT`` is the context id the model receives for its
windows (default 0 = nuclear; must fit the model's context table). ``HELD`` names its held-out contigs by chromosome,
joined with ``+`` (``chr8+chr9``), or ``none`` for no held-out set; omitted, the genome's ``held_out`` boolean column
in contigs.parquet is used and must exist. Windows are drawn from the training contigs by length (every base equally)
and a contig shorter than the window is taken whole and padded, exactly as for the primary genome.

Everything here is pure bookkeeping (no torch, no GPU) so it can be unit-tested; ``genome_train`` draws the windows.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl

CONTIG_COLUMNS = ("chrom", "start", "end", "offset")


class GenomeSpecError(ValueError):
    """A --genome option that cannot be used (genome_train turns it into a typer.BadParameter)."""


@dataclass(frozen=True)
class GenomeSpec:
    name: str
    share: float
    context: int = 0
    held_out: tuple[str, ...] | None = None        # None: the contigs.parquet 'held_out' column; (): none held out


@dataclass
class FurtherGenome:
    """A loaded further genome: its bases, training and held-out contig tables (each with a ``len`` column) and the
    contig-by-length draw probabilities of each."""
    spec: GenomeSpec
    bases: np.ndarray
    train: pl.DataFrame
    val: pl.DataFrame
    train_p: np.ndarray
    val_p: np.ndarray

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def share(self) -> float:
        return self.spec.share

    @property
    def context(self) -> int:
        return self.spec.context


def parse_genome_spec(text: str) -> GenomeSpec:
    """``NAME:SHARE[:CONTEXT[:HELD]]`` -> GenomeSpec (field checks only; files are checked by load_further_genomes)."""
    parts = text.strip().split(":")
    if not 2 <= len(parts) <= 4:
        raise GenomeSpecError(f"--genome {text!r}: expected NAME:SHARE[:CONTEXT[:HELD1+HELD2|none]]")
    name = parts[0].strip()
    if not name or "/" in name or name in (".", ".."):
        raise GenomeSpecError(f"--genome {text!r}: NAME must be a directory name under data/interim/genome/")
    try:
        share = float(parts[1])
    except ValueError:
        raise GenomeSpecError(f"--genome {text!r}: SHARE {parts[1]!r} is not a number") from None
    if not (0.0 < share < 1.0) or not np.isfinite(share):
        raise GenomeSpecError(f"--genome {text!r}: SHARE must be in (0, 1), the fraction of training windows")
    context = 0
    if len(parts) >= 3 and parts[2].strip() != "":
        try:
            context = int(parts[2])
        except ValueError:
            raise GenomeSpecError(f"--genome {text!r}: CONTEXT {parts[2]!r} is not an integer") from None
        if context < 0:
            raise GenomeSpecError(f"--genome {text!r}: CONTEXT must be >= 0 (0 = nuclear)")
    held: tuple[str, ...] | None = None
    if len(parts) == 4 and parts[3].strip() != "":
        h = parts[3].strip()
        if h.lower() == "none":
            held = ()
        else:
            held = tuple(x.strip() for x in h.split("+"))
            if any(not x for x in held):
                raise GenomeSpecError(f"--genome {text!r}: empty name in HELD (join names with '+')")
            if len(set(held)) != len(held):
                raise GenomeSpecError(f"--genome {text!r}: HELD names a contig twice")
    return GenomeSpec(name=name, share=share, context=context, held_out=held)


def parse_genome_specs(texts: list[str], primary: str = "", other_share: float = 0.0) -> list[GenomeSpec]:
    """Parse every --genome and check them together: distinct names, none the primary genome, and the further
    genomes' shares plus ``other_share`` (the --mito share) below 1 so the primary genome keeps some windows."""
    specs = [parse_genome_spec(t) for t in texts if t.strip()]
    names = [s.name for s in specs]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise GenomeSpecError(f"--genome lists {', '.join(dup)} more than once")
    if primary and primary in names:
        raise GenomeSpecError(f"--genome {primary}: that is the primary genome (--name); it keeps the remaining share")
    total = sum(s.share for s in specs) + other_share
    if specs and total >= 1.0:
        raise GenomeSpecError(f"--genome shares sum to {sum(s.share for s in specs):g}"
                              + (f" (+ --mito-share {other_share:g})" if other_share else "")
                              + "; they must stay below 1 so the primary genome keeps a share")
    return specs


def contig_table(path: Path) -> pl.DataFrame:
    """contigs.parquet with a ``len`` column, checked for the hg38 layout."""
    t = pl.read_parquet(path)
    missing = [c for c in CONTIG_COLUMNS if c not in t.columns]
    if missing:
        raise GenomeSpecError(f"{path} lacks columns {missing} (expected the hg38 layout {list(CONTIG_COLUMNS)})")
    return t.with_columns((pl.col("end") - pl.col("start")).alias("len"))


def load_genome(spec: GenomeSpec, src: Path) -> FurtherGenome:
    """Open one prepared genome and split its contigs into training and held-out sets."""
    if not src.is_dir():
        raise GenomeSpecError(f"--genome {spec.name}: no prepared genome at {src}")
    for f in ("bases.u8", "contigs.parquet"):
        if not (src / f).is_file():
            raise GenomeSpecError(f"--genome {spec.name}: {src / f} is missing (prepare it like hg38)")
    table = contig_table(src / "contigs.parquet")
    if table.height == 0:
        raise GenomeSpecError(f"--genome {spec.name}: {src / 'contigs.parquet'} has no contigs")
    if int(table["len"].min()) <= 0:
        raise GenomeSpecError(f"--genome {spec.name}: a contig has end <= start")
    bases = np.memmap(src / "bases.u8", dtype=np.uint8, mode="r")
    need = int((table["offset"] + table["len"]).max())
    if len(bases) < need:
        raise GenomeSpecError(f"--genome {spec.name}: bases.u8 holds {len(bases):,} bases but contigs.parquet "
                              f"reaches {need:,} (still being written?)")
    if spec.held_out is None:
        if "held_out" not in table.columns:
            raise GenomeSpecError(f"--genome {spec.name}: name its held-out contigs (NAME:SHARE:CONTEXT:chr8+chr9) "
                                  f"or 'none'; its contigs.parquet has no 'held_out' column")
        if table.schema["held_out"] != pl.Boolean:
            raise GenomeSpecError(f"--genome {spec.name}: the 'held_out' column must be boolean, "
                                  f"not {table.schema['held_out']}")
        is_held = pl.col("held_out").fill_null(False)
    else:
        unknown = sorted(set(spec.held_out) - set(table["chrom"].to_list()))
        if unknown:
            raise GenomeSpecError(f"--genome {spec.name}: held-out {unknown} match no contig "
                                  f"(have e.g. {table['chrom'].unique(maintain_order=True).to_list()[:8]})")
        is_held = pl.col("chrom").is_in(list(spec.held_out))
    val = table.filter(is_held)
    train = table.filter(~is_held)
    if train.height == 0:
        raise GenomeSpecError(f"--genome {spec.name}: every contig is held out, nothing left to train on")
    return FurtherGenome(spec=spec, bases=bases, train=train, val=val,
                         train_p=by_length(train), val_p=by_length(val))


def by_length(table: pl.DataFrame) -> np.ndarray:
    """Draw probabilities of a contig table's rows by length (every base equally), as genome_train's sample()."""
    if table.height == 0:
        return np.zeros(0)
    return table["len"].to_numpy() / table["len"].sum()


def load_further_genomes(specs: list[GenomeSpec], resolve: Callable[[str], Path]) -> list[FurtherGenome]:
    return [load_genome(s, resolve(s.name)) for s in specs]


def check_contexts(genomes: list[FurtherGenome], n_contexts: int, mito_codes: set[int] | None = None) -> None:
    """Every context id must index the model's context table, and none may be a genetic code table that --mito
    windows already use (those ids would conflate a nuclear genome with organelles)."""
    for g in genomes:
        c = g.context
        if c == 0:
            continue
        if n_contexts <= 0:
            raise GenomeSpecError(f"--genome {g.name}: context {c} needs a context table, but this model has none "
                                  f"(it gets one with --mito, 34 entries, or with --n-contexts N; ids in genome.CONTEXT_RANGES)")
        if c >= n_contexts:
            raise GenomeSpecError(f"--genome {g.name}: context {c} is beyond the model's context table "
                                  f"({n_contexts} entries, ids 0..{n_contexts - 1})")
        if mito_codes and c in mito_codes:
            raise GenomeSpecError(f"--genome {g.name}: context {c} is a genetic code table that --mito windows use")


def cumulative_shares(genomes: list[FurtherGenome]) -> np.ndarray:
    return np.cumsum([g.share for g in genomes]) if genomes else np.zeros(0)


def pick_genome(u: float, cum: np.ndarray) -> int | None:
    """The further genome a uniform draw ``u`` lands in (genome k owns [cum[k-1], cum[k])), else None (the window
    goes on to the --mito / --ntv3-extra / primary draws). One uniform draw decides, so no second one is spent."""
    if len(cum) == 0 or u >= cum[-1]:
        return None
    return int(np.searchsorted(cum, u, side="right"))


def window_shares(genome_shares: list[float], mito_share: float = 0.0, extra_share: float = 0.0,
                  extra_p: list[float] | None = None) -> dict[str, float]:
    """The expected fraction of training windows from each source, as genome_train draws them: further genomes
    first (absolute shares), then --mito with ``mito_share / (1 - S)`` among the rest (so its share stays absolute),
    then --ntv3-extra with ``extra_share`` among what is left (its 3c90c8c meaning), then the primary genome."""
    s = float(sum(genome_shares))
    rest = 1.0 - s - mito_share
    out = {f"genome{k}": g for k, g in enumerate(genome_shares)}
    out["mito"] = mito_share
    for k, p in enumerate(extra_p or []):
        out[f"extra{k}"] = rest * extra_share * p
    out["primary"] = rest * (1.0 - (extra_share if extra_p else 0.0))
    return out
