"""genome-train ``--short-sources pack``: training rows for sources shorter than the window, without padding.

``sample()`` used to place every drawn contig in a row of the full window and fill the rest with N. At a 524 kb
window a mitochondrial genome (~16.5 kb) is then one genome and 97 % padding, and a species whose assembly is
mostly short contigs trains on little else. With ``pack``, a window whose contig is shorter than the window becomes
a *pack*: several shorter rows that together hold about one window of real bases.

The row-length rule (``row_ladder``): the allowed row lengths for a window ``W`` are ``W * m / 8 / 2**k`` for
``m`` in 8, 7, 6, 5 and ``k`` = 0, 1, 2, ..., kept when (a) the length is a whole number of ``unit`` bases (the
model's ``2**n_down``) and at least ``MIN_UNITS`` of them, and (b) the ``W // length`` rows that fit one window
fill at least ``FILL_MIN`` of it. For W = 524,288 that is 524288 x1, 458752 x1, 262144 x2, 229376 x2, 163840 x3,
131072 x4, ..., 16384 x32, 14336 x36, 12288 x42, 10240 x51, 8192 x64, ... A contig of length ``L < W`` uses the
longest allowed length ``R <= L`` (so a row reads at least 4/7 of its contig, usually more than 7/8, and a pack
holds at least 7/8 of a window's bases, none of them padding); the pack's other rows are drawn the way the first
contig was, from the contigs of the same source whose length selects the same ``R`` (its *bucket*). A linear contig
is cropped at a uniform offset; a circular one is read from a uniform origin across its end, never twice round.
Only a sequence shorter than the shortest allowed length is still padded (to that length).

Nothing here touches a random stream unless a short contig is drawn, so a run in which none is (every phase whose
window is no longer than the shortest contig) draws exactly what ``pad`` draws.
"""
from __future__ import annotations

import numpy as np

FILL_MIN = 7 / 8          # the rows of one pack hold at least this share of a window's bases
MIN_UNITS = 8             # no row shorter than this many core steps (2**n_down bases each)
EIGHTHS = (8, 7, 6, 5)    # row lengths per halving of the window, in eighths

# Tests and simulations set this to a list; sample() then appends one tuple per training row:
# (source label, window, row length, real bases, rows in this row's pack, 1 for the pack's first row else 0).
TRACE: list | None = None


def row_ladder(window: int, unit: int) -> list[tuple[int, int]]:
    """``[(row length, rows per window)]``, longest first; the first entry is always ``(window, 1)``."""
    floor = min(MIN_UNITS * unit, window)
    out, k = [], 0
    while (window >> k) >= floor:
        for m in EIGHTHS:
            if (window * m) % (8 << k):
                continue
            r = window * m // (8 << k)
            n = window // r
            if r % unit == 0 and r >= floor and n * r >= FILL_MIN * window:
                out.append((r, n))
        k += 1
    return out


class Bucket:
    """The contigs of one source whose length selects one row length: ``draw`` picks one the way the source's
    table is drawn (by ``p`` renormalised over the bucket, or uniformly when the table is drawn uniformly)."""

    def __init__(self, row: int, n: int, members: np.ndarray, p: np.ndarray | None) -> None:
        self.row, self.n, self.members = row, n, members
        self.p = None if p is None else p[members] / p[members].sum()

    def draw(self, gen: np.random.Generator) -> int:
        if self.p is None:
            return int(self.members[int(gen.integers(0, len(self.members)))])
        return int(self.members[int(gen.choice(len(self.members), p=self.p))])


class Buckets:
    """One source's contig table cut into length buckets by a window's row ladder."""

    def __init__(self, lengths: np.ndarray, p: np.ndarray | None, ladder: list[tuple[int, int]]) -> None:
        self.ladder = ladder
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.p = None if p is None else np.asarray(p, dtype=np.float64)
        self._rows_up = np.array([r for r, _ in ladder][::-1], dtype=np.int64)       # ascending, for searchsorted
        self.rung = self.rung_of(self.lengths)
        self._cells: dict[int, Bucket] = {}

    def rung_of(self, length):
        """Index into the ladder of the longest row a contig of ``length`` fills (the shortest row when it fills
        none: that contig is padded, as before, but only to the shortest row)."""
        j = np.searchsorted(self._rows_up, length, side="right") - 1                # last rung <= length
        return len(self.ladder) - 1 - np.maximum(j, 0)

    def cell(self, length: int) -> Bucket:
        k = int(self.rung_of(length))
        if k not in self._cells:
            row, n = self.ladder[k]
            self._cells[k] = Bucket(row, n, np.flatnonzero(self.rung == k), self.p)
        return self._cells[k]


class Tally:
    """What the training sampler drew since the last report, per source: windows (a pack counts as one), rows,
    real and padded bases. Bookkeeping only (no random number), so it runs for ``pad`` too and shows its padding."""

    def __init__(self) -> None:
        self.by: dict[str, list[float]] = {}

    def add(self, label: str, window: int, row: int, real: int, pack_n: int, lead: bool) -> None:
        if TRACE is not None:
            TRACE.append((label, window, row, real, pack_n, int(lead)))
        t = self.by.setdefault(label.split(":")[0], [0, 0, 0, 0, 0])      # windows, rows, real, row bases, budget
        t[0] += int(lead)
        t[1] += 1
        t[2] += real
        t[3] += row
        t[4] += window if lead else 0

    def line(self) -> str:
        total = sum(t[0] for t in self.by.values()) or 1
        parts = [f"{k} {100 * t[0] / total:.1f}% ({t[1] / max(t[0], 1):.1f} rows/window, real bases "
                 f"{100 * t[2] / max(t[4], 1):.1f}% of the window, padding {100 * (1 - t[2] / max(t[3], 1)):.1f}% "
                 f"of its rows)" for k, t in sorted(self.by.items())]
        self.by = {}
        return f"{total:,} windows: " + "; ".join(parts)
