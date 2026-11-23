"""bigWig signal for every track store: calibration (the encoding's scale and the non-zero mean, both on the training
split only) and one scan of a track over a layout that counts NaN, negative and unobserved stretches."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from kron_dna.prep.transforms import encode_u8

CHUNK = 1 << 22


def sample_windows(rows, rng, n, width=100_000, have=None, split=None, tries=200_000):
    """``n`` windows of ``width`` bases at random on contigs the bigWig lists; with ``split``, only windows whose bases
    are at least 90 % training split (code 1), so held-out chromosomes never calibrate an encoding."""
    out = []
    for _ in range(tries):
        if len(out) >= n:
            break
        c, s, e, off = rows[int(rng.integers(0, len(rows)))]
        if e - s < width or (have is not None and c not in have):
            if all(r[2] - r[1] < width for r in rows):
                break
            continue
        a = int(rng.integers(s, e - width))
        g = off + (a - s)
        if split is not None and float((np.asarray(split[g:g + width]) == 1).mean()) < 0.9:
            continue
        out.append((c, a, g))
    return out


def scale_of(bw, rows, percentile: float = 99.99, split=None, n: int = 200) -> float:
    """The encoding's scale: log1p of the ``percentile`` of non-zero signal over ``n`` sampled training windows."""
    have = bw.chroms()
    vals = [np.nan_to_num(bw.values(c, a, a + 100_000, numpy=True))
            for c, a, _ in sample_windows(rows, np.random.default_rng(0), n, have=have, split=split)]
    v = np.concatenate([x[x > 0] for x in vals]) if vals else np.array([1.0])
    return float(max(np.log1p(np.percentile(v, percentile)) if v.size else 1.0, 1e-3))


def nonzero_mean(bw, rows, scale: float, split, n: int = 400) -> float:
    """The mean decoded signal over non-zero training bases of ``n`` sampled windows (the target's divisor)."""
    tot = cnt = 0.0
    for c, a, g in sample_windows(rows, np.random.default_rng(1), n, have=bw.chroms(), split=split):
        x = np.nan_to_num(bw.values(c, a, a + 100_000, numpy=True))
        u = encode_u8(x, scale)
        d = np.expm1(u.astype(np.float64) / 255.0 * scale)
        m = (np.asarray(split[g:g + 100_000]) == 1) & (u > 0)
        tot += float(d[m].sum())
        cnt += float(m.sum())
    return tot / cnt if cnt else 1.0


@dataclass
class ScanStats:
    nan: int = 0
    seen: int = 0
    negative: int = 0
    unobserved: list = field(default_factory=list)

    def info(self) -> dict:
        return {"nan_share": self.nan / max(self.seen, 1), "observed_bases": self.seen,
                "negative_values": self.negative, "unobserved_contigs": self.unobserved}


def scan(bw, rows, stats: ScanStats, chunk: int = CHUNK):
    """Every contig of the layout the bigWig covers, in chunks: yields (global offset, values) with NaN as 0 and
    counts NaN, negative values and the contigs the bigWig does not cover (recorded in ``stats``, left to the store's
    zeros)."""
    have = bw.chroms()
    for c, s, e, off in rows:
        if c not in have or have[c] < e:
            stats.unobserved.append(f"{c}:{s}-{e}")
            continue
        for a in range(s, e, chunk):
            z = min(a + chunk, e)
            x = bw.values(c, a, z, numpy=True)
            bad = ~np.isfinite(x)
            stats.nan += int(bad.sum())
            stats.seen += z - a
            x = np.where(bad, 0.0, x)
            stats.negative += int((x < 0).sum())
            yield off + (a - s), x
