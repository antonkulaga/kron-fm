"""ClinVar SNVs anywhere in the genome, scored by KronGenome: coding, splice, UTR, regulatory and the rest.

Each reviewed pathogenic or benign SNV (:func:`kron_dna.variants.read_clinvar`) is placed on the contig layout of
``genome-prepare`` and given a region from the stage-2 label tracks at its base, in this order: coding (GENCODE CDS),
splice site (the 2 intron bases at an exon end), near a splice site (within ``near`` bases of one), UTR exon (exon,
not CDS), regulatory element (Ensembl promoter, enhancer, CTCF or open chromatin) and otherwise intron or intergenic.

The score is the masked marginal at the variant, ``log p(alt) - log p(ref)``, in a window of the genome around it;
lower should mean more damaging, so AUROC is of ``-score`` with pathogenic positive (:func:`kron_dna.variants.auroc`).
Windows tile each contig by half a window, and a variant is read from the tile whose middle half holds it, so it has
at least a quarter window of context on each side (less only at a contig end). All variants of one tile are masked
together, a small share of the bases next to the 15 % the model trained with.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import torch

from kron_dna import genome as G

REGIONS = ("coding", "splice site", "near splice site", "UTR exon", "regulatory element", "intron or intergenic")


def locate(index: pl.DataFrame, chrom: np.ndarray, pos0: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(global offset, contig offset, contig length)`` of 0-based positions in the contig layout; -1 where the
    position falls in no contig (a gap, a dropped chromosome)."""
    g = np.full(len(pos0), -1, dtype=np.int64)
    c_off, c_len = g.copy(), g.copy()
    for (name,), part in index.group_by("chrom"):
        part = part.sort("start")
        starts, ends, offs = (part[c].to_numpy() for c in ("start", "end", "offset"))
        rows = np.nonzero(chrom == name)[0]
        if not len(rows):
            continue
        k = np.searchsorted(starts, pos0[rows], side="right") - 1
        ok = (k >= 0) & (pos0[rows] < ends[np.maximum(k, 0)])
        r, k = rows[ok], k[ok]
        g[r] = offs[k] + pos0[r] - starts[k]
        c_off[r], c_len[r] = offs[k], ends[k] - starts[k]
    return g, c_off, c_len


def region_of(labels: np.ndarray, g: np.ndarray, near: int = 10) -> np.ndarray:
    """The region (one of ``REGIONS``) of each global offset ``g``, from the ``genome-labels`` bits."""
    bit = lambda name: 1 << G.LABEL_BITS[name]
    here = np.asarray(labels[g]).astype(np.int64)
    splice = bit("donor") | bit("acceptor")
    around = np.zeros(len(g), dtype=bool)
    for d in range(-near, near + 1):
        at = np.clip(g + d, 0, len(labels) - 1)
        around |= (np.asarray(labels[at]) & splice) > 0
    regulatory = bit("promoter") | bit("enhancer") | bit("CTCF_binding_site") | bit("open_chromatin_region")
    return np.select([here & bit("CDS") > 0, here & splice > 0, around, here & bit("exon") > 0, here & regulatory > 0],
                     list(REGIONS[:5]), REGIONS[5]).astype(object)


def splice_distance(labels: np.ndarray, g: np.ndarray, block: int = 1 << 27) -> np.ndarray:
    """Bases from each global offset ``g`` to the nearest splice-site base (GENCODE donor or acceptor) in the layout:
    a model-free baseline, since pathogenic non-coding variants cluster near exons. Contig ends are ignored."""
    mask = (1 << G.LABEL_BITS["donor"]) | (1 << G.LABEL_BITS["acceptor"])
    sites = np.concatenate([np.nonzero(np.asarray(labels[a:a + block]) & mask)[0] + a
                            for a in range(0, len(labels), block)])
    k = np.searchsorted(sites, g)
    left = np.abs(g - sites[np.clip(k - 1, 0, len(sites) - 1)])
    right = np.abs(sites[np.clip(k, 0, len(sites) - 1)] - g)
    return np.minimum(left, right).astype(float)


def tile_of(g: np.ndarray, c_off: np.ndarray, c_len: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """``(tile start, tile length)`` in the layout for each variant: tiles every ``window // 2`` along its contig,
    the one whose middle half holds it; a contig shorter than ``window`` is one tile."""
    half = window // 2
    local = g - c_off
    k = np.maximum(local - window // 4, 0) // half
    start = np.minimum(k * half, np.maximum(c_len - window, 0))
    return c_off + start, np.minimum(c_len, window)


@torch.no_grad()
def masked_marginals(model: G.KronGenome, bases: np.ndarray, g: np.ndarray, ref: np.ndarray, alt: np.ndarray,
                     tile_start: np.ndarray, tile_len: np.ndarray, window: int, device: torch.device,
                     batch_size: int = 2, rc: bool = False) -> np.ndarray:
    """``log p(alt) - log p(ref)`` at each variant (``ref``/``alt`` as base codes 0..3), all variants of a tile masked
    in one pass; tiles shorter than ``window`` are padded with N as a segment of their own."""
    unit = 2 ** model.config.n_down
    out = np.full(len(g), np.nan)
    keys = np.stack([tile_start, tile_len], 1)
    tiles, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    order = np.argsort(inverse, kind="stable")
    bounds = np.searchsorted(inverse[order], np.arange(len(tiles) + 1))
    autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    for b0 in range(0, len(tiles), batch_size):
        chunk = range(b0, min(b0 + batch_size, len(tiles)))
        seqs, segs, rows, cols = [], [], [], []
        for j, t in enumerate(chunk):
            start, take = (int(v) for v in tiles[t])
            codes = np.full(window, 4, dtype=np.int64)
            codes[:take] = bases[start:start + take]
            members = order[bounds[t]:bounds[t + 1]]
            pos = g[members] - start
            if rc:                                       # the other strand: the tile reversed and complemented
                codes[:take] = G.reverse_complement(codes[:take].astype(np.uint8))
                pos = take - 1 - pos
            codes[pos] = G.MASK_CHANNEL
            seg = np.zeros(window // unit, dtype=np.int64)
            seg[(take + unit - 1) // unit:] = 1
            seqs.append(codes)
            segs.append(seg)
            rows.append(members)
            cols.append(np.full(len(members), j))
        seq = torch.from_numpy(np.stack(seqs)).to(device)
        with autocast:
            logits, _, _ = model(seq, torch.from_numpy(np.stack(segs)).to(device))
        logp = torch.log_softmax(logits.float(), -1).cpu().numpy()
        members, which = np.concatenate(rows), np.concatenate(cols)
        at = g[members] - tiles[np.array(list(chunk))[which], 0]
        r, a = ref[members], alt[members]
        if rc:
            at = tiles[np.array(list(chunk))[which], 1] - 1 - at
            r, a = 3 - r, 3 - a
        out[members] = logp[which, at, a] - logp[which, at, r]
    return out


@torch.no_grad()
def track_deltas(model: G.KronGenome, bases: np.ndarray, g: np.ndarray, ref: np.ndarray, alt: np.ndarray,
                 c_off: np.ndarray, c_len: np.ndarray, window: int, device: torch.device, batch_size: int = 8,
                 flank: int = 50, rc: bool = False) -> np.ndarray:
    """``[n, n_tracks]``: for each variant, the largest ``|p(alt) - p(ref)|`` of each track within ``flank`` bases of
    it, from two unmasked passes (reference and alternative) over a ``window`` centred on it (clamped to its contig;
    a shorter contig is padded with N as a segment of its own)."""
    unit = 2 ** model.config.n_down
    out = np.zeros((len(g), model.config.n_tracks), dtype=np.float32)
    autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    local = g - c_off
    start = np.clip(local - window // 2, 0, np.maximum(c_len - window, 0))
    take = np.minimum(c_len, window)
    for b0 in range(0, len(g), batch_size):
        idx = np.arange(b0, min(b0 + batch_size, len(g)))
        seqs, segs = [], []
        for i in idx:
            codes = np.full(window, 4, dtype=np.int64)
            codes[:take[i]] = bases[c_off[i] + start[i]:c_off[i] + start[i] + take[i]]
            seg = np.zeros(window // unit, dtype=np.int64)
            seg[(take[i] + unit - 1) // unit:] = 1
            mutant = codes.copy()
            mutant[local[i] - start[i]] = alt[i]
            if rc:                                       # the other strand; the tracks are not strand-specific
                codes[:take[i]] = G.reverse_complement(codes[:take[i]].astype(np.uint8))
                mutant[:take[i]] = G.reverse_complement(mutant[:take[i]].astype(np.uint8))
            seqs += [codes, mutant]
            segs += [seg, seg]
        with autocast:
            _, _, tracks = model(torch.from_numpy(np.stack(seqs)).to(device), torch.from_numpy(np.stack(segs)).to(device))
        prob = torch.sigmoid(tracks.float())
        for j, i in enumerate(idx):
            at = int(local[i] - start[i])
            if rc:
                at = int(take[i]) - 1 - at
            lo, hi = max(0, at - flank), min(int(take[i]), at + flank + 1)
            out[i] = (prob[2 * j + 1, lo:hi] - prob[2 * j, lo:hi]).abs().amax(0).cpu().numpy()
    return out


def place(variants: pl.DataFrame, index: pl.DataFrame, bases: np.ndarray, labels: np.ndarray,
          held_chroms: set[str], near: int = 10) -> tuple[pl.DataFrame, dict[str, int]]:
    """``read_clinvar`` rows on the contig layout: adds ``g``, ``c_off``, ``c_len``, base codes ``ref_code`` and
    ``alt_code``, ``region`` and ``held_out`` (chromosome held out of training); drops variants outside every contig
    and those whose reference base disagrees with the genome, counting both."""
    chrom = variants["chrom"].cast(pl.Utf8).to_numpy().astype(str)
    chrom = np.where(np.char.startswith(chrom, "chr"), chrom, np.char.add("chr", chrom))
    pos0 = variants["pos"].to_numpy().astype(np.int64) - 1
    g, c_off, c_len = locate(index, chrom, pos0)
    code = {b: i for i, b in enumerate("ACGT")}
    ref = np.array([code.get(b, 4) for b in variants["ref"].to_list()])
    alt = np.array([code.get(b, 4) for b in variants["alt"].to_list()])
    placed = g >= 0
    match = placed & (ref < 4) & (alt < 4)
    match[match] = np.asarray(bases[g[match]]) == ref[match]
    dropped = {"outside the contigs": int((~placed).sum()), "reference mismatch": int((placed & ~match).sum())}
    keep = np.nonzero(match)[0]
    table = variants[keep].with_columns(
        chrom=pl.Series(chrom[keep]), g=pl.Series(g[keep]), c_off=pl.Series(c_off[keep]),
        c_len=pl.Series(c_len[keep]), ref_code=pl.Series(ref[keep]), alt_code=pl.Series(alt[keep]),
        region=pl.Series(region_of(labels, g[keep], near).astype(str)),
        held_out=pl.Series(np.isin(chrom[keep], list(held_chroms))))
    return table, dropped


def summarize(table: pl.DataFrame, scores: dict[str, np.ndarray], n_boot: int = 1000) -> list[dict]:
    """AUROC of every score per region (and all non-coding together), on training and held-out chromosomes, with a
    95 % bootstrap interval as ``<score>_ci``."""
    from kron_dna.variants import auroc, bootstrap_ci

    label = table["label"].to_numpy()
    region = table["region"].to_numpy()
    held = table["held_out"].to_numpy()
    out = []
    for name in (*REGIONS, "all non-coding"):
        in_region = region != "coding" if name == "all non-coding" else region == name
        for split, keep in (("training chromosomes", ~held), ("held-out chromosomes", held)):
            m = in_region & keep
            row = {"region": name, "chromosomes": split, "pathogenic": int((label[m] == 1).sum()),
                   "benign": int((label[m] == 0).sum())}
            for run, sc in scores.items():
                row[run] = auroc(sc[m], label[m])
                if n_boot:
                    row[f"{run}_ci"] = bootstrap_ci(auroc, sc[m], label[m], n=n_boot)
            out.append(row)
    return out
