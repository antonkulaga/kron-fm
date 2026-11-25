"""bigWig tracks straight to a pre-binned store (128-bp cells), without the per-base store in between.

    uv run --with pyBigWig python scripts/kron_dna/prep_bigwig_binned.py SELECT.tsv BIGWIG_DIR OUT_DIR \
        [--bin 128] [--scale-percentile 99.99] [--workers 4] [--convert-only] [--provenance provenance.jsonl]

The per-base route (prep_bigwig_tracks.py, then bin_track_store.py) writes T bytes per base first: 1.45 TB for 492
ChIP-seq tracks, only to average it away. Here each track is read once and reduced on the way to two float16 cell
means per track, the same quantities as bin_track_store.py writes:

- ``target``: the mean over the cell's bases of the per-base training target (``xheads.targets_of`` of the u8-encoded
  signal: decoded, over the track's non-zero mean, the 0.75 power for RNA-seq, the soft clip above 10);
- ``log1p``: the mean over the cell's bases of log1p of the u8-encoded signal.

The encoding is the per-base stores' (u8 of log1p over the track's scale, the scale its ``--scale-percentile`` of
non-zero signal over 200 sampled 100 kb windows). The non-zero mean (the target's divisor) is estimated before the
full pass, from 400 sampled 100 kb windows of the prepared genome's training split (``ntv3/split.u8`` == 1); the
per-base stores compute it after, over 1 base in 64 of that split. A NaN in a listed contig counts as zero (its share
is recorded); a contig the bigWig does not list is unobserved and recorded.

Cells are ``[k * BIN, (k + 1) * BIN)`` of the global base layout, as bin_track_store.py writes them, so the output
is read by ``xheads.BinnedTrackStore`` like any pre-binned store. Per-track results go to ``OUT_DIR/cells/`` (a rerun
skips finished tracks, every file written under a temporary name and renamed); ``--convert-only`` stops there, so
several jobs can convert parts of one selection; a last run on the whole list assembles ``target.f16``, ``log1p.f16``
and ``binned.json``, then checks random cells against the bigWig read again.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import socket
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import polars as pl
import typer

from kron_core.paths import interim_dir

CHUNK = 1 << 23
PART = f".part.{socket.gethostname()}.{os.getpid()}"
_G: dict = {}


def one(fid: str) -> dict:
    import pyBigWig

    from kron_dna.prep.bigwig import ScanStats, Signed, nonzero_mean, scale_of, scan
    from kron_dna.prep.transforms import decode_u8, encode_u8, targets_np

    rows, total, src, out, b = _G["rows"], _G["total"], _G["src"], _G["out"], _G["bin"]
    side = out / "cells" / f"{fid}.json"
    if side.exists():
        return json.loads(side.read_text())
    bw = Signed(pyBigWig.open(str(src / f"{fid}.bigWig")))      # minus-strand files may be negative
    split = _G["split"]
    sc = scale_of(bw, rows, _G["percentile"], split)        # the shared calibration (kron_dna.prep.bigwig)
    nz = nonzero_mean(bw, rows, sc, split)
    rna = _G["rna"][fid]
    cells = -(-total // b)
    s_t = np.zeros(cells, dtype=np.float64)
    s_l = np.zeros(cells, dtype=np.float64)
    stats = ScanStats()
    for g0, x in scan(bw, rows, stats, CHUNK):
        u = encode_u8(x, sc)
        lg = u.astype(np.float32) / 255.0 * sc
        t = targets_np(decode_u8(u, sc) / nz, rna)
        k = (g0 + np.arange(len(x))) // b                                 # each base's global cell
        lo = int(k[0])
        s_t[lo:lo + int(k[-1]) - lo + 1] += np.bincount(k - lo, weights=t)
        s_l[lo:lo + int(k[-1]) - lo + 1] += np.bincount(k - lo, weights=lg)
    bw.close()
    n = np.full(cells, float(b))
    n[-1] = total - (cells - 1) * b
    for name, arr in (("target", s_t / n), ("log1p", s_l / n)):
        dst = out / "cells" / f"{fid}.{name}.f16"
        tmp = str(dst) + PART
        arr.astype(np.float16).tofile(tmp)
        os.replace(tmp, dst)
    info = {"file": fid, "log1p_scale": sc, "nonzero_mean": nz, "sign": bw.sign, **stats.info()}
    Path(str(side) + PART).write_text(json.dumps(info))
    os.replace(str(side) + PART, side)
    return info


def _init(g: dict) -> None:
    _G.update(g)


def main(select: Path, bigwigs: Path, out: Path, name: str = "hg38", bin: int = 128, workers: int = 4,
         scale_percentile: float = typer.Option(99.99, help="percentile of non-zero signal the encoding clips at"),
         convert_only: bool = typer.Option(False, help="write the per-track cells only, no store"),
         provenance: Path = typer.Option(None), check_cells: int = 2000,
         split: Path = typer.Option(None, help="uint8 per base of the layout, 1 = training (the non-zero mean is "
                                                "sampled there); default: the genome's ntv3/split.u8")) -> None:
    import pyBigWig

    from kron_dna.prep.bigwig import Signed

    genome = interim_dir("genome", name)
    if genome.resolve() in [out.resolve(), *out.resolve().parents]:
        raise typer.BadParameter("OUT_DIR lies inside the prepared genome; choose a new path")
    idx = pl.read_parquet(genome / "contigs.parquet").sort("offset")
    rows = [(r["chrom"], r["start"], r["end"], r["offset"]) for r in idx.iter_rows(named=True)]
    total = int((idx["end"] - idx["start"]).sum())
    with open(select) as f:
        sel = list(csv.DictReader(f, delimiter="\t"))
    files = [r["file"] for r in sel]
    ids = [r.get("channel") or r["file"] for r in sel]
    assert len(set(files)) == len(files) and len(set(ids)) == len(ids), "a file or channel is listed twice"
    (out / "cells").mkdir(parents=True, exist_ok=True)
    g = dict(rows=rows, total=total, src=bigwigs, out=out, bin=bin, percentile=scale_percentile,
             split=np.memmap(split or genome / "ntv3" / "split.u8", dtype=np.uint8, mode="r"),
             rna={r["file"]: "RNA-seq" in r["assay"] for r in sel})
    infos = {}
    with Pool(workers, initializer=_init, initargs=(g,)) as p:
        for info in p.imap_unordered(one, files):
            infos[info["file"]] = info
            typer.echo(f"  {info['file']} scale={info['log1p_scale']:.3f} nz={info['nonzero_mean']:.3f} "
                       f"nan={info['nan_share']:.3f} unobserved={len(info['unobserved_contigs'])}")
    if convert_only:
        typer.echo(f"converted {len(infos)} tracks into {out / 'cells'}; no store assembled (--convert-only)")
        return
    t, cells = len(files), -(-total // bin)
    for kind in ("target", "log1p"):
        dst = out / f"{kind}.f16"
        if dst.exists() and dst.stat().st_size == cells * t * 2:
            continue
        mm = np.memmap(str(dst) + PART, dtype=np.float16, mode="w+", shape=(cells, t))
        src_ = [np.memmap(out / "cells" / f"{fid}.{kind}.f16", dtype=np.float16, mode="r") for fid in files]
        step = 1 << 20
        for a in range(0, cells, step):
            z = min(a + step, cells)
            mm[a:z] = np.stack([s_[a:z] for s_ in src_], 1)
        mm.flush()
        del mm
        os.replace(str(dst) + PART, dst)
    # check: random cells of each track against the bigWig read again (log1p of the raw signal, unencoded)
    lg = np.memmap(out / "log1p.f16", dtype=np.float16, mode="r", shape=(cells, t))
    rng = np.random.default_rng(2)
    pick = [r for r in rows if r[2] - r[1] > 50 * bin]
    pear = np.zeros(t)
    for k, fid in enumerate(files):
        bw = Signed(pyBigWig.open(str(bigwigs / f"{fid}.bigWig")))
        xs, ys = [], []
        for _ in range(check_cells // 50):
            c, s, e, off = pick[int(rng.integers(0, len(pick)))]
            if c not in bw.chroms():
                continue
            g0 = off + int(rng.integers(0, e - s - 50 * bin))
            c0 = -(-g0 // bin)                                             # 50 whole cells inside the contig
            a = s + c0 * bin - off
            if a + 50 * bin > e:
                continue
            raw = np.nan_to_num(bw.values(c, a, a + 50 * bin, numpy=True))
            xs.append(np.log1p(np.maximum(raw, 0)).reshape(50, bin).mean(1))
            ys.append(np.asarray(lg[c0:c0 + 50, k], dtype=np.float64))
        bw.close()
        x, y = np.concatenate(xs), np.concatenate(ys)
        pear[k] = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
    prov = {}
    if provenance and provenance.exists():
        prov = {r["file"]: r for r in map(json.loads, provenance.read_text().splitlines())}
    stranded = any(i.endswith(("_P", "_M")) for i in ids)
    meta = dict(ids=ids, total_bases=total, n_tracks=t, bin=bin, cells=cells, genome=name,
                scales=[infos[f]["log1p_scale"] for f in files], nonzero_mean=[infos[f]["nonzero_mean"] for f in files],
                assays=[r["assay"] for r in sel], scale_percentile=scale_percentile,
                encoding="uint8 = log1p(signal)/log1p_scale*255, clipped; cells: float16 means of the training target "
                         "and of log1p over 128 bases",
                nonzero_mean_method="400 sampled 100 kb windows of the training split, before the full pass",
                grid=f"cell k = global bases [k*{bin}, (k+1)*{bin}) of the genome's layout",
                strand="stranded pairs by channel id (_P plus, _M minus)" if stranded else "unstranded",
                tracks=[dict(channel=k, id=ids[k], **{c: r.get(c) for c in ("experiment", "file", "assay", "biosample",
                                                                             "target", "output_type")},
                             **{c: infos[r["file"]][c] for c in ("log1p_scale", "nonzero_mean", "nan_share",
                                                                 "observed_bases", "negative_values", "unobserved_contigs")},
                             sign=infos[r["file"]].get("sign", 1.0),
                             check_pearson_cells_vs_bigwig=float(pear[k]),
                             **{c: prov.get(r["file"], {}).get(c) for c in ("assembly", "md5", "size", "lab")})
                        for k, r in enumerate(sel)])
    (out / "binned.json").write_text(json.dumps(meta, indent=1))
    empty = [ids[k] for k in range(t) if not np.isfinite(pear[k])]
    typer.echo(f"wrote {out / 'binned.json'}: {t} tracks, {cells:,} cells; check r (cells vs bigWig) min "
               f"{np.nanmin(pear):.4f} median {np.nanmedian(pear):.4f}; {sum(infos[f].get('sign', 1.0) < 0 for f in files)} "
               f"tracks read with the sign turned (negative minus-strand files)")
    if empty:                                        # a track without signal in every checked cell: say so, loudly
        typer.echo(f"WARNING: {len(empty)} tracks show no signal in the checked cells: {empty[:8]}", err=True)


if __name__ == "__main__":
    typer.run(main)
