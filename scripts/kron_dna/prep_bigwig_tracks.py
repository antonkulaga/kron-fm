"""Measured bigWig tracks as a new per-base track store on a prepared genome's contig layout.

    uv run --with pyBigWig python scripts/kron_dna/prep_bigwig_tracks.py SELECT.tsv BIGWIG_DIR OUT_DIR \
        [--name hg38] [--provenance provenance.jsonl] [--workers 4]

``SELECT.tsv`` lists the tracks in channel order (columns ``experiment``, ``file``, ``assay``, ``biosample``,
``output_type``, optionally ``channel``); the channel id is ``channel`` when given, else the file accession, and the
order of the rows is the order of the channels, fixed from then on. Stranded signal comes as two rows whose channel
ids end ``_P`` and ``_M`` (plus and minus strand): the store's loader exchanges such a pair on a window read on the
other strand.

``--convert-only`` writes the per-track files and stops: several jobs, each given part of the rows, convert in
parallel into one ``OUT_DIR``, and a last run on the whole list assembles the store (converting what is missing). Each ``BIGWIG_DIR/<file>.bigWig`` becomes ``OUT_DIR/tracks/<file>.u8``, one byte per contig base
laid out exactly like the genome's ``bases.u8``, in the encoding of the NTv3 benchmark store so one loader reads
both: ``uint8 = log1p(signal) / log1p_scale * 255``, clipped, the scale being the track's own 99.9th percentile of
non-zero log1p signal. Then ``OUT_DIR/dense.u8`` (base-major, all channels) and ``dense.json``.

Never writes into the prepared genome: ``OUT_DIR`` must lie outside it. Every file is written under a temporary
name and renamed, so a killed run leaves nothing half-written in place and a rerun skips finished tracks.

Missing coverage against a measured zero: a bigWig holds no value (NaN) where the pipeline wrote none. A contig the
bigWig does not list at all is *unobserved* for that track and listed in ``dense.json`` (``unobserved_contigs``), for
the loss to mask. Inside a listed contig a NaN is stored as 0, and its share per track is recorded
(``nan_share``): ENCODE's signal files leave out runs without reads, so there a NaN is a zero, but the share is kept
so a track that is mostly empty is seen. The held-out chromosomes are stored too (evaluation reads them); the
training mask is the prepared genome's own split, named in ``dense.json`` and never copied.
"""
from __future__ import annotations

import csv
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
# temporary files carry the host and process, so two runs of one conversion (a preempted job's rerun beside a second
# submission) never write the same file; the rename is atomic and the later one replaces an identical result
PART = f".part.{socket.gethostname()}.{os.getpid()}"
ENCODING = "uint8 = log1p(signal)/log1p_scale*255, clipped"
_G: dict = {}


def scale_of(bw, rows: list[tuple], percentile: float = 99.9, split=None) -> float:
    """The ``percentile`` of non-zero log1p signal over 200 sampled 100 kb windows (99.9: as the NTv3 store; the
    DNase pilot showed that it clips peak summits where a track's scale is low, so peaky assays take 99.99). With
    ``split``, a draw counts only when its window is at least 90 % training split (code 1): held-out chromosomes never
    calibrate the encoding (review, 23 Nov)."""
    rng = np.random.default_rng(0)
    vals = []
    have = set(bw.chroms())
    for _ in range(200):
        c, s, e, off = rows[int(rng.integers(0, len(rows)))]
        if e - s < 100_000 or c not in have:
            continue
        a = int(rng.integers(s, e - 100_000))
        if split is not None and float((np.asarray(split[off + a - s:off + a - s + 100_000]) == 1).mean()) < 0.9:
            continue
        v = np.nan_to_num(bw.values(c, a, a + 100_000, numpy=True))
        vals.append(v[v > 0])
    v = np.concatenate(vals) if vals else np.array([1.0])
    return float(max(np.log1p(np.percentile(v, percentile)) if v.size else 1.0, 1e-3))


def one(fid: str) -> dict:
    import pyBigWig

    rows, total, src, out = _G["rows"], _G["total"], _G["src"], _G["out"]
    dst = out / "tracks" / f"{fid}.u8"
    side = out / "tracks" / f"{fid}.json"
    if dst.exists() and dst.stat().st_size == total and side.exists():
        return json.loads(side.read_text())
    bw = pyBigWig.open(str(src / f"{fid}.bigWig"))
    have = bw.chroms()
    sc = scale_of(bw, rows, _G.get("percentile", 99.9), _G.get("split"))
    part = str(dst) + f".part.{socket.gethostname()}.{os.getpid()}"
    arr = np.memmap(part, dtype=np.uint8, mode="w+", shape=(total,))
    nan = seen = negative = 0
    unobserved = []
    for c, s, e, off in rows:
        if c not in have or have[c] < e:
            unobserved.append(f"{c}:{s}-{e}")
            continue
        for a in range(s, e, CHUNK):
            b = min(a + CHUNK, e)
            v = bw.values(c, a, b, numpy=True)
            bad = ~np.isfinite(v)
            nan += int(bad.sum())
            seen += b - a
            v = np.where(bad, 0.0, v)
            negative += int((v < 0).sum())
            arr[off + (a - s): off + (b - s)] = np.clip(np.log1p(np.maximum(v, 0)) / sc * 255.0, 0, 255).astype(np.uint8)
    arr.flush()
    del arr
    bw.close()
    os.replace(part, dst)
    info = {"file": fid, "log1p_scale": sc, "nan_share": nan / max(seen, 1), "observed_bases": seen,
            "negative_values": negative, "unobserved_contigs": unobserved}
    Path(str(side) + PART).write_text(json.dumps(info))
    os.replace(str(side) + PART, side)
    return info


def _init(g: dict) -> None:
    _G.update(g)


def main(select: Path, bigwigs: Path, out: Path, name: str = "hg38", provenance: Path = typer.Option(None),
         workers: int = 4, check_windows: int = 40,
         scale_percentile: float = typer.Option(99.9, help="percentile of non-zero signal the encoding clips at"),
         convert_only: bool = typer.Option(False, help="write the per-track files only, no store")) -> None:
    import pyBigWig

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
    assert len(set(files)) == len(files), "a file is listed twice"
    assert len(set(ids)) == len(ids), "a channel id is listed twice"
    for i in ids if not convert_only else []:         # a part of the list may hold one strand of a pair
        if i.endswith(("_P", "_M")) and i[:-1] + ("M" if i.endswith("_P") else "P") not in ids:
            raise typer.BadParameter(f"stranded channel {i} has no partner")
    stranded = any(i.endswith(("_P", "_M")) for i in ids)
    prov = {}
    if provenance and provenance.exists():
        prov = {r["file"]: r for r in map(json.loads, provenance.read_text().splitlines())}
    (out / "tracks").mkdir(parents=True, exist_ok=True)
    sp_ = genome / "ntv3" / "split.u8"                 # the training split, so the scale is calibrated on it only
    g = dict(rows=rows, total=total, src=bigwigs, out=out, percentile=scale_percentile,
             split=np.memmap(sp_, dtype=np.uint8, mode="r") if sp_.exists() else None)
    with Pool(workers, initializer=_init, initargs=(g,)) as p:
        infos = {}
        for info in p.imap_unordered(one, files):
            infos[info["file"]] = info
            typer.echo(f"  {info['file']} scale={info['log1p_scale']:.4f} nan={info['nan_share']:.3f} "
                       f"unobserved contigs={len(info['unobserved_contigs'])}", )
    if convert_only:
        typer.echo(f"converted {len(infos)} tracks into {out / 'tracks'}; no store assembled (--convert-only)")
        return
    T = len(ids)
    src = [np.memmap(out / "tracks" / f"{i}.u8", dtype=np.uint8, mode="r") for i in files]
    dense_path = out / "dense.u8"
    if not (dense_path.exists() and dense_path.stat().st_size == total * T):
        # resumable: one fixed temporary file, flushed block by block, and a marker of the blocks done, so a run
        # killed at a time limit is continued by the next (a block written but not yet marked is written again)
        part, mark = Path(str(dense_path) + ".part"), Path(str(dense_path) + ".part.blocks")
        step = 1 << 25
        start = int(mark.read_text()) if part.exists() and mark.exists() else 0
        if start and part.stat().st_size != total * T:
            start = 0
        dst = np.memmap(part, dtype=np.uint8, mode="r+" if start else "w+", shape=(total, T))
        if start:
            typer.echo(f"continuing the store at block {start} of {-(-total // step)}")
        for i, a in enumerate(range(0, total, step)):
            if i < start:
                continue
            b = min(a + step, total)
            blk = np.empty((b - a, T), dtype=np.uint8)
            for k, s_ in enumerate(src):
                blk[:, k] = s_[a:b]
            dst[a:b] = blk
            dst.flush()
            Path(str(mark) + PART).write_text(str(i + 1))
            os.replace(str(mark) + PART, mark)
        del dst
        os.replace(part, dense_path)
        mark.unlink(missing_ok=True)
    dense = np.memmap(dense_path, dtype=np.uint8, mode="r", shape=(total, T))
    scales = np.array([infos[i]["log1p_scale"] for i in files])

    # non-zero mean of the decoded signal on the genome's training split, over 1 run of 4,096 bases in 64 (a run,
    # not a base: a strided read of single bases touches every page of a store this wide)
    split = np.memmap(genome / "ntv3" / "split.u8", dtype=np.uint8, mode="r")
    tot, cnt = np.zeros(T), np.zeros(T)
    run = 4096
    for a in range(0, total, run * 64):
        b = min(a + run, total)
        m = np.asarray(split[a:b]) == 1
        if m.any():
            q = np.asarray(dense[a:b])[m].astype(np.float64)
            nz = q > 0
            tot += np.where(nz, np.expm1(q / 255.0 * scales), 0).sum(0)
            cnt += nz.sum(0)
    nzmean = np.where(cnt > 0, tot / np.maximum(cnt, 1), 1.0)

    # decoded bins against the source: random 128 bp bins on random contigs, each channel
    rng = np.random.default_rng(1)
    worst, pear = np.zeros(T), np.zeros(T)
    for k, fid in enumerate(files):
        bw = pyBigWig.open(str(bigwigs / f"{fid}.bigWig"))
        have = bw.chroms()
        a_src, a_dec = [], []
        ok = [r for r in rows if r[0] in have and r[2] - r[1] > (1 << 17)]
        for _ in range(check_windows):
            c, s, e, off = ok[int(rng.integers(0, len(ok)))]
            a = int(rng.integers(s, e - (1 << 17)))
            v = np.nan_to_num(bw.values(c, a, a + (1 << 17), numpy=True))
            d = np.expm1(np.asarray(dense[off + a - s: off + a - s + (1 << 17), k]) / 255.0 * scales[k])
            a_src.append(np.log1p(np.maximum(v, 0)).reshape(-1, 128).mean(1))
            a_dec.append(np.log1p(d).reshape(-1, 128).mean(1))
        x, y = np.concatenate(a_src), np.concatenate(a_dec)
        below = x < scales[k]                                   # bins the clip at the scale does not touch
        worst[k] = float(np.abs(x - y)[below].max()) if below.any() else 0.0
        pear[k] = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 else float("nan")
        bw.close()

    meta = {"ids": ids, "total_bases": total, "n_tracks": T, "scales": scales.tolist(),
            "assays": [r["assay"] for r in sel], "nonzero_mean": nzmean.tolist(), "mean_stride": 64,
            "mean_sample": "1 run of 4096 bases in 64",
            "encoding": ENCODING, "scale_percentile": scale_percentile, "genome": name, "split": f"interim/genome/{name}/ntv3/split.u8 (1 = train; the "
            "held-out chromosomes are 0 there)",
            "strand": "stranded pairs by channel id (_P plus, _M minus)" if stranded else "unstranded",
            "tracks": [dict(channel=k, id=ids[k], **{c: r.get(c) for c in ("experiment", "file", "assay", "biosample",
                                                               "output_type")},
                            **{c: infos[r["file"]][c] for c in ("log1p_scale", "nan_share", "observed_bases",
                                                                "negative_values", "unobserved_contigs")},
                            nonzero_mean=float(nzmean[k]), check_pearson_128bp=float(pear[k]),
                            check_max_abs_log1p_error_128bp=float(worst[k]),
                            **{c: prov.get(r["file"], {}).get(c) for c in ("assembly", "md5", "size", "lab",
                                                                           "award", "date_released",
                                                                           "biological_replicates")})
                       for k, r in enumerate(sel)]}
    (out / "dense.json").write_text(json.dumps(meta, indent=1))
    for t in meta["tracks"]:
        typer.echo(f"  ch{t['channel']:2d} {t['file']} {t['biosample'][:24]:24s} nz_mean={t['nonzero_mean']:.3f} "
                   f"r128={t['check_pearson_128bp']:.5f} max_err={t['check_max_abs_log1p_error_128bp']:.4f}")
    typer.echo(f"wrote {out / 'dense.json'}")


if __name__ == "__main__":
    typer.run(main)
