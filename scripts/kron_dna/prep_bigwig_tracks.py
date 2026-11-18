"""Measured bigWig tracks as a new per-base track store on a prepared genome's contig layout.

    uv run --with pyBigWig python scripts/kron_dna/prep_bigwig_tracks.py SELECT.tsv BIGWIG_DIR OUT_DIR \
        [--name hg38] [--provenance provenance.jsonl] [--workers 4]

``SELECT.tsv`` lists the tracks in channel order (columns ``experiment``, ``file``, ``assay``, ``biosample``,
``output_type``); the channel id is the file accession, and the order of the rows is the order of the channels,
fixed from then on. Each ``BIGWIG_DIR/<file>.bigWig`` becomes ``OUT_DIR/tracks/<file>.u8``, one byte per contig base
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


def scale_of(bw, rows: list[tuple]) -> float:
    """99.9th percentile of non-zero log1p signal over 200 sampled 100 kb windows (as the NTv3 store)."""
    rng = np.random.default_rng(0)
    vals = []
    have = set(bw.chroms())
    for _ in range(200):
        c, s, e, _ = rows[int(rng.integers(0, len(rows)))]
        if e - s < 100_000 or c not in have:
            continue
        a = int(rng.integers(s, e - 100_000))
        v = np.nan_to_num(bw.values(c, a, a + 100_000, numpy=True))
        vals.append(v[v > 0])
    v = np.concatenate(vals) if vals else np.array([1.0])
    return float(max(np.log1p(np.percentile(v, 99.9)) if v.size else 1.0, 1e-3))


def one(fid: str) -> dict:
    import pyBigWig

    rows, total, src, out = _G["rows"], _G["total"], _G["src"], _G["out"]
    dst = out / "tracks" / f"{fid}.u8"
    side = out / "tracks" / f"{fid}.json"
    if dst.exists() and dst.stat().st_size == total and side.exists():
        return json.loads(side.read_text())
    bw = pyBigWig.open(str(src / f"{fid}.bigWig"))
    have = bw.chroms()
    sc = scale_of(bw, rows)
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
         workers: int = 4, check_windows: int = 40) -> None:
    import pyBigWig

    genome = interim_dir("genome", name)
    if genome.resolve() in [out.resolve(), *out.resolve().parents]:
        raise typer.BadParameter("OUT_DIR lies inside the prepared genome; choose a new path")
    idx = pl.read_parquet(genome / "contigs.parquet").sort("offset")
    rows = [(r["chrom"], r["start"], r["end"], r["offset"]) for r in idx.iter_rows(named=True)]
    total = int((idx["end"] - idx["start"]).sum())
    with open(select) as f:
        sel = list(csv.DictReader(f, delimiter="\t"))
    ids = [r["file"] for r in sel]
    assert len(set(ids)) == len(ids), "a file is listed twice"
    prov = {}
    if provenance and provenance.exists():
        prov = {r["file"]: r for r in map(json.loads, provenance.read_text().splitlines())}
    (out / "tracks").mkdir(parents=True, exist_ok=True)
    g = dict(rows=rows, total=total, src=bigwigs, out=out)
    with Pool(workers, initializer=_init, initargs=(g,)) as p:
        infos = {}
        for info in p.imap_unordered(one, ids):
            infos[info["file"]] = info
            typer.echo(f"  {info['file']} scale={info['log1p_scale']:.4f} nan={info['nan_share']:.3f} "
                       f"unobserved contigs={len(info['unobserved_contigs'])}")
    T = len(ids)
    src = [np.memmap(out / "tracks" / f"{i}.u8", dtype=np.uint8, mode="r") for i in ids]
    dense_path = out / "dense.u8"
    if not (dense_path.exists() and dense_path.stat().st_size == total * T):
        dst = np.memmap(str(dense_path) + PART, dtype=np.uint8, mode="w+", shape=(total, T))
        step = 1 << 25
        for a in range(0, total, step):
            b = min(a + step, total)
            blk = np.empty((b - a, T), dtype=np.uint8)
            for k, s_ in enumerate(src):
                blk[:, k] = s_[a:b]
            dst[a:b] = blk
        dst.flush()
        del dst
        os.replace(str(dense_path) + PART, dense_path)
    dense = np.memmap(dense_path, dtype=np.uint8, mode="r", shape=(total, T))
    scales = np.array([infos[i]["log1p_scale"] for i in ids])

    # non-zero mean of the decoded signal on the genome's training split, 1 base in 64 (as the NTv3 store)
    split = np.memmap(genome / "ntv3" / "split.u8", dtype=np.uint8, mode="r")
    tot, cnt = np.zeros(T), np.zeros(T)
    for a in range(0, total, 1 << 25):
        b = min(a + (1 << 25), total)
        m = np.asarray(split[a:b:64]) == 1
        if m.any():
            q = np.asarray(dense[a:b:64])[m].astype(np.float64)
            nz = q > 0
            tot += np.where(nz, np.expm1(q / 255.0 * scales), 0).sum(0)
            cnt += nz.sum(0)
    nzmean = np.where(cnt > 0, tot / np.maximum(cnt, 1), 1.0)

    # decoded bins against the source: random 128 bp bins on random contigs, each channel
    rng = np.random.default_rng(1)
    worst, pear = np.zeros(T), np.zeros(T)
    for k, fid in enumerate(ids):
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
            "encoding": ENCODING, "genome": name, "split": f"interim/genome/{name}/ntv3/split.u8 (1 = train; the "
            "held-out chromosomes are 0 there)", "strand": "unstranded",
            "tracks": [dict(channel=k, **{c: r.get(c) for c in ("experiment", "file", "assay", "biosample",
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
