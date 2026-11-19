"""AlphaGenome v1's DNase and splice-site predictions on the further heads' benchmark windows (evaluation only).

    $AGENV/bin/python scripts/kron_dna/xhead_bench_ag.py BENCH --ag-dir AG_DIR --model all-folds [--subset all]

Runs in AlphaGenome's own environment, from the local weights, with the helpers of the earlier AlphaGenome track
evaluation (``AG_DIR/ag_common.py``: the hg38 reader and the model loader). For each window it gives the model the
1,048,576-base interval centred on it (the ``context_start``/``context_end`` of ``windows.csv``) and keeps the
window's 131,072 bases. ``--subset`` is ``all`` or a boolean column of ``AG_DIR/meta/window_folds.csv`` (a fold
model's held-out test windows, e.g. ``test_strict_fold1``).

Writes ``BENCH/pred/alphagenome-<model>__<subset>/`` in the form ``xhead_bench.py report`` reads:
``dnase_128.npy`` (per 128-bp bin, the mean of log1p of AlphaGenome's 1-bp DNase output, for the track matched to
each target channel by ENCODE biosample term in ``BENCH/dnase_match.json``; NaN for a channel without one),
``splice.npy`` (its SPLICE_SITES output, channels donor +, acceptor +, donor -, acceptor -), ``index.json``.
AlphaGenome's weights and outputs are non-commercial: nothing written here feeds a training job.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ap = argparse.ArgumentParser()
ap.add_argument("bench", type=Path)
ap.add_argument("--ag-dir", type=Path, required=True)
ap.add_argument("--model", default="all-folds")
ap.add_argument("--subset", default="all")
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()
sys.path.insert(0, str(args.ag_dir))
from ag_common import CTX, MODELS, WIN, GpuWatch, Hg38, card_used, make_model  # noqa: E402

BIN = 128
card = os.environ["CUDA_VISIBLE_DEVICES"]
watch = GpuWatch(card)
watch.start()
assert (args.bench / "READY").exists(), "the benchmark's targets are not frozen"
win = pl.read_csv(args.bench / "windows.csv").with_row_index("w")
if args.subset != "all":
    keep = pl.read_csv(args.ag_dir / "meta" / "window_folds.csv").filter(pl.col(args.subset))["w"].to_list()
    win = win.filter(pl.col("w").is_in(keep))
if args.limit:
    win = win.head(args.limit)
match = json.loads((args.bench / "dnase_match.json").read_text())
n, t = win.height, len(match)
label = f"alphagenome-{args.model}__{args.subset}" + (f"__first{args.limit}" if args.limit else "")
out = args.bench / "pred" / label
if (out / "index.json").exists():
    raise SystemExit(f"{out} exists")
out.mkdir(parents=True, exist_ok=True)
print(f"{label}: {n} windows, {sum(m['ag_track'] is not None for m in match)} of {t} DNase channels matched; card {card_used(card)}",
      flush=True)

from alphagenome.models.v1 import dna_output  # noqa: E402

model = make_model(MODELS / f"alphagenome-{args.model}")
g = Hg38()
terms = sorted({m["encode_term_id"] for m in match if m["ag_track"] is not None})
requested = [dna_output.OutputType.DNASE, dna_output.OutputType.SPLICE_SITES]
d128 = np.full((n, WIN // BIN, t), np.nan, dtype=np.float32)
sp = np.zeros((n, WIN, 4), dtype=np.float16)
lo = (CTX - WIN) // 2
cols, times = None, []
for k, r in enumerate(win.iter_rows(named=True)):
    a, b = r["start"] - lo, r["start"] - lo + CTX
    assert (a, b) == (r["context_start"], r["context_end"])
    seq = g.seq(r["chrom"], a, b)
    assert len(seq) == CTX and "N" not in seq[lo:lo + WIN]
    t0 = time.time()
    for attempt in range(3):                       # XLA autotuning can fail on a first call under a capped pool
        try:
            o = model.predict_sequence(seq, requested_outputs=requested, ontology_terms=terms)
            break
        except Exception as ex:                    # noqa: BLE001
            print(f"window {k}: predict failed ({type(ex).__name__}: {str(ex)[:200]}), try {attempt + 1}", flush=True)
            if attempt == 2:
                raise
    times.append(time.time() - t0)
    if cols is None:                               # each matched DNase track's column, located once by name
        md = o.dnase.metadata.reset_index(drop=True)
        cols = {}
        for m in match:
            if m["ag_track"] is None:
                continue
            hit = md.index[(md["name"] == m["ag_track"]).to_numpy()].tolist()
            assert len(hit) == 1, (m, hit)
            cols[m["channel"]] = hit[0]
        sm = o.splice_sites.metadata.reset_index(drop=True)
        want = [("donor", "+"), ("acceptor", "+"), ("donor", "-"), ("acceptor", "-")]
        splice_cols = [int(sm.index[(sm["name"] == nm).to_numpy() & (sm["strand"] == st).to_numpy()][0]) for nm, st in want]
        assert o.dnase.resolution == 1 and o.splice_sites.resolution == 1
    dv = np.asarray(o.dnase.values)[lo:lo + WIN]
    for c, i in cols.items():
        d128[k, :, c] = np.log1p(np.maximum(dv[:, i].astype(np.float64), 0)).reshape(-1, BIN).mean(1)
    sp[k] = np.asarray(o.splice_sites.values)[lo:lo + WIN][:, splice_cols].astype(np.float16)
    if k % 8 == 0 or k == n - 1:
        print(f"window {k + 1}/{n} {times[-1]:.1f} s, own {watch.now} MiB (peak {watch.peak})", flush=True)
np.save(out / "dnase_128.npy", d128)
np.save(out / "splice.npy", sp)
from importlib.metadata import version  # noqa: E402

(out / "index.json").write_text(json.dumps(dict(
    label=label, kind="AlphaGenome", model=args.model, subset=args.subset, windows=win["w"].to_list(), context=CTX,
    dnase_channels={str(c): match[c]["ag_track"] for c in cols}, dnase_terms=terms, splice_columns=splice_cols,
    seconds_first=round(times[0], 1), seconds_median=round(float(np.median(times)), 2), own_gpu_mib_peak=watch.peak,
    versions={p: version(p) for p in ("alphagenome", "alphagenome-research", "jax")}), indent=1))
print(f"done: {label}, {n} windows, median {np.median(times):.2f} s, peak {watch.peak} MiB", flush=True)
