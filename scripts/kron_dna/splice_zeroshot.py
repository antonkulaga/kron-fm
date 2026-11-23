"""Splice-site average precision of a KronGenome's splice head on any prepared genome: the zero-shot test of a species
that is in no training genome, scored the same way as on human held-out windows.

    uv run python scripts/kron_dna/splice_zeroshot.py windows OUT --name mm10 [--n 96] [--seed 0]
    uv run python scripts/kron_dna/splice_zeroshot.py score OUT --name mm10 --splice STORE LABEL=CHECKPOINT [...]
    uv run python scripts/kron_dna/splice_zeroshot.py score OUT --name hg38 --splice STORE --windows BENCH/windows.csv LABEL=...

``windows``: ``n`` windows of 131,072 bases with no N inside, drawn at random positions over every gap-free contig
long enough for one, a contig chosen by its length (every base equally likely), as the human benchmark's windows were
drawn on chr8 and chr9; the same columns as that benchmark's ``windows.csv``.

``score``: for each window the model reads 1,048,576 bases centred on it (bases beyond a contig are N) and its splice
head runs on the window's bases only, as in training and in ``xhead_bench.py score``. Average precision is exact over
every base of every window, per channel (donor +, acceptor +, donor -, acceptor -), with a 95 % bootstrap interval over
windows. Writes ``OUT/<label>.json`` and the per-base scores ``OUT/<label>.splice.npy`` (float16 logits).
"""
from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import typer

from kron_core.paths import interim_dir
from kron_dna import genome as G
from kron_dna import xheads as X

_spec = importlib.util.spec_from_file_location("xhead_bench", Path(__file__).with_name("xhead_bench.py"))
XB = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(XB)

app = typer.Typer(add_completion=False)
WIN = XB.WIN


@app.command()
def windows(out: Path, name: str = typer.Option(...), n: int = 96, seed: int = 0,
            context: int = 1048576,
            chroms: str = typer.Option("", help="only these chromosomes, '+'-joined (e.g. chr8+chr9: held out)"),
            keep: Path = typer.Option(None, help="a windows.csv whose windows on these chromosomes are kept first"),
            max_n: float = typer.Option(0.0, help="the share of N a window may hold (a fragmented assembly's scaffolds "
                                                  "hold gaps; 0: N-free)")) -> None:
    """Random N-free windows over the genome's gap-free contigs (or those of ``chroms``); ``keep``'s come first."""
    src = interim_dir("genome", name)
    bases = np.memmap(src / "bases.u8", dtype=np.uint8, mode="r")
    contigs = pl.read_parquet(src / "contigs.parquet").filter((pl.col("end") - pl.col("start")) >= WIN)
    if chroms:
        contigs = contigs.filter(pl.col("chrom").is_in(chroms.split("+")))
    lens = (contigs["end"] - contigs["start"]).to_numpy()
    rng = np.random.default_rng(seed)
    rows, tries = [], 0
    if keep is not None:
        old = pl.read_csv(keep)
        if chroms:
            old = old.filter(pl.col("chrom").is_in(chroms.split("+")))
        rows = [dict(r, window_id=f"w{k:03d}") for k, r in enumerate(old.drop("window_id").iter_rows(named=True))]
    while len(rows) < n:
        tries += 1
        assert tries < 100 * n, "too few N-free windows"
        r = contigs.row(int(rng.choice(len(lens), p=lens / lens.sum())), named=True)
        s = r["start"] + int(rng.integers(0, r["end"] - r["start"] - WIN + 1))
        a = r["offset"] + s - r["start"]
        if float((np.asarray(bases[a:a + WIN]) >= 4).mean()) > max_n or any(abs(s - q["start"]) < WIN and q["chrom"] == r["chrom"] for q in rows):
            continue
        flank = (context - WIN) // 2
        lo, hi = s - flank, s + WIN + flank
        inside = sum(max(0, min(hi, c["end"]) - max(lo, c["start"])) for c in pl.read_parquet(src / "contigs.parquet")
                     .filter(pl.col("chrom") == r["chrom"]).iter_rows(named=True))
        w = np.asarray(bases[a:a + WIN])
        rows.append(dict(window_id=f"w{len(rows):03d}", chrom=r["chrom"], start=s, end=s + WIN, bases_offset=a,
                         context_start=lo, context_end=hi, context_gap_bases=context - inside,
                         gc_frac=round(float(np.isin(w, (1, 2)).mean()), 4)))
    out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).sort("chrom", "start").write_csv(out / "windows.csv")
    typer.echo(f"{n} windows on {len({r['chrom'] for r in rows})} chromosomes -> {out / 'windows.csv'}")


@app.command()
def score(out: Path, checkpoints: list[str] = typer.Argument(..., help="LABEL=CHECKPOINT"),
          name: str = typer.Option(...), splice: Path = typer.Option(..., help="the genome's splice label store"),
          windows_csv: Path = typer.Option(None, "--windows", help="default: OUT/windows.csv"),
          splice_head: str = "splice", context: int = 1048576, device: str = "cuda", n_boot: int = 200,
          seed: int = 0) -> None:
    win = pl.read_csv(windows_csv or out / "windows.csv")
    assert (win["end"] - win["start"] == WIN).all()
    ss = X.open_store(splice)
    assert ss.kind == "binary" and list(ss.ids) == XB.SPLICE_IDS, (ss.kind, ss.ids)
    src = interim_dir("genome", name)
    bases = np.memmap(src / "bases.u8", dtype=np.uint8, mode="r")
    contigs = pl.read_parquet(src / "contigs.parquet")
    lab = np.stack([ss.window(int(r["bases_offset"]), WIN, WIN) for r in win.iter_rows(named=True)])   # [n, WIN, 4]
    dev = torch.device(device)
    flank = (context - WIN) // 2
    rng = np.random.default_rng(seed)
    for item in checkpoints:
        label, where = item.split("=", 1)
        model, state = G.load_checkpoint(where)
        if splice_head not in [h[0] for h in model.config.xheads]:
            raise typer.BadParameter(f"{label}: no head {splice_head!r}")
        model = model.to(dev).eval()
        model.x_active, model.x_span = {splice_head}, (flank, flank + WIN)
        unit = 2 ** model.config.n_down
        sp = np.zeros((win.height, WIN, 4), dtype=np.float16)
        t0 = time.time()
        for k, r in enumerate(win.iter_rows(named=True)):
            seq = torch.from_numpy(XB.codes_of(contigs, bases, r["chrom"], r["start"] - flank, r["end"] + flank)).to(dev)
            seg = torch.zeros(1, context // unit, dtype=torch.long, device=dev)
            ctx = torch.zeros(1, dtype=torch.long, device=dev) if model.config.n_contexts > 0 else None
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                model(seq[None], seg, ctx)
            assert model.last_x_at == (flank, flank + WIN), model.last_x_at
            sp[k] = model.last_x[splice_head][0].float().clamp(-6e4, 6e4).cpu().numpy().astype(np.float16)
        ranked = [XB.RankedSplice(sp[:, :, c], lab[:, :, c]) for c in range(4)]
        aps = [rs.ap() for rs in ranked]
        boots = np.array([[rs.ap(np.bincount(rng.integers(0, win.height, win.height), minlength=win.height)
                                 .astype(float)) for rs in ranked] for _ in range(n_boot)])
        res = dict(label=label, checkpoint=str(where), step=int(state.get("step", -1)), genome=name,
                   splice_store=str(splice), n_windows=win.height, positives=lab.sum((0, 1)).tolist(),
                   ap=dict(zip(XB.SPLICE_IDS, aps)), ap_mean=float(np.mean(aps)),
                   ap_mean_ci=[float(np.percentile(boots.mean(1), 2.5)), float(np.percentile(boots.mean(1), 97.5))],
                   seconds=round(time.time() - t0, 1), context=context)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / f"{label}.splice.npy", sp)
        (out / f"{label}.json").write_text(json.dumps(res, indent=1))
        typer.echo(f"{label} on {name}: AP {', '.join(f'{a:.3f}' for a in aps)}; mean {res['ap_mean']:.3f} "
                   f"[{res['ap_mean_ci'][0]:.3f}, {res['ap_mean_ci'][1]:.3f}], {win.height} windows")
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    app()
