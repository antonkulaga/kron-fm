"""In-silico enhancer knockouts against CRISPRi enhancer-gene screens (the ENCODE-rE2G benchmark: Fulco 2019,
Gasperini 2019, ...; ``prep_crispri_pairs.py``), as Enformer, Borzoi and AlphaGenome report them.

    uv run python scripts/kron_dna/knockout_bench.py score OUT LABEL=CHECKPOINT [...] [--pairs PAIRS.parquet]
        [--max-pairs N] [--e5 200] [--dtype bf16|fp16|fp32] [--shard i/n] [--chunk 131072]
    uv run python scripts/kron_dna/knockout_bench.py merge OUT LABEL --shards n      (after every --shard i/n)
    uv run python scripts/kron_dna/knockout_bench.py report OUT

``score``: one window of ``--context`` bases centred on each element (snapped to the 128-bp grid), read three times:
reference, the element masked (its bases N) and the element shuffled (a permutation of its own bases, seeded by the
element, so composition is kept). The RNA head (``--head``, the stranded RNA-seq store) runs on the span holding the
element's target TSSs; each pair's readout is the predicted RNA signal (``expm1`` of the head's log1p, summed over
the 128-bp bins within ``--tss-flank`` of the TSS) on the cell type's channel of the gene's strand, and its effect is
``log2((ko + eps) / (ref + eps))``. Pairs whose TSS lies outside the window's middle (``--edge`` from either end),
or whose cell type has no RNA channel in the head, are left out and counted.

``--e5 N`` (the JEPA plan's E5, residual reuse, Balestriero and LeCun 2026) also, for N sampled pairs (half
regulated where there are enough), hides a block of ``--block`` bases over the TSS, runs encoder and core on the
reference and on the masked knockout with that block hidden, and takes the predictor's states there, ``mu(ref)`` and
``mu(ko)``. The counterfactual core is ``z_obs + s * (mu(ko) - mu(ref))`` inside the block (``z_obs`` the clean
reference core; ``s`` per bin, ``|z_obs| / |mu(ref)|``, since the cosine loss leaves the predictor's norm free),
decoded with the reference skips through the heads: ``e5_effect``. ``e5_latent`` is the mean relative size of
``mu(ko) - mu(ref)`` in the block, and ``e5_placebo`` the same for a placebo knockout of the element's length at
the mirror position across the TSS (the noise floor the plan asks the latent change to beat).

``report``: average precision of ``-effect`` against ``regulated`` (significant and negative), beside the
distance-only baseline (``-|distance|``) and the fraction of positives (chance), with 95 % bootstrap intervals; per
study and cell type, and hg38 chr8/chr9 (the held-out split) apart from the rest. On the E5 pairs, every score on
the same pairs.
"""
from __future__ import annotations

import gzip
import json
import time
import zlib
from pathlib import Path

import numpy as np
import polars as pl
import torch
import typer

from kron_core.paths import interim_dir
from kron_dna import evalkit as EK
from kron_dna import genome as G
from kron_dna import xheads as X

app = typer.Typer(add_completion=False)
BIN = 128
CELL_RNA = {"K562": "ENCSR000CPY", "HCT116": "ENCSR698RPL", "WTC11": "ENCSR450EXF", "GM12878": "ENCSR000CVT"}
EPS = 1.0


def codes_of(contigs: pl.DataFrame, bases: np.ndarray, chrom: str, lo: int, hi: int) -> np.ndarray:
    out = np.full(hi - lo, 4, dtype=np.int64)
    for r in contigs.filter((pl.col("chrom") == chrom) & (pl.col("start") < hi) & (pl.col("end") > lo)).iter_rows(named=True):
        a, b = max(lo, r["start"]), min(hi, r["end"])
        out[a - lo:b - lo] = bases[r["offset"] + a - r["start"]:r["offset"] + b - r["start"]]
    return out


def strands_of(gtf: Path) -> dict:
    out = {}
    with gzip.open(gtf, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split("\t", 9)
            if p[2] == "gene":
                out[p[8].split('gene_id "', 1)[1].split('"', 1)[0].split(".")[0]] = p[6]
    return out


def channels_of(store) -> dict:
    """(cell type, strand) -> channel of the RNA head's store."""
    meta = json.loads((Path(store.path) / "binned.json").read_text())
    ids = {t["id"]: t["channel"] for t in meta["tracks"]}
    return {(ct, s): ids[f"{exp}_{'P' if s == '+' else 'M'}"] for ct, exp in CELL_RNA.items() for s in "+-"
            if f"{exp}_{'P' if s == '+' else 'M'}" in ids}


class Runner:
    """A model's pieces, so a forward can be run whole or as encoder+core and decoder apart (E5)."""

    def __init__(self, model, head: str, dev, dtype: str = "bf16", chunk: int = 0) -> None:
        self.m, self.head, self.dev, self.dtype, self.chunk = model, head, dev, dtype, chunk
        self.unit = 2 ** model.config.n_down
        self.ctx = torch.zeros(1, dtype=torch.long, device=dev) if model.config.n_contexts > 0 else None

    def ac(self):
        return EK.autocast(self.dev, self.dtype)

    def full(self, seq: torch.Tensor) -> torch.Tensor:
        L = seq.shape[-1]
        seg = torch.zeros(1, L // self.unit, dtype=torch.long, device=self.dev)
        with torch.no_grad(), self.ac():
            if self.chunk:
                EK.forward_capped(self.m, seq, self.ctx, self.m.x_span, self.chunk)
            else:
                self.m(seq[None], seg, self.ctx)
        return self.m.last_x[self.head][0].float()

    def core(self, seq: torch.Tensor):
        m = self.m
        if self.chunk:                              # skips are rebuilt over the head span at decode
            with torch.no_grad(), self.ac():
                return EK.core_capped(m, seq, self.ctx, self.chunk), seq
        with torch.no_grad(), self.ac():
            x, skips = m.encode(seq[None])
            h = x.transpose(1, 2)
            if m.context is not None:
                h = h + m.context_rows(self.ctx, 1, self.dev).unsqueeze(1).to(h.dtype)
            cu = torch.tensor([0, h.shape[1]], dtype=torch.int32, device=self.dev)
            return m.run_core(h, cu), skips

    def decode(self, core: torch.Tensor, skips) -> torch.Tensor:
        m = self.m
        if self.chunk:                              # ``skips`` is the window's sequence here
            with torch.no_grad(), self.ac():
                EK.decode_span(m, skips, core, m.x_span)
            return m.last_x[self.head][0].float()
        with torch.no_grad(), self.ac():
            x = m.add_local(m.decode(core.transpose(1, 2), skips), skips)
            m.last_x, m.last_x_at = m.run_xheads(x)
        return m.last_x[self.head][0].float()

    def pred(self, core: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(), self.ac():
            return self.m.predict_jepa(core).float()


def readout(p_sig: np.ndarray, rel_tss: int, span_lo: int, ch: int, flank: int) -> float:
    a = max((rel_tss - flank - span_lo) // BIN, 0)
    b = min((rel_tss + flank - span_lo) // BIN + 1, p_sig.shape[0])
    return float(p_sig[a:b, ch].sum())


@app.command()
def score(out: Path, checkpoints: list[str] = typer.Argument(..., help="LABEL=CHECKPOINT"),
          pairs: Path = typer.Option(None, help="prep_crispri_pairs.py's table (default: interim eval/crispri-eg-v1)"),
          gtf: Path = typer.Option(None, help="GENCODE GTF for gene strands (default: input regulation/gencode)"),
          head: str = "rna462b", name: str = "hg38", context: int = 1048576, edge: int = 8192,
          tss_flank: int = 1024, max_pairs: int = typer.Option(0, help="all regulated pairs plus negatives up to N in "
                                                                       "all (whole elements kept); 0: every pair"),
          e5: int = 0, block: int = 4096, seed: int = 0, device: str = "cuda",
          dtype: str = typer.Option("bf16", help="bf16 | fp16 | fp32 (bf16 is emulated, slow, on Turing cards)"),
          chunk: int = typer.Option(0, help="bases per encoder chunk (memory cap: e.g. 131072 runs a 1 Mb context "
                                            "on an 11 GB card; 0: the whole window at once)"),
          shard: str = typer.Option("", help="i/n: windows k % n == i only, written under OUT/shards/; then merge")
          ) -> None:
    sh = EK.parse_shard(shard)
    root = interim_dir("genome", name).parent.parent
    pairs = pairs or root / "eval" / "crispri-eg-v1" / "pairs.parquet"
    gtf = gtf or root.parent / "input" / "regulation" / "gencode" / "gencode.basic.gtf.gz"
    t = pl.read_parquet(pairs).filter("valid").with_row_index("pair")
    strand = strands_of(gtf)
    t = t.with_columns(strand=pl.col("gene_id").replace_strict(strand, default=None),
                       e_mid=(pl.col("start") + pl.col("end")) // 2)
    t = t.with_columns(w_lo=(pl.col("e_mid") // BIN) * BIN - context // 2)
    ok = ((pl.col("chrom") == pl.col("chrom_tss")) & (pl.col("tss") >= pl.col("w_lo") + edge)
          & (pl.col("tss") < pl.col("w_lo") + context - edge) & pl.col("strand").is_not_null()
          & pl.col("cell_type").is_in(list(CELL_RNA)))
    dropped = dict(total=t.height, kept=int(t.filter(ok).height),
                   no_rna_channel=int(t.filter(~pl.col("cell_type").is_in(list(CELL_RNA))).height),
                   tss_outside=int(t.filter((pl.col("chrom") != pl.col("chrom_tss")) | (pl.col("tss") < pl.col("w_lo") + edge)
                                            | (pl.col("tss") >= pl.col("w_lo") + context - edge)).height))
    t = t.filter(ok)
    rng = np.random.default_rng(seed)
    if max_pairs and t.height > max_pairs:                      # whole elements: one window serves all their pairs
        el = t.group_by("chrom", "start", "end").agg(reg=pl.col("regulated").any(), n=pl.len()).sort("chrom", "start", "end")
        keep = el.filter("reg")
        rest = el.filter(~pl.col("reg"))
        rest = rest[rng.permutation(rest.height)]
        room = max_pairs - int(keep["n"].sum())
        take = rest.filter(pl.col("n").cum_sum() <= max(room, 0))
        t = t.join(pl.concat([keep, take]).select("chrom", "start", "end"), on=["chrom", "start", "end"])
    e5_ids = set()
    if e5:
        pos, neg = t.filter("regulated")["pair"].to_numpy(), t.filter(~pl.col("regulated"))["pair"].to_numpy()
        n_pos = min(len(pos), e5 // 2)
        e5_ids = set(rng.choice(pos, n_pos, replace=False).tolist()) | set(
            rng.choice(neg, min(len(neg), e5 - n_pos), replace=False).tolist())
    src = interim_dir("genome", name)
    bases = np.memmap(src / "bases.u8", dtype=np.uint8, mode="r")
    contigs = pl.read_parquet(src / "contigs.parquet")
    dev = torch.device(device)
    els = t.group_by("chrom", "start", "end", "w_lo", maintain_order=True).agg(pl.all().exclude("chrom", "start", "end", "w_lo"))
    els = els.sort("chrom", "start")
    for item in checkpoints:
        label, where = item.split("=", 1)
        dest = out / "pred" / label
        part = out / "shards" / label / f"{EK.tag(sh)}.parquet"
        if (dest / "pairs.parquet").exists() or (sh[1] > 1 and part.exists()):
            raise typer.BadParameter(f"{dest if not sh[1] > 1 else part} exists; choose another label")
        model, state = G.load_checkpoint(where)
        heads = [h[0] for h in model.config.xheads]
        if head not in heads:
            raise typer.BadParameter(f"{label}: no head {head!r} ({heads})")
        rec = (((state.get("run_record") or {}).get("xheads") or {}).get("stores") or {}).get(head, {})
        store = X.open_store(rec.get("path") or str(root / "tracks" / "hg38" / "rnaseq-stranded-v1-bin128"))
        chans = channels_of(store)
        e5_ok = bool(e5) and model.jepa_pred is not None and not model.config.chunk_ratio and model.config.arch != "v3"
        if e5 and not e5_ok:
            typer.echo(f"{label}: no JEPA predictor (or a chunked/v3 core): E5 skipped")
        model = model.to(dev).eval()
        model.x_active = {head}
        if chunk and not EK.capped_ok(model):
            raise typer.BadParameter(f"{label}: --chunk needs a v2/v2l core without learned chunking")
        R = Runner(model, head, dev, dtype, chunk)
        consts = store.consts(dev)
        n_ch = len(store.ids)

        def sig(raw: torch.Tensor) -> np.ndarray:
            p_log, _ = X.eval_pair(store, raw[:, :n_ch], torch.zeros_like(raw[:, :n_ch]).unsqueeze(-1).expand(
                *raw[:, :n_ch].shape, 2), consts)
            return torch.expm1(p_log.float()).clamp_min(0).cpu().numpy()

        rows, t0 = [], time.time()
        for k, e in enumerate(els.iter_rows(named=True)):
            if not EK.mine(k, sh):
                continue
            lo = e["w_lo"]
            ref = torch.from_numpy(codes_of(contigs, bases, e["chrom"], lo, lo + context)).to(dev)
            a, b = e["start"] - lo, e["end"] - lo
            ko_n = ref.clone()
            ko_n[a:b] = 4
            ko_s = ref.clone()
            g = torch.Generator().manual_seed(zlib.crc32(f"{e['chrom']}:{e['start']}-{e['end']}:{seed}".encode()))
            ko_s[a:b] = ref[a:b][torch.randperm(b - a, generator=g).to(dev)]
            rel = [x - lo for x in e["tss"]]
            s_lo = max(((min(rel) - tss_flank - BIN) // BIN) * BIN, 0)
            s_hi = min(((max(rel) + tss_flank + 2 * BIN) // BIN) * BIN, context)
            model.x_span = (s_lo, s_hi)
            p = {v: sig(R.full(s)) for v, s in (("ref", ref), ("mask", ko_n), ("shuffle", ko_s))}
            for j, pid in enumerate(e["pair"]):
                ch = chans.get((e["cell_type"][j], e["strand"][j]))
                if ch is None:
                    continue
                r = dict(pair=pid, **{f"rna_{v}": readout(p[v], rel[j], s_lo, ch, tss_flank) for v in p})
                for v in ("mask", "shuffle"):
                    r[f"effect_{v}"] = float(np.log2((r[f"rna_{v}"] + EPS) / (r["rna_ref"] + EPS)))
                if e5_ok and pid in e5_ids:
                    r.update(e5_pair(R, ref, ko_n, rel[j], a, b, block, s_lo, s_hi, ch, tss_flank, sig, r["rna_ref"]))
                rows.append(r)
            if k % 50 == 0:
                typer.echo(f"{label}: element {k + 1}/{els.height}, {len(rows)} pairs, {time.time() - t0:.0f} s")
        res = t.select("pair", "chrom", "start", "end", "gene", "tss", "strand", "cell_type", "study", "set",
                       "distance", "effect", "regulated", "held_out").join(pl.DataFrame(rows), on="pair")
        info = dict(label=label, source=str(where), step=int(state.get("step", -1)), context=context, head=head,
                    store=str(store.path), channels={f"{c}{s}": v for (c, s), v in chans.items()}, tss_flank=tss_flank,
                    edge=edge, max_pairs=max_pairs, seed=seed, e5=e5 if e5_ok else 0, block=block,
                    pairs_table=str(pairs), dtype=dtype, chunk=chunk, windows=els.height, pairs_scored=res.height, dropped=dropped,
                    seconds=round(time.time() - t0, 1),
                    commit=((state.get("run_record") or {}).get("source") or {}).get("commit"))
        if sh[1] > 1:                                   # a shard: its part, joined by merge once all n are there
            part.parent.mkdir(parents=True, exist_ok=True)
            res.write_parquet(part)
            part.with_suffix(".json").write_text(json.dumps(dict(info, shard=list(sh)), indent=1))
            typer.echo(f"{label}: shard {EK.tag(sh)}, {res.height} pairs in {time.time() - t0:.0f} s -> {part}")
        else:
            dest.mkdir(parents=True, exist_ok=True)
            res.write_parquet(dest / "pairs.parquet")
            (dest / "index.json").write_text(json.dumps(info, indent=1))
            typer.echo(f"{label}: {res.height} pairs from {els.height} windows in {time.time() - t0:.0f} s -> {dest}")
        del model
        torch.cuda.empty_cache() if dev.type == "cuda" else None


def e5_pair(R: Runner, ref, ko, rel_tss, a, b, block, s_lo, s_hi, ch, flank, sig, rna_ref) -> dict:
    u = R.unit
    c_lo, c_hi = (rel_tss - block // 2) // u, (rel_tss + block // 2) // u + 1
    if c_lo * u < b and c_hi * u > a:                           # the element inside the block: not a counterfactual
        return dict(e5_skipped=1)
    hide = lambda s: s.clone().index_fill_(0, torch.arange(c_lo * u, c_hi * u, device=s.device), G.MASK_CHANNEL)
    z_obs, skips = R.core(ref)
    mu_a = R.pred(R.core(hide(ref))[0])[0, c_lo:c_hi]
    mu_b = R.pred(R.core(hide(ko))[0])[0, c_lo:c_hi]
    pb = ref.clone()                                            # placebo: the same length at the mirror position
    m_lo = 2 * rel_tss - b
    if 0 <= m_lo and m_lo + (b - a) <= len(ref):
        pb[m_lo:m_lo + (b - a)] = 4
        mu_p = R.pred(R.core(hide(pb))[0])[0, c_lo:c_hi]
    else:
        mu_p = None
    zo = z_obs[0, c_lo:c_hi].float()
    s = zo.norm(dim=-1, keepdim=True) / mu_a.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    z_cf = z_obs.clone()
    z_cf[0, c_lo:c_hi] = (zo + s * (mu_b - mu_a)).to(z_cf.dtype)
    R.m.x_span = (s_lo, s_hi)
    cf = readout(sig(R.decode(z_cf, skips)), rel_tss, s_lo, ch, flank)
    rd = readout(sig(R.decode(z_obs, skips)), rel_tss, s_lo, ch, flank)   # the reference through the same path
    rel = lambda m: float(((m - mu_a).norm(dim=-1) / mu_a.norm(dim=-1).clamp_min(1e-6)).mean())
    return dict(e5_skipped=0, rna_e5_ref=rd, rna_e5_cf=cf, e5_effect=float(np.log2((cf + EPS) / (rd + EPS))),
                e5_latent=rel(mu_b), e5_placebo=rel(mu_p) if mu_p is not None else float("nan"))


@app.command()
def merge(out: Path, label: str, shards: int = typer.Option(..., help="n, as in --shard i/n")) -> None:
    """Joins the n shards of LABEL into pred/LABEL (refuses while one is missing or they disagree on settings)."""
    d = out / "shards" / label
    parts = [d / f"{i}-of-{shards}.parquet" for i in range(shards)]
    missing = [p.name for p in parts if not p.exists()]
    if missing:
        raise typer.BadParameter(f"{label}: {len(missing)} of {shards} shards missing: {missing[:5]}")
    infos = [json.loads(p.with_suffix(".json").read_text()) for p in parts]
    fixed = ("source", "step", "context", "head", "tss_flank", "edge", "max_pairs", "seed", "e5", "block", "dtype",
             "pairs_table", "windows")
    for k in fixed:
        if len({json.dumps(i.get(k)) for i in infos}) > 1:
            raise typer.BadParameter(f"{label}: shards differ in {k}")
    res = pl.concat([pl.read_parquet(p) for p in parts], how="diagonal_relaxed").sort("pair")
    if res["pair"].n_unique() != res.height:
        raise typer.BadParameter(f"{label}: a pair is in two shards")
    dest = out / "pred" / label
    if (dest / "pairs.parquet").exists():
        raise typer.BadParameter(f"{dest} exists")
    dest.mkdir(parents=True, exist_ok=True)
    res.write_parquet(dest / "pairs.parquet")
    info = dict(infos[0], pairs_scored=res.height, shards=shards, seconds=sum(i["seconds"] for i in infos))
    info.pop("shard", None)
    (dest / "index.json").write_text(json.dumps(info, indent=1))
    typer.echo(f"{label}: {shards} shards, {res.height} pairs -> {dest}")


def ap(score: np.ndarray, y: np.ndarray) -> float:
    o = np.argsort(-score, kind="stable")
    y = y[o].astype(np.float64)
    if y.sum() == 0:
        return float("nan")
    prec = np.cumsum(y) / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def ap_ci(score, y, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    v = [ap(score[i], y[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))]
    return ap(score, y), float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))


@app.command()
def report(out: Path, n_boot: int = 1000) -> None:
    lines, table = [], []
    for d in sorted((out / "pred").iterdir()):
        if not (d / "pairs.parquet").exists():
            continue
        r = pl.read_parquet(d / "pairs.parquet")
        subsets = [("all", r), ("K562", r.filter(pl.col("cell_type") == "K562")),
                   ("chr8/chr9 (held out)", r.filter("held_out")),
                   ("chr3/chr10/chr16 (pair test)", r.filter(pl.col("chrom").is_in(["chr3", "chr10", "chr16"]))),   # no
                   ("other chromosomes", r.filter(~pl.col("held_out")))]      # counterfactual pair is trained there (prep_cf_pairs.py)
        subsets += [(f"study {s}", r.filter(pl.col("study") == s)) for s in
                    r.group_by("study").len().filter(pl.col("len") >= 200)["study"].sort()]
        e = r.filter(pl.col("e5_skipped") == 0) if "e5_skipped" in r.columns else r.clear()
        if e.height:
            subsets.append(("E5 pairs", e))
        for nm, s in subsets:
            y = s["regulated"].to_numpy()
            scores = {"mask": -s["effect_mask"].to_numpy(), "shuffle": -s["effect_shuffle"].to_numpy(),
                      "distance": -np.abs(s["distance"].to_numpy()).astype(float)}
            if nm == "E5 pairs":
                scores |= {"e5_effect": -s["e5_effect"].to_numpy(), "e5_latent": s["e5_latent"].to_numpy()}
            row = dict(label=d.name, subset=nm, pairs=s.height, regulated=int(y.sum()), chance=float(y.mean()) if len(y) else 0)
            for k, v in scores.items():
                a, lo_, hi_ = ap_ci(v, y, n_boot)
                row[k] = a
                row[f"{k}_ci"] = [lo_, hi_]
            if nm == "E5 pairs":
                row["e5_latent_over_placebo"] = float(np.nanmedian(s["e5_latent"].to_numpy() / s["e5_placebo"].to_numpy()))
            table.append(row)
            lines.append(f"{d.name:24s} {nm:28s} n={s.height:5d} reg={int(y.sum()):4d} chance={row['chance']:.3f} "
                         + " ".join(f"{k}={row[k]:.3f}[{row[k + '_ci'][0]:.3f},{row[k + '_ci'][1]:.3f}]" for k in scores))
    (out / "report.json").write_text(json.dumps(table, indent=1))
    typer.echo("\n".join(lines))


if __name__ == "__main__":
    app()
