"""Pretraining kron-ppg and fitting its age clock on whole streams, not windows.

KronLSTM is a streaming model, so it is trained as one: truncated backpropagation through time over the
streams of :mod:`kron_ppg.streams`. ``lanes`` streams are read side by side, ``chunk_s`` seconds at a time;
each lane carries its recurrent memory (:class:`kron_core.blocks.LayerStates`) from one chunk to the next,
detached between chunks, and its memory is reset only when the lane moves to another stream. Nothing is cut
out or dropped: a gap is masked (its patches leave the recurrence), motion noise stays in and its loss is
weighted by the block's pulse periodicity.

Scaling is causal, as on a watch: each sample is divided by the median robust scale of the previous
``scale_blocks`` 10 s blocks at least 80 % valid (the stream's first block uses its own), then clipped to
``clip`` and marked invalid beyond it.

* **pretraining**: next-patch L1 (masked, periodicity-weighted) plus, with ``subject_weight``, the
  participant InfoNCE of the window trainer: lanes come in pairs reading two places of the same person, and the
  pooled embeddings of a chunk are the positives. A pair reads ``pair_chunks`` chunks, then both lanes move to
  a new person. Sources are drawn with weight ``sqrt(valid hours)``, so a 49-person wrist set is not drowned by
  6,000 surgical cases nor the reverse; a person within a source uniformly; a stream within a person by length.
* **clock**: the age head reads the running mean of the hidden states since the lane's stream began (the pool
  a phone keeps); loss at the end of every chunk. Evaluation streams each person's recordings whole (up to
  ``eval_hours``) and reads the age at the end, then the bias correction and gap associations of
  :mod:`kron_ppg.clock`.

Splits are the window trainer's (``subject_split`` over ``<source>/<subject>``, key ``kron-ppg``), so a
stream clock and a window clock meet the same test people.
"""
from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from kron_core.blocks import LayerStates
from kron_ppg.clock import fit_bias, gap_associations, json_safe, scores
from kron_ppg.data import subject_split
from kron_ppg.model import AgeHead, KronPPG, KronPPGConfig, PPGAgeClock, patchify
from kron_ppg.streams import Streams
from kron_ppg.train import _lr_at, _param_groups, load_encoder


# ---- the streams of several sources as one --------------------------------------------------------
class StreamPool:
    """Streams of several variants: per stream its source, ``<source>/<subject>`` key and length; per key its
    age and subject-table row; the window trainer's split."""

    def __init__(self, folders: list[Path], scale_blocks: int = 30, clip: float = 20.0) -> None:
        self.sets = [Streams(f) for f in folders]
        self.scale_blocks, self.clip = scale_blocks, clip
        rows = []
        for k, s in enumerate(self.sets):
            src = s.info["source"]
            for i, r in enumerate(s.table.iter_rows(named=True)):
                rows.append({"set": k, "i": i, "source": src, "key": f"{src}/{r['subject']}",
                             "length": r["length"], "valid_hours": r["valid_hours"]})
        self.streams = pl.DataFrame(rows)
        if not len(self.streams):
            raise ValueError("no streams in " + ", ".join(map(str, folders)))
        self.block = int(round(self.sets[0].info["block_s"] * self.sets[0].info["sample_rate"]))
        self.sample_rate = float(self.sets[0].info["sample_rate"])
        if {s.info["sample_rate"] for s in self.sets} != {self.sample_rate}:
            raise ValueError("the streams differ in sample rate")
        self.table: dict[str, dict[str, str]] = {}
        self.age: dict[str, float] = {}
        for s in self.sets:
            src = s.info["source"]
            for r in s.subjects.iter_rows(named=True):
                key = f"{src}/{r['subject']}"
                self.table[key] = {k: ("" if v is None else str(v)) for k, v in r.items()}
                try:
                    age = float(r.get("age")) if r.get("age") not in (None, "") else math.nan
                except (TypeError, ValueError):
                    age = math.nan
                self.age[key] = age
        keys = sorted(set(self.streams["key"]))
        self.side = subject_split(keys, "kron-ppg")

    def keys(self, split: str, need_age: bool = False) -> list[str]:
        out = sorted({k for k in self.streams["key"] if self.side[k] == split})
        return [k for k in out if math.isfinite(self.age.get(k, math.nan))] if need_age else out

    def read(self, stream: int, start: int, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``n`` samples of stream ``stream`` (a row of :attr:`streams`) from ``start``, causally scaled:
        ``(x float32, valid bool, weight float32)``; past the stream's end the samples are invalid zeros."""
        r = self.streams.row(stream, named=True)
        sig, valid, blocks = self.sets[r["set"]][r["i"]]
        stop = min(start + n, r["length"])
        x = np.zeros(n, np.float32)
        ok = np.zeros(n, bool)
        w = np.zeros(n, np.float32)
        if stop <= start:
            return x, ok, w
        x[:stop - start] = sig[start:stop]
        ok[:stop - start] = valid[start:stop]
        b0, b1 = start // self.block, (stop - 1) // self.block + 1
        lo = max(0, b0 - self.scale_blocks)
        past = blocks[lo:b1]
        scale = np.empty(b1 - b0, np.float32)
        period = np.empty(b1 - b0, np.float32)
        for j, b in enumerate(range(b0, b1)):
            prev = past[max(0, b - self.scale_blocks) - lo:b - lo]
            good = prev[(prev[:, 2] >= 0.8) & (prev[:, 0] > 0), 0]
            own = blocks[b, 0] if b < len(blocks) else 0.0
            scale[j] = np.median(good) if len(good) else (own if own > 0 else 1.0)
            period[j] = blocks[b, 1] if b < len(blocks) else 0.0
        idx = (np.arange(start, stop) // self.block) - b0
        x[:stop - start] /= scale[idx]
        w[:stop - start] = np.clip(period[idx], 0.0, 1.0)
        far = np.abs(x) > self.clip
        x[far] = 0.0
        ok &= ~far
        return x, ok, w

    def next_valid(self, stream: int, start: int) -> int:
        """The first sample at or after ``start`` inside a block with signal (the stream's length if none)."""
        r = self.streams.row(stream, named=True)
        _, _, blocks = self.sets[r["set"]][r["i"]]
        b = start // self.block
        ahead = np.flatnonzero(np.asarray(blocks[b:, 2]) > 0)
        return r["length"] if not len(ahead) else max(start, (b + int(ahead[0])) * self.block)


class Lanes:
    """``lanes`` (even) readers in same-person pairs, each with a stream and a position."""

    def __init__(self, pool: StreamPool, keys: list[str], lanes: int, chunk: int, pair_chunks: int,
                 rng: np.random.Generator) -> None:
        if lanes % 2:
            raise ValueError("lanes come in pairs: give an even number")
        self.pool, self.chunk, self.pair_chunks, self.rng = pool, chunk, pair_chunks, rng
        st = pool.streams.with_row_index("row").filter(pl.col("key").is_in(keys) & (pl.col("valid_hours") > 0))
        if not len(st):
            raise ValueError("no stream with signal for these people")
        self.by_key = {(k[0] if isinstance(k, tuple) else k): (g["row"].to_numpy(), g["length"].to_numpy())
                       for k, g in st.group_by("key", maintain_order=True)}
        by_source: dict[str, list[str]] = {}
        hours: dict[str, float] = {}
        for k in self.by_key:
            src = k.split("/", 1)[0]
            by_source.setdefault(src, []).append(k)
        for r in st.group_by("source").agg(pl.col("valid_hours").sum()).iter_rows(named=True):
            hours[r["source"]] = r["valid_hours"]
        self.sources = sorted(by_source)
        self.source_keys = {s: sorted(v) for s, v in by_source.items()}
        p = np.array([math.sqrt(hours[s]) for s in self.sources])
        self.source_p = p / p.sum()
        n = lanes
        self.stream = np.zeros(n, np.int64)
        self.pos = np.zeros(n, np.int64)
        self.key = [""] * n
        self.left = np.zeros(n // 2, np.int64)
        self.fresh = np.ones(n, bool)
        for pair in range(n // 2):
            self._new_pair(pair)

    def _place(self, lane: int) -> None:
        rows, lengths = self.by_key[self.key[lane]]
        j = self.rng.choice(len(rows), p=lengths / lengths.sum())
        stream = int(rows[j])
        start = int(self.rng.integers(0, max(1, lengths[j] - self.chunk)))
        self.stream[lane] = stream
        self.pos[lane] = self.pool.next_valid(stream, start)
        self.fresh[lane] = True

    def _new_pair(self, pair: int) -> None:
        src = self.sources[self.rng.choice(len(self.sources), p=self.source_p)]
        key = self.source_keys[src][self.rng.integers(len(self.source_keys[src]))]
        a, b = 2 * pair, 2 * pair + 1
        for lane in (a, b):
            self.key[lane] = key
            self._place(lane)
        if self.stream[a] == self.stream[b]:  # one stream (a BIDMC record): the other half of it, so the
            length = int(self.pool.streams["length"][int(self.stream[a])])  # positives are not the same samples
            self.pos[b] = self.pool.next_valid(int(self.stream[b]), (int(self.pos[a]) + length // 2) % max(length, 1))
        self.left[pair] = self.pair_chunks

    def batch(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """The next chunk of every lane: ``(x, valid, weight)`` ``(lanes, chunk)``, the lanes whose memory must
        be reset first, and each lane's person; then every lane moves on."""
        n = len(self.pos)
        x = np.zeros((n, self.chunk), np.float32)
        ok = np.zeros((n, self.chunk), bool)
        w = np.zeros((n, self.chunk), np.float32)
        reset = self.fresh.copy()
        keys = list(self.key)
        for lane in range(n):
            x[lane], ok[lane], w[lane] = self.pool.read(int(self.stream[lane]), int(self.pos[lane]), self.chunk)
        self.fresh[:] = False
        for lane in range(n):
            length = int(self.pool.streams["length"][int(self.stream[lane])])
            nxt = self.pool.next_valid(int(self.stream[lane]), int(self.pos[lane]) + self.chunk)
            if nxt >= length:  # the stream ended: another place of the same person
                self._place(lane)
            else:
                self.pos[lane] = nxt  # skipping a stretch without signal needs no reset: masked patches leave
                # the recurrence untouched
        self.left -= 1
        for pair in np.flatnonzero(self.left <= 0):
            self._new_pair(int(pair))
        return x, ok, w, reset, keys


def _weighted_next_loss(model: KronPPG, h: torch.Tensor, patches: torch.Tensor, ok: torch.Tensor,
                        weight: torch.Tensor) -> torch.Tensor:
    """:meth:`KronPPG.next_patch_loss` with each target patch weighted by its periodicity (0.05 floor)."""
    c = model.config
    batch, steps, _ = patches.shape
    pred = model.next_head(h).view(batch, steps, c.horizon, c.patch)
    total = h.new_zeros(())
    count = h.new_zeros(())
    for k in range(1, min(c.horizon, steps - 1) + 1):
        m = ((ok[:, :-k] & ok[:, k:]).float() * weight[:, k:].clamp_min(0.05)).unsqueeze(-1)
        total = total + ((pred[:, :-k, k - 1] - patches[:, k:]).abs() * m).sum()
        count = count + m.sum() * c.patch
    return total / count.clamp_min(1.0)


def _tensors(x, ok, w, patch, dev):
    xt = torch.from_numpy(x).to(dev, non_blocking=True)
    okt = torch.from_numpy(ok).to(dev, non_blocking=True)
    patches, pv = patchify(xt, patch, okt)
    steps = patches.shape[1]
    wt = torch.from_numpy(w[:, :steps * patch].reshape(len(w), steps, patch).mean(-1)).to(dev)
    return patches, pv, wt


@dataclass
class StreamPretrainArgs:
    folders: list[Path]
    out: Path
    config: KronPPGConfig = field(default_factory=KronPPGConfig)
    lanes: int = 64
    chunk_s: float = 120.0
    pair_chunks: int = 30
    subject_weight: float = 0.1
    steps: int = 10_000
    lr: float = 1e-3
    warmup_fraction: float = 0.05
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    eval_every: int = 250
    eval_lanes: int = 32
    eval_chunks: int = 8
    scale_blocks: int = 30
    seed: int = 0
    device: str = "cuda"


def _evaluate_next(model: KronPPG, pool: StreamPool, keys: list[str], a, chunk: int, dev) -> list[float]:
    """Next-patch L1 per chunk index on fixed places of validation people, the memory carried over
    ``eval_chunks`` chunks: index 0 reads from an empty memory, later ones from what came before."""
    lanes = Lanes(pool, keys, a.eval_lanes, chunk, 10 ** 9, np.random.default_rng(12345))
    states = LayerStates.empty(model.config.n_layers)
    model.eval()
    per_chunk = []
    with torch.no_grad():
        for c in range(a.eval_chunks):
            x, ok, w, reset, _ = lanes.batch()
            patches, pv, wt = _tensors(x, ok, w, model.config.patch, dev)
            states = states.reset(torch.from_numpy(reset).to(dev)) if c else states
            h = model.encode(patches, pv, states)
            per_chunk.append(float(model.next_patch_loss(h, patches, pv)))
    model.train()
    return per_chunk


def pretrain_streams(a: StreamPretrainArgs) -> dict:
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = torch.device(a.device)
    pool = StreamPool(a.folders, a.scale_blocks)
    if abs(pool.sample_rate - a.config.sample_rate) > 1e-6:
        raise ValueError(f"the streams are at {pool.sample_rate} Hz and the model reads {a.config.sample_rate} Hz")
    chunk = int(round(a.chunk_s * pool.sample_rate)) // a.config.patch * a.config.patch
    train_keys, val_keys = pool.keys("train"), pool.keys("val")
    lanes = Lanes(pool, train_keys, a.lanes, chunk, a.pair_chunks, rng)
    model = KronPPG(a.config).to(dev)
    a.out.mkdir(parents=True, exist_ok=True)
    record = {"task": "pretrain-streams", "model": a.config.to_dict(),
              "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(a).items()
                       if k not in ("config", "folders", "out")},
              "data": [str(f) for f in a.folders], "chunk_samples": chunk,
              "people": {s: len(pool.keys(s)) for s in ("train", "val", "test")},
              "source_weights": dict(zip(lanes.sources, map(float, lanes.source_p), strict=True)),
              "parameters": model.n_parameters()}
    (a.out / "config.json").write_text(json.dumps(record, indent=2) + "\n")
    opt = torch.optim.AdamW(_param_groups(model, a.weight_decay), lr=a.lr, betas=(0.9, 0.95))
    warmup = max(1, int(a.warmup_fraction * a.steps))
    states = LayerStates.empty(a.config.n_layers)
    log, best, t0, seen = [], float("inf"), time.time(), 0.0
    model.train()
    for step in range(1, a.steps + 1):
        x, ok, w, reset, keys = lanes.batch()
        patches, pv, wt = _tensors(x, ok, w, a.config.patch, dev)
        states = states.reset(torch.from_numpy(reset).to(dev))
        for g in opt.param_groups:
            g["lr"] = _lr_at(step - 1, a.steps, warmup, a.lr)
        h = model.encode(patches, pv, states)
        next_loss = _weighted_next_loss(model, h, patches, pv, wt)
        subj_loss = next_loss.new_zeros(())
        if a.subject_weight > 0:
            ids = {k: i for i, k in enumerate(sorted(set(keys)))}
            has = pv.any(1)
            if int(has.sum()) > 2:
                subj = torch.tensor([ids[k] for k in keys], device=dev)
                subj_loss = model.subject_loss(model.embedding(h, pv)[has], subj[has])
        loss = next_loss + a.subject_weight * subj_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
        opt.step()
        states = states.detach()
        seen += float(ok.sum()) / pool.sample_rate
        if step % a.eval_every == 0 or step == a.steps:
            per_chunk = _evaluate_next(model, pool, val_keys, a, chunk, dev)
            val = float(np.mean(per_chunk[1:])) if len(per_chunk) > 1 else per_chunk[0]
            rate = seen / (time.time() - t0)
            log.append({"step": step, "lr": opt.param_groups[0]["lr"], "train_next": next_loss.item(),
                        "train_subject": subj_loss.item(), "val_next": val, "val_next_chunk0": per_chunk[0],
                        "val_next_last": per_chunk[-1], "signal_s_per_s": round(rate, 1),
                        "signal_hours": round(seen / 3600, 1)})
            pl.DataFrame(log).write_parquet(a.out / "metrics.parquet")
            print(f"step {step}: next {next_loss.item():.4f}, subject {subj_loss.item():.4f}, val next {val:.4f}, "
                  f"(cold {per_chunk[0]:.4f}, warm {per_chunk[-1]:.4f}), {seen / 3600:,.0f} h read, {rate:,.0f} s of signal/s", flush=True)
            if val < best:
                best = val
                torch.save({"model": model.state_dict(), "config": a.config.to_dict(), "step": step,
                            "objective": "streams", "val_next": val}, a.out / "checkpoint.pt")
    return {**record, "best_val_next": best}


def memory_check(encoder: KronPPG, pool: StreamPool, keys: list[str], lanes: int, chunk: int, chunks: int,
                 dev, seed: int = 12345) -> list[dict]:
    """The same validation chunks scored twice: from an empty memory (``cold``) and with the memory carried in
    from the chunks before them (``warm``). Per chunk index ``c``: both next-patch losses on identical samples,
    so their difference is what the carried memory is worth after ``c`` chunks of context."""
    lane = Lanes(pool, keys, lanes, chunk, 10 ** 9, np.random.default_rng(seed))
    batches = [lane.batch() for _ in range(chunks)]
    encoder.eval()
    out = []
    with torch.no_grad():
        states = LayerStates.empty(encoder.config.n_layers)
        for c, (x, ok, w, reset, _) in enumerate(batches):
            patches, pv, _ = _tensors(x, ok, w, encoder.config.patch, dev)
            if c:
                states = states.reset(torch.from_numpy(reset).to(dev))
            warm = float(encoder.next_patch_loss(encoder.encode(patches, pv, states), patches, pv))
            cold = float(encoder.next_patch_loss(encoder.encode(patches, pv, LayerStates.empty(encoder.config.n_layers)),
                                                 patches, pv))
            out.append({"chunk": c, "context_s": c * chunk / pool.sample_rate, "cold": cold, "warm": warm,
                        "warm_minus_cold": warm - cold, "resets": int(reset.sum()) if c else len(reset)})
    return out


# ---- the clock -------------------------------------------------------------------------------------
@dataclass
class StreamClockArgs:
    folders: list[Path]
    out: Path
    encoder: Path | None = None
    config: KronPPGConfig = field(default_factory=KronPPGConfig)
    freeze: bool = False
    lanes: int = 64
    chunk_s: float = 120.0
    pair_chunks: int = 15
    steps: int = 4_000
    lr: float = 3e-4
    head_lr: float = 1e-3
    warmup_fraction: float = 0.05
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    eval_every: int = 500
    eval_hours: float = 4.0
    eval_lanes: int = 64
    scale_blocks: int = 30
    #: ``uniform``: every valid patch counts the same in a person's pool; ``periodicity``: weighted by the pulse
    #: periodicity of its 10 s block (:data:`POOL_GATE`), a soft quality gate inside the stream
    pool: str = "uniform"
    seed: int = 0
    device: str = "cuda"


#: Pulse periodicity below which a stretch adds nothing to a person's pooled state (the window gate's 0.35);
#: above it the weight rises linearly to 1. A soft quality gate inside the stream: nothing is cut out.
POOL_GATE = 0.35


def _pool_weights(w: np.ndarray, valid: np.ndarray, patch: int, mode: str) -> np.ndarray:
    """Per-patch pooling weights ``(B, T)``: the validity (``uniform``), or validity times
    ``clip((periodicity - POOL_GATE) / (1 - POOL_GATE), 0, 1)`` (``periodicity``)."""
    steps = valid.shape[1] // patch
    ok = valid[:, :steps * patch].reshape(len(valid), steps, patch).all(-1).astype(np.float32)
    if mode == "uniform":
        return ok
    if mode != "periodicity":
        raise ValueError(f"pool must be uniform or periodicity, not {mode!r}")
    per = w[:, :steps * patch].reshape(len(w), steps, patch).mean(-1)
    return ok * np.clip((per - POOL_GATE) / (1 - POOL_GATE), 0.0, 1.0)


def _pool_step(clock: PPGAgeClock, patches, pv, weight, states, psum, pcount):
    """One chunk of a stream clock: the encoder carries ``states``; the pool adds the chunk's hidden states
    weighted by ``weight`` ``(B, T)``; returns ``(age, psum, pcount)``."""
    h = clock.encoder.encode(patches, pv, states)
    wt = weight.to(h.dtype).unsqueeze(-1)
    psum = psum + (h * wt).sum(1)
    pcount = pcount + wt.sum(1)
    return clock.head(psum / pcount.clamp_min(1e-6)), psum, pcount


def stream_ages(clock: PPGAgeClock, pool: StreamPool, keys: list[str], hours: float, chunk: int, lanes: int,
                dev, pool_mode: str = "uniform") -> dict[str, tuple[float, float, int]]:
    """Each person's age read at the end of up to ``hours`` of their streams, one after another, the memory and
    the pool carried across them: ``key -> (predicted, age, patches pooled)``."""
    clock.eval()
    out = {}
    budget = int(hours * 3600 * pool.sample_rate)
    plan = []
    for k in keys:
        rows = pool.streams.with_row_index("row").filter(pl.col("key") == k)
        segs, left = [], budget
        for r in rows.iter_rows(named=True):
            if left <= 0:
                break
            n = min(r["length"], left)
            segs += [(r["row"], s, min(chunk, n - (s - 0))) for s in range(0, n, chunk)]
            left -= n
        plan.append((k, segs))
    patch = clock.encoder.config.patch
    with torch.no_grad():
        for g in range(0, len(plan), lanes):
            group = plan[g:g + lanes]
            states = LayerStates.empty(clock.encoder.config.n_layers)
            d = clock.encoder.config.d_model
            psum = torch.zeros(len(group), d, device=dev)
            pcount = torch.zeros(len(group), 1, device=dev)
            age = torch.zeros(len(group), device=dev)
            for c in range(max(len(s) for _, s in group)):
                x = np.zeros((len(group), chunk), np.float32)
                ok = np.zeros((len(group), chunk), bool)
                w = np.zeros((len(group), chunk), np.float32)
                for j, (_, segs) in enumerate(group):
                    if c < len(segs):
                        row, start, n = segs[c]
                        xx, oo, ww = pool.read(int(row), int(start), chunk)
                        x[j, :n], ok[j, :n], w[j, :n] = xx[:n], oo[:n], ww[:n]
                patches, pv, _ = _tensors(x, ok, w, patch, dev)
                weight = torch.from_numpy(_pool_weights(w, ok, patch, pool_mode)).to(dev)
                age, psum, pcount = _pool_step(clock, patches, pv, weight, states, psum, pcount)
            for j, (k, _) in enumerate(group):
                out[k] = (float(age[j]), pool.age[k], int(pcount[j]))
    clock.train()
    return out


def fit_stream_clock(a: StreamClockArgs) -> dict:
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = torch.device(a.device)
    pool = StreamPool(a.folders, a.scale_blocks)
    encoder = load_encoder(a.encoder, dev) if a.encoder else KronPPG(a.config).to(dev)
    config = encoder.config
    chunk = int(round(a.chunk_s * pool.sample_rate)) // config.patch * config.patch
    split = {s: pool.keys(s, need_age=True) for s in ("train", "val", "test")}
    if min(map(len, split.values())) < 2:
        raise ValueError(f"need two people with an age in each split: { {k: len(v) for k, v in split.items()} }")
    head = AgeHead(config.d_model).to(dev)
    train_ages = np.array([pool.age[k] for k in split["train"]])
    head.set_scale(torch.tensor(train_ages))
    clock = PPGAgeClock(encoder, head).to(dev)
    if a.freeze:
        for p in encoder.parameters():
            p.requires_grad_(False)
    lanes = Lanes(pool, split["train"], a.lanes, chunk, a.pair_chunks, rng)
    a.out.mkdir(parents=True, exist_ok=True)
    record = {"task": "clock-streams", "model": config.to_dict(), "encoder": str(a.encoder) if a.encoder else None,
              "freeze": a.freeze, "data": [str(f) for f in a.folders], "chunk_samples": chunk,
              "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(a).items()
                       if k not in ("config", "folders", "out", "encoder")},
              "people": {k: len(v) for k, v in split.items()},
              "train_age": {"mean": float(train_ages.mean()), "sd": float(train_ages.std()),
                            "min": float(train_ages.min()), "max": float(train_ages.max())}}
    (a.out / "config.json").write_text(json.dumps(record, indent=2) + "\n")
    groups = [{"params": list(head.parameters()), "lr": a.head_lr, "weight_decay": 0.0, "base": a.head_lr}]
    if not a.freeze:
        groups += [{**g, "lr": a.lr, "base": a.lr} for g in _param_groups(encoder, a.weight_decay)]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95))
    warmup = max(1, int(a.warmup_fraction * a.steps))
    states = LayerStates.empty(config.n_layers)
    psum = torch.zeros(a.lanes, config.d_model, device=dev)
    pcount = torch.zeros(a.lanes, 1, device=dev)

    def evaluate() -> dict:
        subj = {k: stream_ages(clock, pool, v, a.eval_hours, chunk, a.eval_lanes, dev, a.pool) for k, v in split.items()}
        arrays = {k: (np.array([p for p, _, _ in d.values()]), np.array([g for _, g, _ in d.values()]))
                  for k, d in subj.items()}
        bias = fit_bias(*arrays["val"])
        out = {"bias": {"a": bias[0], "b": bias[1]}, **{k: scores(p, g, bias) for k, (p, g) in arrays.items()}}
        gaps = {k: {s: p - (bias[0] + bias[1] * g) for s, (p, g, _) in d.items()} for k, d in subj.items()}
        out["outcomes"] = {k: gap_associations(gaps[k], pool.table, ages={s: g for s, (_, g, _) in subj[k].items()})
                           for k in ("val", "test")}
        return {"scores": out, "subjects": subj, "gaps": gaps}

    log, best, best_eval = [], float("inf"), None
    clock.train()
    if a.freeze:
        encoder.eval()
    for step in range(1, a.steps + 1):
        x, ok, w, reset, keys = lanes.batch()
        patches, pv, _ = _tensors(x, ok, w, config.patch, dev)
        weight = torch.from_numpy(_pool_weights(w, ok, config.patch, a.pool)).to(dev)
        r = torch.from_numpy(reset).to(dev)
        states = states.reset(r)
        keep = (~r).float().unsqueeze(-1)
        psum, pcount = psum * keep, pcount * keep
        for g in opt.param_groups:
            g["lr"] = _lr_at(step - 1, a.steps, warmup, g["base"])
        pred, psum, pcount = _pool_step(clock, patches, pv, weight, states, psum, pcount)
        age = torch.tensor([pool.age[k] for k in keys], device=dev, dtype=torch.float32)
        seen = pcount.squeeze(-1) > 1e-3
        loss = F.smooth_l1_loss(pred.float()[seen], age[seen], beta=2.0) if bool(seen.any()) else pred.sum() * 0
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], a.grad_clip)
        opt.step()
        states, psum, pcount = states.detach(), psum.detach(), pcount.detach()
        if step % a.eval_every == 0 or step == a.steps:
            ev = evaluate()
            s = ev["scores"]
            log.append({"step": step, "train_l1": loss.item(), "val_mae": s["val"]["mae"], "val_r": s["val"]["r"],
                        "test_mae": s["test"]["mae"], "test_r": s["test"]["r"]})
            pl.DataFrame(log).write_parquet(a.out / "metrics.parquet")
            print(f"step {step}: train l1 {loss.item():.2f}, val MAE {s['val']['mae']:.2f} (r {s['val']['r']:.2f}), "
                  f"test MAE {s['test']['mae']:.2f} (r {s['test']['r']:.2f})", flush=True)
            if s["val"]["mae"] < best:
                best, best_eval = s["val"]["mae"], {**ev, "step": step}
                torch.save({"encoder": encoder.state_dict(), "head": head.state_dict(), "config": config.to_dict(),
                            "step": step, "bias": s["bias"]}, a.out / "checkpoint.pt")
    assert best_eval is not None
    result = {**record, "best_step": best_eval["step"], **best_eval["scores"]}
    (a.out / "metrics.json").write_text(json.dumps(json_safe(result), indent=2) + "\n")
    pl.DataFrame(
        [{"split": k, "subject": s, "age": g, "predicted": p, "gap_corrected": best_eval["gaps"][k][s], "patches": n}
         for k, d in best_eval["subjects"].items() for s, (p, g, n) in sorted(d.items())],
        schema={"split": pl.String, "subject": pl.String, "age": pl.Float64, "predicted": pl.Float64,
                "gap_corrected": pl.Float64, "patches": pl.Int64},
    ).write_parquet(a.out / "predictions.parquet")
    return result


def _cli(argv: list[str]) -> None:
    """``python -m kron_ppg.stream_train pretrain|clock OUT FOLDER... [--key value ...]``"""
    if len(argv) >= 3 and argv[0] == "memory":  # memory CKPT OUT.json FOLDER... : cold vs warm on the same chunks
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        enc = load_encoder(Path(argv[1]), dev)
        pool = StreamPool([Path(f) for f in argv[3:]])
        chunk = int(round(120.0 * pool.sample_rate)) // enc.config.patch * enc.config.patch
        rows = memory_check(enc, pool, pool.keys("val"), 64, chunk, 15, dev)
        Path(argv[2]).write_text(json.dumps(rows, indent=2) + "\n")
        for r in rows:
            print(f"chunk {r['chunk']:2d} ({r['context_s'] / 60:4.0f} min of context): cold {r['cold']:.4f}  "
                  f"warm {r['warm']:.4f}  diff {r['warm_minus_cold']:+.4f}  resets {r['resets']}")
        return
    if len(argv) < 3 or argv[0] not in ("pretrain", "clock"):
        sys.exit("usage: python -m kron_ppg.stream_train pretrain|clock OUT FOLDER... [--steps N --lanes N "
                 "--chunk-s S --encoder CKPT --freeze --seed N --device cuda]")
    task, out, rest = argv[0], Path(argv[1]), argv[2:]
    folders, opts = [], {}
    i = 0
    while i < len(rest):
        if rest[i].startswith("--"):
            name = rest[i][2:].replace("-", "_")
            if name == "freeze":
                opts[name] = True
                i += 1
                continue
            opts[name] = rest[i + 1]
            i += 2
        else:
            folders.append(Path(rest[i]))
            i += 1
    cls = StreamPretrainArgs if task == "pretrain" else StreamClockArgs
    types = {f: type(getattr(cls(folders=[], out=out), f)) for f in cls.__dataclass_fields__}
    kwargs = {}
    for k, v in opts.items():
        if k == "encoder":
            kwargs[k] = Path(v)
        elif k == "freeze":
            kwargs[k] = True
        else:
            kwargs[k] = types[k](v)
    args = cls(folders=folders, out=out, **kwargs)
    result = (pretrain_streams if task == "pretrain" else fit_stream_clock)(args)
    print(json.dumps(json_safe({k: v for k, v in result.items() if k in ("best_val_next", "best_step", "test", "val")})))


if __name__ == "__main__":
    _cli(sys.argv[1:])
