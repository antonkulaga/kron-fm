"""Training a further per-base head (``KronGenomeConfig.xheads``) on a measured track store, on top of a trained model.

Three parts, each used by ``genome-train`` and testable alone:

- :class:`TrackStore`: a store written by ``scripts/kron_dna/prep_bigwig_tracks.py`` on the primary genome's contig
  layout (``dense.u8`` + ``dense.json``), read per window; its identity (channel ids in order, encoding, scales) is
  what a checkpoint records and a resume checks.
- :func:`head_loss` / :func:`head_loss_parts`: the track objective of the inherited NTv3 head (Poisson total +
  multinomial profile on AlphaGenome-scaled targets) for one head, whole or from the pieces of a split window.
- :class:`Stage`: where a continuation stands - head-only or joint, how many labelled updates each head has taken,
  when each was activated - and the learning rate of every optimizer group from it. It is a plain dict in the
  checkpoint, so a resumed job picks the same phase, data and rates.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from kron_dna import genome as G


# ------------------------------------------------------------------------------------------------ the store

@dataclass
class TrackStore:
    """A per-base track store on a prepared genome's layout: ``dense[at, k]`` is channel ``k`` at global base ``at``."""
    path: Path
    ids: list[str]
    dense: np.ndarray                 # [total_bases, T] uint8
    scales: np.ndarray                # [T] log1p scale of the encoding
    nonzero_mean: np.ndarray          # [T] mean decoded signal over non-zero training bases
    is_rna: np.ndarray                # [T] bool: the 0.75 power of the target transform
    swap: np.ndarray                  # [T] the strand partner of each channel (itself when unstranded)
    unobserved: list[list[str]]       # per channel: contigs the source did not cover ("chrom:start-end")
    genome: str
    identity: dict = field(default_factory=dict)

    @classmethod
    def open(cls, path: str | Path) -> "TrackStore":
        path = Path(path)
        raw = (path / "dense.json").read_bytes()
        j = json.loads(raw)
        ids = list(j["ids"])
        total, t = int(j["total_bases"]), int(j["n_tracks"])
        if t != len(ids) or len(set(ids)) != t:
            raise ValueError(f"{path}: {t} tracks but {len(set(ids))} distinct ids")
        size = (path / "dense.u8").stat().st_size
        if size != total * t:
            raise ValueError(f"{path / 'dense.u8'} has {size:,} bytes, its record says {total:,} x {t}")
        at = {i: k for k, i in enumerate(ids)}
        swap = np.arange(t)
        for k, i in enumerate(ids):
            if i.endswith(("_P", "_M")):
                swap[k] = at.get(i[:-1] + ("M" if i.endswith("_P") else "P"), k)
        tracks = j.get("tracks") or [{} for _ in ids]
        identity = dict(ids=ids, encoding=j["encoding"], scales=[float(s) for s in j["scales"]],
                        nonzero_mean=[float(s) for s in j["nonzero_mean"]], genome=j.get("genome"),
                        total_bases=total, record_sha256=hashlib.sha256(raw).hexdigest())
        return cls(path=path, ids=ids, dense=np.memmap(path / "dense.u8", dtype=np.uint8, mode="r", shape=(total, t)),
                   scales=np.asarray(j["scales"], dtype=np.float32),
                   nonzero_mean=np.asarray(j["nonzero_mean"], dtype=np.float32),
                   is_rna=np.array(["RNA-seq" in a for a in j["assays"]], dtype=bool), swap=swap,
                   unobserved=[list(tr.get("unobserved_contigs") or []) for tr in tracks],
                   genome=str(j.get("genome")), identity=identity)

    def window(self, at: int, take: int, width: int, flip: bool = False) -> np.ndarray:
        """Targets ``[width, T]`` uint8 of bases ``at:at + take`` (zero-padded to ``width``); ``flip``: the other
        strand, reversed with the strand partners swapped, as the window's bases are reverse-complemented."""
        out = np.zeros((width, len(self.ids)), dtype=np.uint8)
        out[:take] = self.dense[at:at + take]
        if flip:
            out[:take] = out[:take][::-1][:, self.swap]
        return out

    def differences(self, saved: dict) -> list[str]:
        """What differs between this store and the identity a checkpoint recorded (empty: the same targets)."""
        now, out = self.identity, []
        for k in ("ids", "encoding", "genome", "total_bases"):
            if saved.get(k) != now.get(k):
                out.append(f"{k}: checkpoint {saved.get(k)!r}, now {now.get(k)!r}")
        for k in ("scales", "nonzero_mean"):
            a, b = np.asarray(saved.get(k, []), dtype=np.float64), np.asarray(now[k], dtype=np.float64)
            if a.shape != b.shape or not np.allclose(a, b, rtol=1e-6, atol=0):
                out.append(f"{k} differ")
        return out


# ------------------------------------------------------------------------------------------------- the loss

def targets_of(want: torch.Tensor, scales: torch.Tensor, nonzero_mean: torch.Tensor,
               is_rna: torch.Tensor) -> torch.Tensor:
    """Stored bytes ``[..., T]`` as the training target: decoded signal over the track's non-zero mean, to the
    power 0.75 for RNA-seq, soft square-root clip above 10 (AlphaGenome's and NTv3's scaling, as the NTv3 head)."""
    x = torch.expm1(want.float() / 255.0 * scales).clamp_min(0) / nonzero_mean
    x = torch.where(is_rna, x.clamp_min(1e-12).pow(0.75), x)
    return torch.where(x > 10, 2 * torch.sqrt(10 * x) - 10, x)


def signal_of(pred: torch.Tensor, nonzero_mean: torch.Tensor, is_rna: torch.Tensor) -> torch.Tensor:
    """The inverse of :func:`targets_of` up to the encoding: a non-negative prediction as log1p signal, the units
    track Pearson is scored in."""
    x = torch.where(pred > 10, ((pred + 10) / 2).pow(2) / 10, pred)
    x = torch.where(is_rna, x.clamp_min(0).pow(1 / 0.75), x)
    return torch.log1p((x * nonzero_mean).clamp_min(0))


def supervised(split: torch.Tensor, center: float = 1.0, skip: torch.Tensor | None = None) -> torch.Tensor:
    """``[B, L]`` 0/1: the positions a head trains on - the genome's training split (code 1, which leaves the
    held-out chromosomes and the benchmark's own validation and test regions out), the central ``center`` fraction
    of the window, and not ``skip`` (bases inside hidden JEPA blocks)."""
    ok = (split == 1).float()
    if center < 1.0:
        edge = int(round(ok.shape[1] * (1 - center) / 2))
        if edge:
            ok[:, :edge] = 0
            ok[:, -edge:] = 0
    if skip is not None:
        ok = ok * (~skip).float().to(ok.device)
    return ok


def head_loss(pred: torch.Tensor, want: torch.Tensor, ok: torch.Tensor, consts: tuple) -> torch.Tensor:
    """The head's loss on a whole window: ``pred`` [B, L, T] raw outputs, ``want`` [B, L, T] stored bytes, ``ok``
    [B, L] from :func:`supervised`, ``consts`` = (scales, nonzero_mean, is_rna) on ``pred``'s device."""
    return G.poisson_multinomial_loss(F.softplus(pred.float()), targets_of(want, *consts), ok.unsqueeze(-1))


def head_loss_parts(pred: torch.Tensor, want: torch.Tensor, ok: torch.Tensor, consts: tuple) -> dict:
    """The same for one piece of a split window, on the piece's card: the sums
    :func:`kron_dna.genome.poisson_multinomial_combine` joins on the first card into :func:`head_loss`'s value."""
    return G.poisson_multinomial_parts(F.softplus(pred.float()), targets_of(want, *consts), ok.unsqueeze(-1))


def split_head_loss(pieces: list, name: str, want: torch.Tensor, ok: torch.Tensor, consts_on, device) -> torch.Tensor:
    """:func:`head_loss` of head ``name`` from the pieces of ``forward_split``: targets go from the host to each
    piece's card, only per-(row, track) sums cross to ``device``. ``consts_on(device)`` gives that card's constants."""
    parts = [(p.rows, head_loss_parts(p.x[name], want[p.rows, p.start:p.end].to(p.device),
                                      ok[p.rows, p.start:p.end].to(p.device), consts_on(p.device))) for p in pieces]
    return G.poisson_multinomial_combine(parts, want.shape[0], device)


class Pearson:
    """Running per-channel Pearson of predicted against measured log1p signal at ``bin`` bases (128: the track
    benchmark's resolution), over the positions given."""

    def __init__(self, n: int, bin: int = 128) -> None:
        self.bin, self.n = bin, 0
        self.s = [torch.zeros(n, dtype=torch.float64) for _ in range(5)]

    def add(self, pred_log1p: torch.Tensor, want_log1p: torch.Tensor) -> None:
        """``[B, L, T]`` each; whole bins only."""
        b, length, t = pred_log1p.shape
        keep = length - length % self.bin
        if not keep:
            return
        x = pred_log1p[:, :keep].double().reshape(b, -1, self.bin, t).mean(2).reshape(-1, t).cpu()
        y = want_log1p[:, :keep].double().reshape(b, -1, self.bin, t).mean(2).reshape(-1, t).cpu()
        self.n += x.shape[0]
        for acc, v in zip(self.s, (x.sum(0), y.sum(0), (x * y).sum(0), (x * x).sum(0), (y * y).sum(0))):
            acc += v

    def value(self) -> np.ndarray:
        """Per channel; NaN where either side is constant or nothing was added."""
        if not self.n:
            return np.full(len(self.s[0]), np.nan)
        sx, sy, sxy, sxx, syy = (v.numpy() for v in self.s)
        cov = sxy / self.n - sx * sy / self.n ** 2
        vx, vy = sxx / self.n - (sx / self.n) ** 2, syy / self.n - (sy / self.n) ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            r = cov / np.sqrt(vx * vy)
        return np.where((vx > 1e-12) & (vy > 1e-12), r, np.nan)


# ------------------------------------------------------------------------------------------------ the stage

@dataclass
class StagePlan:
    """The settings of a continuation stage (the command line's; a resume must give the same)."""
    heads: list[str]                       # every registered head, in registration order
    head_only: int = 200                   # labelled updates with the inherited weights frozen, before joint training
    lr_inherited: float = 4e-5             # peak rate of the inherited groups in joint training
    lr_head: float = 4e-4                  # peak rate of each new head
    warmup_inherited: int = 200            # joint updates to the inherited peak
    warmup_head: int = 200                 # a head's own labelled updates to its peak

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class Stage:
    """Where a continuation stands. ``state`` is what the checkpoint stores:

    - ``phase``: "head-only" or "joint";
    - ``stage_step``: optimizer steps taken in this stage;
    - ``joint_step``: of them, in joint training;
    - ``updates[name]``: labelled optimizer updates head ``name`` has taken;
    - ``active[name]``: the stage step at which the head's data was adopted (absent: registered, no data yet);
    - ``stores[name]``: the identity of the store it trains on (:attr:`TrackStore.identity`).
    """

    def __init__(self, plan: StagePlan, state: dict | None = None) -> None:
        self.plan = plan
        self.state = state if state is not None else dict(
            phase="head-only" if plan.head_only > 0 else "joint", stage_step=0, joint_step=0,
            updates={h: 0 for h in plan.heads}, active={}, stores={})
        if state is not None:
            unknown = set(self.state["updates"]) ^ set(plan.heads)
            if unknown:
                raise ValueError(f"the checkpoint's stage has other heads than this run: {sorted(unknown)}")

    def activate(self, name: str, identity: dict) -> None:
        """Adopt head ``name``'s store at this step (a checkpoint boundary); a second call must give the same store."""
        if name not in self.state["updates"]:
            raise ValueError(f"head {name!r} is not registered: {self.plan.heads}")
        saved = self.state["stores"].get(name)
        if saved is not None and saved != identity:
            raise ValueError(f"head {name!r} was activated on another store (ids, scales or record differ)")
        if saved is None:
            self.state["stores"][name] = identity
            self.state["active"][name] = self.state["stage_step"]

    @property
    def active(self) -> list[str]:
        return [h for h in self.plan.heads if h in self.state["active"]]

    @property
    def inherited_trains(self) -> bool:
        return self.state["phase"] == "joint"

    def lr_inherited(self, factor: float = 1.0) -> float:
        """The inherited groups' rate at the next update (0 in the head-only phase). ``factor`` is the run's own
        schedule at this step as a share of its peak (genome_lr / --lr: warmup, cosine, cooldown), so the stage
        scales the scheduled rate and never replaces it."""
        if not self.inherited_trains:
            return 0.0
        return self.plan.lr_inherited * factor \
            * min(1.0, (self.state["joint_step"] + 1) / max(self.plan.warmup_inherited, 1))

    def lr_head(self, name: str, factor: float = 1.0) -> float:
        """Head ``name``'s rate at its next labelled update (0 before its data is adopted), under the same ``factor``."""
        if name not in self.state["active"]:
            return 0.0
        return self.plan.lr_head * factor \
            * min(1.0, (self.state["updates"][name] + 1) / max(self.plan.warmup_head, 1))

    def stepped(self, labelled: list[str]) -> str | None:
        """Record one optimizer step in which the heads ``labelled`` had supervised positions. Returns
        "joint" at the step the stage leaves the head-only phase (the caller saves a checkpoint and scores there)."""
        s = self.state
        for h in labelled:
            s["updates"][h] += 1
        s["stage_step"] += 1
        if s["phase"] == "joint":
            s["joint_step"] += 1
            return None
        first = self.active[:1]
        if first and s["updates"][first[0]] >= self.plan.head_only:
            s["phase"] = "joint"
            return "joint"
        return None


def param_groups(model: G.KronGenome, weight_decay: float = 0.1) -> list[dict]:
    """AdamW groups for a stage: the inherited weights (decayed, then norms/biases/embeddings as genome-train splits
    them) and one pair per further head, each tagged ``role`` ("inherited" or the head's name) so a rate goes to it."""
    def decays(n: str, p: torch.Tensor) -> bool:
        return p.ndim >= 2 and "context" not in n

    groups = []
    named = [(n, p) for n, p in model.named_parameters()]
    inherited = [(n, p) for n, p in named if not n.startswith("xheads.")]
    for role, rows in [("inherited", inherited)] + [
            (h, [(n, p) for n, p in named if n.startswith(f"xheads.{h}.")]) for h in model.xheads]:
        groups.append(dict(params=[p for n, p in rows if decays(n, p)], weight_decay=weight_decay, role=role))
        groups.append(dict(params=[p for n, p in rows if not decays(n, p)], weight_decay=0.0, role=role))
    return groups


def set_rates(opt: torch.optim.Optimizer, stage: Stage, labelled: list[str], factor: float = 1.0) -> dict[str, float]:
    """Set every group's rate for the coming step (``factor``: the run's schedule as a share of its peak, see
    :meth:`Stage.lr_inherited`): the inherited groups from the stage, each head's own schedule
    when it has labels in this step, else 0 (AdamW then leaves it alone: no gradient, and no decay at rate 0).
    In the head-only phase the caller also turns the inherited parameters' ``requires_grad`` off (:func:`freeze`),
    so their moment estimates do not move either. Returns the rates by role."""
    rates = {"inherited": stage.lr_inherited(factor), **{h: (stage.lr_head(h, factor) if h in labelled else 0.0)
                                                          for h in stage.plan.heads}}
    for g in opt.param_groups:
        g["lr"] = rates[g["role"]]
    return rates


def freeze(model: G.KronGenome, stage: Stage) -> None:
    """``requires_grad`` of the inherited parameters by the stage's phase (off while only the heads train)."""
    on = stage.inherited_trains
    for n, p in model.named_parameters():
        if not n.startswith("xheads."):
            p.requires_grad_(on)
