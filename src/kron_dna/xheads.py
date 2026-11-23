"""Training a further per-base head (``KronGenomeConfig.xheads``) on a measured track store, on top of a trained model.

Three parts, each used by ``genome-train`` and testable alone:

- :class:`TrackStore`: a store written by ``scripts/kron_dna/prep_bigwig_tracks.py`` on the primary genome's contig
  layout (``dense.u8`` + ``dense.json``), read per window; its identity (channel ids in order, encoding, scales) is
  what a checkpoint records and a resume checks. :class:`LabelStore`: 0/1 labels (stranded splice sites) the same
  way; :func:`open_store` picks by what the folder holds.
- :func:`head_loss` / :func:`head_loss_parts`: the track objective of the inherited NTv3 head (Poisson total +
  multinomial profile on AlphaGenome-scaled targets) for one head, whole or from the pieces of a split window.
- :class:`Stage`: where a continuation stands - head-only or joint, how many labelled updates each head has taken,
  when each was activated - and the learning rate of every optimizer group from it. It is a plain dict in the
  checkpoint, so a resumed job picks the same phase, data and rates.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from kron_dna import genome as G


# ------------------------------------------------------------------------------------------------ the store

_FDS: dict = {}


def read_rows(mm, a: int, b: int) -> np.ndarray:
    """Rows ``a:b`` of a row-major memmap read with ``os.pread`` at their offset, not through the mapping: a run
    touches hundreds of GB of stores at random, and every page read through a mapping stays mapped in the process
    (the trainer measured F at 342 GB RssFile, 17 of 20 s in the kernel per step, at 15 s/step instead of 2.8). The
    page cache serves both alike. Anything that is not a file-backed ``np.memmap`` (a test's stand-in) is sliced."""
    fn = getattr(mm, "filename", None)
    # an ndarray memmap only: a test's recording stand-in (not an ndarray, even where it replaces np.memmap) keeps
    # its reads going through its own __getitem__
    if not (isinstance(mm, np.ndarray) and isinstance(mm, np.memmap)) or fn is None:
        return np.asarray(mm[a:b])
    b = min(b, mm.shape[0])
    if b <= a:
        return np.zeros((0,) + mm.shape[1:], dtype=mm.dtype)
    key = (os.getpid(), fn)                           # one descriptor per file and process (a fork reopens)
    fd = _FDS.get(key)
    if fd is None:
        fd = _FDS[key] = os.open(fn, os.O_RDONLY)
    row = int(np.prod(mm.shape[1:], dtype=np.int64)) * mm.dtype.itemsize
    want = (b - a) * row
    buf = os.pread(fd, want, int(mm.offset) + a * row)
    while len(buf) < want:                            # pread may return short; read the rest
        more = os.pread(fd, want - len(buf), int(mm.offset) + a * row + len(buf))
        if not more:
            raise OSError(f"{fn}: short read at row {a}")
        buf += more
    return np.frombuffer(buf, dtype=mm.dtype).reshape((b - a,) + mm.shape[1:])


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
    kind = "tracks"                   # quantitative: Poisson-multinomial loss, Pearson of log1p signal at 128 bp

    def consts(self, device) -> tuple:
        """(scales, non-zero means, RNA flags) on ``device``."""
        return tuple(torch.as_tensor(np.asarray(v)).to(device) for v in (self.scales, self.nonzero_mean, self.is_rna))

    def scorer(self, bin: int = 1):
        """Pearson at 128 bp; a head predicting at ``bin`` bases gives bins already (``bin`` divides 128)."""
        return Pearson(len(self.ids), max(1, 128 // bin))

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
        out = self.empty(width)
        out[:take] = read_rows(self.dense, at, at + take)
        if flip:
            out[:take] = out[:take][::-1][:, self.swap]
        return out

    def empty(self, width: int) -> np.ndarray:
        """The targets of a row with no supervision (another genome's window): zeros of a window's shape."""
        return np.zeros((width, len(self.ids)), dtype=np.uint8)

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


@dataclass
class LabelStore:
    """Per-base 0/1 labels on a prepared genome's layout, one bit of ``labels[at]`` per channel: ``splice.json`` +
    ``splice.u8`` (``scripts/kron_dna/prep_splice_labels.py``: donor and acceptor per strand), or ``labels.json`` +
    ``labels.u8``/``labels.u16`` (its ``dtype``; ``prep_gene_structure_labels.py``). Every channel id ends in
    ``_plus`` or ``_minus`` (or neither, unstranded); ``swap`` gives the channel a window read on the other strand uses."""
    path: Path
    ids: list[str]
    labels: np.ndarray                # [total_bases] uint8, bit k = channel k
    swap: np.ndarray                  # [T] the other strand's channel
    positives: np.ndarray             # [T] labelled bases per channel
    genome: str
    identity: dict = field(default_factory=dict)
    train_bases: int = 0              # bases of the training split the positives were counted on (0: the whole store)
    kind = "binary"                   # classification: weighted BCE, average precision on the held-out chromosomes

    @classmethod
    def open(cls, path: str | Path) -> "LabelStore":
        path = Path(path)
        stem = "labels" if (path / "labels.json").exists() else "splice"
        raw = (path / f"{stem}.json").read_bytes()
        j = json.loads(raw)
        dtype = np.dtype(j.get("dtype", "uint8"))
        if dtype not in (np.dtype("uint8"), np.dtype("uint16")):
            raise ValueError(f"{path}: label dtype {dtype} (uint8 or uint16)")
        data = path / (f"{stem}.u8" if dtype == np.uint8 else f"{stem}.u16")
        ids = sorted(j["bits"], key=j["bits"].get)
        if len(ids) > 8 * dtype.itemsize:
            raise ValueError(f"{path}: {len(ids)} channels do not fit {dtype}")
        if [j["bits"][i] for i in ids] != list(range(len(ids))):
            raise ValueError(f"{path}: label bits must be 0..{len(ids) - 1}, got {j['bits']}")
        total = int(j["total_bases"])
        size = data.stat().st_size
        if size != total * dtype.itemsize:
            raise ValueError(f"{data} has {size:,} bytes, its record says {total:,} bases of {dtype}")
        at = {i: k for k, i in enumerate(ids)}
        swap = np.arange(len(ids))
        for k, i in enumerate(ids):
            other = i[:-4] + "minus" if i.endswith("_plus") else i[:-5] + "plus" if i.endswith("_minus") else i
            if other not in at:
                raise ValueError(f"{path}: stranded channel {i} has no partner")
            swap[k] = at[other]
        # the class weights count the training split only (review, 23 Nov): train_sites and train_bases where the
        # label script recorded them, else (a store made before) the whole store
        positives = np.array([int(j["check"][i].get("train_sites", j["check"][i]["sites"])) for i in ids],
                             dtype=np.int64)
        train_bases = int(j.get("train_bases") or total)
        identity = dict(ids=ids, encoding="one bit per channel", genome=j.get("genome"), total_bases=total,
                        definition=j.get("definition"), labels_sha256=j.get(f"{stem}_sha256"),
                        record_sha256=hashlib.sha256(raw).hexdigest())
        return cls(path=path, ids=ids, labels=np.memmap(data, dtype=dtype, mode="r"), swap=swap,
                   positives=positives, genome=str(j.get("genome")), identity=identity, train_bases=train_bases)

    @property
    def dense(self) -> np.ndarray:                    # for the size check a store's user makes against the genome
        return self.labels

    def window(self, at: int, take: int, width: int, flip: bool = False) -> np.ndarray:
        """Targets ``[width, T]`` uint8 (0 or 1) of bases ``at:at + take``; ``flip``: the other strand."""
        out = np.zeros((width, len(self.ids)), dtype=np.uint8)
        bits = read_rows(self.labels, at, at + take)
        out[:take] = (bits[:, None] >> np.arange(len(self.ids), dtype=bits.dtype)[None, :]) & 1
        if flip:
            out[:take] = out[:take][::-1][:, self.swap]
        return out

    def empty(self, width: int) -> np.ndarray:
        return np.zeros((width, len(self.ids)), dtype=np.uint8)

    def differences(self, saved: dict) -> list[str]:
        return [f"{k}: checkpoint {saved.get(k)!r}, now {self.identity.get(k)!r}"
                for k in ("ids", "encoding", "genome", "total_bases", "definition", "labels_sha256")
                if saved.get(k) != self.identity.get(k)]

    def consts(self, device) -> tuple:
        """(positive weights,) on ``device``: sqrt(negatives / positives) per channel, between 1 and 100, as the
        prepared genome's own rare tracks are weighted."""
        share = self.positives / max(self.train_bases or len(self.labels), 1)
        w = np.clip(np.sqrt((1 - share) / np.maximum(share, 1e-12)), 1, 100).astype(np.float32)
        return (torch.as_tensor(w).to(device),)

    def scorer(self, bin: int = 1):
        if bin != 1:
            raise ValueError("a binary head predicts per base")
        return AveragePrecision(len(self.ids))


@dataclass
class BinnedTrackStore:
    """A pre-binned track store (``scripts/kron_dna/bin_track_store.py``): per cell of ``bin`` bases on the global
    grid, the mean training target and the mean log1p signal of each channel. For a head that predicts at ``bin``
    bases; a window must start on the grid (``genome-train`` snaps primary-genome windows to it)."""
    path: Path
    ids: list[str]
    target: np.ndarray                # [cells, T] float16: mean of targets_of over the cell
    log1p: np.ndarray                 # [cells, T] float16: mean of log1p signal over the cell
    bin: int
    scales: np.ndarray
    nonzero_mean: np.ndarray
    is_rna: np.ndarray
    swap: np.ndarray
    genome: str
    identity: dict = field(default_factory=dict)
    kind = "tracks_binned"

    def consts(self, device) -> tuple:
        return tuple(torch.as_tensor(np.asarray(v)).to(device) for v in (self.scales, self.nonzero_mean, self.is_rna))

    def scorer(self, bin: int = 1):
        """Pearson over the cells as given (at ``self.bin`` bases already)."""
        return Pearson(len(self.ids), max(1, 128 // self.bin) if self.bin < 128 else 1)

    @classmethod
    def open(cls, path: str | Path) -> "BinnedTrackStore":
        path = Path(path)
        raw = (path / "binned.json").read_bytes()
        j = json.loads(raw)
        ids, cells, b = list(j["ids"]), int(j["cells"]), int(j["bin"])
        at = {i: k for k, i in enumerate(ids)}
        swap = np.arange(len(ids))
        for k, i in enumerate(ids):
            if i.endswith(("_P", "_M")):
                swap[k] = at.get(i[:-1] + ("M" if i.endswith("_P") else "P"), k)
        identity = dict(ids=ids, encoding=f"float16 cell means at {b} bp (target, log1p)", bin=b,
                        scales=[float(v) for v in j["scales"]], nonzero_mean=[float(v) for v in j["nonzero_mean"]],
                        genome=j.get("genome"), total_bases=int(j["total_bases"]),
                        record_sha256=hashlib.sha256(raw).hexdigest())
        mm = lambda n: np.memmap(path / f"{n}.f16", dtype=np.float16, mode="r", shape=(cells, len(ids)))
        return cls(path=path, ids=ids, target=mm("target"), log1p=mm("log1p"), bin=b,
                   scales=np.asarray(j["scales"], dtype=np.float32),
                   nonzero_mean=np.asarray(j["nonzero_mean"], dtype=np.float32),
                   is_rna=np.array(["RNA-seq" in a for a in j["assays"]], dtype=bool), swap=swap,
                   genome=str(j.get("genome")), identity=identity)

    @property
    def total_bases(self) -> int:
        return int(self.identity["total_bases"])

    def empty(self, width: int) -> np.ndarray:
        return np.zeros((width // self.bin, len(self.ids), 2), dtype=np.float16)

    def window(self, at: int, take: int, width: int, flip: bool = False) -> np.ndarray:
        """``[width / bin, T, 2]`` float16 (target, log1p) of the cells in bases ``at:at + take``. The window must
        start on the grid; cells past ``take // bin`` stay zero (their bases are padding, never supervised)."""
        out = self.empty(width)
        n = take // self.bin
        if not n:
            return out
        if at % self.bin:
            raise ValueError(f"a window at base {at} is off the store's {self.bin}-base grid")
        c = at // self.bin
        out[:n, :, 0] = read_rows(self.target, c, c + n)
        out[:n, :, 1] = read_rows(self.log1p, c, c + n)
        if flip:
            out[:n] = out[:n][::-1][:, self.swap]
        return out

    def differences(self, saved: dict) -> list[str]:
        return [f"{k}: checkpoint {saved.get(k)!r}, now {self.identity.get(k)!r}"
                for k in ("ids", "encoding", "bin", "genome", "total_bases") if saved.get(k) != self.identity.get(k)]


def channel_keys(store) -> list:
    """What makes two stores' channels the same output: for a label store its channel id (the same definition in
    every genome); for a track store (assay, target, biosample, strand) from its record, the assay folded so that
    the RNA-seq kinds agree (biosample terms are ENCODE's ontology names, shared by human and mouse tissues)."""
    if store.kind == "binary":
        return list(store.ids)
    rec = json.loads((Path(store.path) / ("binned.json" if store.kind == "tracks_binned" else "dense.json")).read_text())
    keys = []
    for i, t in zip(store.ids, rec.get("tracks") or [{}] * len(store.ids)):
        assay = "RNA-seq" if "RNA-seq" in str(t.get("assay", "")) else str(t.get("assay", ""))
        strand = i[-2:] if i.endswith(("_P", "_M")) else ""
        keys.append((assay, str(t.get("target") or ""), str(t.get("biosample") or i), strand))
    return keys


class SpeciesStores:
    """One further head's stores on several genomes (``--xhead-species``). The head's channels are the primary
    store's, then each further genome's channels that match none of them (an ``<genome>:``-prefixed id); a channel
    of a further genome that matches a primary one (:func:`channel_keys`: the same label, or the same assay, target,
    biosample and strand, e.g. DNase in liver) is that channel, trained by both genomes' windows. A row of genome
    ``g`` reads ``g``'s store into ``g``'s channels, zeros elsewhere; its channel mask (:meth:`mask`) is those
    channels, so the loss and the held-out score see only what the row has. Every part is of one kind; channels are
    shared only between stores whose targets are already normalised per track (pre-binned or label stores), since
    a per-base store's bytes decode with its own track's scale."""

    def __init__(self, primary, extra: dict, share: bool = True) -> None:
        self.parts = [(None, primary)] + list(extra.items())
        kinds = {st.kind for _, st in self.parts}
        if len(kinds) != 1:
            raise ValueError(f"the stores of one head must be of one kind, got {sorted(kinds)}")
        self.kind = primary.kind
        if self.kind == "tracks_binned" and len({st.bin for _, st in self.parts}) != 1:
            raise ValueError("the pre-binned stores of one head must share a bin")
        share = share and self.kind != "tracks"
        self.primary, self.path, self.genome = primary, primary.path, primary.genome
        first = {}
        for k, key in enumerate(channel_keys(primary)):
            first.setdefault(key, k)
        self.cols = {None: np.arange(len(primary.ids))}
        self.ids = list(primary.ids)
        for g, st in extra.items():
            cols = np.empty(len(st.ids), dtype=np.int64)
            taken = set()                             # a primary channel is shared with one channel of each genome:
            for k, key in enumerate(channel_keys(st)):  # a second experiment of that biosample gets its own channel
                if share and key in first and first[key] not in taken:
                    cols[k] = first[key]
                    taken.add(first[key])
                else:
                    cols[k] = len(self.ids)
                    self.ids.append(f"{g}:{st.ids[k]}")
            self.cols[g] = cols
        # each channel's strand partner, in head columns; a channel two genomes share must have one partner
        sw = np.full(len(self.ids), -1)
        for g, st in self.parts:
            cols, mine = self.cols[g], self.cols[g][np.asarray(st.swap)]
            clash = (sw[cols] >= 0) & (sw[cols] != mine)
            if clash.any():
                raise ValueError(f"--xhead-species {g}: a shared stranded channel's partner differs between genomes")
            sw[cols] = mine
        self.swap = sw
        self.shared = {g: int((c < len(primary.ids)).sum()) for g, c in self.cols.items() if g is not None}
        if self.kind == "binary":
            pos = np.zeros(len(self.ids), dtype=np.int64)
            for g, st in self.parts:
                np.add.at(pos, self.cols[g], st.positives)
            self.positives = pos
        if self.kind == "tracks_binned":
            self.bin = primary.bin
        self.identity = dict(primary.identity, species={g: dict(st.identity, path=str(st.path),
                                                                channels=self.cols[g].tolist())
                                                       for g, st in self.parts if g is not None})

    @property
    def dense(self):                                  # the primary's, for the size check against the primary genome
        return self.primary.dense

    @property
    def total_bases(self) -> int:
        return self.primary.total_bases

    def store_of(self, genome: str | None):
        return dict(self.parts)[genome]

    def mask(self, genome: str | None) -> np.ndarray:
        """``[T]`` bool: the channels a row of ``genome`` (None: the primary) supervises."""
        m = np.zeros(len(self.ids), dtype=bool)
        m[self.cols[genome]] = True
        return m

    def _place(self, part: np.ndarray, genome: str | None) -> np.ndarray:
        out = np.zeros(part.shape[:1] + (len(self.ids),) + part.shape[2:], dtype=part.dtype)
        cols = self.cols[genome]
        if len(cols) and cols[-1] - cols[0] + 1 == len(cols) and (np.diff(cols) == 1).all():
            out[:, cols[0]:cols[-1] + 1] = part       # a contiguous range (the primary's): a plain slice copy
        else:
            out[:, cols] = part
        return out

    def window(self, at: int, take: int, width: int, flip: bool = False, genome: str | None = None) -> np.ndarray:
        return self._place(self.store_of(genome).window(at, take, width, flip), genome)

    def empty(self, width: int) -> np.ndarray:
        """Zeros of a window's shape at the head's full width, left untouched (np.zeros maps them lazily: a row
        that reads nothing costs nothing, as with a plain store)."""
        e = self.primary.empty(width)
        return np.zeros(e.shape[:1] + (len(self.ids),) + e.shape[2:], dtype=e.dtype)

    def consts(self, device) -> tuple:
        """Per channel, the primary store's constants where it has the channel, else the further genome's."""
        out = None
        for g, st in reversed(self.parts):            # the primary written last, so its values stand on shared channels
            c = st.consts(device)
            if out is None:
                out = [torch.zeros((len(self.ids),) + tuple(v.shape[1:]), dtype=v.dtype, device=v.device) for v in c]
            cols = torch.as_tensor(self.cols[g], device=c[0].device)
            for o, v in zip(out, c):
                o[cols] = v
        return tuple(out)

    def scorer(self, bin: int = 1):
        sc = self.primary.scorer(bin)
        return Pearson(len(self.ids), sc.bin) if isinstance(sc, Pearson) else AveragePrecision(len(self.ids))

    def differences(self, saved: dict) -> list[str]:
        out = self.primary.differences({k: v for k, v in saved.items() if k != "species"})
        if saved.get("species") != self.identity.get("species"):
            out.append(f"species: checkpoint {sorted(saved.get('species') or {})}, now {sorted(self.identity['species'])}")
        return out


def train_counts(labels: np.ndarray, split: np.ndarray, bits: dict, step: int = 1 << 26) -> tuple[dict, int]:
    """Per channel of a bit-packed label array, its positives on the training split (code 1), and the split's bases."""
    pos = {k: 0 for k in bits}
    n = 0
    for a in range(0, len(labels), step):
        m = np.asarray(split[a:a + step]) == 1
        lab = np.asarray(labels[a:a + step])[m]
        n += int(m.sum())
        for k, b in bits.items():
            pos[k] += int(np.count_nonzero(lab & (1 << b)))
    return pos, n


def store_of_record(entry: dict):
    """A head's store as a run record lists it (``xheads.stores[name]``): the store at its path, or, for a head
    with further genomes, the same :class:`SpeciesStores` (the channel map is checked against the record)."""
    st = open_store(entry["path"])
    if not entry.get("species"):
        return st
    out = SpeciesStores(st, {g: open_store(e["path"]) for g, e in entry["species"].items()})
    for g, e in entry["species"].items():
        if out.cols[g].tolist() != list(e["channels"]):
            raise ValueError(f"{entry['path']}: genome {g}'s channels moved since the run ({e['channels'][:6]} ...)")
    return out


def open_store(path: str | Path):
    """The store at ``path``: measured tracks (``dense.json``), pre-binned tracks (``binned.json``) or 0/1 labels
    (``splice.json``)."""
    path = Path(path)
    if (path / "binned.json").exists():
        return BinnedTrackStore.open(path)
    if (path / "dense.json").exists():
        return TrackStore.open(path)
    if (path / "splice.json").exists() or (path / "labels.json").exists():
        return LabelStore.open(path)
    raise ValueError(f"{path} holds none of binned.json, dense.json (tracks), splice.json or labels.json (labels)")


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


def _bce_sum(pred: torch.Tensor, want: torch.Tensor, ok: torch.Tensor, consts: tuple,
             cmask: torch.Tensor | None = None) -> torch.Tensor:
    per = F.binary_cross_entropy_with_logits(pred.float(), want.float(), pos_weight=consts[0], reduction="none")
    return (per * _values(ok.unsqueeze(-1), cmask)).sum()


def _values(m: torch.Tensor, cmask: torch.Tensor | None) -> torch.Tensor:
    """A position mask ``[B, L, 1]`` times the rows' channel mask ``[B, T]`` (None: every channel)."""
    return m if cmask is None else m * cmask.to(m.dtype).unsqueeze(1)


def _n_values(ok: torch.Tensor, t: int, cmask: torch.Tensor | None) -> torch.Tensor:
    """The number of supervised (position, channel) values: ``ok`` [B, L] summed over the channels each row has."""
    return (ok.sum() * t if cmask is None else (ok.sum(1) * cmask.to(ok.dtype).sum(1)).sum()).clamp_min(1)


def span_of(ok: torch.Tensor, unit: int) -> tuple[int, int] | None:
    """The smallest stretch of window bases, widened to multiples of ``unit``, that holds every supervised position
    of ``ok`` [B, L]; None when nothing is supervised. The heads run on it alone (``KronGenome.x_span``)."""
    cols = torch.nonzero(ok.sum(0) > 0)
    if not cols.numel():
        return None
    lo, hi = int(cols[0]) // unit * unit, -(-(int(cols[-1]) + 1) // unit) * unit
    return lo, min(hi, ok.shape[1])


def pool_bins(x: torch.Tensor, bin: int) -> torch.Tensor:
    """``[B, L, T]`` -> ``[B, L / bin, T]``: the mean over each bin (``bin`` divides L, as spans are whole core steps)."""
    if bin == 1:
        return x
    b, length, t = x.shape
    return x.reshape(b, length // bin, bin, t).mean(2)


def pool_ok(ok: torch.Tensor, bin: int) -> torch.Tensor:
    """``[B, L]`` -> ``[B, L / bin]``: a bin is supervised when all its bases are."""
    if bin == 1:
        return ok
    b, length = ok.shape
    return ok.reshape(b, length // bin, bin).amin(2)


def binned_targets(want: torch.Tensor, ok: torch.Tensor, consts: tuple, bin: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A track head's training targets and mask at ``bin`` bases: the per-base targets (:func:`targets_of`) averaged
    over each bin, so a binned head learns the bin's mean signal in the same units as a per-base head."""
    return pool_bins(targets_of(want, *consts), bin), pool_ok(ok, bin)


def head_loss(pred: torch.Tensor, want: torch.Tensor, ok: torch.Tensor, consts: tuple,
              kind: str = "tracks", bin: int = 1, cmask: torch.Tensor | None = None) -> torch.Tensor:
    """The head's loss on a whole window: ``pred`` [B, L, T] raw outputs, ``want`` [B, L, T] stored targets, ``ok``
    [B, L] from :func:`supervised`, ``consts`` the store's on ``pred``'s device. ``kind`` "tracks": the NTv3 head's
    Poisson-multinomial on scaled signal; "binary": weighted BCE, the mean over supervised values. ``cmask`` [B, T]:
    the channels each row supervises (a further species' rows, ``SpeciesStores``); None: all."""
    if kind == "binary":
        if bin != 1:
            raise ValueError("a binary head predicts per base")
        return _bce_sum(pred, want, ok, consts, cmask) / _n_values(ok, pred.shape[-1], cmask)
    y, m = (want[..., 0].float(), pool_ok(ok, bin)) if kind == "tracks_binned" else binned_targets(want, ok, consts, bin)
    return G.poisson_multinomial_loss(F.softplus(pred.float()), y, _values(m.unsqueeze(-1), cmask))


def head_loss_parts(pred: torch.Tensor, want: torch.Tensor, ok: torch.Tensor, consts: tuple, bin: int = 1,
                    kind: str = "tracks", cmask: torch.Tensor | None = None) -> dict:
    """The same for one piece of a split window, on the piece's card: the sums
    :func:`kron_dna.genome.poisson_multinomial_combine` joins on the first card into :func:`head_loss`'s value."""
    y, m = (want[..., 0].float(), pool_ok(ok, bin)) if kind == "tracks_binned" else binned_targets(want, ok, consts, bin)
    return G.poisson_multinomial_parts(F.softplus(pred.float()), y, _values(m.unsqueeze(-1), cmask))


def split_head_loss(pieces: list, name: str, want: torch.Tensor, ok: torch.Tensor, consts_on, device,
                    kind: str = "tracks", bin: int = 1, cmask: torch.Tensor | None = None) -> torch.Tensor:
    """:func:`head_loss` of head ``name`` from the pieces of ``forward_split``: targets go from the host to each
    piece's card, only per-(row, track) sums cross to ``device``. ``consts_on(device)`` gives that card's constants.
    Each piece's outputs cover its ``x_at`` bases (the model's ``x_span`` within the piece); ``ok`` must be zero
    outside the span, which :func:`span_of` guarantees."""
    cm = (lambda p: None) if cmask is None else (lambda p: cmask[p.rows].to(p.device))
    if kind == "binary":
        tot = sum(_bce_sum(p.x[name], want[p.rows, p.x_at[0]:p.x_at[1]].to(p.device),
                           ok[p.rows, p.x_at[0]:p.x_at[1]].to(p.device), consts_on(p.device), cm(p)).to(device)
                  for p in pieces)
        return tot / _n_values(ok, want.shape[-1], cmask).to(device)
    cell = bin if kind == "tracks_binned" else 1      # a pre-binned store's targets are per cell, not per base
    parts = [(p.rows, head_loss_parts(p.x[name], want[p.rows, p.x_at[0] // cell:p.x_at[1] // cell].to(p.device),
                                      ok[p.rows, p.x_at[0]:p.x_at[1]].to(p.device), consts_on(p.device), bin, kind,
                                      cm(p)))
             for p in pieces]
    return G.poisson_multinomial_combine(parts, want.shape[0], device)


class Pearson:
    """Running per-channel Pearson of predicted against measured log1p signal at ``bin`` bases (128: the track
    benchmark's resolution), over the positions given."""

    def __init__(self, n: int, bin: int = 128) -> None:
        self.bin = bin
        self.n = torch.zeros(n, dtype=torch.float64)  # bins added, per channel
        self.s = [torch.zeros(n, dtype=torch.float64) for _ in range(5)]

    def add(self, pred_log1p: torch.Tensor, want_log1p: torch.Tensor, cmask: torch.Tensor | None = None,
            pmask: torch.Tensor | None = None) -> None:
        """``[B, L, T]`` each; whole bins only. ``cmask`` [B, T]: the channels each row counts for (None: all).
        ``pmask`` [B, L]: the positions that count (None: all); a bin counts only when all its positions do, so
        padding and positions outside the held-out contigs never enter the score."""
        b, length, t = pred_log1p.shape
        keep = length - length % self.bin
        if not keep:
            return
        w = torch.ones(b, t, dtype=torch.float64) if cmask is None else cmask.double().cpu()
        w = w.unsqueeze(1).expand(b, keep // self.bin, t)
        if pmask is not None:
            pm = pmask[:, :keep].bool().cpu().reshape(b, -1, self.bin).all(2)
            w = w * pm.unsqueeze(-1).double()
        w = w.reshape(-1, t)
        x = pred_log1p[:, :keep].double().reshape(b, -1, self.bin, t).mean(2).reshape(-1, t).cpu()
        y = want_log1p[:, :keep].double().reshape(b, -1, self.bin, t).mean(2).reshape(-1, t).cpu()
        self.n += w.sum(0)
        for acc, v in zip(self.s, ((w * x).sum(0), (w * y).sum(0), (w * x * y).sum(0), (w * x * x).sum(0),
                                   (w * y * y).sum(0))):
            acc += v

    def value(self) -> np.ndarray:
        """Per channel; NaN where either side is constant or nothing was added."""
        n = self.n.numpy()
        sx, sy, sxy, sxx, syy = (v.numpy() for v in self.s)
        with np.errstate(divide="ignore", invalid="ignore"):
            cov = sxy / n - sx * sy / n ** 2
            vx, vy = sxx / n - (sx / n) ** 2, syy / n - (sy / n) ** 2
            r = cov / np.sqrt(vx * vy)
        return np.where((n > 0) & (vx > 1e-12) & (vy > 1e-12), r, np.nan)


class AveragePrecision:
    """Per-channel average precision of a binary head's logits against its 0/1 labels over the positions added
    (positives are about 1 base in 16,000 for splice sites, so AUROC would say little). Every positive is kept;
    of the negatives, those scoring above a floor that keeps about the top 2 % of each batch, which decide the
    ranking near the positives; the rest count as ranked below everything kept."""

    def __init__(self, n: int, keep: float = 0.02) -> None:
        self.keep, self.n = keep, 0                    # n: positives seen, all channels
        self.pos = [[] for _ in range(n)]
        self.neg = [[] for _ in range(n)]
        self.neg_total = [0] * n
        self.floor = [float("-inf")] * n              # above it every negative of every batch was kept

    def add(self, pred: torch.Tensor, want: torch.Tensor, cmask: torch.Tensor | None = None,
            pmask: torch.Tensor | None = None) -> None:
        """``[B, L, T]`` logits and 0/1 labels; ``cmask`` [B, T]: the channels each row counts for (None: all);
        ``pmask`` [B, L]: the positions that count (None: all)."""
        b, length, t = pred.shape
        x, y = pred.detach().float().reshape(-1, t).cpu(), want.reshape(-1, t).cpu() > 0
        w = (torch.ones(b, t, dtype=torch.bool) if cmask is None else cmask.bool().cpu()).unsqueeze(1)
        w = w.expand(b, length, t)
        if pmask is not None:
            w = w & pmask.bool().cpu().unsqueeze(-1)
        w = w.reshape(-1, t)
        for k in range(t):
            p, q = x[y[:, k] & w[:, k], k], x[~y[:, k] & w[:, k], k]
            self.pos[k].append(p)
            self.neg_total[k] += int(q.numel())
            if q.numel():
                top = max(1, int(q.numel() * self.keep))
                kept = torch.topk(q, top).values
                self.neg[k].append(kept)
                if top < q.numel():
                    self.floor[k] = max(self.floor[k], float(kept.min()))
            self.n += int(p.numel())

    def value(self) -> np.ndarray:
        """Per channel; NaN without a positive. Exact for positives above the floor (the highest of the batches' lowest
        kept negatives); a positive below it is placed after every negative (a lower bound on its precision)."""
        out = np.full(len(self.pos), np.nan)
        for k in range(len(self.pos)):
            p = torch.cat(self.pos[k]) if self.pos[k] else torch.zeros(0)
            if not p.numel():
                continue
            q = torch.cat(self.neg[k]) if self.neg[k] else torch.zeros(0)
            floor = self.floor[k]
            p = torch.sort(p, descending=True).values
            above = torch.searchsorted(-torch.sort(q, descending=True).values, -p, right=False).double()  # negatives > p
            above = torch.where(p >= floor, above, torch.full_like(above, float(self.neg_total[k])))
            rank = torch.arange(1, p.numel() + 1, dtype=torch.float64)
            out[k] = float((rank / (rank + above)).mean())
        return out


def positions_at(ok: torch.Tensor, length: int) -> torch.Tensor:
    """A per-base mask ``ok`` [B, L] at the resolution of a prediction of ``length`` positions (L a multiple of it):
    a position counts only when every base it covers does."""
    f = max(1, ok.shape[1] // max(1, length))
    return ok[:, :length * f].reshape(ok.shape[0], length, f).min(-1).values > 0


def scored_n(scorer) -> int:
    """What a scorer counted, as one number: a Pearson's bins (of its most-counted channel; channels differ when rows
    carry channel masks), an average precision's positives."""
    n = scorer.n
    return int(n.max()) if torch.is_tensor(n) else int(n)


def eval_pair(store, pred: torch.Tensor, want: torch.Tensor, consts: tuple, bin: int = 1
              ) -> tuple[torch.Tensor, torch.Tensor]:
    """What the store's scorer is given for held-out rows: predicted and measured log1p signal for tracks, logits and
    labels for binary heads."""
    if store.kind == "binary":
        return pred.float(), want
    if store.kind == "tracks_binned":                 # the cell means of log1p signal, stored
        return signal_of(F.softplus(pred.float()), consts[1], consts[2]), want[..., 1].float()
    # a binned head's prediction is its bin's mean; the target side is the bin's mean of log1p signal
    return signal_of(F.softplus(pred.float()), consts[1], consts[2]), pool_bins(want.float() / 255.0 * consts[0], bin)


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


def added_heads(saved_plan: dict, plan: StagePlan) -> list[str]:
    """The heads ``plan`` registers beyond a checkpoint's ``saved_plan``, when every other setting is the same and
    the checkpoint's heads come first in the same order; raises ValueError otherwise (a head removed, reordered, or
    another setting changed: that is another stage, not a continuation)."""
    old, new = list(saved_plan["heads"]), list(plan.heads)
    rest_old = {k: v for k, v in saved_plan.items() if k != "heads"}
    rest_new = {k: v for k, v in plan.to_dict().items() if k != "heads"}
    if rest_old != rest_new:
        raise ValueError(f"the stage was started with other settings: {rest_old}, now {rest_new}")
    if new[:len(old)] != old:
        raise ValueError(f"the checkpoint's heads {old} must come first and in this order, got {new}")
    return new[len(old):]


def extend_state(state: dict, added: list[str]) -> dict:
    """A checkpoint's stage ``state`` with ``added`` heads registered: no labelled update yet, not active; the
    phase and every other head's counts stand, so a head added in joint training starts on its own warmup while
    nothing else restarts."""
    out = dict(state, updates=dict(state["updates"]))
    for h in added:
        out["updates"][h] = 0
    return out


def merge_optimizer_state(saved: dict, fresh: dict, n_old_groups: int) -> dict:
    """An optimizer state for a model that gained heads: ``saved`` (the checkpoint's) for its first ``n_old_groups``
    groups, ``fresh`` (this run's new optimizer) for the groups after them. The groups are role-ordered
    (:func:`param_groups`: inherited, then each head in registration order), so a head appended at the end leaves
    every earlier parameter's index unchanged; that is checked, group by group."""
    if len(saved["param_groups"]) != n_old_groups or len(fresh["param_groups"]) < n_old_groups:
        raise ValueError(f"the checkpoint's optimizer has {len(saved['param_groups'])} groups, expected {n_old_groups}")
    for k in range(n_old_groups):
        a, b = saved["param_groups"][k], fresh["param_groups"][k]
        if a["params"] != b["params"] or a.get("role") != b.get("role"):
            raise ValueError(f"optimizer group {k} ({b.get('role')}) does not hold the checkpoint's parameters")
    return dict(state=dict(saved["state"]),
                param_groups=list(saved["param_groups"]) + list(fresh["param_groups"][n_old_groups:]))


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


def parse_cycle(spec: str, heads: list[str]) -> list[set[str]]:
    """``--xhead-cycle``: groups split by ';', heads by ','. Every name a registered head, every head in a group."""
    groups = [{h.strip() for h in g.split(",") if h.strip()} for g in spec.split(";")]
    if any(not g for g in groups):
        raise ValueError(f"--xhead-cycle {spec!r}: an empty group")
    unknown = sorted(set().union(*groups) - set(heads))
    missing = [h for h in heads if not any(h in g for g in groups)]
    if unknown or missing:
        raise ValueError(f"--xhead-cycle {spec!r}: " + "; ".join(
            ([f"not a head: {unknown}"] if unknown else []) + ([f"in no group: {missing}"] if missing else [])))
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
