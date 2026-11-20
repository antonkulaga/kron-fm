"""KronGenome as one ONNX graph: base codes in; base logits, NTv3 tracks and the further heads' outputs out.

The Triton kernels neither run on a CPU nor export. As for the codon model (:mod:`kron_dna.onnx_export`), the graph
reads the trained weights through :class:`kron_core.onnx.ChunkedKronMixer`, the KronLSTM recurrence in plain
PyTorch, 16 steps at a time with matrix products, its loop over chunks unrolled at export. Here every
bidirectional KronLSTM block of the model (the core, the local track if any, and KronLSTM output heads) has its
``mix`` replaced by that plain scan; the rest of KronGenome's forward pass (stem, pooling, decoder, heads) runs
unchanged, so the graph computes what the model computes, up to the kernels' round-off.

Each row is one document: the recurrence runs over the whole row, as KronGenome does for a window without
segments. The graph reads a fixed number of bases, chosen at export (a multiple of the core's step); a batch of
rows is free. Fixed inside the graph: the context id is 0 (nuclear DNA) and the reserved extra input channels are
zero (no repeat mask, no variant dosage), as in evaluation.

Outputs (float32): ``base_logits`` ``[B, L, 4]`` (A, C, G, T); ``ntv3`` ``[B, L, T]`` the NTv3 tracks in natural
(bigWig) units, the model's AlphaGenome-scaled output turned back with each track's non-zero mean and RNA flag;
for each further head, ``x_<name>``: a track head's signal in natural units ``[B, L / bin, T]``, or a binary head's
probabilities ``[B, L, T]`` (splice sites: donor +, acceptor +, donor -, acceptor -). ``onnx_meta.json`` names
every channel and records the checks run: ONNX Runtime against the PyTorch graph here; on CUDA, the PyTorch graph
against the model's own Triton kernels (:func:`kernel_parity`). These are conversion checks, not quality.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from kron_core.bidirectional import BiKronLSTMBlock
from kron_core.onnx import ChunkedKronMixer
from kron_dna import genome as G
from kron_dna import xheads as XH


def natural(raw: torch.Tensor, nonzero_mean: torch.Tensor, is_rna: torch.Tensor) -> torch.Tensor:
    """A poisson track head's raw output -> signal in natural units: softplus gives the AlphaGenome-scaled rate
    (:func:`kron_dna.xheads.targets_of`), turned back (the soft clip above 10, the 0.75 power for RNA-seq, times the
    track's non-zero mean). Equal to ``expm1(xheads.signal_of(...))``, without log1p and expm1 (no ONNX operators)."""
    y = F.softplus(raw.float())
    x = torch.where(y > 10, ((y + 10) / 2).pow(2) / 10, y)
    x = torch.where(is_rna, x.clamp_min(0).pow(1 / 0.75), x)
    return (x * nonzero_mean).clamp_min(0)


def _plain_mix(fwd: ChunkedKronMixer, bwd: ChunkedKronMixer, length: int):
    """A ``BiKronLSTMBlock.mix`` over documents of ``length`` steps (the graph's rows), through the plain chunked
    scans. ``length`` is fixed when the graph is built, from the window: nothing about the documents is read from
    traced values (a document count read from ``cu_seqlens`` corrupted an unrelated reshape in the exported graph)."""

    def scan(mixer: ChunkedKronMixer, x: torch.Tensor) -> torch.Tensor:
        layer = mixer.layer
        rec = x.new_zeros(x.size(0), layer.num_heads, layer.k_head_dim, layer.v_head_dim)
        conv = x.new_zeros(x.size(0), layer.qkv_conv1d.weight.size(0), layer.qkv_conv1d.weight.size(-1))
        out, _, _ = mixer(x, rec, conv)
        return out

    def mix(self, h: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        if not (torch.jit.is_tracing() or torch.compiler.is_exporting()):   # documents = rows of this length
            want = torch.arange(0, h.shape[0] * h.shape[1] + 1, length, device=cu_seqlens.device)
            if not torch.equal(cu_seqlens.long(), want):
                raise ValueError(f"the ONNX graph runs documents of {length} steps only (one per row, no segments)")
        width = int(fwd.layer.hidden_size) if hasattr(fwd.layer, "hidden_size") else int(h.shape[-1])
        # blocks get [1, documents x length, width] (packed rows): every size here is a Python constant, none traced
        x = h.reshape(-1, length, width)
        out = scan(fwd, x) + scan(bwd, x.flip(1)).flip(1)
        return out.reshape(1, -1, width).to(h.dtype)

    return mix


class KronGenomeGraph(nn.Module):
    """``bases`` ``[B, L]`` int64 (0-3 = A, C, G, T; 4 = N) -> the outputs listed in :attr:`output_names`.

    The model is changed in place (each KronLSTM block's ``mix`` becomes the plain scan); pass a model loaded for
    export. ``ntv3_nz`` / ``ntv3_rna``: the NTv3 tracks' non-zero means and RNA flags (None: no NTv3 output);
    ``x_consts``: per track head, its store's (non-zero means, RNA flags)."""

    def __init__(self, model: G.KronGenome, ntv3_nz: np.ndarray | None = None, ntv3_rna: np.ndarray | None = None,
                 x_consts: dict | None = None, chunk: int = 16, window: int = 131072) -> None:
        super().__init__()
        self.model = model.eval()
        c = model.config
        if c.arch not in ("v2", "v2l") or c.chunk_ratio:
            raise ValueError("the ONNX graph covers arch v2/v2l without learned chunking")
        # each block's document length at this window: the core over 2**n_down-base bins, the local track over its
        # span, a KronLSTM head over its bin times its own step
        lengths = {id(b): window // 2 ** c.n_down for b in model.core}
        for b in getattr(model, "local_core", None) or []:
            lengths[id(b)] = min(c.local_span, window)
        for h in c.xheads:
            head = model.xheads[h[0]]
            if isinstance(head, G.KronHead):
                for b in head.blocks:
                    lengths[id(b)] = window // ((int(h[2]) if len(h) > 2 else 1) * head.bin)
        # a plain list, not a ModuleList: each plain mixer holds the model's own KronLSTM layer, and registering it
        # again put one layer under two module paths, which made the TorchScript exporter write wrong shapes
        # (a reshape to [1, 16, 2] for [1, 16, 32]); the mixers own no weights, the layers stay where they are
        self.mixers: list[ChunkedKronMixer] = []
        for block in list(model.modules()):
            if isinstance(block, BiKronLSTMBlock):
                if id(block) not in lengths:
                    raise ValueError("a KronLSTM block of the model has no known document length")
                f = ChunkedKronMixer(block.forward_mixer, chunk)
                b = ChunkedKronMixer(block.forward_mixer if block.backward_mixer is None else block.backward_mixer, chunk)
                self.mixers += [f, b]
                block.mix = types.MethodType(_plain_mix(f, b, lengths[id(block)]), block)
        self.window = window
        model.x_active, model.x_span = None, None
        self.with_ntv3 = model.ntv3 is not None and ntv3_nz is not None
        if self.with_ntv3:
            self.register_buffer("ntv3_nz", torch.as_tensor(ntv3_nz, dtype=torch.float32))
            self.register_buffer("ntv3_rna", torch.as_tensor(ntv3_rna, dtype=torch.bool))
        self.x_heads = [h[0] for h in model.config.xheads]
        self.x_kind = {}
        for n in self.x_heads:
            if x_consts and n in x_consts:
                self.x_kind[n] = "tracks"
                self.register_buffer(f"x_nz_{n}", torch.as_tensor(x_consts[n][0], dtype=torch.float32))
                self.register_buffer(f"x_rna_{n}", torch.as_tensor(x_consts[n][1], dtype=torch.bool))
            else:
                self.x_kind[n] = "binary"
        self.output_names = ["base_logits"] + (["ntv3"] if self.with_ntv3 else []) + [f"x_{n}" for n in self.x_heads]

    def forward(self, bases: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.outputs_of(self.model, bases)

    @staticmethod
    def run_model(m: G.KronGenome, bases: torch.Tensor) -> torch.Tensor:
        """KronGenome's forward pass (arch v2/v2l, no segments, nuclear context, no extra channels) with the core's
        document boundaries as plain row starts. ``KronGenome.forward`` derives them from segments through
        ``nonzero``, a data-dependent step that corrupted an unrelated reshape in the exported graph; one document
        per row needs none of it. Sets ``m.last_ntv3`` and ``m.last_x`` as the model's own forward does."""
        b, length = bases.shape[0], bases.shape[-1]
        x, skips = m.encode(bases, None)
        h = x.transpose(1, 2)                                       # [B, bins, d_model]
        bins = h.shape[1]
        if m.context is not None:
            h = h + m.context_rows(None, b, bases.device).unsqueeze(1).to(h.dtype)
        cu = torch.arange(0, b * bins + 1, bins, device=bases.device, dtype=torch.int32)
        core = m.run_core(h.reshape(1, -1, h.shape[-1]), cu).reshape(b, bins, -1)
        x = m.add_local(m.decode(core.transpose(1, 2), skips), skips)
        m.last_ntv3 = m.ntv3(x).transpose(1, 2) if m.ntv3 is not None else None
        m.last_x, m.last_x_at = m.run_xheads(x)
        return m.head(x).transpose(1, 2)

    def outputs_of(self, m: G.KronGenome, bases: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Run ``m`` (this graph's patched model, or an unpatched copy with its kernels) and convert its outputs."""
        dev = bases.device
        m.x_active, m.x_span = None, None
        logits = self.run_model(m, bases)
        out = [logits.float()]
        if self.with_ntv3:
            out.append(natural(m.last_ntv3, self.ntv3_nz.to(dev), self.ntv3_rna.to(dev)))
        for n in self.x_heads:
            p = m.last_x[n].float()
            if self.x_kind[n] == "tracks":
                out.append(natural(p, getattr(self, f"x_nz_{n}").to(dev), getattr(self, f"x_rna_{n}").to(dev)))
            else:
                out.append(torch.sigmoid(p))
        return tuple(out)


def ntv3_constants(record: dict, ids: list[str], interim) -> tuple[np.ndarray, np.ndarray]:
    """Non-zero means and RNA flags of the NTv3 head's channels ``ids`` (in head order), from the stores the run
    trained on (its ``--ntv3-dir`` and ``--ntv3-extra`` genomes), as ``genome-train`` builds them."""
    opt = record["options"]
    table = {}
    nd = Path(opt["ntv3_dir"])
    dj = json.loads((nd / "dense.json").read_text())
    assay = {t["file_id"]: t["assay"] for t in json.loads((nd / "tracks.json").read_text())["tracks"]}
    for i, nz in zip(dj["ids"], dj["nonzero_mean"]):
        table[i] = (nz, "RNA-seq" in assay.get(i, ""))
    for xname in [x.strip() for x in (opt.get("ntv3_extra") or "").split(",") if x.strip()]:
        xj = json.loads((interim("genome", xname) / "ntv3" / "dense.json").read_text())
        for i, nz, a in zip(xj["ids"], xj["nonzero_mean"], xj["assays"]):
            table[f"{xname}:{i}"] = (nz, "RNA-seq" in a)
    missing = [i for i in ids if i not in table]
    if missing:
        raise ValueError(f"{len(missing)} NTv3 channels not found in the run's stores: {missing[:5]}")
    return (np.array([table[i][0] for i in ids], dtype=np.float32), np.array([table[i][1] for i in ids], dtype=bool))


def x_constants(record: dict) -> tuple[dict, dict]:
    """Per further head of the run: (non-zero means, RNA flags) for a track head, and each head's channel ids."""
    consts, ids = {}, {}
    for n, st in (record.get("xheads") or {}).get("stores", {}).items():
        ids[n] = list(st["ids"])
        store = XH.open_store(st["path"])
        if store.kind == "tracks":
            consts[n] = (np.asarray(store.nonzero_mean), np.asarray(store.is_rna))
    return consts, ids


def example_bases(batch: int, length: int, seed: int = 0) -> torch.Tensor:
    """Random bases with a run of N, for the conversion checks."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 4, (batch, length), generator=g)
    x[:, length // 3:length // 3 + 100] = 4
    return x


def kernel_parity(checkpoint: str | Path, graph: KronGenomeGraph, bases: torch.Tensor,
                  want: tuple[torch.Tensor, ...]) -> list[dict]:
    """The model with its own Triton kernels on CUDA (a fresh, unpatched load of ``checkpoint``) against the plain
    graph's outputs ``want`` for ``bases``, through the same output conversion."""
    fresh, _ = G.load_checkpoint(checkpoint)
    fresh = fresh.float().cuda().eval()
    with torch.no_grad():
        got = graph.outputs_of(fresh, bases.cuda())
    rows = []
    for name, a, b in zip(graph.output_names, want, got):
        d = (a.float() - b.float().cpu()).abs()
        rows.append({"what": f"Triton kernels (CUDA) vs PyTorch plain graph (CPU), fp32, {name}",
                     "max_abs_diff": float(d.max()), "max_abs_value": float(a.abs().max())})
    del fresh
    torch.cuda.empty_cache()
    return rows


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def export_genome_onnx(checkpoint: str | Path, out_dir: str | Path, window: int = 131072, opset: int = 18,
                       interim=None, check_kernels: bool = True, with_ntv3: bool = True, optimize: bool = False) -> dict:
    """Export a ``genome-train`` checkpoint (or run directory) to ``out_dir/model.onnx`` reading ``window`` bases,
    check it with ONNX Runtime, write ``onnx_meta.json``."""
    import onnxruntime as ort
    from kron_core.paths import interim_dir
    interim = interim or interim_dir
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model, state = G.load_checkpoint(checkpoint)
    model = model.float().eval()
    unit = 2 ** model.config.n_down
    if window % unit:
        raise ValueError(f"window {window} must be a multiple of the core step {unit}")
    record = state.get("run_record") or {}
    nt = state.get("ntv3")
    nz = rna = None
    if with_ntv3 and model.ntv3 is not None and nt is not None:
        if nt.get("target") != "alphagenome" or nt.get("loss") != "poisson":
            raise ValueError(f"NTv3 head convention {nt.get('target')}/{nt.get('loss')} is not exported (only "
                             "alphagenome/poisson)")
        nz, rna = ntv3_constants(record, list(nt["ids"]), interim)
    xc, xids = x_constants(record)
    graph = KronGenomeGraph(model, nz, rna, xc, window=window).eval()
    names = graph.output_names
    example = example_bases(2, window, seed=0)          # two rows: an example batch of one would fix it at one
    started = time.time()
    # optimize=False by default: the exporter's ONNX optimisation pass did not finish in 3 h on F at 131 kb (the
    # unrolled recurrence is a large graph); ONNX Runtime optimises a graph itself when it loads it
    # the torch.export route: the TorchScript exporter mis-resolved ranks and shapes in this graph (negative dims
    # against stale ranks, a reshape constant written wrong, a crash in its shape arithmetic with a free batch)
    with torch.no_grad():
        torch.onnx.export(graph, (example,), out / "model.onnx", input_names=["bases"], output_names=names,
                          dynamic_shapes=({0: torch.export.Dim("batch", min=1, max=1024)},),
                          opset_version=opset, dynamo=True, external_data=False, optimize=optimize)
    export_seconds = time.time() - started

    check = example_bases(2, window, seed=1)
    with torch.no_grad():
        want = graph(check)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 0
    started = time.time()
    sess = ort.InferenceSession(str(out / "model.onnx"), opts, providers=["CPUExecutionProvider"])
    got = sess.run(None, {"bases": check.numpy()})
    ort_seconds = time.time() - started
    verification = []
    for name, a, b in zip(names, want, got):
        d = (a - torch.from_numpy(b)).abs()
        verification.append({"what": f"ONNX Runtime vs PyTorch plain graph (CPU, fp32), {name}",
                             "max_abs_diff": float(d.max()), "max_abs_value": float(a.abs().max())})
    if check_kernels and torch.cuda.is_available():
        verification.extend(kernel_parity(checkpoint, graph, check, want))

    path = out / "model.onnx"
    c = model.config
    outputs = [{"name": "base_logits", "shape": ["batch", window, 4], "channels": ["A", "C", "G", "T"]}]
    if graph.with_ntv3:
        outputs.append({"name": "ntv3", "shape": ["batch", window, len(nt["ids"])], "units": "natural (bigWig) units",
                        "channels": list(nt["ids"])})
    for n in graph.x_heads:
        b = next((int(h[2]) for h in c.xheads if h[0] == n and len(h) > 2), 1)
        tracks = graph.x_kind[n] == "tracks"
        outputs.append({"name": f"x_{n}", "shape": ["batch", window // b, len(xids.get(n, []))],
                        "units": "natural units, mean over each bin" if tracks else "probability",
                        "bin": b, "channels": xids.get(n, [])})
    meta = {"graph": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size, "opset": opset,
            "export_route": "torch.export (dynamo)", "exporter_optimize": optimize, "export_seconds": round(export_seconds, 1),
            "ort_two_rows_seconds_with_load": round(ort_seconds, 1), "window": window,
            "inputs": [{"name": "bases", "shape": ["batch", window], "dtype": "int64",
                        "meaning": "hg38-style base codes 0-3 = A, C, G, T, 4 = N; one window per row, forward strand"}],
            "fixed_in_graph": {"context": "0 (nuclear DNA)", "extra_input_channels": "zero (no repeat mask, no dosage)"},
            "outputs": outputs, "model": c.to_dict(), "step": int(state.get("step", -1)),
            "source_commit": (record.get("source") or {}).get("commit"), "checkpoint": str(checkpoint),
            "n_parameters": sum(p.numel() for p in model.parameters()),
            "verification": verification}
    (out / "onnx_meta.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
    return meta


__all__ = ["KronGenomeGraph", "example_bases", "export_genome_onnx", "kernel_parity", "ntv3_constants"]
