"""kron-dna as one ONNX graph: codon tokens in, logits, per-codon hidden states and a sequence embedding out.

The Triton kernels neither run on a CPU nor export, so the graph reads the trained weights through
:class:`kron_core.onnx.ChunkedKronMixer`, the layer's recurrence in plain PyTorch, 16 steps at a time with
matrix products. Its loop over chunks is unrolled when exported, so the graph reads a fixed number of codon
tokens, ``seq_len``, chosen at export; shorter sequences are padded with ``<pad>`` after their end.

Each row holds one sequence (``<cls>`` codons ``<eos>``, then padding): the graph does not restart the
recurrence inside a row as the packed training does. The backward scan reverses each row's real tokens in
place (a per-row gather), so padding always comes after the tokens a scan reads and never touches them;
no padding mask is needed inside the recurrence.

:func:`export_onnx` writes ``kron_dna.onnx`` and ``onnx_meta.json``. The meta records the checks: ONNX
Runtime against the PyTorch graph, and, when a CUDA device is present, the PyTorch graph against the
model's own Triton kernels. These are conversion checks; they do not measure model quality.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import torch
from torch import nn

from kron_core.bidirectional import BiKronLSTMBlock, SegmentAttentionBlock
from kron_core.onnx import ChunkedKronMixer, rms
from kron_dna import codons as C
from kron_dna.model import KronDNA, KronDNAConfig


def reverse_index(valid: torch.Tensor) -> torch.Tensor:
    """``[B, T]`` bool, true on each row's real tokens (a prefix) -> the index that reverses that prefix in place."""
    steps = valid.size(1)
    t = torch.arange(steps, device=valid.device).unsqueeze(0)
    length = valid.long().sum(1, keepdim=True)
    return torch.where(t < length, length - 1 - t, t)


def _gather_steps(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return x.gather(1, index.unsqueeze(-1).expand(-1, -1, x.size(-1)))


class KronDNAGraph(nn.Module):
    """``tokens`` ``[B, T]`` int64 -> ``logits`` ``[B, T, V]``, ``hidden`` ``[B, T, D]``, ``embedding`` ``[B, D]``.

    ``embedding`` is the mean of the hidden states over the row's real tokens, ``<cls>`` and ``<eos>``
    included. The weights are the model's own: nothing is copied.
    """

    def __init__(self, model: KronDNA, chunk: int = 16) -> None:
        super().__init__()
        self.model = model
        self.forward_mixers = nn.ModuleDict()
        self.backward_mixers = nn.ModuleDict()
        for i, layer in enumerate(model.layers):
            if isinstance(layer, BiKronLSTMBlock):
                self.forward_mixers[str(i)] = ChunkedKronMixer(layer.forward_mixer, chunk)
                back = layer.forward_mixer if layer.backward_mixer is None else layer.backward_mixer
                self.backward_mixers[str(i)] = ChunkedKronMixer(back, chunk)

    @staticmethod
    def _scan(mixer: ChunkedKronMixer, x: torch.Tensor) -> torch.Tensor:
        layer = mixer.layer
        batch = x.size(0)
        recurrent = x.new_zeros(batch, layer.num_heads, layer.k_head_dim, layer.v_head_dim)
        conv = x.new_zeros(batch, layer.qkv_conv1d.weight.size(0), layer.qkv_conv1d.weight.size(-1))
        out, _, _ = mixer(x, recurrent, conv)
        return out

    @staticmethod
    def _attention(block: SegmentAttentionBlock, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, steps, width = h.shape
        q, k, v = block.qkv(rms(block.attn_norm, h)).view(batch, steps, 3, block.n_heads, -1).permute(2, 0, 3, 1, 4).unbind(0)
        scores = q @ k.transpose(-1, -2) / (q.size(-1) ** 0.5)
        scores = scores.masked_fill(~mask, float("-inf"))
        o = scores.softmax(-1) @ v
        h = h + block.out(o.transpose(1, 2).reshape(batch, steps, width))
        return h + block.mlp(rms(block.mlp_norm, h))

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        valid = tokens != C.PAD_ID
        rev = reverse_index(valid)
        # one document per row: the real tokens attend to each other, padding only to padding
        mask = (valid.unsqueeze(2) == valid.unsqueeze(1)).unsqueeze(1)
        h = self.model.embed(tokens)
        for i, layer in enumerate(self.model.layers):
            if isinstance(layer, BiKronLSTMBlock):
                normed = rms(layer.mixer_norm, h)
                fwd = self._scan(self.forward_mixers[str(i)], normed)
                bwd = _gather_steps(self._scan(self.backward_mixers[str(i)], _gather_steps(normed, rev)), rev)
                h = h + fwd + bwd
                h = h + layer.mlp(rms(layer.mlp_norm, h))
            else:
                h = self._attention(layer, h, mask)
        h = rms(self.model.final_norm, h)
        logits = self.model.lm_head(h)
        weight = valid.to(h.dtype).unsqueeze(-1)
        embedding = (h * weight).sum(1) / weight.sum(1).clamp(min=1.0)
        return logits, h, embedding


def load_run(run_dir: str | Path, map_location: str | torch.device = "cpu") -> tuple[KronDNA, dict]:
    """The model and the checkpoint record of a ``kron-dna train`` run directory (or a checkpoint path)."""
    path = Path(run_dir)
    path = path / "checkpoint.pt" if path.is_dir() else path
    blob = torch.load(path, map_location=map_location, weights_only=False)
    model = KronDNA(KronDNAConfig(**blob["config"]))
    model.load_state_dict(blob["model"])
    return model.eval(), {k: v for k, v in blob.items() if k != "model"} | {"checkpoint": str(path)}


def example_tokens(batch: int, seq_len: int, seed: int = 0) -> torch.Tensor:
    """Random coding sequences of varied lengths as padded rows, for the conversion checks."""
    gen = torch.Generator().manual_seed(seed)
    sense = torch.tensor([C.TOKEN_ID[c] for c in C.CODONS if C.GENETIC_CODE[c] != "*"])
    rows = torch.full((batch, seq_len), C.PAD_ID, dtype=torch.long)
    for b in range(batch):
        n = int(torch.randint(max(3, seq_len // 3), seq_len - 1, (1,), generator=gen)) if seq_len > 4 else seq_len - 2
        body = sense[torch.randint(len(sense), (n,), generator=gen)]
        rows[b, 0], rows[b, 1:n + 1], rows[b, n + 1] = C.CLS_ID, body, C.EOS_ID
    return rows


def kernel_parity(model: KronDNA, tokens: torch.Tensor) -> dict:
    """The plain graph against the model's Triton kernels on CUDA, over the rows' real tokens."""
    device = torch.device("cuda")
    model = model.to(device).eval()
    tokens = tokens.to(device)
    with torch.no_grad():
        kernel_logits, _ = model(tokens)
        plain_logits, _, _ = KronDNAGraph(model)(tokens)
    valid = tokens != C.PAD_ID
    diff = (kernel_logits - plain_logits).abs()[valid]
    model.cpu()
    return {"what": "PyTorch plain graph vs Triton kernels (fp32), logits over real tokens",
            "max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean())}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def export_onnx(model: KronDNA, out_dir: str | Path, seq_len: int = 256, opset: int = 17,
                record: dict | None = None, check_kernels: bool = True) -> dict:
    """Export ``model`` to ``out_dir/kron_dna.onnx`` reading ``seq_len`` tokens, check it, write ``onnx_meta.json``."""
    import onnxruntime as ort

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = model.cpu().float().eval()
    graph = KronDNAGraph(model).eval()
    example = example_tokens(3, seq_len, seed=0)
    started = time.time()
    with torch.no_grad():
        torch.onnx.export(graph, (example,), out / "kron_dna.onnx", input_names=["tokens"],
                          output_names=["logits", "hidden", "embedding"],
                          dynamic_axes={"tokens": {0: "batch"}, "logits": {0: "batch"}, "hidden": {0: "batch"},
                                        "embedding": {0: "batch"}},
                          opset_version=opset, dynamo=False)
    export_seconds = time.time() - started

    check = example_tokens(4, seq_len, seed=1)
    with torch.no_grad():
        torch_logits, _, torch_embedding = graph(check)
    options = ort.SessionOptions()
    # The cores this process may use, not the machine's: ONNX Runtime pins one thread per machine core,
    # which fails (and floods the log) inside a scheduler's CPU allocation.
    options.intra_op_num_threads = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 0
    session = ort.InferenceSession(str(out / "kron_dna.onnx"), options, providers=["CPUExecutionProvider"])
    ort_logits, _, ort_embedding = session.run(None, {"tokens": check.numpy()})
    valid = check != C.PAD_ID
    verification = [{"what": "ONNX Runtime vs PyTorch plain graph, logits over real tokens",
                     "max_abs_diff": float((torch.from_numpy(ort_logits) - torch_logits).abs()[valid].max())},
                    {"what": "ONNX Runtime vs PyTorch plain graph, embedding",
                     "max_abs_diff": float((torch.from_numpy(ort_embedding) - torch_embedding).abs().max())}]
    if check_kernels and torch.cuda.is_available():
        verification.append(kernel_parity(model, check))

    path = out / "kron_dna.onnx"
    meta = {"graph": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size, "opset": opset,
            "export_route": "torchscript", "export_seconds": round(export_seconds, 1),
            "seq_len": seq_len, "vocab": list(C.VOCAB), "pad_id": C.PAD_ID, "cls_id": C.CLS_ID, "eos_id": C.EOS_ID,
            "inputs": [{"name": "tokens", "shape": ["batch", seq_len], "dtype": "int64",
                        "meaning": "one coding sequence per row: <cls>, codon tokens, <eos>, then <pad>"}],
            "outputs": [{"name": "logits", "shape": ["batch", seq_len, C.VOCAB_SIZE]},
                        {"name": "hidden", "shape": ["batch", seq_len, model.config.d_model]},
                        {"name": "embedding", "shape": ["batch", model.config.d_model],
                         "meaning": "mean hidden state over the real tokens"}],
            "model": model.config.to_dict(), "n_parameters": model.n_parameters(),
            "verification": verification, **({"run": record} if record else {})}
    (out / "onnx_meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    return meta


__all__ = ["KronDNAGraph", "example_tokens", "export_onnx", "kernel_parity", "load_run", "reverse_index"]
