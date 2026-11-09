"""The KronLSTM layer in plain PyTorch, for ONNX export and for running a trained model on a CPU.

The trained layer runs FLA's Triton kernels, which neither run on a CPU nor export to ONNX. The
:class:`PlainKronMixer` here reads the same weights (it holds the layer itself) through the token-by-token
recurrence of ``vendor/fla/ops/kronlstm/naive.py``, the short convolution as the cache-update kernel
computes it, and the sigmoid-gated RMSNorm. It is moved verbatim from glucose_forecasting_metabonet's
``kronlstm/onnx_export.py`` (commit 6e9bbbe91), where the streamed glucose forecaster's ONNX bundle is
built on it, and :mod:`krontime.onnx_export` and :mod:`kron_dna.onnx_export` both use it.

The recurrence is written as a Python loop over steps, so an exported graph unrolls it: the graph grows
with the number of steps it reads, which is why the exports read fixed-length chunks.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def rms(norm: nn.RMSNorm, x: torch.Tensor) -> torch.Tensor:
    """``norm(x)`` written out: the exporters have no ONNX operator for ``aten::rms_norm`` before opset 23."""
    eps = norm.eps if norm.eps is not None else torch.finfo(x.dtype).eps
    out = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps)
    return out * norm.weight if norm.weight is not None else out


class PlainKronMixer(nn.Module):
    """The vendored KronLSTM layer's forward in plain PyTorch, its memory passed in and out as tensors.

    It holds the layer itself, so the weights are the trained ones and nothing is copied. ``valid``
    ``(B, T)``, when given, takes a step out of the recurrence and the convolution as the layer's own
    padding mask does by packing: the memory is left as it was at that step.
    """

    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        if not layer.use_short_conv:
            raise ValueError("the ONNX path reads layers with the short convolution only, as every KronLSTM run has")
        self.layer = layer

    def project(self, x: torch.Tensor, valid: torch.Tensor | None, conv: torch.Tensor):
        """The layer's inputs to the recurrence: ``q, k, v, beta, g, target_gate`` and the new convolution state."""
        layer = self.layer
        batch, steps, _ = x.shape
        heads, kd, vd = layer.num_heads, layer.k_head_dim, layer.v_head_dim
        qkv = layer.qkv_proj(x)
        weight = layer.qkv_conv1d.weight[:, 0, :]  # (D, W)
        bias = layer.qkv_conv1d.bias
        if valid is None:
            # Every step is read: one causal depthwise convolution over the cached inputs and the chunk.
            width = weight.size(-1)
            seq = torch.cat([conv[:, :, 1:], qkv.transpose(1, 2)], dim=2)
            mixed = F.conv1d(seq, weight.unsqueeze(1), bias, groups=weight.size(0))
            conv = seq[:, :, -width:]
            qkv = F.silu(mixed).transpose(1, 2)
        else:
            outs = []
            for t in range(steps):
                shifted = torch.cat([conv[:, :, 1:], qkv[:, t].unsqueeze(-1)], dim=2)
                y = (shifted * weight).sum(-1)
                if bias is not None:
                    y = y + bias
                outs.append(F.silu(y))
                conv = torch.where(valid[:, t].view(-1, 1, 1), shifted, conv)
            qkv = torch.stack(outs, dim=1)
        q, k, v = qkv.split((layer.key_dim, layer.key_dim, layer.value_dim), dim=-1)
        q = F.normalize(q.reshape(batch, steps, heads, kd), dim=-1)
        k = F.normalize(k.reshape(batch, steps, heads, kd), dim=-1)
        v = v.reshape(batch, steps, heads, vd)
        if layer.normalize_v:
            v = F.normalize(v, dim=-1)
        beta = layer.b_proj(x).sigmoid() if layer.use_beta else x.new_ones(batch, steps, heads)
        g = F.softplus(layer.f_proj(x).float() + layer.dt_bias)
        if layer.decay_gate == "vector":
            g = -layer.A_log.float().exp().unsqueeze(-1) * g.reshape(batch, steps, heads, kd)
        else:
            g = (-layer.A_log.float().exp() * g).unsqueeze(-1)
        target_gate = layer.c_proj(x).float().sigmoid() if layer.use_target_gate else None
        if layer.decouple_erase and target_gate is None:
            raise ValueError("decouple_erase needs the target gate")
        return q, k, v, beta, g, target_gate, conv

    def output(self, o: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """The gated RMSNorm and the output projection over the recurrence's ``(B, T, H, V)`` reading."""
        layer = self.layer
        batch, steps, heads, vd = o.shape
        norm = layer.o_norm
        o = o * torch.rsqrt((o * o).mean(-1, keepdim=True) + norm.eps)
        if norm.weight is not None:
            o = o * norm.weight
        if layer.use_output_gate:
            o = o * torch.sigmoid(layer.g_proj(x).reshape(batch, steps, heads, vd))
        return layer.o_proj(o.reshape(batch, steps, heads * vd))

    def forward(self, x: torch.Tensor, valid: torch.Tensor | None, recurrent: torch.Tensor, conv: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        layer = self.layer
        steps = x.size(1)
        kd = layer.k_head_dim
        q, k, v, beta, g, target_gate, conv = self.project(x, valid, conv)

        # vendor/fla/ops/kronlstm/naive.py, naive_recurrent_kronlstm, step for step.
        beta_max_scaler = 2.0 if layer.allow_neg_eigval else 1.0
        scale = kd ** -0.5
        v_norm = (v * v).sum(-1)
        beta_max = beta_max_scaler / v_norm.clamp_min(1e-18)
        beta_star = beta_max * beta
        state = recurrent
        outs = []
        for t in range(steps):
            b_q, b_k, b_v = q[:, t], k[:, t], v[:, t]
            decayed = state * g[:, t].exp().unsqueeze(-1)
            b_y = ((b_k.unsqueeze(-1) * decayed).sum(-2) * b_v).sum(-1)
            if layer.decouple_erase:
                b_u = beta_max[:, t] * (v_norm[:, t] * target_gate[:, t]) - beta_star[:, t] * b_y
            elif target_gate is not None:
                b_u = beta_star[:, t] * (v_norm[:, t] * target_gate[:, t] - b_y)
            else:
                b_u = beta_star[:, t] * (v_norm[:, t] - b_y)
            updated = decayed + b_u[..., None, None] * b_k.unsqueeze(-1) * b_v.unsqueeze(-2)
            outs.append(((b_q * scale).unsqueeze(-1) * updated).sum(-2))
            state = updated if valid is None else torch.where(valid[:, t].view(-1, 1, 1, 1), updated, state)
        return self.output(torch.stack(outs, dim=1), x), state, conv


def write_coefficients(layer: nn.Module, v: torch.Tensor, beta: torch.Tensor, target_gate: torch.Tensor | None
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    """``(a, b)`` with the naive recurrence's write ``u_t = a_t - b_t y_t``, per step and head."""
    beta_max = (2.0 if layer.allow_neg_eigval else 1.0) / (v * v).sum(-1).clamp_min(1e-18)
    v_norm = (v * v).sum(-1)
    beta_star = beta_max * beta
    if layer.decouple_erase:
        return beta_max * v_norm * target_gate, beta_star
    if target_gate is not None:
        return beta_star * v_norm * target_gate, beta_star
    return beta_star * v_norm, beta_star


class ChunkedKronMixer(PlainKronMixer):
    """The same layer over a whole sequence, read ``chunk`` steps at a time with matrix products.

    :class:`PlainKronMixer` loops over steps, which an exported graph unrolls step by step: a 4-layer
    bidirectional model over 256 codons did not finish exporting in ten minutes. Every write of the
    KronLSTM recurrence is rank one along ``k_t v_t^T`` with a scalar strength, ``S_t = D_t S_{t-1} +
    u_t k_t v_t^T`` with ``u_t = a_t - b_t y_t`` and ``y_t = k_t^T D_t S_{t-1} v_t``, so within a chunk the
    strengths solve one unit lower-triangular system,

        (I + diag(b) M) u = a - b * y0,   M_ts = (k_t^T Gamma_{s->t} k_s) (v_t . v_s) for s < t,

    where ``Gamma_{s->t}`` is the decay from step ``s`` to ``t`` and ``y0`` is what the chunk's starting
    memory contributes. The inverse is a product of ``log2(chunk)`` matrix factors (the matrix is
    nilpotent), so the graph has matrix products only, and a loop over chunks rather than steps. Every
    decay is formed as ``exp`` of a difference that is at most zero, so nothing overflows.

    Every step is read (no padding mask): put padding after the real steps, where it cannot reach them.
    """

    def __init__(self, layer: nn.Module, chunk: int = 16) -> None:
        super().__init__(layer)
        if chunk < 2 or chunk & (chunk - 1):
            raise ValueError(f"chunk must be a power of two, got {chunk}")
        self.chunk = chunk

    def forward(self, x: torch.Tensor, recurrent: torch.Tensor, conv: torch.Tensor  # type: ignore[override]
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        layer = self.layer
        # The sequence length is a Python int, so an exported graph reads a fixed length with constant
        # chunk shapes; a traced length turned the padding into a Pad node ONNX Runtime evaluated wrongly.
        steps = int(x.shape[1])
        q, k, v, beta, g, target_gate, conv = self.project(x, None, conv)
        a, b = write_coefficients(layer, v, beta, target_gate)
        q = q * layer.k_head_dim ** -0.5
        c = self.chunk
        n = -(-steps // c)
        pad = n * c - steps

        # no squeeze or unsqueeze: the TorchScript exporter turns a squeeze of an unknown dim into an If node
        def chunks(t: torch.Tensor) -> torch.Tensor:  # (B, T, H, ...) -> (B, H, N, C, ...)
            t = t.float()
            if pad:
                t = F.pad(t, (0, 0) * (t.dim() - 2) + (0, pad))
            t = t.reshape(-1, n, c, *(int(s) for s in t.shape[2:]))
            return t.movedim(3, 1)

        q, k, v, g = chunks(q), chunks(k), chunks(v), chunks(g)
        a, b = chunks(a), chunks(b)                                          # (B, H, N, C)
        G = g.cumsum(3)                                                     # (B, H, N, C, Kg)
        upper = torch.ones(c, c, dtype=torch.bool, device=x.device).triu(1)
        diff = (G.unsqueeze(4) - G.unsqueeze(3)).masked_fill(upper.unsqueeze(-1), float("-inf"))
        decay = diff.exp()                                                  # Gamma_{s->t}, zero for s > t
        if decay.size(-1) == 1:
            kk = (k @ k.transpose(-1, -2)) * decay[..., 0]
            qk = (q @ k.transpose(-1, -2)) * decay[..., 0]
        else:
            kk = torch.einsum("bhntd,bhnsd,bhntsd->bhnts", k, k, decay)
            qk = torch.einsum("bhntd,bhnsd,bhntsd->bhnts", q, k, decay)
        strict = upper.transpose(0, 1)                                      # s < t
        m = kk * (v @ v.transpose(-1, -2)) * strict
        nil = -b.unsqueeze(-1) * m                                          # strictly lower, nilpotent
        eye = torch.eye(c, dtype=nil.dtype, device=x.device)
        inverse, power = eye + nil, nil
        for _ in range(c.bit_length() - 2):                                 # (I-N)^-1 = prod (I + N^(2^j))
            power = power @ power
            inverse = inverse @ (eye + power)
        state = recurrent.float()
        outs = []
        for i in range(n):
            gi, ki, vi = G[:, :, i], k[:, :, i], v[:, :, i]
            y0 = torch.einsum("bhck,bhkv,bhcv->bhc", ki * gi.exp(), state, vi)
            u = (inverse[:, :, i] @ (a[:, :, i] - b[:, :, i] * y0).unsqueeze(-1)).squeeze(-1)
            o = (q[:, :, i] * gi.exp()) @ state + (qk[:, :, i] * u.unsqueeze(-2)) @ vi
            last = gi[:, :, -1:]                                            # (B, H, 1, Kg)
            written = ki * (last - gi).exp() * u.unsqueeze(-1)              # (B, H, C, K)
            state = last.transpose(-1, -2).exp() * state + written.transpose(-1, -2) @ vi
            outs.append(o)
        o = torch.stack(outs, dim=2).reshape(-1, layer.num_heads, n * c, layer.v_head_dim)[:, :, :steps]
        return self.output(o.transpose(1, 2).to(x.dtype), x), state, conv


