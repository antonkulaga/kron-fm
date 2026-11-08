"""A CPU stand-in for the vendored KronLSTM layer, for tests on machines without a CUDA device.

The real layer runs Triton kernels only. :class:`CausalMixer` keeps its call and its rules, so a model's
own logic (padding, packing, decode layouts, directions) can be tested on the CPU:

* it is called as ``mixer(h, attention_mask=mask)`` or ``mixer(h, cu_seqlens=cu)`` and returns
  ``(out, None, None)``;
* a step whose mask is 0 is taken out of the sequence: it reads nothing, nothing reads it, and its
  output is zero;
* with ``cu_seqlens`` (batch 1), each segment is read from a zero state and never reads another.

Its output at a step is a causal decayed sum, ``out_t = sum over read steps s <= t of 0.98 ** (read
steps in (s, t]) * v_s``, written as a matrix: quadratic in length, so keep test sequences short.
Install it with ``monkeypatch.setattr(kron_core.blocks, "kronlstm_mixer", CausalMixer)``.
"""
from __future__ import annotations

import torch


class CausalMixer(torch.nn.Module):
    def __init__(self, d_model: int, n_heads: int, layer_idx: int) -> None:
        super().__init__()
        self.inp = torch.nn.Linear(d_model, d_model)
        self.out = torch.nn.Linear(d_model, d_model)
        # the glucose family's checkpoint fingerprint looks for A_log
        self.A_log = torch.nn.Parameter(torch.zeros(n_heads))

    def forward(self, h, attention_mask=None, cu_seqlens=None, **_):
        batch, steps = h.shape[:2]
        read = (torch.ones(batch, steps, dtype=torch.bool, device=h.device) if attention_mask is None
                else attention_mask.bool())
        if cu_seqlens is None:
            segment = torch.zeros(batch, steps, dtype=torch.long, device=h.device)
        else:
            if batch != 1:
                raise ValueError("cu_seqlens needs a batch of one packed row")
            lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).long()
            segment = torch.repeat_interleave(torch.arange(len(lengths), device=h.device), lengths).unsqueeze(0)
        count = read.to(torch.int64).cumsum(dim=1)
        lag = (count.unsqueeze(2) - count.unsqueeze(1)).clamp(min=0).to(h.dtype)
        causal = torch.ones(steps, steps, dtype=torch.bool, device=h.device).tril()
        same = segment.unsqueeze(2) == segment.unsqueeze(1)
        weight = 0.98 ** lag * (causal & same & read.unsqueeze(1) & read.unsqueeze(2))
        return self.out(weight @ self.inp(h)) * read.unsqueeze(-1), None, None
