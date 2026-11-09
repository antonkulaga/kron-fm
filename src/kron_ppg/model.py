"""kron-ppg: a causal, streaming KronLSTM encoder over raw PPG, with a next-patch objective and an age head.

The input is the scaled PPG of :mod:`kron_ppg.signal` at ``sample_rate`` (25 Hz by default, the rate the
no-setup watch tier delivers), cut into patches of ``patch`` samples (0.2 s). Each patch is one token,
so a 30 s window is 150 steps and an hour 18,000; the recurrence costs the same per step at any depth
of the stream, and its memory (:class:`kron_core.blocks.LayerStates`) carries a person's stream from one
chunk to the next, on the GPU through the Triton kernels and on a phone through the ONNX graph
(:mod:`kron_ppg.onnx_export`).

Pretraining (see ``reports/PPG foundation models and aging clocks 2026.md``):

* **next-patch prediction** (``next``): from the memory after patch ``t``, predict the next
  ``horizon`` patches, scored with an L1 loss: a Laplace likelihood at a fixed scale. GPT-PPG
  (arXiv 2503.08015) pretrains on one-step-ahead patches with a logit-Laplace likelihood whose scale it
  also predicts. It is the one self-supervised objective a causal model reads exactly as it is served, and it
  asks for the pulse's shape: the next beat's upstroke, notch and diastolic wave.
* **participant contrast** (``next+subject``): Apple's PPG foundation model's positive pairs, two
  windows of the same person, scored with InfoNCE on the pooled stream embedding. It pushes the pooled
  embedding toward what is stable within a person (age, vascular state) and away from the moment.

The **age head** (:class:`AgeHead`) reads the mean of the final hidden states over a stream's valid
patches. Streaming keeps that mean as a running sum and count, so a phone can read the age at any time.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from kron_core.blocks import KronLSTMBlock, LayerStates

OBJECTIVES = ("next", "next+subject")


@dataclass
class KronPPGConfig:
    #: Samples per second of the scaled input; the interim windows must be stored at this rate.
    sample_rate: float = 25.0
    #: Samples per token: 5 at 25 Hz is 0.2 s, about a fifth of a beat.
    patch: int = 5
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 6
    ff_units: int = 768
    dropout: float = 0.1
    #: Patches ahead the next-patch head predicts from each step (5 x 0.2 s = 1 s).
    horizon: int = 5
    #: Width of the contrastive projection (``next+subject``).
    proj_dim: int = 128

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def patch_seconds(self) -> float:
        return self.patch / self.sample_rate


def patchify(x: torch.Tensor, patch: int, valid: torch.Tensor | None = None
             ) -> tuple[torch.Tensor, torch.Tensor]:
    """``(B, L)`` samples -> ``(B, T, patch)`` tokens and their ``(B, T)`` validity.

    A trailing part patch is dropped. A patch is valid when every sample in it is; an invalid one is
    zero, and the encoder takes it out of the recurrence.
    """
    batch, length = x.shape
    steps = length // patch
    tokens = x[:, :steps * patch].reshape(batch, steps, patch)
    if valid is None:
        ok = torch.ones(batch, steps, dtype=torch.bool, device=x.device)
    else:
        ok = valid[:, :steps * patch].reshape(batch, steps, patch).all(-1)
    return tokens.masked_fill(~ok.unsqueeze(-1), 0.0), ok


class KronPPG(nn.Module):
    """``(B, T, patch)`` scaled PPG patches -> ``(B, T, d_model)`` causal hidden states."""

    def __init__(self, config: KronPPGConfig) -> None:
        super().__init__()
        self.config = c = config
        self.embed = nn.Linear(c.patch, c.d_model)
        self.embed_norm = nn.RMSNorm(c.d_model, eps=1e-5)
        self.blocks = nn.ModuleList(KronLSTMBlock(c.d_model, c.n_heads, c.ff_units, c.dropout, i)
                                    for i in range(c.n_layers))
        self.final_norm = nn.RMSNorm(c.d_model, eps=1e-5)
        self.next_head = nn.Linear(c.d_model, c.horizon * c.patch)
        self.proj = nn.Sequential(nn.Linear(c.d_model, c.d_model), nn.GELU(), nn.Linear(c.d_model, c.proj_dim))
        self._init_weights()

    @torch.no_grad()
    def _init_weights(self) -> None:
        for name, module in self.named_modules():
            # the vendored layer initialises its own decay and gates
            if isinstance(module, nn.Linear) and ".mixer." not in f".{name}.":
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def tokens(self, patches: torch.Tensor) -> torch.Tensor:
        return self.embed_norm(self.embed(patches))

    def encode(self, patches: torch.Tensor, valid: torch.Tensor | None = None,
               states: LayerStates | None = None) -> torch.Tensor:
        """The hidden states of ``patches``, zero where a patch is invalid.

        With ``states`` the stream starts from the memory they hold and leaves them holding its end, so
        a stream read in chunks gives the hidden states it gives read whole (up to the kernels' rounding).
        """
        if patches.dim() != 3 or patches.size(-1) != self.config.patch:
            raise ValueError(f"expected (batch, steps, {self.config.patch}) patches, got {tuple(patches.shape)}")
        if states is not None and len(states) != len(self.blocks):
            raise ValueError(f"the states hold {len(states)} layers and the model has {len(self.blocks)}")
        if valid is None:
            valid = torch.ones(patches.shape[:2], dtype=torch.bool, device=patches.device)
        pad = ~valid
        h = self.tokens(patches).masked_fill(pad.unsqueeze(-1), 0.0)
        mask = None if bool(valid.all()) else valid.to(torch.int32)
        with torch.autocast(device_type=patches.device.type, dtype=torch.bfloat16, enabled=patches.is_cuda):
            for blk in self.blocks:
                h = blk(h, mask, states)
        return self.final_norm(h.float()).masked_fill(pad.unsqueeze(-1), 0.0)

    def forward(self, patches: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        return self.encode(patches, valid)

    # ---- pretraining -----------------------------------------------------------------------
    def next_patch_loss(self, h: torch.Tensor, patches: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Mean L1 error of the next ``horizon`` patches predicted from each step, over the pairs where both
        the step and the patch it predicts are valid."""
        c = self.config
        batch, steps, _ = patches.shape
        pred = self.next_head(h).view(batch, steps, c.horizon, c.patch)
        total = h.new_zeros(())
        count = h.new_zeros(())
        for k in range(1, min(c.horizon, steps - 1) + 1):
            target = patches[:, k:]
            ok = (valid[:, :-k] & valid[:, k:]).unsqueeze(-1).float()
            err = (pred[:, :-k, k - 1] - target).abs() * ok
            total = total + err.sum()
            count = count + ok.sum() * c.patch
        return total / count.clamp_min(1.0)

    def embedding(self, h: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """The mean hidden state over each stream's valid patches, ``(B, d_model)``."""
        w = valid.unsqueeze(-1).to(h.dtype)
        return (h * w).sum(1) / w.sum(1).clamp_min(1.0)

    def subject_loss(self, pooled: torch.Tensor, subject: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
        """InfoNCE over the batch: every other window of the same subject is a positive.

        Batches are built with at least two windows per subject (:class:`kron_ppg.data.SubjectPairs`);
        windows whose subject appears once contribute nothing.
        """
        z = F.normalize(self.proj(pooled), dim=-1)
        logits = z @ z.T / temperature
        eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
        logits = logits.masked_fill(eye, float("-inf"))
        positive = (subject.unsqueeze(0) == subject.unsqueeze(1)) & ~eye
        has = positive.any(1)
        if not bool(has.any()):
            return pooled.new_zeros(())
        log_prob = logits.log_softmax(dim=1)
        per_row = -(log_prob.masked_fill(~positive, 0.0).sum(1) / positive.sum(1).clamp_min(1))
        return per_row[has].mean()

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class AgeHead(nn.Module):
    """Pooled embedding -> age in years: ``mean + scale * MLP(LayerNorm(z))``.

    ``mean`` and ``scale`` are buffers set from the training ages, so the MLP starts by predicting the
    training mean and learns a deviation in units of the age spread.
    """

    def __init__(self, d_model: int, hidden: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.net = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.register_buffer("age_mean", torch.tensor(50.0))
        self.register_buffer("age_scale", torch.tensor(15.0))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def set_scale(self, ages: torch.Tensor) -> None:
        self.age_mean.fill_(float(ages.float().mean()))
        self.age_scale.fill_(max(float(ages.float().std()), 1.0))

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.age_mean + self.age_scale * self.net(self.norm(pooled)).squeeze(-1)


class PPGAgeClock(nn.Module):
    """Encoder + age head: scaled PPG samples ``(B, L)`` -> age ``(B,)``.

    :meth:`stream` reads a stream chunk by chunk, carrying the recurrent memory and the running pool,
    and returns the age so far: what the ONNX graph does on a phone.
    """

    def __init__(self, encoder: KronPPG, head: AgeHead | None = None) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head or AgeHead(encoder.config.d_model)

    def forward(self, samples: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        patches, ok = patchify(samples, self.encoder.config.patch, valid)
        h = self.encoder.encode(patches, ok)
        return self.head(self.encoder.embedding(h, ok))

    def stream(self, patches: torch.Tensor, valid: torch.Tensor, states: LayerStates,
               pool_sum: torch.Tensor, pool_count: torch.Tensor
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One chunk of a stream: returns ``(age, pool_sum, pool_count)``; ``states`` is left at the chunk's end."""
        h = self.encoder.encode(patches, valid, states)
        pool_sum = pool_sum + (h * valid.unsqueeze(-1).to(h.dtype)).sum(1)
        pool_count = pool_count + valid.to(h.dtype).sum(1, keepdim=True)
        return self.head(pool_sum / pool_count.clamp_min(1.0)), pool_sum, pool_count
