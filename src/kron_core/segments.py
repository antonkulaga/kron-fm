"""Packed sequences: several documents laid end to end in one row, read as separate sequences.

The vendored KronLSTM layer takes a packed batch as ``[1, total_tokens, D]`` with ``cu_seqlens``, the
cumulative document lengths (int32, ``[N + 1]``, starting at 0 and ending at ``total_tokens``), and runs
the recurrence per document from a zero state. A ``[B, T]`` batch of packed rows is therefore flattened
to one row of ``B * T`` tokens, with a boundary at every row start and every change of document id;
no document crosses a row, so the flattening never joins two documents.
"""
from __future__ import annotations

import torch


def cu_seqlens_from_doc_ids(doc_ids: torch.Tensor) -> torch.Tensor:
    """``[B, T]`` document ids -> int32 ``cu_seqlens`` of the ``B * T`` flattened tokens.

    A new segment starts at every row start and wherever the id changes from the previous step, so
    padding given its own id (for example -1) is a segment of its own and reads nothing real.
    """
    if doc_ids.dim() != 2:
        raise ValueError(f"doc_ids must be [B, T], got shape {tuple(doc_ids.shape)}")
    batch, steps = doc_ids.shape
    start = torch.ones_like(doc_ids, dtype=torch.bool)
    start[:, 1:] = doc_ids[:, 1:] != doc_ids[:, :-1]
    idx = start.flatten().nonzero().squeeze(1).to(torch.int32)
    return torch.cat([idx, idx.new_tensor([batch * steps])])


def reverse_cu_seqlens(cu_seqlens: torch.Tensor) -> torch.Tensor:
    """The boundaries of the same packed row read back to front.

    Flipping a packed row reverses the order of its documents and each document in place, so the
    flipped row's boundaries are ``total - cu_seqlens`` in reverse order.
    """
    total = cu_seqlens[-1]
    return (total - cu_seqlens.flip(0)).to(cu_seqlens.dtype)


def same_segment_mask(doc_ids: torch.Tensor) -> torch.Tensor:
    """``[B, T]`` document ids -> ``[B, 1, T, T]`` bool, true where query and key share a document.

    For :func:`torch.nn.functional.scaled_dot_product_attention`; every query sees at least itself, so
    no row of the mask is empty.
    """
    return (doc_ids.unsqueeze(2) == doc_ids.unsqueeze(1)).unsqueeze(1)
