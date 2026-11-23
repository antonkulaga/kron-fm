"""The value transforms every store shares, in numpy, each with its torch twin in ``kron_dna.xheads``."""
from __future__ import annotations

import numpy as np


def targets_np(decoded_over_nz: np.ndarray, is_rna) -> np.ndarray:
    """The training target from decoded signal already divided by the track's non-zero mean: the 0.75 power for
    RNA-seq, then the soft square-root clip above 10 (AlphaGenome's and NTv3's scaling). The numpy twin of
    :func:`kron_dna.xheads.targets_of` after its decoding step; ``tests/test_kron_dna_prep_base.py`` checks them equal."""
    x = np.maximum(decoded_over_nz, 0)
    x = np.where(is_rna, np.power(np.maximum(x, 1e-12), 0.75), x)
    return np.where(x > 10, 2 * np.sqrt(10 * x) - 10, x)


def decode_u8(u8: np.ndarray, scale) -> np.ndarray:
    """log1p-u8 encoded bytes back to signal: expm1(u8 / 255 * scale), clipped at 0."""
    return np.maximum(np.expm1(u8.astype(np.float32) / 255.0 * scale), 0)


def encode_u8(signal: np.ndarray, scale: float) -> np.ndarray:
    """Signal to the stores' u8 encoding: log1p(signal) / scale * 255, clipped to 0..255."""
    return np.clip(np.log1p(np.maximum(signal, 0)) / scale * 255.0, 0, 255).astype(np.uint8)
