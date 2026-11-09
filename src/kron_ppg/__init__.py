"""kron-ppg: a causal, streaming KronLSTM encoder over raw PPG, with a PPG aging clock as its first task.

See ``reports/PPG foundation models and aging clocks 2026.md`` for the design and its sources.
"""
from kron_ppg.model import AgeHead, KronPPG, KronPPGConfig, PPGAgeClock, patchify

__all__ = ["AgeHead", "KronPPG", "KronPPGConfig", "PPGAgeClock", "patchify"]
