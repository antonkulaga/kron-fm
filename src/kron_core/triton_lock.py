"""One lock around Triton's autotuner, so two threads never tune the same kernel at once.

With a window split over two cards (``KronGenome.forward_split``) PyTorch runs each card's backward pass in its own
thread, and both can enter the same autotuned kernel (fla's fused norm-gate backward). An ``Autotuner`` keeps its
arguments in ``self.nargs`` while it benchmarks and clears them when it returns, so a second thread's call can clear
them under the first: ``TypeError: 'NoneType' object is not a mapping`` in ``autotuner.benchmark``. It shows only
while a shape is being tuned (new layer widths; a cold cache), which is why runs on cached shapes never met it
(G-a, 24 Nov). The lock is reentrant (tuning launches the kernel through the same path) and costs nothing once every
shape is tuned, beyond taking an uncontended lock per launch.
"""
from __future__ import annotations

import threading

_LOCK = threading.RLock()
_PATCHED = False


def install() -> bool:
    """Wrap ``triton.runtime.autotuner.Autotuner.run`` once; False when Triton is not installed."""
    global _PATCHED
    if _PATCHED:
        return True
    try:
        from triton.runtime import autotuner
    except ImportError:
        return False
    original = autotuner.Autotuner.run

    def run(self, *args, **kwargs):
        with _LOCK:
            return original(self, *args, **kwargs)

    run.__wrapped__ = original
    autotuner.Autotuner.run = run
    _PATCHED = True
    return True
