"""The autotuner lock (kron_core.triton_lock): installed once by importing kron_core.blocks, reentrant, and it
serialises two threads that would otherwise run Autotuner.run at the same time."""
from __future__ import annotations

import threading
import time

import pytest

triton = pytest.importorskip("triton")


def test_installed_once_and_reentrant():
    import kron_core.blocks  # noqa: F401  (installs it)
    from kron_core import triton_lock
    from triton.runtime import autotuner
    assert triton_lock._PATCHED and hasattr(autotuner.Autotuner.run, "__wrapped__")
    wrapped = autotuner.Autotuner.run
    assert triton_lock.install() and autotuner.Autotuner.run is wrapped          # not wrapped twice
    with triton_lock._LOCK:
        with triton_lock._LOCK:                                                  # reentrant
            pass


def test_two_threads_do_not_overlap(monkeypatch):
    """install() over a stand-in for Autotuner.run that records overlap: through the wrapper, eight threads never
    overlap; without it they do (the check is not vacuous)."""
    from kron_core import triton_lock
    from triton.runtime import autotuner
    inside, peak = [0], [0]
    lock = threading.Lock()

    def fake(self, *a, **k):
        with lock:
            inside[0] += 1
            peak[0] = max(peak[0], inside[0])
        time.sleep(0.02)
        with lock:
            inside[0] -= 1

    def hammer(fn):
        peak[0] = 0
        threads = [threading.Thread(target=fn, args=(None,)) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return peak[0]

    assert hammer(fake) > 1                                     # unlocked, they overlap
    monkeypatch.setattr(autotuner.Autotuner, "run", fake)
    monkeypatch.setattr(triton_lock, "_PATCHED", False)
    assert triton_lock.install()
    assert hammer(autotuner.Autotuner.run) == 1
