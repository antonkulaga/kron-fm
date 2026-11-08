"""Shared pieces of the kron-fm models.

* :mod:`kron_core.vendor` holds NX-AI's KronLSTM layer and Triton kernels (MIT), vendored at a pinned commit.
* :mod:`kron_core.blocks` holds the causal KronLSTM residual block and the recurrent-state cache, as the
  glucose forecaster uses them.
* :mod:`kron_core.bidirectional` holds the forward + backward block and a document-masked attention block
  for masked-language models over packed documents.
* :mod:`kron_core.segments` turns document ids into the layer's ``cu_seqlens``.
* :mod:`kron_core.testing` holds a CPU stand-in for the layer.
"""
