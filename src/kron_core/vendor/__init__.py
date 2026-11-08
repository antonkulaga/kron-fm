"""NX-AI's KronLSTM kernels and layer, vendored from their flash-linear-attention snapshot.

Source: https://github.com/NX-AI/KronLSTM at commit 3d800c9a153a88ee9624fe6acf7f39a58017a17e (MIT,
see ``LICENSE``). KronLSTM is described in Hartl et al., "Kronecker Delta Rule for Long Context Linear
Attention", arXiv:2610.26834 (2026). Only the import closure of ``fla.layers.kronlstm`` is copied, with
the package ``__init__`` files trimmed and ``fla.`` rewritten to ``kron_core.vendor.fla.``; nothing else
is edited. The tree is identical, apart from that namespace, to glucose_forecasting_metabonet's
``kronlstm.vendor.fla``. Regenerate with ``scripts/vendor_kronlstm.py`` rather than by hand.
"""
