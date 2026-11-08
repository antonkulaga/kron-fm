#!/usr/bin/env python3
"""Regenerate ``src/kron_core/vendor/fla`` from a checkout of NX-AI's KronLSTM release.

The KronLSTM kernels are vendored rather than installed, as in glucose_forecasting_metabonet
(``src/kronlstm/vendor``, the script this one was copied from): only the modules the KronLSTM layer
imports are copied, the import closure measured by running the layer's training, padded, inference
and short-decoding paths with the package ``__init__`` files trimmed (the list below). Imports and the
module names held in strings are rewritten from ``fla.`` to ``kron_core.vendor.fla.``, so the snapshot
never shadows, or is shadowed by, an installed flash-linear-attention or the glucose repository's
``kronlstm.vendor.fla``.

Usage, from the repository root, with the release checked out at the pinned commit::

    git clone https://github.com/NX-AI/KronLSTM.git /tmp/kronlstm-release
    git -C /tmp/kronlstm-release checkout 3d800c9a153a88ee9624fe6acf7f39a58017a17e
    uv run python scripts/vendor_kronlstm.py /tmp/kronlstm-release
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

PINNED_COMMIT = "3d800c9a153a88ee9624fe6acf7f39a58017a17e"
TARGET = Path("src/kron_core/vendor/fla")
NAMESPACE = "kron_core.vendor.fla"

#: The import closure of ``fla.layers.kronlstm`` with the package ``__init__`` files trimmed.
CLOSURE = """
fla fla.backends fla.layers fla.layers.kronlstm fla.layers.utils fla.modules fla.modules.causal_conv1d
fla.modules.causal_conv1d.backends fla.modules.causal_conv1d.backends.cuda fla.modules.causal_conv1d.backends.gluon
fla.modules.causal_conv1d.cp fla.modules.causal_conv1d.ops fla.modules.norm fla.modules.norm.fused_norm_gate
fla.modules.norm.l2norm fla.modules.norm.layernorm fla.modules.norm.layernorm_gated fla.ops fla.ops.common
fla.ops.common.backends fla.ops.common.backends.intracard fla.ops.common.backends.tilelang
fla.ops.common.backends.triton_ascend fla.ops.common.chunk_delta_h fla.ops.common.chunk_h
fla.ops.common.fused_recurrent fla.ops.common.gate fla.ops.cp fla.ops.cp.chunk_delta_h fla.ops.cp.comm
fla.ops.cp.context fla.ops.gla fla.ops.gla.backends fla.ops.gla.backends.triton_ascend fla.ops.gla.chunk
fla.ops.gla.fused_chunk fla.ops.gla.fused_recurrent fla.ops.kda fla.ops.kda.backends fla.ops.kda.backends.flash_kda
fla.ops.kda.backends.tilelang fla.ops.kda.backends.triton_ascend fla.ops.kda.chunk fla.ops.kda.chunk_bwd
fla.ops.kda.chunk_fwd fla.ops.kda.chunk_intra fla.ops.kda.chunk_intra_token_parallel fla.ops.kda.fused_recurrent
fla.ops.kda.gate fla.ops.kda.wy_fast fla.ops.kronlstm fla.ops.kronlstm.chunk fla.ops.kronlstm.chunk_bwd
fla.ops.kronlstm.chunk_fwd fla.ops.kronlstm.fused_recurrent fla.ops.kronlstm.naive fla.ops.utils
fla.ops.utils.backends fla.ops.utils.backends.triton_ascend fla.ops.utils.cache fla.ops.utils.constant
fla.ops.utils.csr fla.ops.utils.cumsum fla.ops.utils.graph fla.ops.utils.index fla.ops.utils.logsumexp
fla.ops.utils.matmul fla.ops.utils.op fla.ops.utils.pack fla.ops.utils.pooling fla.ops.utils.softmax
fla.ops.utils.softplus fla.ops.utils.solve_tril fla.utils fla.utils.decorators fla.utils.env fla.utils.hardware
fla.utils.testing
""".split()

#: Package ``__init__`` files replaced so that importing the layer loads the closure and nothing else.
TRIMMED_INITS = {
    "fla/__init__.py": '__version__ = "0.6.0"\n',
    "fla/layers/__init__.py": "",
    "fla/ops/__init__.py": "",
    "fla/modules/__init__.py": (
        f"from {NAMESPACE}.modules.causal_conv1d import ShortConvolution\n"
        f"from {NAMESPACE}.modules.norm import FusedRMSNormGated, RMSNorm\n"
    ),
}

_REWRITES = (
    (re.compile(r"(?m)^(\s*)from fla(\.|\s)"), rf"\1from {NAMESPACE}\2"),
    (re.compile(r"(?m)^(\s*)import fla\."), rf"\1import {NAMESPACE}."),
    (re.compile(r"""(['"])fla\."""), rf"\1{NAMESPACE}."),
)


def module_file(root: Path, module: str) -> Path:
    path = root / module.replace(".", "/")
    return path / "__init__.py" if path.is_dir() else path.with_suffix(".py")


def rewrite(text: str) -> str:
    for pattern, repl in _REWRITES:
        text = pattern.sub(repl, text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("release", type=Path, help="checkout of https://github.com/NX-AI/KronLSTM")
    args = parser.parse_args()
    head = subprocess.run(["git", "-C", str(args.release), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()
    if head != PINNED_COMMIT:
        raise SystemExit(f"the release checkout is at {head}, not the pinned {PINNED_COMMIT}")
    if TARGET.exists():
        shutil.rmtree(TARGET)
    for module in CLOSURE:
        source = module_file(args.release, module)
        relative = source.relative_to(args.release).as_posix()
        dest = TARGET.parent / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        text = TRIMMED_INITS.get(relative)
        dest.write_text(text if text is not None else rewrite(source.read_text()))
    shutil.copy(args.release / "LICENSE", TARGET.parent / "LICENSE")
    print(f"vendored {len(CLOSURE)} modules of {PINNED_COMMIT[:8]} into {TARGET}")


if __name__ == "__main__":
    main()
