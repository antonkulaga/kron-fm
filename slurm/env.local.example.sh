#!/bin/bash
# Copy to slurm/env.local.sh (git-ignored) on the cluster and fill in. Read by slurm/env.sh.
export KRON_GPU_PARTITION=""          # the GPU partition you may use
export KRON_CPU_PARTITION=""          # a CPU partition, if different
export KRON_ACCOUNT=""                # SLURM account, if required
export KRON_GPU_CONSTRAINT=""         # node feature, e.g. the GPU model
export KRON_GPU_NODES=""              # optional comma-separated GPU nodes
export KRON_CPU_EXCLUDE=""            # GPU nodes CPU jobs should avoid
export KRON_DATA_ROOT_HINT=""         # unused by scripts; note where data/ lives
export UV_CACHE_DIR=""                # a large filesystem, not a small home quota
export UV_PYTHON_INSTALL_DIR=""
