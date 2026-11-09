#!/bin/bash
# Cluster settings for kron-fm's SLURM scripts. Everything that names a cluster (partition, account,
# node features) lives in slurm/env.local.sh, which is git-ignored, because this repository is public.
# Copy slurm/env.local.example.sh to slurm/env.local.sh on the cluster and fill it in.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/env.local.sh" ] && source "$HERE/env.local.sh"

export REPO="${REPO:-$(cd "$HERE/.." && pwd)}"
export KRON_GPU_PARTITION="${KRON_GPU_PARTITION:-}"     # required for GPU jobs
export KRON_CPU_PARTITION="${KRON_CPU_PARTITION:-$KRON_GPU_PARTITION}"
export KRON_ACCOUNT="${KRON_ACCOUNT:-}"
export KRON_GPU_CONSTRAINT="${KRON_GPU_CONSTRAINT:-}"   # e.g. a GPU model feature
export KRON_GPU_NODES="${KRON_GPU_NODES:-}"             # optional --nodelist for GPU jobs
export KRON_CPU_EXCLUDE="${KRON_CPU_EXCLUDE:-}"         # keep CPU jobs off the GPU nodes
export KRON_GPU_CPUS="${KRON_GPU_CPUS:-8}" KRON_GPU_MEM="${KRON_GPU_MEM:-96G}"
export KRON_CPU_CPUS="${KRON_CPU_CPUS:-8}" KRON_CPU_MEM="${KRON_CPU_MEM:-64G}"
# No default --time: a limit chosen because it looked round kills jobs; the partition default applies.
export KRON_TIME="${KRON_TIME:-}"
export LOG_DIR="${LOG_DIR:-$REPO/slurm/logs}"
export PYTHONUNBUFFERED=1
# uv's cache and Python builds go wherever env.local.sh points them (a large scratch filesystem).
[ -n "${UV_CACHE_DIR:-}" ] && export UV_CACHE_DIR
[ -n "${UV_PYTHON_INSTALL_DIR:-}" ] && export UV_PYTHON_INSTALL_DIR
export PATH="$HOME/.local/bin:$PATH"
