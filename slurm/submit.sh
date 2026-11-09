#!/bin/bash
# slurm/submit.sh gpu|cpu NAME COMMAND...   e.g.
#   slurm/submit.sh gpu dna-ecoli kron-dna train ecoli-k12 --steps 20000
#   NODE=gpu13 slurm/submit.sh gpu ...    (pin a node; a GPU job exits 3 when every card on its node is busy)
#   slurm/submit.sh cpu prepare-syncodon kron-dna prepare syncodonlm --keep-set train
# Builds sbatch flags from slurm/env.sh (and the git-ignored env.local.sh) and runs job.sbatch.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
kind="${1:?gpu or cpu}"; name="${2:?job name}"; shift 2
[ $# -gt 0 ] || { echo "no command given" >&2; exit 2; }
mkdir -p "$LOG_DIR"
flags=(--job-name="kron-$name" --nodes=1 --ntasks=1 --output="$LOG_DIR/%x-%j.out")
[ -n "$KRON_ACCOUNT" ] && flags+=(--account="$KRON_ACCOUNT")
[ -n "$KRON_TIME" ] && flags+=(--time="$KRON_TIME")
if [ "$kind" = gpu ]; then
  [ -n "$KRON_GPU_PARTITION" ] || { echo "set KRON_GPU_PARTITION in slurm/env.local.sh" >&2; exit 2; }
  flags+=(--partition="$KRON_GPU_PARTITION" --cpus-per-task="$KRON_GPU_CPUS" --mem="$KRON_GPU_MEM")
  [ -n "$KRON_GPU_CONSTRAINT" ] && flags+=(--constraint="$KRON_GPU_CONSTRAINT")
  nodes="${NODE:-$KRON_GPU_NODES}"; [ -n "$nodes" ] && flags+=(--nodelist="$nodes")
else
  [ -n "$KRON_CPU_PARTITION" ] || { echo "set KRON_CPU_PARTITION in slurm/env.local.sh" >&2; exit 2; }
  flags+=(--partition="$KRON_CPU_PARTITION" --cpus-per-task="$KRON_CPU_CPUS" --mem="$KRON_CPU_MEM")
  [ -n "$KRON_CPU_EXCLUDE" ] && flags+=(--exclude="$KRON_CPU_EXCLUDE")
fi
sbatch "${flags[@]}" --export="ALL,REPO=$REPO,KRON_JOB_KIND=$kind,CUDA_DEVICE=${CUDA_DEVICE:-}" "$HERE/job.sbatch" "$@"
