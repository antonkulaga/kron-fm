#!/bin/bash
# One kron-fm command as a SLURM job: `uv run --no-sync <command...>` in the repository.
# Submit through slurm/submit.sh, which builds the scheduler flags from slurm/env.sh.
# GPU jobs (KRON_JOB_KIND=gpu): the nodes may have no GPU GRES, so the job picks a card itself,
# the first with no compute process and no other job's lease, unless CUDA_DEVICE is set; it exits 3 when
# every card is busy or leased.
set -euo pipefail
source "${REPO:?}/slurm/env.sh"
cd "$REPO"
echo "host=$(hostname) job=${SLURM_JOB_ID:-local} kind=${KRON_JOB_KIND:-cpu} commit=$(git rev-parse --short HEAD 2>/dev/null || echo ?)"
if [ "${KRON_JOB_KIND:-cpu}" = gpu ]; then
  # A card is leased with a node-local flock on its own lock file, held by fd 9 through the exec below until the
  # job's processes exit (the kernel drops it on exit or kill), so two jobs starting together cannot both pick a
  # card that has no compute process yet (audit 2026-11-18, OP-03).
  LOCKS="${KRON_GPU_LOCK_DIR:-/tmp/kron-gpu-locks-$(id -u)}"; mkdir -p "$LOCKS"
  if [ -z "${CUDA_DEVICE:-}" ]; then
    busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
    CUDA_DEVICE=""
    while IFS=, read -r idx uuid; do
      uuid=$(echo "$uuid" | xargs); idx=$(echo "$idx" | xargs)
      grep -q "$uuid" <<<"$busy" && continue
      exec 9>"$LOCKS/gpu$idx.lock"
      if flock -n 9; then CUDA_DEVICE=$idx; break; fi
      exec 9>&-
    done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)
    [ -z "$CUDA_DEVICE" ] && { echo "every GPU on $(hostname) is busy or leased" >&2; exit 3; }
  else
    exec 9>"$LOCKS/gpu$CUDA_DEVICE.lock"
    flock -n 9 || echo "warning: card $CUDA_DEVICE is leased by another job; using it as asked" >&2
  fi
  export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv -i "$CUDA_DEVICE"
else
  export CUDA_VISIBLE_DEVICES=""
fi
exec uv run --no-sync "$@"
