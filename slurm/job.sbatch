#!/bin/bash
# One kron-fm command as a SLURM job: `uv run --no-sync <command...>` in the repository.
# Submit through slurm/submit.sh, which builds the scheduler flags from slurm/env.sh.
# GPU jobs (KRON_JOB_KIND=gpu): the nodes may have no GPU GRES, so the job picks a card itself,
# the first with no compute process, unless CUDA_DEVICE is set; it exits 3 when every card is busy.
set -euo pipefail
source "${REPO:?}/slurm/env.sh"
cd "$REPO"
echo "host=$(hostname) job=${SLURM_JOB_ID:-local} kind=${KRON_JOB_KIND:-cpu} commit=$(git rev-parse --short HEAD 2>/dev/null || echo ?)"
if [ "${KRON_JOB_KIND:-cpu}" = gpu ]; then
  if [ -z "${CUDA_DEVICE:-}" ]; then
    busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
    CUDA_DEVICE=""
    while IFS=, read -r idx uuid; do
      uuid=$(echo "$uuid" | xargs)
      if ! grep -q "$uuid" <<<"$busy"; then CUDA_DEVICE=$(echo "$idx" | xargs); break; fi
    done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)
    [ -z "$CUDA_DEVICE" ] && { echo "every GPU on $(hostname) is busy" >&2; exit 3; }
  fi
  export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv -i "$CUDA_DEVICE"
else
  export CUDA_VISIBLE_DEVICES=""
fi
exec uv run --no-sync "$@"
