#!/bin/bash
# One KronGenome arm as a SLURM job; submit through slurm/genome.sh, which resolves the recipe and the scheduler
# flags. Here: an optional handover (OLD), the cards (KRON_NCARDS, leased with node-local locks as job.sbatch does),
# a Triton cache of this job's own, thread counts from the allocation, then genome-train.
set -euo pipefail
source "${REPO:?}/slurm/env.sh"
cd "$REPO"
: "${KRON_ARM:?}" "${KRON_OUT:?}" "${KRON_ARGS_FILE:?}"
mapfile -t lines < "$KRON_ARGS_FILE"
eval "args=(${lines[0]})"
eval "extra=(${lines[1]:-})"
echo "host=$(hostname) job=${SLURM_JOB_ID:-local} arm=$KRON_ARM out=$KRON_OUT commit=$(git rev-parse --short HEAD 2>/dev/null || echo ?) $(date '+%F %T')"

if [ -n "${OLD:-}" ]; then          # handover: the old job's next checkpoint, then its cards
  : "${OLD_RUN:?}" "${SNAP:?}"
  t0=$(stat -c %Y "$OLD_RUN/checkpoint.pt")
  echo "waiting for the next checkpoint of $OLD (last $(date -d @"$t0" +%T))"
  while [ "$(stat -c %Y "$OLD_RUN/checkpoint.pt")" = "$t0" ]; do squeue -h -j "$OLD" | grep -q . || break; sleep 20; done
  sleep 30
  scancel "$OLD"
  for _ in $(seq 1 120); do squeue -h -j "$OLD" 2>/dev/null | grep -q . || break; sleep 5; done
  [ -e "$SNAP" ] || { cp "$OLD_RUN/checkpoint.pt" "$SNAP" && chmod a-w "$SNAP"; }
  echo "handover done; snapshot $SNAP; $(date +%T)"
fi

# cards: the ones given, or the first KRON_NCARDS free ones, each leased with a node-local flock (fds 20+)
LOCKS="${KRON_GPU_LOCK_DIR:-/tmp/kron-gpu-locks-$(id -u)}"; mkdir -p "$LOCKS"
need="${KRON_NCARDS:-1}"; picked=()
wait_s=0
while :; do
  picked=(); fd=20
  if [ -n "${CARDS:-}" ]; then cand=(${CARDS//,/ }); else mapfile -t cand < <(nvidia-smi --query-gpu=index --format=csv,noheader); fi
  for k in "${cand[@]}"; do
    k=$(echo "$k" | xargs)
    used=$(nvidia-smi -i "$k" --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "$used" -lt 1000 ] || continue
    eval "exec $fd>\"$LOCKS/gpu$k.lock\""
    if flock -n "$fd"; then picked+=("$k"); fd=$((fd + 1)); else eval "exec $fd>&-"; fi
    [ ${#picked[@]} -ge "$need" ] && break
  done
  [ ${#picked[@]} -ge "$need" ] && break
  for ((f = 20; f < fd; f++)); do eval "exec $f>&-"; done
  # given cards wait for their owner to leave (a handover's old job, another job's last seconds); free ones do not
  if [ -n "${CARDS:-}" ] && [ "$wait_s" -lt "${KRON_CARD_WAIT:-600}" ]; then sleep 15; wait_s=$((wait_s + 15)); continue; fi
  echo "not enough free cards on $(hostname): need $need, have ${#picked[@]}" >&2; exit 3
done
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${picked[*]}")
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv -i "$CUDA_VISIBLE_DEVICES"

threads="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS=$threads MKL_NUM_THREADS=$threads OPENBLAS_NUM_THREADS=$threads POLARS_MAX_THREADS=$threads
# a Triton cache of this job's own: two jobs compiling into one shared cache on a network filesystem have raced
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/kron-triton-${SLURM_JOB_ID:-$$}"; mkdir -p "$TRITON_CACHE_DIR"
trap 'rm -rf "$TRITON_CACHE_DIR"' EXIT

init=()
if [ -n "${FROM:-}" ] && [ ! -e "$KRON_OUT/checkpoint.pt" ]; then     # a new run from a checkpoint
  step=$(uv run --no-sync python -c "import torch; print(torch.load('$FROM', map_location='cpu', weights_only=False, mmap=True)['step'])")
  [ "${START0:-0}" = 1 ] && step=0
  init=(--init "$FROM" --start-step "$step")
  echo "start from $FROM at step $step"
fi
stop=()
if [ -n "${STOP:-}" ]; then
  base=${step:-0}
  [ -e "$KRON_OUT/checkpoint.pt" ] && base=$(uv run --no-sync python -c "import torch; print(torch.load('$KRON_OUT/checkpoint.pt', map_location='cpu', weights_only=False, mmap=True)['step'])")
  stop=(--stop-step $((base + STOP)))
fi
uv run --no-sync python -m kron_dna.cli genome-train "$KRON_OUT" "${args[@]}" "${init[@]}" "${stop[@]}" "${extra[@]}"
