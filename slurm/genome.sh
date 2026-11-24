#!/bin/bash
# slurm/genome.sh ARM OUT [recipe options...] [-- extra genome-train options...]
# Submit one KronGenome arm (kron_dna.recipes) as a SLURM job, with the cluster settings of slurm/env.sh and the
# git-ignored slurm/env.local.sh. Examples:
#   slurm/genome.sh F runs/F-new --genomes all --cards 2
#   FROM=path/to/checkpoint.pt slurm/genome.sh G runs/G-b --cards 2 --edits --pairs -- --stop-step 20000
#   OLD=<jobid> OLD_RUN=<its run dir> SNAP=<copy path> slurm/genome.sh F runs/F-s2 --cards 2 ...   (handover: wait
#     for the old job's next checkpoint, cancel it, keep a read-only copy, then start on its cards)
# Environment: FROM (a checkpoint to start from: --init, its step continues the schedule unless START0=1),
# CARDS (card indices on the node, default the first N free), NODE (pin a node), NAME (job name), COMMENT (the job's
# --comment), STOP (stop after this many steps from the start). OUT is a run directory; an existing one resumes.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
arm="${1:?arm (A, A1m, A3, D, F, G)}"; out="${2:?run directory}"; shift 2
recipe_opts=(); extra=()
while [ $# -gt 0 ]; do
  if [ "$1" = "--" ]; then shift; extra=("$@"); break; fi
  recipe_opts+=("$1"); shift
done
n_cards=1
for ((i = 0; i < ${#recipe_opts[@]}; i++)); do
  [ "${recipe_opts[$i]}" = "--cards" ] && n_cards="${recipe_opts[$((i + 1))]}"
done
# the recipe is resolved now, on the submitting host, so the job runs exactly what was printed here
args=$(cd "$REPO" && uv run --no-sync python -m kron_dna.recipes "$arm" "${recipe_opts[@]}")
[ -n "$args" ] || { echo "recipe $arm gave no arguments" >&2; exit 2; }
mkdir -p "$LOG_DIR"
[ -n "$KRON_GPU_PARTITION" ] || { echo "set KRON_GPU_PARTITION in slurm/env.local.sh" >&2; exit 2; }
flags=(--job-name="${NAME:-kron-dna-$arm}" --nodes=1 --ntasks=1 --output="$LOG_DIR/%x-%j.out"
       --partition="$KRON_GPU_PARTITION" --cpus-per-task="${KRON_GENOME_CPUS:-4}" --mem="${KRON_GENOME_MEM:-128G}")
[ -n "$KRON_ACCOUNT" ] && flags+=(--account="$KRON_ACCOUNT")
[ -n "$KRON_TIME" ] && flags+=(--time="$KRON_TIME")
[ -n "$KRON_GPU_CONSTRAINT" ] && flags+=(--constraint="$KRON_GPU_CONSTRAINT")
nodes="${NODE:-$KRON_GPU_NODES}"; [ -n "$nodes" ] && flags+=(--nodelist="$nodes")
[ -n "${COMMENT:-}" ] && flags+=(--comment="$COMMENT")
printf '%s\n' "$args" "${extra[*]:-}" > "$LOG_DIR/.genome-args-$$"
sbatch "${flags[@]}" --export="ALL,REPO=$REPO,KRON_ARM=$arm,KRON_OUT=$out,KRON_NCARDS=$n_cards,KRON_ARGS_FILE=$LOG_DIR/.genome-args-$$" \
  "$HERE/genome.sbatch"
