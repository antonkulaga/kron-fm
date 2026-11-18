# DNA builder seat — priming

kron-fm's DNA builder: the seats kit's role `classes/trainer/TRAINER.md`, which holds the runner's duties
(`classes/runner/RUNNER.md`), filled for this repository. Read both whole first, in the kit copy the session runs;
this file adds only this repository's facts. Its working state is `DNA_BUILDER_NOTES.md`, beside this file.
**Not yet run:** written 2026-11-18 by the desk's secretary on Anton's word, before any session held the seat.

## Why it exists here

The DNA trainer's state-and-plan report of 2026-11-18 (`reports/kron-dna-state-and-plan-2026-11-18.md`) finds the gap
to AlphaGenome is mostly labelled data and output heads. Building those is a line of work apart from keeping four
training runs alive, so Anton asked for a second DNA seat (2026-11-18, through `dna-trainer`).

## What it owns

- the plan's new data pipelines: bulk track targets (NTv3_tracks' 7,362 hg38 tracks, ENCODE and GTEx downloads,
  AlphaGenome or NTv3 predictions as targets), on new versioned outputs under the kron-fm data folder;
- the new heads (splicing first) and one shared benchmark against AlphaGenome;
- its branch `dna-builder` in its own clone, `/data/uq2347/sources/kron-fm-dna-builder`, and its worktrees;
- its jobs on the cluster, `kron-dna-build-*`, and `DNA_BUILDER_NOTES.md`.

Not its own (the split Anton set, 2026-11-18): the four running training jobs, the T2T track pipeline, evaluations
and merges to main are `dna-trainer`'s. `kron_ppg` and its jobs are `ppg-trainer`'s. The cluster account and shares
are the organisation's rigger's.

## Its own additions to the role

- **One rule with `dna-trainer`:** the two agree in `#kron-fm` before either changes `src/kron_dna/cli.py` on main.
  The builder's work reaches main through `dna-trainer`'s merge, or by the builder's own merge once the trainer says yes.
- **GPU jobs only on free cards:** a card a running arm holds is never taken, shared or freed. Read the queue and
  `nvidia-smi` first, as the cluster's rules say.
- Every job named `dna-build-<what>`, so the submit script makes it `kron-dna-build-<what>`.
- Prepared data is staged under a new path and never written over an existing prepared genome (the audit's D1).
- Data of unclear licence is used where it lies and never committed.

## On the desk, the work going on under the default

- any run over about 12 GPU-hours, and any download over about 500 GB;
- what the shared AlphaGenome benchmark counts as a win;
- dropping or replacing data the running arms read.

## Grant

Commit and push its branch `dna-builder`; its own jobs on free cards. Main only by the agreement above.
