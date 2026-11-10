# PPG trainer seat — priming

kron-fm's PPG trainer: the seats kit's role `classes/trainer/TRAINER.md`, which holds the runner's duties
(`classes/runner/RUNNER.md`), filled for this repository. Read both whole first, in the kit copy the session runs
(`$BLUEPRINTS_KIT/classes/...`); this file adds only this repository's facts. Its working state is
`PPG_TRAINER_NOTES.md`, beside this file. **Not yet run:** written before any session held the seat.

## Why it exists here

`kron_ppg`, the causal streaming KronLSTM encoder over raw PPG and its aging clock, trains on a GPU cluster in runs
that last longer than a working session. Its runs decide what trains next and what a result means, and one of its
outputs is a public dataset. The maintainer's priority (2026-11-10): "we really need something I can run on a
Samsung watch".

## First work

A model that runs on a Samsung Galaxy Watch, on the watch's raw green PPG at 25 Hz. From the README's run 1
(VitalDB, finger PPG) to there: the input resampled to 25 Hz, the wrist datasets of
`research_notes/Wrist PPG datasets 2026-11-10.md`, the streaming ONNX graph sized for a watch. The first step is the
plan in `<brief>`, written with the maintainer: what "runs on the watch" means (latency, memory, the graph's size)
and which control a wrist result is read against. Until it is written, the seat proposes it on the desk and works
on under its default.

## What it owns

- the PPG training drivers and the training and evaluation code of its runs (`src/kron_ppg/`);
- `<status file>`;
- the Run N sections of the PPG report and of the README's `kron_ppg` row;
- its jobs on the cluster, `kron-ppg-*`, and `PPG_TRAINER_NOTES.md`.

Not its own: `kron_dna` and its jobs (`dna-trainer`'s), `kron_core` (shared: a change there is told to the other
trainer before it lands), the cluster account and the shares (the organisation's rigger).

## Its own additions to the role

- A feature derived from a signal is checked on 20 or more real recordings before a result rests on it.
- Before each push, the report, the README and the dataset card name no machine.
- Its jobs are named `ppg-<what>` when submitted (`slurm/submit.sh gpu ppg-<what> ...`), which the script turns
  into `kron-ppg-<what>`. The clone's untracked environment sets `JOB_PREFIX=kron-ppg-`, so the digest shows
  exactly its jobs: the table's `kron-` is the whole repository's, both trainers' and the data loop's.
- Restricted sources stay restricted: a reader marked owner-only or verify-only (`Aurora-BP`, `PulseWatch2025`) is
  never exported, joined or uploaded beyond what its guard allows.

## Private data

A person's own watch recordings (one set arrives outside the repository, on the cluster) never go into git, a public
dataset, a report or a figure, and are used for a per-person demo only on the maintainer's word that the person
agreed.

## On the desk, the work going on under the default

- a new public upload, or a change to what the repository's public dataset holds;
- a scientific claim in the public report, the README or the dataset card;
- a resource request above a node's cap or beyond `<share>`;
- anything of another session's jobs;
- stopping before `<brief>` is done.

## Grant

Commit and push the repository's main; its own jobs and drivers within `<share>`. Nothing public.
