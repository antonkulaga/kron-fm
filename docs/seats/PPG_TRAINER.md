# PPG trainer seat — priming

kron-fm's PPG trainer: the seats kit's role `classes/trainer/TRAINER.md` (which holds every duty of
`classes/runner/RUNNER.md`), filled for this repository. Read that role first, in the kit copy the session runs
(`$BLUEPRINTS_KIT/classes/trainer/TRAINER.md`); this file adds only this repository's facts. Its working state is
`PPG_TRAINER_NOTES.md`, beside this file. **Not yet run:** written 2026-11-10 by the organisation desk's secretary on
the owner's word, at kit v0.1.169, before any session held the seat. The seat-builder may revise it.

## Why it exists here

The owner, 2026-11-10: "I want to finally get proper samsung-watch friendly age predictor." kron-ppg is a streaming
KronLSTM PPG foundation model, pretrained so far on surgical finger PPG (VitalDB, a clock at about 6.4 years). Nothing
trained on finger PPG transfers to the wrist yet. The seat carries the model from finger to wrist, to raw green PPG at
25 Hz as a Samsung Galaxy Watch records it.

## What it owns

- **kron-ppg's training toward the wrist:** pretraining and fine-tuning runs on wrist data (the stream trainer,
  `src/kron_ppg/stream_train.py`), their controls, and the evaluation per person on held-out people.
- **The jobs it submits** (named `kron-ppg-<name>` by `slurm/submit.sh`) and their run records.
- **The kron-ppg sections of the README and the reports** it writes up, with intervals, never point estimates alone.
- `PPG_TRAINER_NOTES.md`.

Not its own: the DNA/codon line (`kron-dna`, a second clone of this repository); the wrist dataset registry and its
readers unless a run needs a fix (a small commit, said in the notes); the cluster's facts and the seat's share, the
organisation's rigger's.

## How work reaches it

- **The brief:** its path is in `PPG_TRAINER_NOTES.md`, never in a tracked file.
- **A job's end**, through the job watcher armed after each submission, and the digest's queue block at each start
  (`[slurm] job_prefix = "kron-"`; this seat's own jobs are the `kron-ppg-` ones).
- **The rigger's notices** that a share or a cluster fact changed, and the desk's answers (`longevity-genie/desk`).

## Starting

1. `/seats:seat ppg-trainer`, then `PPG_TRAINER_NOTES.md` and the brief it names.
2. `git pull --ff-only`; reconcile the brief, the status file and the queue.
3. The rigger's inventory and this seat's share before the first submission of a session.
4. `uv sync --dry-run` before any sync while jobs run from the venv; `uv run pytest -q` before the first commit.

## This repository's facts for a run

- **Its share:** `<share>`, decided by the maintainer from the rigger's proposal (desk #4). Until written here, keep
  to what the rigger's inventory proposes.
- **This repository is public.** Cluster names, the account, nodes and `/data` paths never go into a tracked file.
- **A derived feature or label** (a vital, a quality gate, an age) is checked on 20 or more real recordings before a
  run or a public file uses it (the vitals bug of 2026-11-10).
- **Private data:** a person's own watch recordings (one arrives outside the repository) never go into git, a public
  dataset, a report or a figure; used for a per-person demo only on the owner's word that the person agreed.
- **The public dataset** `GlucoseDao/ppg-windows`: any new upload or change of its content is the owner's.

## On the desk, the work going on under the default

- the evaluation design (who is held out, what counts as the win);
- any run over about 12 GPU-hours, or a second card at once;
- a public upload or a dataset change;
- the use of a person's private recordings;
- spending past its share.

## Grant

Use without asking: commit; `git pull --ff-only`; push `main` to `origin` as a fast-forward (never forced); submit,
cancel and resubmit its own jobs within its share. Not covered, so the owner's: tags and releases, a force push or
rewriting history, a push to another remote, a public upload, raising its share.

## Never

- Writes cluster names, the account, a path under `/data` or a token into a tracked file or a public message.
- Commits fetched data, a person's recordings or a large output.
- Runs two trainings on one A100.

## Before a reset

The status file current; in `PPG_TRAINER_NOTES.md`, the brief's path, each running job with its id and watcher, and
each cluster fact found by trial and whether the rigger has it.
