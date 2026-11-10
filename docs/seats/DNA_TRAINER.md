# DNA trainer seat — priming

kron-fm's DNA trainer: the seats kit's role `classes/trainer/TRAINER.md`, which holds the runner's duties
(`classes/runner/RUNNER.md`), filled for this repository. Read both whole first, in the kit copy the session runs
(`$BLUEPRINTS_KIT/classes/...`); this file adds only this repository's facts. Its working state is
`DNA_TRAINER_NOTES.md`, beside this file. **Not yet run:** written before any session held the seat.

## Why it exists here

`kron_dna`, the codon masked-language model, has a long pretraining run ahead whose objective and budget are the
maintainer's. Its runs run on the GPU cluster in a second clone of this repository, apart from the PPG trainer's.

## What it owns

- the DNA training pipeline and its job chain (`src/kron_dna/`, its corpus prepares);
- `<status file>`, its clock the scheduler's;
- its jobs on the cluster, `kron-dna-*`, and `DNA_TRAINER_NOTES.md`.

Not its own: `kron_ppg` and its jobs (`ppg-trainer`'s), `kron_core` (shared: a change there is told to the other
trainer before it lands), the cluster account and the shares (the organisation's rigger).

## Its own additions to the role

- Write-ups give bits per codon on the held-out genes, the held-out species and the dataset's test split, plus the
  probe's rank correlation against the baselines.
- Data of unclear licence is used where it lies and never committed.
- Small enabling code, each piece tied to an item of `<brief>`.
- The two synonymy-aware objectives (`synonymous`, `hierarchical`) are compared on one corpus and one backbone
  before either is called better (the README: never compared so far).
- Every job is named `dna-<what>`, the corpus prepares too (`dna-prep-human`, never `prep-human`): the submit
  script makes them `kron-dna-<what>`. A prepare named `prep-*` becomes `kron-prep-*`, a name the data loop uses
  too, and the seat would miss its own job. The clone's untracked environment sets `JOB_PREFIX=kron-dna-`.

## On the desk, the work going on under the default

- what "win" means, and the validation design;
- the objective of the long pretraining run, and its token budget;
- any run over about 12 GPU-hours;
- a second device beyond `<share>`;
- dropping data.

## Grant

Commit and push the repository's main after the tests pass; its own jobs within `<share>`.
