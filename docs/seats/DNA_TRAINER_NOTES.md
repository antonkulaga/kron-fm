# DNA trainer notes

The state the DNA trainer's last holder left. State, not rules: the rules are the priming, `DNA_TRAINER.md` beside
this file, and the kit's role `classes/trainer/TRAINER.md`.

_Seated 2026-11-10 22:30 CET by the session that ran the kron-dna loop since the morning (handout from the desk's
secretary, on the maintainer's word)._

## The placeholders

- `<brief>` = `/data/uq2347/briefs/kron-dna.md` (outside the repository: it names cluster details).
- `<status file>` = `/data/uq2347/briefs/kron-dna-status.md`, CET (the scheduler's clock). The results, decisions
  and incidents of the run so far are there, not repeated here.
- `<share>` = the rigger's inventory until desk #4 is answered: a soft share of **25 CPU cores** on the
  account's shared core cap (`[shares]` of the rigger's queue map, desk #4 default); GPU jobs outside it. Devices
  as the brief and status file hold them: **one A100 card at a time** (gpu12 card 1, training), never the PPG
  trainer's last free card; short inference (evaluate, probe, ClinVar scoring) may share a card only after checking
  its free memory.
- `JOB_PREFIX=kron-dna-` set in the clone's untracked `slurm/env.local.sh`.

## Running

- `kron-dna-syncodon-mlm` (6427115): SynCodonLM pretraining, `mlm`, 300k steps, gpu12 card 1, resumable,
  `--max-hours 70`. Watcher: a background wait on the step-50k snapshot file (session task), then evaluate,
  ClinVar and mRFP on it.

## To file

Nothing yet.
