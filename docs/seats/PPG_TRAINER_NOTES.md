# PPG trainer notes

The state the PPG trainer's last holder left. State, not rules: the rules are the priming, `PPG_TRAINER.md` beside
this file, and the kit's role `classes/trainer/TRAINER.md`.

Seated 2026-11-10 ~23:30 CET by the session already working the Samsung-watch brief (handed over by the desk's
secretary on the maintainer's word).

## Placeholders

- `<brief>`: the Samsung-watch brief, outside the repository (it names cluster details); its path is in the
  session's start line and in the secretary's handout.
- `<status file>`: beside the brief, `kron-samsung-status.md`. Run log and results live there.
- `<share>`: the rigger's inventory default until desk #4 is answered: 25 CPU cores for this seat; GPU jobs outside
  the core share; GPU as the loops found it (kron-fm on gpu13). This seat's first jobs used gpu12 card 0, free at
  the time and named free by the brief; short inference jobs, 4 CPUs each.

## State

- Work so far: `kron-ppg wrist-age` (per-person age from a frozen encoder, ridge head, grouped CV, constant and
  source-median baselines, permutation null, leave-one-source-out, `--lowpass`, `transfer`), commits 1cbae51,
  8642cbc, 68eaf93, 735a86f. `wrist-age-export` (ONNX graph with the linear head, timed) uncommitted until its
  first job (wa5) passes.
- Jobs wa3/wa4/wa5 (`kron-ppg-wa*`), watched by a background waiter.
- A public research note on the wrist results waits on the desk (a scientific claim in a public file); the numbers
  are in `<status file>` meanwhile.
- Aurora-BP: research only, never product development (its DUA, read 2026-11-10); a head fitted on it is never the
  deployed model. Open-data heads may be scored on Aurora alone.

## Items to file
- (none yet)
