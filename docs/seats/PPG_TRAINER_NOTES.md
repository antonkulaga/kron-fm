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
  8642cbc, 68eaf93, 735a86f, e8c0e63, ffc1cd4, 550e798 (wrist-age-export: ONNX with the linear head, timed).
- Owns since 2026-11-11 00:35 the watch-data work (wrist datasets, stream trainer; its handout and status file sit
  beside the brief). Jobs: s5 done 2026-11-11 00:52; 6428195 kron-w1-pretrain (eta 03:50), a plain waiter armed. Then probe-full.sh and the w1/s5 transfer test. Deployable predictor and draft note: see `<status file>`.
- Desk: #7, #8 answered (published; the target is a prediction on his own watch). Open: #20 data access requests, #21 UK Biobank fee.
- The two Samsung Health exports (Anton's, Livia's) are reported privately (paths in `<status file>`); no PPG in
  either. `src/kron_ppg/samsung_health.py` reads them; its R-peak detector failed on real watch ECGs.
- Aurora-BP: research only, never product development (its DUA, read 2026-11-10); a head fitted on it is never the
  deployed model. Open-data heads may be scored on Aurora alone.

## Reporting (Anton, 2026-11-11)
- Results page (tabbed artifact, republish the same file at each milestone): https://claude.ai/artifact/GJDyDo8i87LqDiT8s9EHip;
  its source sits beside the brief's private files (kron-samsung). Kept state report in #kron-fm > state (keep name
  ppg-state), milestones in #kron-fm > milestones, questions in #kron-fm > decisions. House style: the desk's
  docs/REPORTING.md (finding first, one headline number with ± and n, marks, tables). No person's values.
- Every step asked of Anton carries its URL (dataset page, agreement to sign, desk issue, form); Anton, 2026-11-11:
  "always give urls".

## Cluster rules taken up
- Every sbatch carries `--comment 'seat=kron-fm/ppg-trainer; what=<one line, no person or private data>;
  eta=<ISO-8601 with offset, or ?>; cuda=<card or ->'`; `scontrol update JobId=<id> Comment='...'` when the
  estimate moves (the rigger, agent-chat 2026-11-11).

## Items to file
- filed: kit #422 (until-event ends on watch-job's armed banner); a plain until-loop on squeue meanwhile.
- `canceled: both work and are tested; they share only a ridge fit, and a merge risks two CLIs for no gain` (2026-11-11): fold `wrist_probe.py` (14ab76c, another session) and `wrist_age.py` into one module.
