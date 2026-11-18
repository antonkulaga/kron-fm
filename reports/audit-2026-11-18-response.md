# Response to the independent code audit of 18 Nov 2026

The audit (`$SCRATCH/kron-dna/reviews/codex-20261118/` on the cluster) listed 17 open findings: 1 P1, 15 P2
and 1 P3. All 17 were reproduced or checked against the code, fixed, and covered by a regression test that fails on the
previous code and passes on the fix. For each finding, the last column says whether it changed a result already reported,
checked against the run directories, checkpoints and evaluation outputs on 18 Nov 2026, 02:30-03:00 CET.

The training arms (A, D, E, F) run from frozen source copies (`kron-fm-dna-wt-*-run`) and were not touched. None of
the findings is serious enough to restart a running arm (see TR04).

## Repository fixes (this commit series; tests in `tests/test_kron_dna_audit_20261118.py`)

| ID | What was wrong | Fix | Changed a reported result? |
|---|---|---|---|
| D1 (P1) | `genome-prepare` opened the final `bases.u8` before parsing the FASTA: a failed or mistaken run left an empty or partial genome beside the old index | Refuses an existing genome unless `--overwrite`; parses first; builds both files in a staging folder; checks array length = index extent; publishes the pair, keeping the old one as `*.prev` | No. All 14 prepared genomes have `bases.u8` size equal to their index extent |
| D2 | A streamed rebuild into a folder holding an old `tokens.npy` was silently ignored by the loader | The stream writer refuses a folder with `tokens.npy`, the monolithic writer refuses a sharded folder, and the loader refuses a folder holding both | No. None of the 4 monolithic datasets sits beside a shard list |
| TR01 | Resume loaded the NTv3 head and its optimizer rows by position, without checking channel identity | Resume refuses a different ordered channel-id list or target/centre/loss, and refuses dropping the NTv3 head. Older checkpoints without ids get a note | No. A, D and F carry the same 155 ids in the same order (one hash) and the same target settings in every checkpoint, from the oldest backup to the current one. E has no NTv3 head |
| TR02 | A `--stop-step` off the evaluation grid exited without saving that step | The stop step is an evaluation-and-save boundary, like the curriculum end, on every rank | No. No run record has a non-aligned stop step; the only `--stop-step 1` launches used `--eval-every 1` |
| TR03 | `--seed` did not seed the model's initial weights or the training masks | `torch.manual_seed(seed)` before the model is built; the mask/dropout stream is seeded per card and start step | No number is wrong. Arms started from scratch (A, B, C, D, F) had different initial weights despite `--seed 0`, so differences between arms include an unmeasured initialisation component. E continues the control's weights |
| TR04 | Each resume restarted the main window stream, replaying the windows drawn after the previous resume | A resumed run draws from step-keyed streams (`[seed, rank, start step]`); a fresh run keeps its stream. Recorded as `rng` in the run record | Held-out and benchmark numbers are honest measurements, but the arms trained on partly repeated window positions: A redrew 44k steps' windows (steps 134k-178k repeat steps 90k-134k), D about 19k steps, F about 28k of its 44k steps. Each arm's current segment is past its earlier ones (F within about 3k steps), so new windows are drawn from now on. Not a reason to restart |
| TR05 | Resuming with nonzero `--jepa-ema` discarded the EMA teacher | The teacher is saved (`jepa_target`) and restored. Older checkpoints restart it with a note; EMA 0 unchanged | No. No arm had a teacher: A, B, C, E train without JEPA; D and F use EMA 0 |
| TR06 | The optional local recurrent track's restart points could join two batch rows | Restart points per row; the same as before when the row length is a multiple of the span | No. No arm uses the local track |
| EV-01 | The mapped ClinVar table was cached in one file, whatever the VCF, CDS, gene split or seed | Cache file keyed by the VCF and CDS identity (path, size, mtime), held-out fraction, seed and mapping version | No. All 17 ClinVar evaluations used the same VCF, CDS and seed with the default split; the VCF predates the cache |
| EV-02 | The AlphaGenome cache was keyed by ClinVar id alone, so a changed condition or scorer list reused old scores | One row per variant (id, position, alleles), model version and condition, one column per scorer. The score is the maximum over the requested conditions, and NaN unless all of them are in. The old id-keyed file is not read | No visible distortion: one AlphaGenome 2 result file and one cache, and the README's condition set (muscle, frontal cortex, liver) was written once. No log survives to rule out an aborted earlier run with other conditions |
| EV-03 | `genome-clinvar` named a checkpoint file's scores after its parent folder, so two files of one folder overwrote each other | `run_keys`: folder name, extended by the file stem where names collide; duplicates refused before scoring | No. Both genome-ClinVar outputs scored whole run folders with distinct names |
| OP-03 | `slurm/job.sbatch` picked the first GPU with no process, without a lock | A node-local `flock` lease per card, held through the job's lifetime and released by the kernel on exit | No. The arms pin their cards by hand |

## Scratch-script fixes (cluster scripts; tests in `$SCRATCH/kron-dna/auditfix/tests/`)

| ID | What was wrong | Fix | Changed a reported result? |
|---|---|---|---|
| D3 | T2T stage 4 accepted a tagAlign after a failed dedup (`$(rep_ta ...)` runs without errexit) | Every helper propagates its failure; each call site checks its substitution; `inherit_errexit` | No production output existed. Tested with the real stage-4 flow and an injected dedup failure |
| D4 | `nonzero_mean.py` wrote `nonzero_frac` as a ratio to the densest track | Divides by the number of NTv3 train bases | No. The human `dense.json` has no `nonzero_frac`; the species files come from the newer, correct script; no training code reads it |
| EV-04 | `trackeval.py` kept an existing prediction array beside new metrics computed from different weights | A differing array goes to its own file (suffix or step); the metadata names the file its metrics came from | No. The two recorded repeats differ by 0.0 |
| OP-01 | The launchers started a code change even when the checkpoint backup failed or was partial | `lib_backup.sh`: copy to `.part`, compare, rename; an existing backup is reused only when byte-identical; on failure the launcher exits 4 before training | No. Every `checkpoint-before-*` backup loads and holds the step its name says |
| OP-02 | `evalround3.sbatch` deleted its snapshots and printed DONE whatever the evaluators returned | Each exit code goes to `bench/evalround3_manifest.tsv`; snapshots are kept and the round marked INCOMPLETE if any evaluator failed; training resumes either way | No. Round 3's track evals exited 0 and all three masked-base runs finished |

Note on OP-01: arm A's running bash reads `A_524_nopad.sbatch` from the scratch path (the eval-round job exec'd it).
The edit first rewrote that file in place, which bash would have misread once training ends. Its open inode was
restored byte for byte (md5 checked through `/proc/<pid>/fd/255`), and the new version was installed as a new file.

## Tests

- `tests/test_kron_dna_audit_20261118.py`: 29 tests (D1, D2, TR01-TR06, EV-01-EV-03, OP-03). The bug tests fail on
  commit 4426f66, and the full suite passes on the fixed code (CPU; GPU kernels not exercised).
- Scratch: `test_launchers_op01_op02.py` (copies of the live launchers in a sandbox, stub commands),
  `test_ev04.py` (the evaluator's real block), `test_d4_nonzero_frac.py` (the real script on a fixture). D3's harness
  is described in `t2t_tracks/qc/CODE_REVIEW_RESPONSE_20261118.md`.

Not validated: GPU kernels, multi-GPU collectives at the new TR02 save boundary, full-size runs.
