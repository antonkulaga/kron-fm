# DNA builder — working state

Cluster paths are written relative to two roots that this file does not name: `$SCRATCH` (the account's scratch
folder) and `$KRON_FM_DATA` (the kron-fm data folder). The builder's own scratch is `$SCRATCH/kron-dna-build/`; the
trainer's is `$SCRATCH/kron-dna/`.

## Started 2026-11-18

Sources read: the priming, the kit's trainer and runner roles, the trainer's plan
(`reports/kron-dna-state-and-plan-2026-11-18.md`), the trainer's handoff (kept in this clone's
`.git/seats/handoffs/dna-builder/delivered/`) and its addendum on the ENCODE download pilot.

**Base (agreed with dna-trainer in `#kron-fm>dna-builder`, 18 Nov):** trainer main `9a1d784` merged into
`dna-builder` (`8a2a757`), then `7fdd30a` (with `lr-cooldown`). The trainer lands `lr-cooldown` (`c2d205b`, touches `cli.py`: `genome_lr()`, `--lr-total`,
`--cooldown-from`) and a test-helper fix before pushing main, and says so in the topic: merge main again then.
Main was pushed 18 Nov (`6bc790b`); the branch is pushed from then on. **Before every push: scan the diff against
`origin/main` for account, partition, host names and absolute paths** (this file is public).

**Environment:** `slurm/env.local.sh` copied from the trainer's clone with `JOB_PREFIX=kron-dna-build-`. The venv has
the `genomics` and `baselines` groups (`uv sync --group genomics --group baselines`; the audit tests need
`alphagenome`, ClinVar reading needs `polars-bio`).

**Handover (Anton through the trainer, 18 Nov):** the arms restart from published main after E finishes, about
17:30. By then the trainer needs, in the topic: the tested commit to merge and which data manifests are validated.
What is not ready waits for the next restart.

Given to the trainer 18 Nov: commit `4045a7a`, pushed as `origin/dna-builder`, one commit on published main
`6bc790b` (the 5 formerly failing cases and the xheads tests pass on CPU on the same code, 38 passed, 12 CUDA cases
skipped; the commits before it named cluster accounts in this file, so they stay in the clone under
`refs/local/dna-builder-raw` and are never pushed) and the DNase pilot store as validated. Order on E's freed
card (the trainer's plan): (1) the CUDA cases of `tests/test_kron_dna_xheads*.py` and a single-card memory
estimate; (2) the two-card 1 Mb bench with the new head when F hands over its cards (`--xhead-hidden 128` if it
does not fit); (3) the ClinVar pass, `$SCRATCH/kron-dna-build/clinvar/v1/score.sh`. Full suite on `58a25de`: 475
passed, 5 failed, all five failing on trainer main `7fdd30a` too (parent-equivalence tests hit by the audit's
mask-seed change; the trainer is fixing the helpers). Merge trainer main again when that lands.

**Restart of all four arms (Anton through the trainer, 18 Nov 16:30):** A, D, E and F restart with both new heads,
from published main; the trainer writes the launchers. Sent to the trainer: E has no NTv3 head, so it needs
`--xhead-split <split.u8>` (`3d5d1db`, to merge before E launches); D runs data-parallel, which `--xhead` refuses,
so D runs as one process with `--split-cards 2` or waits (a data-parallel `--xhead` is not written: each process
would need the labelled sets and rates synced). Recommended options: `--xhead-hidden 128`, `--xhead-weight 0.5`,
`--xhead-lr` = the arm's `--lr`, `--xhead-warmup 200`, `--xhead-only 1000` (so an evaluation scores the heads on
the frozen model). `--inherit-lr` is the trainer's and Anton's call: 4e-5 (a tenth of the schedule for the rest of
the run) or the arm's own `--lr` with a 500-update warmup; asked of Anton in the session.

**Split gradients on a card, 18 Nov (the trainer's CUDA run failed 2 of my tests):** not a defect. With IEEE
kernels the split and whole-window gradients of a head's loss agree to 3e-6..1e-5 upstream of the core, the same
as two whole passes of one input; the heads and decoder agree to 1e-6. The KronLSTM kernels are not bit-repeatable
and amplify a 2e-7 change about a thousandfold backwards; kernels compiled before `TRITON_F32_DEFAULT=ieee` is set
(any pytest process that ran other tests first) give about 1e-3. My first guess, max-pool rerouting, was wrong:
measured, it changes nothing. Tests now use 5e-3 upstream of the core on CUDA; `scripts/kron_dna/diag/split_grad.py`
reproduces it in 10 s on a card. The CUDA cases of the changed tests are still to be rerun.

**Two slips of mine to not repeat:** a commit id written into a message before the commit existed (twice): take the
id from `git log` after the push, then write the message. A file created by a shell redirect whose command failed
was committed empty: check a generated file's size before `git add`.

**Restarts with both heads (the trainer's, 18 Nov evening, code `780de73`):** A from step 244k (run directory
`A-best-xh`), D from 166k as one process with `--split-cards 2` (`D-jepa-xh`), E as a frozen-backbone head probe
first, F at its next checkpoint (71k). Read their `xheads.csv` when evaluations exist: the `head-only` rows are
the heads on the frozen model, the `joint` rows what updating the shared layers adds; DNase is Pearson at 128 bp,
splice is average precision, both on chr8/chr9. Early check (the trainer role's duty, for my heads): a head whose
held-out score does not move over its first evaluations, or a loss that is not finite, goes to the trainer at once.

**Input length (Anton, 18 Nov evening):** "keep A and E on one card but D and F will be two cards with 1Mb". Forwarded
to the trainer, who owns the launchers: D changes from two 524 kb rows to one 1 Mb window cut by length; A, E and F
stay as they were. So from this restart A against D is no longer one recipe at one context.

## Items

| item | state |
|---|---|
| B1 ClinVar cohort manifest | done: `$SCRATCH/kron-dna-build/clinvar/v1/manifest.{parquet,json}`, by `scripts/kron_dna/clinvar_manifest.py` |
| B2 ClinVar coverage check | done, below |
| B3 ClinVar scoring of A, D, F and AlphaGenome on the held-out sample | waits on a card: the trainer says in `#kron-fm>dna-builder` when E's is free (E ends about 18:00 on 18 Nov). Snapshots: `$SCRATCH/kron-dna/snapshots/` (A 220k, D 150k, F v2.4 61k; `MANIFEST.json` has sha256). Scores and directions fixed in `clinvar/v1/PREREG.md`; Anton, 18 Nov: compare with both AlphaGenome 1 and 2. That part needs only the API: `scripts/kron_dna/clinvar_alphagenome.py` (resumable cache in `clinvar/v1/alphagenome/`); **waits on Anton for the API key file's path** (asked 18 Nov in the session; the trainer does not know it and its search of old logs was refused by auto mode, so nobody searches for it). AlphaGenome 2 conditions fixed in `PREREG.md`: muscle `UBERON:0001134`, frontal cortex `UBERON:0001870`, liver `UBERON:0001114`. The trainer's 12 Nov rates: v2 15,397 variants x 3 conditions in 2 h 36 min, 359 unanswered; v1 about 41 min. Then: run `--limit 100` first for the rate, then size the whole pass; write the v2 condition list into `PREREG.md` before reading any score |
| B4 DNase pilot download (13 files, 8.7 GB) | done 18 Nov: 13 of 13 verified by size and md5, 8.68 GB in 536 s (16.2 MB/s, one stream, in the seat's session; a job for it was cancelled unstarted behind the group's CPU cap) |
| B5 ENCODE 100-file pilot | done 18 Nov 20:49: 100 of 100 verified by size and md5, no failure, 79.7 GB in 1,752 s over three streams (45.5 MB/s; 17.0 MB/s per stream, so streams add up). In `$SCRATCH/kron-dna/encode_bw/pilot100/` |
| B10 Bulk ENCODE download | Anton, 18 Nov 21:00: validated subsets go into training as soon as they exist, without waiting for all downloads, and disk is no constraint (300 TB); "but heads limitations are important". Running in the seat's session, three checked streams each: stage 1 since about 20:55, every DNase-seq and RNA-seq file not yet held (1,889 files, 456 GB, about 2.8 h), into `$SCRATCH/kron-dna/encode_bw/stage1/`; stage 2, ChIP-seq (3,566 files, 4.24 TB, about 26 h), starts by itself when `stage1/ended` appears, into `stage2/`. Both are resumable by rerunning the same commands; they die with the session, so after a restart of the seat rerun them first |
| B11 Binned heads and adding a head on resume | **next, the critical path for every further subset.** (1) A head kind that predicts at bins from pooled decoder states (`xheads` entries with a bin size, 32 bp for DNase and RNA, 128 bp for ChIP), with a binned store (mean signal on a 16 bp grid per contig, u8 log1p with a scale high enough to keep peak summits; a window's bins are read with its start snapped to the grid), the loss on bin totals, Pearson at 128 bp. Reason: memory and IO per step, not disk. A per-base head costs about 12 MB per output per card at 1 Mb in the loss alone (my estimate, to measure). (2) A resume that finds more `--xhead` heads than the checkpoint has: load the old ones, start the new ones at their own warmup, keep the optimizer state of every old group, no second head-only phase. Then: convert each validated subset (all DNase first, 533 tracks, when stage 1 has them), hand the trainer a store and a commit per subset |
| B6 DNase pilot track store (new versioned path, never over the prepared hg38) | done 18 Nov: one CPU job, 2 cores, 9 min 19 s (a second queued copy cancelled unstarted). `$KRON_FM_DATA/interim/tracks/hg38/dnase-pilot-v1/` (`dense.u8` 38.2 GB, 13 channels, `dense.json`). Checks below |
| B7 Extra track head + phases, per-group LR, per-head counters in the checkpoint | done on CPU, `58a25de` on top of trainer main `7fdd30a`: `genome-train --xhead NAME:STORE` (or `NAME:N`), `--xhead-only`, `--xhead-lr`, `--inherit-lr`, warmups; `src/kron_dna/xheads.py`; tests `test_kron_dna_xheads{,_stage,_cli}.py`. Rates scale `genome_lr` (agreed with the trainer). Head-only keeps the inherited rate at 0 (weights stand still; AdamW's moments for them do warm up, by design). Not yet done: the CUDA test cases and the 1 Mb memory check, which need a card (F peaks at 34.2 and 31.7 GiB per card of a 37.5 limit; a 13-output head with a 512-wide hidden layer at 1 Mb is the open risk, `--head-hidden` is shared with the inherited heads so it cannot be lowered for the new one alone) |
| B8 Splice head | labels done 18 Nov: `scripts/kron_dna/prep_splice_labels.py` wrote `$KRON_FM_DATA/interim/tracks/hg38/splice-sites-v1/` (`splice.u8`, four bits: donor +, acceptor +, donor -, acceptor -; one base each, the intron's first base for a donor and its last for an acceptor in the transcript's direction; GENCODE v50 basic). 504,780 of 504,846 introns placed, 734,330 distinct sites. Motif check from `bases.u8`: donors GT 0.972 + GC 0.019 on +, 0.971 + 0.019 on -; acceptors AG 0.991 and 0.990. The prepared genome's own `labels.u8` definition (unstranded, two bases) is untouched and stays the baseline. Head done on CPU 18 Nov: `xheads.LabelStore` and a binary kind (BCE, positive weight sqrt(neg/pos) capped at 100, strand bits exchanged under `--rc`, the training split's code 1 as for the tracks); `--xhead splice:<dir>` picks the kind from the folder; the held-out score is average precision per channel on chr8/chr9 (`xheads.csv`, column `heldout_metric`). Tests in `test_kron_dna_xheads_stage.py` and `_cli.py`. Not on main yet: it is on the branch after the commit the trainer merged, so it goes into a restart only if the trainer takes it and its bench passes. Still to do: near-splice ClinVar from this head against the distance baseline (a splice-delta score from the new head) |
| B9 Teacher outputs (AlphaGenome 2 API terms; NTv3_tracks licence is Anton's call) | not started; measured data goes first |

## B1/B2 — the ClinVar cohort, 18 Nov

- ClinVar GRCh38 VCF of 2026-11-08; GENCODE v50 basic (Ensembl 116); held out chr8, chr9; seed 0; 2,000 per
  (region, label, split); near-splice = 10 bases. Checksums in `manifest.json`.
- 1,495,501 reviewed pathogenic or benign SNVs; 1,493,774 placed; 1,719 outside the contigs (1,707 on chrMT, the
  rest on two unplaced scaffolds); 8 reference mismatches. Excluded rows stay in the manifest with the reason.
- The sample `genome-clinvar` would score: 31,131 rows, 13,275 of them held-out. Held-out pathogenic counts are
  small outside coding and splice sites: near splice 174, UTR exon 50, intron or intergenic 41, regulatory 3.
- **Coverage:** `output/eval/clinvar/genome{,-rc}.parquet` hold 17,131 of the sample (7,275 held-out), but only for
  the stage-1 and stage-2 checkpoints at a 262,144 window: nothing for A, D or F. The AlphaGenome caches
  (`interim/cds/clinvar/alphagenome{,2}.parquet`) are the legacy id-only layout and cover 88 and 87 of the 13,275
  held-out rows (they were built for the codon benchmark). So every model needs a full pass on the held-out sample.

## B4 — DNase pilot selection

NTv3_tracks' hg38 ENCODE list has no ATAC-seq: its accessibility assay is DNase-seq (533 experiments, 269 GB as
read-depth normalised signal). The pilot head therefore takes DNase. One file per biosample, the one pooling the
most replicates, outside the trainer's 100 random pilot rows: A549, B cell, CD14 monocyte, H1, HeLa-S3, HepG2,
IMR-90, K562, MCF-7, foreskin keratinocyte, heart left ventricle, hepatocyte, liver.
Selection `$SCRATCH/kron-dna-build/tracks/dnase-pilot-v1/select.tsv`; downloads beside it in `files/`, with
`provenance.jsonl` (accessions, assembly, replicates, lab, award, md5, seconds) and `status.tsv`.

## B6 — the DNase pilot store, checked 18 Nov

- Decoded 128 bp bins against the source bigWigs, 40 random 131 kb windows per channel: Pearson 0.982 to 0.999
  (lowest A549 0.982 and hepatocyte 0.982). The loss is the encoding's clip at each track's 99.9th percentile of
  non-zero signal (the NTv3 store's rule): it cuts peak summits, most where the scale is low (HeLa-S3 1.01, MCF-7
  1.02, heart 1.11, hepatocyte 1.15). A higher percentile for peaky assays is the first thing to change in v2.
- No negative values, no contig missing from any file. Bases without a value (stored as 0): 0 in seven files, 6 to
  12 % in five, 45 % in IMR-90 (ENCFF547YAH), an older file that leaves out zero runs. Kept in `dense.json`.
- Coordinates and labels on known loci (mean decoded signal over the track's non-zero mean): GAPDH and ACTB
  promoters 7 to 67 in every channel; ALB promoter 2.9 HepG2, 1.1 liver, 0.9 hepatocyte, at most 0.4 elsewhere
  bar heart 0.8; CD19 promoter 7.5 in B cell, at most 1.7 elsewhere; POU5F1 2.6 in H1; a chr5 gene desert and a chr1
  olfactory cluster at most 1.0. So positions and channel order are right.
- Training mask: the prepared genome's `ntv3/split.u8` (1 = train; chr8 and chr9 are code 4, verified by a
  strided read), so NTv3's own validation and test regions are left out of the new head's training too.
- Cost, measured: download 8.68 GB in 536 s; conversion 18.6 core-minutes for 13 tracks (9.3 min on 2 cores); store
  2.94 GB per channel at base resolution. Projected for all 5,568 chosen ENCODE files (4.8 TB of bigWigs): about 82 h
  of download on one stream at 16 MB/s, about 133 core-hours to convert, and 16 TB as per-base u8 (the format does
  not scale to thousands of channels: bulk needs binned or sparse storage, to design before any bulk run).

## Cluster facts found by trial

- Anton, 18 Nov: besides the group's capped account there is an uncapped preemptible one, where anyone may cancel jobs, so only jobs whose output survives a kill go there. `slurm/env.local.sh` (git-ignored) takes the account and CPU partition from the environment for that; the names are in the seat's machine memory, not here. The capped account was full that afternoon.
- ENCODE through the proxy: 16 MB/s on one stream.
- Anton, 18 Nov: the running models are behind main; at a restart every relevant change is merged to main and main is published first. Forwarded to dna-trainer. For this seat: the F successor launches only from a published main that holds the new heads.

## To file at an idle turn

- Nothing yet.
