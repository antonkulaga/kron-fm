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

## Items

| item | state |
|---|---|
| B1 ClinVar cohort manifest | done: `$SCRATCH/kron-dna-build/clinvar/v1/manifest.{parquet,json}`, by `scripts/kron_dna/clinvar_manifest.py` |
| B2 ClinVar coverage check | done, below |
| B3 ClinVar scoring of A, D, F and AlphaGenome on the held-out sample | waits on a card: the trainer says in `#kron-fm>dna-builder` when E's is free (E ends about 18:00 on 18 Nov). Snapshots: `$SCRATCH/kron-dna/snapshots/` (A 220k, D 150k, F v2.4 61k; `MANIFEST.json` has sha256). Scores and directions fixed in `clinvar/v1/PREREG.md`; Anton, 18 Nov: compare with both AlphaGenome 1 and 2. That part needs only the API: `scripts/kron_dna/clinvar_alphagenome.py` (resumable cache in `clinvar/v1/alphagenome/`); **waits on Anton for the API key file's path** (asked 18 Nov in the session; the trainer does not know it and its search of old logs was refused by auto mode, so nobody searches for it). AlphaGenome 2 conditions fixed in `PREREG.md`: muscle `UBERON:0001134`, frontal cortex `UBERON:0001870`, liver `UBERON:0001114`. The trainer's 12 Nov rates: v2 15,397 variants x 3 conditions in 2 h 36 min, 359 unanswered; v1 about 41 min. Then: run `--limit 100` first for the rate, then size the whole pass; write the v2 condition list into `PREREG.md` before reading any score |
| B4 DNase pilot download (13 files, 8.7 GB) | done 18 Nov: 13 of 13 verified by size and md5, 8.68 GB in 536 s (16.2 MB/s, one stream, in the seat's session; a job for it was cancelled unstarted behind the group's CPU cap) |
| B5 ENCODE 100-file pilot (trainer's job 6510989) | waits on the job; then md5 check and the cost report for Anton (bytes, CPU-hours, projected total) before any bulk download |
| B6 DNase pilot track store (new versioned path, never over the prepared hg38) | done 18 Nov: one CPU job, 2 cores, 9 min 19 s (a second queued copy cancelled unstarted). `$KRON_FM_DATA/interim/tracks/hg38/dnase-pilot-v1/` (`dense.u8` 38.2 GB, 13 channels, `dense.json`). Checks below |
| B7 Extra track head + phases, per-group LR, per-head counters in the checkpoint | done on CPU, `58a25de` on top of trainer main `7fdd30a`: `genome-train --xhead NAME:STORE` (or `NAME:N`), `--xhead-only`, `--xhead-lr`, `--inherit-lr`, warmups; `src/kron_dna/xheads.py`; tests `test_kron_dna_xheads{,_stage,_cli}.py`. Rates scale `genome_lr` (agreed with the trainer). Head-only keeps the inherited rate at 0 (weights stand still; AdamW's moments for them do warm up, by design). Not yet done: the CUDA test cases and the 1 Mb memory check, which need a card (F peaks at 34.2 and 31.7 GiB per card of a 37.5 limit; a 13-output head with a 512-wide hidden layer at 1 Mb is the open risk, `--head-hidden` is shared with the inherited heads so it cannot be lowered for the new one alone) |
| B8 Splice head | next, for the restart after 18 Nov (not rushed into the 17:30 one). Found: `labels.u8` holds donor and acceptor as unstranded 2-base intervals (the intron's first and last two bases, `genome.splice_sites`), which is what the stage-2 splice-delta used. Plan: (1) a new label store from GENCODE v50 basic with four channels, donor and acceptor per strand, one base each (the intron's first base for a donor, its last for an acceptor, in transcript direction), on a new path, checked on both strands against known genes; (2) a second head kind in `xheads.py`, classification (BCE with positive weights, `--rc` swapping the strand pairs), since the present kind is quantitative only; (3) development scores: site precision-recall on chr8/chr9 and near-splice ClinVar against the distance baseline. The two-base unstranded definition stays as the baseline it is. Say the label definition to the trainer before building |
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
