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

**Head comparison (Anton through the trainer, 18 Nov 21:15: "can KAN or MLP with different params do better?"):**
job 6517859 `kron-dna-build-headcmp`, queued on the card the trainer's E probe (6514768) holds, starting when the
probe ends (`afterany`). `scripts/kron_dna/head_compare.py`: six head variants (linear, the current MLP 128, MLP
512, a 3-layer MLP, a 9-base conv head, a reduced KAN) on E's frozen 400k trunk, 131 kb windows, 3 h; scores on 128
fixed chr8/chr9 windows with a window bootstrap of each variant's difference to the current head. Output in
`$SCRATCH/kron-dna-build/head-compare/E400k-v1/` (`progress.csv`, `results.json`). First run on a card: if it fails
early, fix and resubmit; then post the table to the trainer's topic and to Anton, with the caveats (one trunk, one
seed, 13 DNase tracks and splice sites only, frozen features: a head that mixes positions may matter more or less
once the trunk trains with it). The ClinVar pass takes the card after it.

**Done 18 Nov evening, on `origin/dna-builder`:** a plain resume may add `--xhead` heads at the end of the list
(`0366e4e`: old weights, optimizer moments, phase and counts stand); the model side of binned heads (`[name,
outputs, bin]`, a head on the mean decoder state per bin). Still to do for binned heads: the binned store, the
loss on bins and the wiring in `genome-train` (B11).

**State at the end of 18 Nov (read this first after a restart of the seat):**
- The account's 7-day quota passed 85 % at about 22:54: watcher lines (job starts and ends) do not wake this seat
  until the window resets Monday morning; messages from a person or from the trainer still do. So look at the
  queue and the logs yourself at every turn.
- F and D ran out of card memory at 1 Mb with the two heads at hidden 128 (the trainer restarted both at hidden
  32). The fix is pushed, `4f47980`: in training the heads run on the supervised stretch only (0.375 of the bases
  at `--ntv3-center 0.375`), same loss and gradients, CPU-tested (90 passed with `test_kron_dna_split.py`), not yet
  run on a card. The trainer benches it and puts F and D back to hidden 128.
- Head comparison, job 6517859, still waits on the trainer's E probe. Eight variants now: linear, MLP 32/128/512,
  3-layer MLP, conv 9, a reduced KAN, and `kron` (a per-base linear readout plus one bidirectional KronLSTM layer
  over 32-base means; Anton asked whether KronLSTM layers can be heads). A variant that cannot run on the card is
  dropped, the rest go on. After it: post the table (scores with intervals, parameters, measured memory) to the
  trainer's topic and to Anton, and replace the computed memory table I gave Anton with the measured one.
- Downloads run in this session (they die with it; rerun the same commands): stage 1 DNase and RNA-seq, then stage 2
  ChIP-seq by itself.
- AlphaGenome for the ClinVar cohort: Anton gave the API key in the session on 18 Nov about 23:20. It is in an
  owner-only file outside every repository (the path is in `clinvar/v1/alphagenome/` logs' command lines and in
  the seat's machine memory, never here), and it is repeated nowhere. Timing on 100 held-out variants: version 1,
  149 s with 4 threads; version 2 with the three fixed tissues, 42 s. The full pass over the 13,275 held-out
  variants runs in this session since about 23:25, version 2 first, then version 1 (about 1.5 h and 3 to 5 h by
  those rates), resumable, into `$SCRATCH/kron-dna-build/clinvar/v1/alphagenome/`. No score has been looked at:
  only counts. When both are done, wait for the KronGenome pass (it needs a card) before reading any AUROC, as
  `PREREG.md` says.
- Anton's open asks to me: a probe of where the
  information lives (heads on the 1 bp decoder states, on the 128 bp core states, on the encoder's 1 bp states), which
  he has not ordered but which answers his question whether the binning is a problem; a chunked head with a
  two-stage backward (I recommended it before the binned store); a proper literature survey on heads only if he
  asks (one search of four hit the usage limit: heads conditioned on a cell or track embedding).

**19 Nov, 00:40:** stage 1 of the download is complete: 1,889 of 1,889 DNase and RNA-seq files verified, 456 GB in
3 h 45 min (34 MB/s over three streams). Stage 2 (ChIP-seq) started by itself. Next subset for training, per
Anton's rule: one DNase file per biosample beyond the pilot's 13, 196 tracks, selection and links in
`$SCRATCH/kron-dna-build/tracks/dnase-biosamples-v1/`; conversion job 6520646 `kron-dna-build-dnase196-prep` on the
uncapped account, to `$KRON_FM_DATA/interim/tracks/hg38/dnase-biosamples-v1/`, clipped at the 99.99th percentile
(the pilot's 99.9 cut peak summits). When it ends: the same checks as the pilot (decoded bins against source, nan
shares, known loci), then offer it to the trainer as a third head added on a plain resume. Memory at 1 Mb with the
span cut, my estimate: about 1 GiB per card for 196 per-base outputs, so A and E first, F and D only if their
bench has room; otherwise it waits for binned heads. The head comparison job 6517859 has been held by the group's
CPU cap since the E probe ended at 00:03, with the card idle.

**19 Nov, about 03:00.** The head comparison's first run (6517859) died at its first evaluation on a device bug of
mine (fixed `9bed2b4`; CPU tests miss CUDA-only faults, so the job script now runs a 2-step smoke run with one
evaluation first). The rerun, 6522985, runs from about 02:00 into `$SCRATCH/kron-dna-build/head-compare/E400k-v2/`.
At step 8000 of about 29,000: KronLSTM heads lead on both tasks (DNase r 0.620, splice AP 0.369 for one layer;
0.621 and 0.365 for two) against the current MLP 128 (0.560, 0.310), conv 9 next (0.577, 0.357). Measured memory
above the features at batch 2 x 131 kb: KronLSTM 2.46 GiB (1 layer) and 2.71 (2 layers), MLP 128 2.06, KAN 6.25.
Anton, 19 Nov: "let's put KronLSTM heads if they are the best" and "we also have to extend training set with
whatever we downloaded and processed"; he reads two layers as the better one so far, "but maybe further evaluation
will change". So: B12 below, built while the comparison finishes; the layer count is a setting, chosen from the
final table. Data: the 196-track DNase conversion and a new stranded RNA-seq selection (one experiment per
biosample with exactly one plus and one minus file: 231 biosamples, 462 channels as `<experiment>_P/_M`, 78.8 GB;
`$SCRATCH/kron-dna-build/tracks/rnaseq-stranded-v1/`) run as kill-safe `--convert-only` jobs, four and six of 4
cores, each store assembled by one job after them (`afterany`; it converts what a killed job left). AlphaGenome 1
and 2 passes are done over the 13,275 held-out variants (unanswered: 1-3.5 % per score for v2, 0.3-2 % for v1,
counted not dropped); no score read, per `PREREG.md`.

**19 Nov, 06:15: the head comparison is final** (6522985, 27,819 steps, 4.0 h; `head-compare/E400k-v2/results.json`).
Held-out DNase r / splice AP, difference to MLP 128 with 95 % window-bootstrap intervals: KronLSTM 1 layer 0.639 /
0.384 (+0.074 [0.061, 0.088] / +0.058 [0.045, 0.074]); 2 layers 0.642 / 0.384 (+0.077 / +0.057); 256 wide over
128-base steps 0.638 / 0.332 (splice gain gone); conv 9 0.584 / 0.378; MLP 3 layers 0.580 / 0.344; KAN 0.575 / 0.337;
MLP 512 0.569 / 0.338; MLP 128 0.565 / 0.327; linear 0.475 / 0.253. One and two KronLSTM layers tie; I recommended
one (less memory and time) to the trainer, who restarts A then E with `--xhead-kind kron`. The conversion jobs on the
uncapped account have not started (no job of that account runs; one moved to another partition pends too); asked
the rigger seat for a route, waiting meanwhile.

**19 Nov, evening.** KronLSTM heads (1 layer) are on main and in every arm (A from 296k, D, F; E's probe). Short
kill-safe CPU work goes to the uncapped account's 3-hour short-job partition (the preemptible partition started
nothing all day); a store's assembly now resumes block by block (`78fd7aa`), so it is chained in 3-hour jobs.
- DNase subset v1 (196 tracks, one per biosample, no overlap with the pilot): `$KRON_FM_DATA/interim/tracks/hg38/
  dnase-biosamples-v1`, validated: decoded 128-bp bins vs source r median 0.9995, min 0.963; no contig missing; 12
  older tracks leave >50 % of bases without a value (stored 0); GAPDH/ACTB promoters open in all 196; CD19 highest in
  four B-cell lines; POU5F1 in iPS and H7; ALB highest in pancreas (no liver sample in this set). In A's training
  since 19:16 (head `dnase196`, added on a plain resume at 318k); D and F wait on A's memory at its first eval.
- Stranded RNA-seq v1 (231 biosamples, 462 channels `_P/_M`, 1.36 TB): assembling, resumed at block 36 of 88 at
  19:10; four chained jobs. Then the same checks plus a memory check before any arm takes it.
- Head benchmark against AlphaGenome (Anton, through the trainer): `scripts/kron_dna/xhead_bench.py` and
  `xhead_bench_ag.py`, data in `$SCRATCH/kron-dna-build/xhead-bench/v1` (96 trackeval windows; 13 pilot DNase
  channels matched to AlphaGenome by ENCODE biosample term; 4 splice channels). AlphaGenome marks splice sites on
  the exon's boundary base, ours on the intron's: its output is read one base off (`bb9a1ee`). Result, AlphaGenome
  fold model minus A 312k on windows the fold never saw: chr8 (23) DNase +0.028 [-0.007, 0.069], splice AP +0.207
  [0.143, 0.260]; chr9 (18) +0.055 [0.024, 0.097], +0.198 [0.118, 0.266]. D and F trail A. AlphaGenome's own
  all-folds model gains about 0.05 on windows it trained on. The trainer draws the charts for Anton.
- Open: ClinVar scoring of A, D, F (needs a card; AlphaGenome 1 and 2 scores are done, unread per `PREREG.md`).

**20 Nov, 05:00.** Per-base wide stores stalled training on disk reads (the trainer: D 2.7 -> 6-7 s/step with the
196-track per-base store, 600 GB, read 196 MB per 1 Mb window). Fix, all on main (81a48c7): binned heads
(`--xhead-bin`, 710609e), pre-binned stores (`scripts/kron_dna/bin_track_store.py`: per 128-bp cell the mean
training target and mean log1p, float16) read by `xheads.BinnedTrackStore`, primary windows snapped to the 128-bp
grid (32fae75). Stores, validated against their per-base sources (cells within float16 rounding of bin means):
`dnase-biosamples-v1-bin128` (196 channels, 15.4 GB) in A and D since 02:34 as `dnase196b`;
`rnaseq-stranded-v1-bin128` (231 biosamples, 462 stranded channels, 9 GB) in A since about 04:50 as `rna462b`;
the trainer adds D and F after reading A's peak. Per-base RNA store checks: r128 vs source median 0.9995; strands
right (GAPDH own strand 34 vs 0.17, ACTB 30 vs 0.09); ALB top 3 liver.
ONNX (Anton: the default publishing format): `kron-dna export-genome-onnx` (41987e1, 9f42ef9), through torch.export
at opset 18 without the exporter's optimisation pass; F 96k at 131 kb exported in 16 min on the card, 373 MB, ORT vs
graph 8.6e-6, Triton vs graph 0.004; on F's Hugging Face repo (the trainer). kron_core/onnx.py's chunked scan now uses
positive dims (the TorchScript exporter resolved negative dims against stale ranks).
Open: ClinVar scoring of A, D, F (needs a card); stage 2 ChIP-seq download; the rest of stage 1's RNA-seq.

**25 Nov, 20:30 (evening).** An outside review (Astra 6.2 through the kit's review engine, three runs; a fourth
on the ablation script is running) and the trainer's error analysis drove the day.

- Review fixes on main (cd9860c) and after: lean masks hold for one forward; `TrackFeatureStats` (running statistics
  of the richer JEPA target; still folds the current batch in before normalising: open, moot while JEPA is dropped);
  `split_plan`'s unowned tail; `pool_bins` ratio; empty selections on a split replica (`_norm_weights`).
- Anton dropped every JEPA arm (no benefit measured). Cards go to architecture arms at 32 bp against arm A:
  K5P (5 core layers), K5 (`n_layers 5, bin_convs 10`), KI (`inter_levels [3], n_layers 5`), a DNase 32-bp head arm,
  and the codon head on F-lean 250k (`codon_dims 64`, `--codon-on masked`). The trainer runs and reads them.
- New in `genome.py` (5ac1ba4, fixes 4be4f36, 9c32f11): `bin_convs` (`BinConv`, `bin_pre`/`bin_post`/`bin_skip`:
  dilated convolutions each side of the core and a skip around it), `inter_levels` (`InterKron`: KronLSTM blocks in
  encoder and decoder at a pooling level, restarting every `local_span`), `codon_dims` (`CodonHead`,
  `protein_targets`, `codon_loss`, `codon_at`), `piece_step`/`piece_bounds` (every piecewise path cuts where a
  restarting recurrence restarts: split forward, `evalkit.core_capped`, `decode_span`, `forward_contig`),
  `layout_fingerprint`. A new module must be listed in `SPLIT_LOCAL` or as whole-batch:
  `tests/test_kron_dna_split_modules.py` fails otherwise (five were missed tonight; a one-device split test cannot
  see it).
- Protein target without inference: `scripts/kron_dna/esm_atlas_fetch.py` reads per-residue ESM C 6B sparse features
  (64 active of 16,384) from the public ESM Atlas by the MD5 of the sequence, range requests only. Shards: 16 per
  three-character hash prefix, in equal-count pieces whose edges run late (a second pass looks one chunk back).
  Human reference proteome: 20,103 of 20,652 proteins. `prep_esm_atlas_codons.py` lays them on codons of the
  transcript whose translated CDS is the protein: `$KRON_FM_DATA/interim/proteins/esm-atlas-v1/hg38-codons-v2`,
  9,545,923 codons, 18,541 genes, 75 % in the training split. Open from the review: how the stored byte maps to a
  value is an assumption; codons cut by a split are dropped; overlapping genes can give a base two targets; other
  species not fetched yet.
- Stores handed over today: `procap-stranded-v1-bin128` (448), `eclip-stranded-v1-bin128` (1,026), both after the
  minus-strand sign fix (429482a); `tss-stranded-v1-bin128` (442) and `tss3-stranded-v1-bin128` (890: CAGE, RAMPAGE,
  PRO-cap in one, the trainer's shared head `tss890`); `dnase-biosamples-v1-bin32` (the head-size arm);
  `tss3-stranded-v1-bin32` building (603 of 890 tracks, no consumer waiting).
- `scripts/kron_dna/focus_map.py`: `interim/genome/hg38/focus-bin128` (annotation bits and, per store, the share of
  tracks at 1x and 4x their typical signal per 128-bp cell); the trainer's `--focus-map` reads it.
- Measured and withdrawn: the soft clip of the binned targets loses nothing (a perfect prediction of the stored
  target scores the measured log1p); peaks "at half height" was a selection effect (AlphaGenome shows the same), so
  no loss change follows.
- From the trainer's checks (read their notes for the tables): masked bases are predicted through the conv tower,
  most through the coarsest skip; core layers 1-5 jointly matter more than 7-9; for DNase the core's value comes
  from within about 32 kb.
- `xhead_bench.py report` writes results.csv and summary.csv before the paired comparisons (7291ce2); the report
  on 61 sources was still in its paired bootstraps after 2.5 h.
- Waiting on Anton: the trust-register row for the aligners; whether track stores go to Hugging Face.

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
| B12 KronLSTM heads in training | Anton's word, 19 Nov. `genome-train --xhead-kind kron` (`--xhead-layers`, default 2; `--xhead-step-bases` 32; `--xhead-width` 128): each further head a per-base readout plus bidirectional KronLSTM layers over 32-base means of the decoder states (`genome.KronHead`, the comparison's head). Its context is the stretch it runs on: the supervised span in training, each card's piece on a split arm. An arm switches by a new run directory with `--init` (the checkpoint's MLP heads are not loaded; a fresh head-only stage), never by a plain resume (refused). CPU-tested; next: the CUDA cases and a 1 Mb bench by the trainer, then the restarts with the new stores as further heads |
| B9 Teacher outputs (AlphaGenome 2 API terms; NTv3_tracks licence is Anton's call) | not started; measured data goes first |
| B13 chip492b, core histone ChIP | done 22 Nov: `chip-histone-core-v1-bin128` (6 marks x 82 biosamples, 492 channels "mark|biosample"). Cells vs bigWig r min 0.9998. Biology (`scratchpad` check, numbers posted in `#kron-fm>dna-builder` 22 Nov): H3K4me3/H3K27ac 7-8x genome mean at GAPDH/ACTB/EEF1A1 promoters in 100% of biosamples, H3K36me3 2.9x over SF3B1's body, H3K27me3 3.4x at HOXD, H3K9me3 1.9x at the chr19 ZNF cluster. Cleared for A (restart) and D (next checkpoint); F has no room. I held it from 20 Nov 13:20 to 22 Nov 13:2x on Anton's "stop and wait" in the session; his "Just go with extension" (via the trainer) ended the hold |
| B14 Human TF and other-histone ChIP | Anton 22 Nov "Just go with extension"; agreed with the trainer: TF as its own head `chiptf`. Binning since 22 Nov ~13:50 on the uncapped partition: `chip-tf-v1-bin128` (1,643 tracks, 599 factors, 136 biosamples; conv 6591312-6591343, store 6591344) and `chip-histone-rest-v1-bin128` (1,431 tracks, 32 marks, 192 biosamples; conv 6591345-6591376, store 6591377). Selections in `$SCRATCH/kron-dna-build/tracks/{chip-tf-v1,chip-histone-rest-v1}/`; a channel duplicated by mark/factor and biosample carries `|ENCSR...`. **chiptf validated 22 Nov ~17:40** and handed to the trainer: cells vs bigWig r min 0.977; S/N median 3.9; POLR2A 3.5x / S5P 7.5x / TAF1, TBP, YY1 4.5-4.8x at housekeeping promoters, EZH2 1.24x; CTCF 5.96x at left-out shared CTCF sites (RAD21 5.47x, POLR2A 1.58x). Checks: scratchpad `chip_tf_check.py`, `ctcf.py` (to move into the repo as a store-check script). **chiphr validated 22 Nov ~20:30** (`chip-histone-rest-v1-bin128`, 1,431 tracks, 32 marks, 192 biosamples; cells vs bigWig r min 0.9995; core marks as chip492b; the 26 further marks at promoters and gene body as expected) and handed to the trainer as its own head `chiphr` |
| B15 Mouse | agreed 22 Nov: mm10 as it is (ENCODE has mm10 only). mm10 prepared 22 Nov (270 contigs, 2.65 Gb), vM25 basic splice labels (`tracks/mm10/splice-sites-v1`, 241,250 introns, GT/GC 0.987, AG 0.986). **Zero-shot test done 22 Nov** (the trainer ran `splice_zeroshot.py` in F's gap): A1m-60000 splice AP mouse 0.234 [0.186, 0.355] vs human chr8/9 0.506 [0.463, 0.545], 96 random windows each; windows in `$SCRATCH/kron-dna-build/zeroshot/mm10-v1/`. Rerun on the same windows once mouse is in training. Mouse gene-structure labels: job 6591932 |
| B21 Mouse heads (`--xhead-species`) | agreed with the trainer 22 Nov, built at d464e7c + 9836e4d: a head's channels are the human store's, then mouse channels that match none (one shared channel per key per genome; Anton: the model reads the species from the DNA, only targets are per experiment); row channel masks in losses and scorers; `--init` widens readouts; mm10 split `tracks/mm10/split-v1` (chr8+chr9 held out). Mouse stores binning behind the download, in `tracks/mm10/{dnase,rna,histone-core,histone-rest,tf}-v1-bin128` (jobs 6592906-6592990): DNase+ATAC 436 tracks (23 shared with dnase196b), RNA 620 (46 with rna462b), core histone 532 (42 with chip492b), other histone 201 (13 with chiphr), TF 196 (8 with chiptf). **All five mouse stores built and validated 23 Nov** (`check_chip_store.py` with `--gtf` vM25, a7fa3c9; numbers in `#kron-fm>dna-builder` 642745021) and handed to the trainer; A3 takes mouse after the training reviews. Held-out mouse windows for before/after: `$SCRATCH/kron-dna-build/zeroshot/mm10-heldout-v1` (96 on chr8+chr9); the trainer scores the 'before' (A1m-60000) in A3's restart gap |
| B22 HPRC release 2 | Anton 22 Nov (642709897, via the trainer 642709946): HPRC release 2 haplotype assemblies as further genomes, masked bases only (no tracks), chr8/chr9 held out in each haplotype. Building 23 Nov: `scripts/kron_dna/prep_pangenome.py`, ONE prepared genome `hprc_r2` (464 haplotypes of 232 samples; sequences assigned to chr1-22/X/Y by each assembly's chromAlias, incl. chrN_*_random scaffolds, named sample#hap#alias; `held_out` column for chr8/chr9). **Built 23 Nov**: `interim/genome/hprc_r2`, 464 haplotypes, 35,854 contigs, 1,392.1 Gb, held out chr8/chr9 131.8 Gb; each haplotype 23 chromosomes, 2.89-3.09 Gb. Parts (1.4 TB) in the work dir, deletable once the trainer has loaded it. Work dir `$SCRATCH/kron-dna-build/hprc-r2`. For the trainer: `--genome hprc_r2:<share>:0` (held_out column). Model cards' 'no individual-level human genetic data' line changes (the trainer's) |
| B24 Review fixes, 23 Nov | done: mouse split by synteny + N for held-out bases in training rows (bc0bb88); label scripts refuse unplaced annotations (267ccea); store windows by pread (521d8b3); SpeciesStores empty windows untouched zeros (eb39d06, A3's mouse slowdown); calibration on the training split only (e02642e). Latent: bigWig contigs a store does not cover are stored as zeros, xheads never uses the observed list (no current store has any) |
| B23 One data pipeline | Anton 23 Nov (642744279, via the trainer 642744464): the data preparation 'more or less universal, not fully different for every dataset'. Plan from the Claude review's duplication map (`$SCRATCH/kron-dna/review/claude.md`, 'Proposed common pipeline'): `src/kron_dna/prep/`. Phases, each a commit with tests, scripts kept as thin wrappers until their output is shown identical: (1) `layout.py` (Layout: contigs, rows, one locate, split, checksums; chromosome-name aliasing that fails below a placed share) + `io.py` (atomic write, digests); (2) one target transform (numpy twin of xheads.targets_of, tested equal; bin_track_store and prep_bigwig_binned use it); (3) `prep labels --kind splice|genes` on one feature reader and one intron rule; (4) `prep tracks --out perbase|binned` on one bigWig source/encoder; (5) `prep encode select|fetch`, `prep check`, `prep genome` (species, pangenome). Unattended, phase by phase between other work. **Phases 1-2 done 23 Nov (f5d38a1); phase 3 done (82723ca)**: both label scripts on prep.features, hg38 stores rebuilt byte-identical. **Phase 4 done (3508b9c)**: both bigWig scripts on prep.bigwig (calibration, scan) and prep.transforms; two real TF tracks byte-identical old vs new. **Phase 5 done (24c3988)**: `scripts/kron_dna/prep.py`, one entry point (genome, encode, tracks, labels, split, check). B23 done; phases 1-4 on main (625e900). Left as its own item: the observed-contig mask (stores keep zeros where a bigWig has no contig, xheads ignores the observed list; no current store has any) |
| B25 Vertebrate widening | Anton 23 Nov (642891891/642892512, via the trainer 642892615): high-quality genomes of birds, reptiles, fish (insects last), masked bases only. **Ready 23 Nov** (642895576): tgut_btaegut7, apla_pekin_t2t, cmyd_rchemyd1, amis_rallmis1, acar_ranocar3, drer_grcz13ab, olat_hdrr31, xtro_xtro10, amel_hav31 (honeybee, later); held_out column (1-2 whole chromosomes, 7-9 %; synteny too fragmented this far), RefSeq gene labels. Annotation-set species excluded. The trainer sets up a controlled pair. **Ortholog leak guard 23 Nov** (trainer 642943534): `prep_ortholog_split.py` (93f6524) masks genes NCBI pairs with (or whose symbol is) an hg38 chr8/chr9 gene, plus own held-out chromosomes; `tracks/<name>/split-orthologs-v1/split.u8`, 11-13 % held out per vertebrate (ortholog spans 3.2-4.7 %), honeybee 9.2 % (no NCBI human pairs for insects); posted 642946861 |
| B26 Counterfactuals / knockouts | Anton 23 Nov (642947295, arXiv 2611.24778 = Balestriero and LeCun, JEPA counterfactuals by residual reuse; not a DNA paper). The trainer asked for sets (642947974), then gave the scorer to the builder (642949831). **CRISPRi:** `prep_crispri_pairs.py` -> `interim/eval/crispri-eg-v1/pairs.parquet` (ENCODE-rE2G benchmark, EngreitzLab/CRISPR_comparison at 5058742: 14,734 pairs, 661 regulated; Fulco2019 3,501, Gasperini2019 5,299; held_out = element or TSS on chr8/chr9; valid + in window 9,203 pairs, 623 regulated, 4,094 element windows). **Scorer:** `knockout_bench.py` score/merge/report: element masked (N) and shuffled (CRC32-seeded), RNA at TSS +-1 kb on the cell type's strand channel of rna462b (K562 = ENCSR000CPY, channels 62/63; HCT116, WTC11, GM12878 too; Jurkat none), AUPRC vs distance; `--e5 N` = the Fable plan's E5 arm (hidden 4 kb block over the TSS, z_obs + s(mu_ko - mu_ref) decoded with the reference skips; e5_latent vs a mirrored placebo). Checked on CPU with a toy model only; the trainer runs it on D first (--max-pairs 1000 --e5 200). **eQTLs:** `prep_eqtl_finemap.py` (eQTL Catalogue r7 GTEx v8 SuSiE, 49 tissues; PIPs recomputed from lbf_variable, equal to the catalogue's credible sets within 0.0006): PIPs in `$SCRATCH/kron-dna-build/eqtl-finemap/pip/`; assemble (positives PIP >= 0.9, a same-gene distance-matched negative never reaching 0.01) is job 6603686 -> `interim/eval/gtex-finemap-v1/pairs.parquet`. Open: read the knockout results; post the eQTL table |
| B27 Scorers on any card | the trainer, 23 Nov (642959882, 642965342): evaluation moves to the 16 idle RTX 2080 Ti cards (bf16 is emulated there, 19x slower). `kron_dna.evalkit`: `--dtype bf16/fp16/fp32`, `--shard i/n` + `merge` (shards under OUT/shards/), `--chunk N` = a memory-capped forward (encoder in chunks with split_halo, core over all bins, decoder on the head span only; equal to the whole forward, tested) in genes_eval, knockout_bench, xhead_bench (97f32b9, 2b4c3ac). Not yet measured on a real card |
| B28 Exon decoders | CDS exon F1 was 0.000 on all 16 annotation runs: not a scorer fault, the genes head's CDS ends are not base-exact (32-bp KronLSTM steps; beluga A3: 13,135 called runs vs 1,248 exons). genes_eval (4a8decd) now saves the splice head's logits and reports exon F1 smoothed, within 10 bp, **snapped** (ends moved to the splice head's sites) and **paired** (exon between two sites with mean CDS probability >= 0.5); with the annotation's own sites as an upper bound paired gives 0.76 on beluga A3, snapped 0.28. `rescore` recomputes from saved logits on CPU (job 6604393 for the 16 runs; originals kept as .orig). Open: real numbers once the trainer reruns with splice logits; a gene-level F1 on top of paired exons; first/last coding exons (codon-bounded) |
| B29 Counterfactual suite | Anton 23 Nov (642997627, 642997901: work with most of them; they validate the new JEPA). The trainer's queue (642997806): (1) allele-specific binding, (2) MPRA sat-mut + lentiMPRA, (3) splicing (Vex-seq, MFASS, APARENT), (4) HAR MPRAs, (5) STARR-seq compatibility. **Done (ca78b31, 25da225, 00b2d6b):** `variant_bench.py` (ref vs alt through any track head; all channels saved per site; `--read-col`, shards, `--dtype`, `--chunk`; reports `report-asb`, `report-mpra`, `report-eqtl`); `prep_asb.py` -> `interim/eval/adastra-asb-v1` (ADASTRA BillCipher cell-type x TF tables in `input/eval/adastra`; all rows are significant, so the task is direction; 91,440 rows on exact chiptf channels); `prep_mpra_satmut.py` -> `interim/eval/mpra-satmut-v1` (Kircher 2019, OSF 75b2m, in `input/eval/kircher2019`; 29 experiments, 21 elements; MYC x2 and FOXE1 held out); `interim/eval/gtex-finemap-v1/pairs.parquet` assembled (11,957 positives, 11,922 matched negatives, 1,011 held-out positives). **Not run on a real checkpoint by me:** the real KronLSTM layer needs a GPU (triton) and my smoke job never started under the shared CPU cap; the trainer runs it. Rerunnable CPU jobs: see the memory note on the uncapped group (they started in seconds there). **Chained exon decoder** (25da225): `chain_exons`, DP over splice sites and call ends; beluga F 0.418 snapped -> 0.564, chicken A3 0.227 -> 0.430; the paired decoder failed on real sites (14x more sites than coding exons). **Added later the same evening (b785a85, 10b805a, 4e6dcd9):** `prep_lentimpra.py` -> `interim/eval/lentimpra-v1` (Zenodo 10558183 Tables S3+S6 in `input/eval/agarwal2025`; 182,859 measured elements, coordinates reproduce the oligos; not a counterfactual; `score --ref-only` + `report-lentimpra` = own-cell channels and an out-of-fold ridge probe on the paper's folds); `prep_mfass.py` -> `interim/eval/mfass-v1` (KosuriLab/MFASS `snv_data_clean.txt` in `input/eval/mfass`; 27,733 variants, 1,050 SDVs; `intron1_len` is the low-coordinate flank on both strands, 99.5-99.9 % of exon ends are annotated sites); variant_bench reads label heads (highest logit per channel, `--flank-col`) with `report-mfass`, and `report-eqtl`. The trainer's first real run (643008887): ADASTRA on D, AUROC 0.678 vs motif 0.604. All 16 annotation runs rescored in place with `cds_exon_f1_chained` (vertebrates 0.25-0.56, about +0.1 to +0.2 over snapped; insects and worm worse than snapped: the splice head rarely passes the site threshold there). **Vex-seq (17559d3):** `prep_vexseq.py` (run with `uv run --with pyliftover`) -> `interim/eval/vexseq-v1` (scottiadamson/Vex-seq `processed_files` and the UCSC hg19ToHg38 chain in `input/eval/vexseq`; 2,055 variants, 110 exons, MFASS's columns so `report-mfass` reads it; the file's splice-site distances have no zero and no strand: strand by annotated sites and AG/GT, all 110 exons AG..GT, 95 annotated; told the trainer, 643062697). The trainer's baselines on D/A3/F (643017963-643046286): ASB 0.678/0.664/0.656; Kircher MPRA on D mean Pearson 0.281 (DNase head); eQTL through the TSS 0.525, DNase at the variant 0.575. **Human vs chimp and ExP STARR (a5c19d4 and the commit after):** `prep_hsub_mpra.py` -> `interim/eval/hsub-mpra-v1` (Uebbing 2021 MPRA 1, GEO GSE140983 barcode counts in `input/eval/uebbing2021`; 45,507 fragment pairs, ref = human span, alt = chimp bases, effect and Welch test are this script's, 1,153 differential at 5 % FDR, 108 held out; the file writes substitutions chimp->human, 0-based position = Start + eVar_Pos - 1; `report-hsub`, neural channels by `--cells`); its second MPRA (single substitutions deconvoluted, `GSE140983_MPRAcounts.tsv.gz`, downloaded) is not parsed. Whalen 2023 left out: tables not open access, GEO GSE110760 is raw barcode counts (700 MB). `prep_exp_starr.py` -> `interim/eval/exp-starr-v1` (Bergman 2022, GSE184426 1Kx1K counts in `input/eval/bergman2022`; additive fit explains 0.865 of pair variance; 875 enhancer and 806 promoter strengths; `--ref-only` + `report-lentimpra`, which now reports per `group`). All three need `uv run --with pyliftover` and the chain in `input/eval/vexseq`. APARENT not built: no polyA head to read it. **24 Nov, night:** `report-lentimpra --also` (all heads' channels in one probe, af4227c; the paper's baselines from Table S6: Enformer 0.831/0.850/0.833, MPRALegNet 0.833/0.866/0.795 for HepG2/K562/WTC11, sent 643088943). **Framed exon decoder** (660c27b): `chain_exons_framed`, no stop inside an exon and the frame kept across introns, `brk` 200 set on beluga F; exon F1 on F: beluga 0.562 -> 0.656, chicken 0.491 -> 0.591, kestrel 0.440 -> 0.547, horse 0.372 -> 0.477, whale shark 0.280 -> 0.376; A3 likewise (+0.10 to +0.17); fly and worm unchanged (near zero); the 16 runs are NOT yet rescored in place (`rescore.sh` does it, it now needs the genome's bases). **Counterfactual training pairs** (Anton 643090504: yes to pairs in JEPA training; trainer's request 643090557): `prep_cf_pairs.py` -> `interim/train/cf-pairs-v1/{asb,eqtl,crispri,hsub2}.parquet` + `counts.json`. The trainer's 1 Mb rule was impossible (96 % of the genome is within 1 Mb of a scored row); agreed instead (643092336): chr8/chr9 never written, chr3/chr10/chr16 = `pair_test`, the rest `train`; benches of a pair-trained arm are quoted on chr8/chr9 and on the pair-test chromosomes only, so every report needs a pair-test subset line (NOT yet added). eQTL direction from the eQTL Catalogue credible sets (`input/eval/eqtl-catalogue-credible`, 49 GTEx tissues). hsub2: allele strings assumed to run along the fragment (`order_assumed`). Done after that: the pair-test line in every report (6187bc0, 7edd62a); the 16 runs rescored with `cds_exon_f1_framed` (table posted, 643096935). **Defect found and fixed (55c96b2):** `score` dropped alleles over 64 bases silently, so the trainer's first human-vs-chimp run covered 778 of 1,143 sites; same-length blocks now pass to 1,024, indels keep 64, and the count left out is printed; rerun asked (643114112), its old numbers (0.700/0.677/0.676) are partial. The ROGEN November file (`kron-fm.md`, the trainer's) covers this seat's work. **Site threshold for non-vertebrates, tried 24 Nov, not adopted:** the splice head scores annotated exon ends above logit 3 in 77-80 % of cases in beluga and chicken, 43-54 % in whale shark, 6-8 % in fly, 1 % in worm (A3). A lower fixed threshold helps those (fly A3 0.031 -> 0.190 at -3, whale shark F 0.376 -> 0.444 at 0) and wrecks mammals and birds (beluga F 0.656 -> 0.466 at 0). A label-free calibration (the strongest site near smoothed-call ends as the genome's level) failed: that level reads 1.6-4.1 where the true median is about 5, so it lowers the bar for vertebrates too. Conclusion: the limit is the splice head on those clades, a training matter, not a decoder one. **First/last coding exons (e515a45):** `chain_exons_framed(..., utr=)`, decoder `terminal` -> `cds_exon_f1_terminal`: a start or stop codon within 100 bases of a smoothed call's end is a candidate end when the UTR channel outside it averages above 1 over 96 bases (set on beluga F); without the UTR gate the codons added as many false exons as true ones. All 16 runs rescored (job 6626790, 22 min): F beluga 0.682, chicken 0.617, kestrel 0.561, horse 0.499, whale shark 0.396; A3 0.635/0.573/0.473/0.485/0.421 (posted 24 Nov). Still missed: exons with one weak or non-canonical site (227 of 1,245 in beluga). **G design questions (Anton, 24 Nov midday, 643224593 on):** DeepSeek V4 has no MLA any more (CSA/HCA: merged keys/values, top-k, a 128-token window; arXiv 2606.19348) and it cures attention's cache, which our KronLSTM core does not have. Measured F's training memory by stage (CPU fp32, stand-in core, growth per base 32 kb -> 64 kb; scripts in the session scratchpad, not kept): encoder 27 %, core 4 %, decoder 36 %, masked-base + 155 NTv3 per-base head 30 %, the 8 xheads 3 %; `stem_channels` 128 instead of 256 -> 51 %, + `head_hidden` 128 -> 44 %, `v2l` -> 77 %. The trainer's precedents: `v3` (1-bp KronLSTM + H-Net router) lost on 15 Nov (1.52 vs 1.30 bits), the full core at 1 bp is ~75 GiB per Mb. Candidate configs CPU-checked in `scratch/kron-dna-build/consult-g/g_candidate_configs.json`; a local track with xheads is refused by a guard in `_init_xheads` (builds and trains a step when relaxed) and by `forward_split` (untested). The trainer benches on gpu13 (D stopped for it). **Codex consultation blocked:** no delegation relay in this machine's kit config (resolver exit 3); reported to rigger/tooling (643227851); brief and prepared copy in `scratch/kron-dna-build/consult-g/`; rerun with spec `codex:gpt-6.2-astra:high` when the relay exists. **Afternoon 24 Nov.** xheads allowed beside a local track (69a71e5, the trainer merges; `forward_split` with a track is the trainer's). **Licence / NTv3 rebuild (Anton 643248450):** the NTv3 benchmark dataset is CC BY-NC; its 121 non-human tracks are 148 public runs (705 GB fastq; per species: cattle 331, rat 200, tomato 105, P. falciparum 37, yeast 16, pombe 5, rice 4, dicty 4, maize 2, arabidopsis 1 GB; six cattle ERX ids unresolved); rice, maize and arabidopsis are ribo-seq only. Plan told to Anton (643249445): pilot on yeast against NTv3's own bigwigs, then the rest, on the newest annotated assemblies. **Blocked on Anton:** aligners (bowtie2, hisat2, samtools, deeptools) are not installed and `docs/TRUST.md` has no row for them; asked for a row (643249806), and reported my unpinned `uv run --with pyliftover|openpyxl` of 23 Nov. **New genomes (Anton 643248871), jobs 6631489-96** via `scratch/kron-dna-build/species-g/run.sh` (wraps `genes-eval/refseq_species.sh`): smic_fli1 (Greenland shark GCF_052327205.1), nfur_grzrimd1, gacu_fgasacu3, trub_ftakrub13, dsim_prin31, dpse_mv25, dyak_tai18e2, agam_nwf1; redone with `HELD=auto` (jobs 6631840-47; the first pass had no own held-out chromosomes) and ortholog splits (6631848): held out 10.4-17.3 %; told the trainer (24 Nov evening), with the caveat that D. simulans/yakuba training weakens the fly annotation benchmark. C. briggsae and P. pacificus still to do (GenBank-only, GFF not GTF). **Greenland shark switched (Anton 643277452, 643279167: higher sequence quality, annotations we make ourselves):** training sequence is the PNAS 2026 assembly `smic_asm5609953v2` (GCA_056099535.2, Univ. of Tokyo; sequence only; job 6633193 via `species-g/run-gca.sh`; 5.88 Gb, 6,683 pieces, 68 % in pieces >= 1 Mb against 52 %); `smic_fli1` stays prepared with RefSeq labels as the annotation answer key, out of training. The other new species keep their annotated references. A kron-fm/writer seat is being asked for (secretary 643280223): it will want this seat's numbers. **Universal pipeline, genomes (Anton 643282123):** `scripts/kron_dna/sources.toml` (22 NCBI genomes) + `prep_source.py` (`prep.py source genome|verify|list`); all 22 verified by rebuild (ff79d9e) after correcting min_seq for 8 hand-prepared ones (1,000,000 for acar, amel, amis, cmyd, dleu, fnau, olat; 10,000 for rtyp) and writing held-out chromosomes by name (the auto rule had changed since amis). `slurm/submit.sh` takes `KRON_COMMENT`/`KRON_NICE` (544e85b) and is how jobs are submitted from now on. Still outside the manifest: hg38, mm10, apes, HPRC, ggal, ecab, dmel, ce, the NTv3 species; tracks and benchmark tables. **Private paper repository (Anton 643291454):** created for the writer seat, with the pipeline record and our cluster's settings moved into it; where it is: this seat's memory note `kron-genome-paper-private-repo` (not written here: this file is public). A kron-fm/writer seat is coming (seat-builder). **25 Nov, the architecture is this seat's (priming 57ae0f9; the trainer benches, runs, merges).** Built and handed over today: `grid_of` (head bins independent of the core's step, up to 1,024) and `head_at`/`ntv3_at` (masked-base and NTv3 heads only where a loss exists; 15.2 -> 4.5 KB per base on CPU) in 10d44e6, wired by the trainer as `--lean-heads` and the grid on its branch `grid`; KronHead pools before its 1x1 map (b9f6744); the richer JEPA target `jepa_track_dims`, `jepa_track_features`, `jepa_track_loss`, `pool_bins` (7d01ec5; the trainer wires the target's clean decode pass and the weight). CPU memory by stage with lean convs (stand-in core): 64 / 72 / 86 KB per base at 128 / 64 / 32 bp. Not built, on purpose: a sampling scheme for the three per-base xheads (the trainer's 25 % profile ran without `--xhead-recompute`; in training they keep almost nothing). Ideas still open: heads reading the decoder level of their own resolution; PRO-cap at 32 bp. **Evo 2 teacher (Anton 643570427):** Apache-2.0 code and weights; pilot `scripts/kron_dna/evo2/evo2_teacher_pilot.py` on a granted A100 (job 6653466): 4.8 kb/s genomic, 13.5-16.8 GiB, true-base log-lik -0.571 spliced vs -0.582 genomic, argmax agree 82 %; proposal: spliced CDS of the training chromosomes (~35 Mb, ~2 h of one A100) as 4 log-probs per base, a second loss on the masked-base head; waits on the trainer's number for our own masked loss on CDS. **New track stores (trainer 643527223):** `encode_select --assays`; PRO-cap (448 channels), CAGE+RAMPAGE (`tss-stranded-v1`, 442), eCLIP (1,026) building as `interim/tracks/hg38/<name>-bin128` (jobs 6652206, 6652614, 6652615); hand each to the trainer after a check. **Exon metric:** `..._isoforms` scoring (d2b182b), 16 runs rescored; pooled vs isoform differs by <= 0.011. **Owed to the writer (25 Nov, 643413915, my 643449568 and 643449609):** (1) `results/published-baselines.md` in the private paper repository, built from the readers' notes there (`notes/evidence/variant-benchmark-baselines.md`, `alphagenome-borzoi-baselines.md`) and checked, not searched again; (2) CRISPRi: our table equals the source benchmark's pair sets (10,356 K562 pairs / 471 positives; 4,378 held-out / 190), but the trainer's knockout run used `--max-pairs` (2,997 pairs, 621 regulated: all positives, a fifth of the negatives), so its AUPRC is not comparable to any published one; to do: distance, ABC and rE2G on the same 2,997 pairs from the source's per-pair predictor scores (not downloaded yet: only the benchmark tables are in `input/regulation/crispri-eg`), and ask the trainer for a run on all 13,919 scoreable pairs; (3) exon F1 is not gffcompare's (isoforms pooled into union runs, windows not genomes, no 3 bp tolerance): add a metric against the distinct CDS exons of each isoform, exact ends, optional 3 bp at terminal exons, and rescore. **State at the usage limit, 25 Nov early:** `dna-builder` is merged to main (1ff5e1d). Downloads of the 22 manifest genomes are recorded in `SOURCE.json` (e0459ee). Owed to the writer (my reply 643408360): `results/published-baselines.md` in the private paper repository, the published numbers on MPRA saturation mutagenesis, MFASS, Vex-seq, ADASTRA, the CRISPRi benchmark and GTEx eQTLs, each with source, subset and metric, marked 'published, not same rows'; a research agent for it was started and stopped unfinished at the usage limit: start it again. Waiting on Anton: the trust-register row for the aligners (643249806, 643252500), the kit move for Codex (643272225), the Hugging Face publishing decision (643251766). Next for the manifest: ggal, ecab, dmel, ce (two annotation releases each, URLs and label options unrecorded), then hg38, mm10, apes, HPRC, the NTv3 species; then tracks and benchmark tables. **Research on 2026 read tools** (Anton's ask): saved in `scratch/kron-dna-build/reads-env/research-2026-11-24.md` (fastp 1.4.0, bowtie2 2.5.5, STAR 2.7.11b / HISAT2 2.2.3, samtools 1.25, deepTools 4.1.0, bigtools 0.5.8); Anton: no plants beyond NTv3's four. **Codex:** the relay cannot be installed until Anton pushes `kit-leaves` (tooling 643255159); Anton chose a by-hand run (643255768), the auto-mode classifier refused it; he has the command to run himself (643256532); answer expected in `consult-g/runs/codex-answer.md`. **HF publishing** of the binned stores (668 GB, raw f16, not parquet) proposed per chromosome; waits on Anton (public/private, licence). Next: (Whalen 2023 if wanted); gene-level F1, STARR-seq compatibility (Bergman 2022); reading frame and first/last exons in the chain decoder; a lower site threshold for non-vertebrates. |
| B16 Gene-structure head | Anton 22 Nov, "Let's add a gene structure head"; and for F too. Design: a binary (multi-label) further head, the splice head's kind and loss, on a 14-channel label store: tss, polya, utr5, cds (stop codon in), utr3, ncexon, intron, per strand; a base takes every label any transcript gives it (isoforms overlap). `LabelStore` now reads `labels.json` + `labels.u16`. `scripts/kron_dna/prep_gene_structure_labels.py` (checks GT/GC..AG introns and ATG starts, refuses below 0.85), tested in `test_kron_dna_gene_structure_labels.py`. hg38 GENCODE v50 basic: job 6591420. Next: mouse vM25 labels (after the zero-shot test), labels for the NTv3 species with annotations, then the head in a run; F after the memory savers (B18). Benchmark metric: Helixer's base-wise F1 (Genic: CDS, UTR, intron; Subgenic: CDS, intron) plus Tiberius's exon and gene F1 |
| B19 Annotation validation set | Anton 22 Nov: species in no training genome, each with an older and a newer, better annotation of the same assembly; does the head agree more with the newer one, and where the two disagree, which does it side with. Candidates (to check that each assembly is in no training genome and that both releases exist): Drosophila melanogaster (FlyBase r6.0x vs r6.6x on Release 6), C. elegans (WormBase WS2xx vs WS29x on WBcel235), zebrafish (RefSeq GRCz11 release 106 vs RS_2024), chicken GRCg7b (Ensembl vs RefSeq with Iso-Seq), dog. Comparable outside numbers: Tiberius tests on human, cow and beluga (held out with Hominidae, Ruminantia, Cetacea; RefSeq; exon and gene F1). Helixer's test species: 11 vertebrates in its figures (Falco naumanni, Xiphias gladius, Chiroxiphia lanceolata, Balaenoptera musculus, Myotis davidii, Rhincodon typus, Desmodus rotundus, Sparus aurata, Pseudonaja textilis, Opisthocomus hoazin, Lagopus leucura) and 15 invertebrates (Atta cephalotes, Apis dorsata, Frankliniella occidentalis, Ctenocephalides felis, Drosophila albomicans, Tribolium castaneum, Rhopalosiphum maidis, Bombus impatiens, Caenorhabditis remanei, Trachymyrmex cornetzi, Hyposmocoma kahamanoa, Drosophila virilis, Schistosoma mansoni, Opisthorchis viverrini, Culex pipiens); RefSeq/Phytozome references, base-wise Genic/Subgenic F1 (bioRxiv 2023.02.06.527280v2, supplement S2.1). Pick a few of each so our numbers sit beside theirs. **Plan (23 Nov):** (1) old vs new on one assembly: D. melanogaster (FlyBase r6.0x vs current r6.6x, s3ftp.flybase.org mirror works), C. elegans (WormBase WS2xx GFF3 vs WS298 GTF; the label script reads GFF3 since ed3f4f7); score where old and new disagree, which one the head sides with. (2) beside Tiberius: beluga (D. leucas, RefSeq), not cow (cattle is an NTv3 training genome, contaminated), not human (trained). (3) beside Helixer: 2-3 of its vertebrate and invertebrate test species absent from training (e.g. Falco naumanni, Rhincodon typus, Drosophila virilis). Metric: Helixer's base-wise Genic/Subgenic F1 from the head's argmax classes, plus exon/gene F1 later. Needs: genome-prepare per species, labels, a scorer like splice_zeroshot for the genes head (a card for scoring). **Built 23 Nov:** `genes_eval.py` (f11f90d); fly `dmel_r6` r6.01 vs r6.69 (identical sequence) and worm `ce_wbcel235` WS250 vs WS298 (identical but for soft-masking; WormBase source only), 96 windows each in `$SCRATCH/kron-dna-build/genes-eval/`; the trainer scores both on F and A3 in A3's 40k gap. Preparing: beluga GCF_002288925.2 (6595471; note cattle is in our training, Tiberius held out Ruminantia), Helixer's whale shark, lesser kestrel, D. virilis at Helixer's own assemblies (`$SCRATCH/kron-dna-build/genes-eval/refseq_species.sh`). **All four prepared 23 Nov** (beluga dleu_asm228892v3, kestrel fnau_bfalnau1, D. virilis dvir_dvirrs2, whale shark rtyp_asm164234v2 with scaffolds from 10 kb and windows with up to 5 % N), 96 windows each; commands with the trainer (642869077). Exon F1 in genes_eval (f5acb88), logits kept. **Mammal and bird old-vs-new added 23 Nov** (Anton: worm/fly are far from our mostly-mammal training): horse ecab_equcab3 RefSeq 103 vs RS_2024_08 (labels differ on only 0.2 % of annotated bases, so windows `ecab_equcab3-diff-v1` centred on differing bases; read only its disagreement lines), chicken ggal_grcg7b 105 vs 106 (12.7 %); fly 4.7 %, worm 18.1 %. Prepared by `genes-eval/release_pair.sh`; with the trainer (642883318). Open: gene-level F1 from the saved logits; results to read when posted |
| B20 Mouse download | 22 Nov: `scripts/kron_dna/encode_select.py --assembly mm10`, one file per experiment (most replicates): 1,985 files, 1.16 TB, in `$SCRATCH/kron-dna/encode_bw/mouse-mm10/`; three streams of four chained 3 h jobs (6591421-6591432), `encode_fetch.py` (size + md5, resumable); 1,102 of 1,985 files (816 GB) verified by 22 Nov ~20:30, no failure |
| B17 CELLxGENE | (a) Census pseudobulk per cell type x tissue as a gene-level head: measure the aggregation's cost first. (b) coverage from raw reads: not started (weeks of CPU) |
| B18 Memory: head round-robin, recompute for per-base heads, heads on card 1 | the trainer's questions, 22 Nov, to make room on F and for the gene-structure head. Answer them from the code |

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

## 26 Nov, 01:40 (night)

- **Stores built and handed over:** `starr-wg-v1-bin128` (6 whole-genome STARR-seq tracks; two cautions: ratio noise
  around 1, and two groups of cell lines that look like two input libraries); `tss3-stranded-v1-bin32` (890 tracks,
  agrees with the 128-bp store). `encode_select.py --assays STARR-seq` (6a721f1).
- **Plasmids (Anton's idea):** Addgene needs a login and a reviewed bulk-data request; Anton sends a login by
  direct message "tomorrow or the day after". Meanwhile `prep_constructs.py` (464e7ba) turns GenBank flat files
  into a small prepared genome with `parts.u8`; GenBank release 274 division SYN is in scratch
  `constructs/genbank-syn`, job 6657460 builds `interim/genome/constructs_gbsyn274`. To do: read its
  `constructs.json`, report yields, hand to the trainer; an Addgene JSON reader when the login comes.
- **Multi-scale arm:** Astra 6.2 answered twice (writer's run and mine, scratch `consult-astra-5`). Agreed plan:
  no 1-bp recurrent branch; `inter_parallel` (637ecd3) = KI's two 8-bp blocks as a side branch, zero at start,
  `inter_branch_off` for rescoring; starts after KI and K5P reach 15k; trainer builds a 512-window bench.
- **Open, mine:** in-tower blocks (`InterKron`) restart on a fixed grid and ignore contig gaps (needs segment ids
  through encode/decode in the whole, split, capped and contig forwards); one `local_span` for every fine-scale
  block. Both only if the in-tower design is kept after 15k.

## 27 Nov, 14:40 (afternoon)

- **Paused 26 Nov 01:33 to 27 Nov 13:00**: a safety classifier stopped a reply while I was starting a script to mark
  vector records that carry held-out human coding sequence. That script is not written and I do not redo it that
  way. `constructs_vectors_gbsyn274` is therefore **not cleared for training** as a whole; the safe subset is
  everything but the ORFeome family (`organism` column). Another route to the mark is open.
- **Retired:** in-tower blocks in both forms (KI, KP below A at 15k); hierarchical JEPA (the coarse screen failed
  its 5 % bar on F; the trainer reruns it on the slow-memory arm).
- **Long range (Anton's top priority, 27 Nov):** F's memory half-lives are short (median 0.5 to 2.7 kb per layer).
  Built, ce75ef0: `slow_layers/slow_heads/slow_range` (init; `apply_slow_heads` after a load) and `coarse_ratio`
  with `coarse_kind` attention or kron (`CoarseBlock`, zero at start, `coarse_off`). The trainer's arm s (checkpoint
  surgery on F 250k, 8k steps) reads out the evening of 27 Nov; flags on the trainer's branch trainer-longrange.
  Next if wanted: a 32-channel pair state at 2 kb with a contact head, on top of the coarse block.
- **Hi-C:** six ENCODE intact Hi-C maps in scratch `hic/intact-v1/files` (425 GB), selection `hic_select.py`.
  Blocked on Anton: `hictk` trust-register row (asked 643729999, repeated 644126394), else my own reader.
- **Codon head:** arm finished (train 1-cosine 0.561 to 0.505, no cost to the trunk); readout `codon_eval.py`
  (7576c42), not yet run on a card; the trainer runs it.
- **Waiting on Anton:** Addgene login; hictk row; HF publishing of track stores.

## 27 Nov, 23:00 (night)

- **Astra's implementation review, model side, done** (Anton: "fix where you agree"): the coarse block keeps the
  core's segments (90334b7; one segment per row gives what it gave before, difference 0.0), `coarse_local` is the
  control without mixing between positions, `set_slow_heads` takes the longest-memory heads, each with the whole
  range, as the trainer's surgery did; `slow_pick` "shortest" (8c48d1a) leaves the long heads alone (arm s2).
  `diag/slow_band.py` (d340100) shortens one band of half-lives in a checkpoint for rescoring.
- **Arms s (slow memory) and ca (coarse attention): level with mask a** on enhancer-gene AUPRC (scorer v2). The
  reading everyone holds now: nothing in training asks for reach. Two routes to a loss that does: the trainer's
  promoter-hole arm p, and contact maps (mine).
- **Hi-C: own reader, no outside code** (`scripts/kron_dna/prep_hic.py`, a1309ed): the hictk question to Anton is
  closed. Stores for 6 cell lines under `interim/contacts/hg38/intact-<line>-5kb` (5 kb cells, 201 diagonals,
  observed / expected, float16). Checked against each file: counts read / file's own sum = 1.000000 everywhere.
  Depth: beyond 250 kb only GM12878 is dense (others 27-48 % empty pairs at 500 kb).
- **Contact head** (`src/kron_dna/contacts.py`, b65bef5): `ContactStores.window`, `ContactHead` on forward's core
  states, `contact_loss`, `contact_correlation`. Handed to the trainer to wire (general chat). Noise ceiling at
  5 kb is low (cell lines correlate 0.26-0.47 with each other on one window): if the loss does not move, pool to
  10 or 25 kb cells before judging.
- Not done: a card run of any of this; ONNX export of a coarse-block model after the segment change.
- Still waiting on Anton: Addgene login (by direct message); publishing track stores.

## 28 Nov, 00:30: catching up with AlphaGenome on tracks (Anton, 644260742)

Counts per kind posted (644260738; AlphaGenome's column is the trainer's, from its output list: 644260689). Built
human stores: 6,156 channels (RNA 462, CAGE+RAMPAGE 442, PRO-cap 448, DNase 196, histone 1,923, TF 1,643, eCLIP
1,026, STARR 6, contacts 6, splice sites 4); mouse 1,989. Missing kinds: human ATAC, splice-site usage and
junctions, most contact maps, mouse CAGE and contacts. Plan promised to Anton (644261218), each store to the
trainer once validated:

| Step | State |
|---|---|
| 1 Human ATAC, ENCODE (464 experiments, 206 biosamples, 659 GB) | downloading since 28 Nov 00:05, jobs 6668024-6, `$SCRATCH/kron-dna-build/tracks/atac-hg38-v1/` (`selection.tsv`, `part{0,1,2}.tsv`); then convert, bin to `tracks/hg38/atac-v1-bin128`, validate; promised Sat 28 Nov ~16:00 |
| 2 More contact maps | ENCODE has MAPQ-thresholded `.hic` for 190 human biosample groups and 16 mouse; select one per single biosample (one LCL only), `prep_hic.py`; promised Sun 29 Nov |
| 3 Rest of ENCODE human DNase and RNA-seq beyond NTv3's list | `encode_select.py --assembly GRCh38`, minus what the stores hold; Sun 29 Nov |
| 4 CAGE from FANTOM5, human and mouse | source and build (hg38/mm10 reprocessed CTSS) to check; Mon 30 Nov |
| 5 Splice junctions and usage | check recount3 junction tables (build, strand, licence) and report to Anton before building; first version Thu 3 Dec |

`slurm/submit.sh cpu NAME` takes the command as words after `uv run` (not one shell string: "Failed to spawn").
