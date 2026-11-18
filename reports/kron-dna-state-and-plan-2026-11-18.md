# kron-dna: where it stands and what could close the gap to AlphaGenome (18 Nov 2026)

Written by the DNA trainer seat for Anton. The numbers come from the run logs and the seat's status record;
every result names its run, so it can be checked. Cluster details are left out on purpose.

## 1. In one paragraph

kron-dna started as a codon masked language model (KronLSTM over coding sequence) and is now mainly **KronGenome**, a
47-48M-parameter base-level genomic model. It predicts masked bases, 155 NTv3 functional tracks and several context
ids, and trains on 524 kb or 1 Mb windows over hg38, the T2T assemblies and ten other species. Four runs are training. Held-out
masked-base loss has flattened at about 1.37-1.47 bits/base. Track Pearson on never-seen chr8/9 is about 0.5 for the
main runs (0.57-0.615 for runs focused on fewer tracks). The codon model reaches ClinVar missense AUROC 0.807. The gap
to AlphaGenome is mostly **labelled data and output heads** (it trains on thousands of tracks, with splicing heads),
and **model size**, not sequence data or context length.

## 2. Where things are

| what | where |
|---|---|
| Code | `src/kron_dna/` (CLI `kron-dna`, model `genome.py`, data `data.py`/`stream.py`, variants `variants.py`), shared layers `src/kron_core/` |
| Repository | `antonkulaga/kron-fm` (public), branch `main`; this seat works in its own clone |
| Running jobs | each runs from a frozen worktree of the commit it started on, so main can move under it |
| Run directories, launchers, eval rounds | the seat's scratch area (`$SCRATCH/kron-dna/v21/`), outside git |
| Shared data | the kron-fm data folder: `interim/genome/<assembly>/` (bases, contigs, NTv3 tracks), `interim/cds/` (codon data, ClinVar caches) |
| T2T track pipeline | `$SCRATCH/kron-dna/t2t_tracks/` (`HANDOVER.md`, `STATUS.md`, `qc/`) |
| Reports | `reports/` (this file, `Small DNA foundation models 2026.md`, `audit-2026-11-18-response.md`) |
| Status record | the seat's status file, outside git (it names cluster details); the source of the numbers here |

## 3. What was built, in order

1. **Codon model** (SynCodonLM data, 17 GB of CDS across species): pretrained to 200k+ steps. ClinVar zero-shot
   missense rose from 0.705 (50k steps) to **0.807 (0.799-0.816) at 200k**. Synonymous stayed at about 0.44. Whole-gene
   context gave no gain over 1,000 codons (0.808 vs 0.807).
2. **Baselines on the same ClinVar sample:** AlphaMissense 0.966, ESM-2 650M 0.906, Evo 2 7B 0.868 (missense, on a
   subsample). AlphaGenome: splice junctions 0.939 on synonymous variants, missense 0.53-0.59 (it is not a missense
   model). A gene-split ensemble (kron + ESM-2 + AlphaGenome scores) reached missense 0.925 and synonymous 0.938, and
   kron added +0.010 to it.
3. **KronGenome** (base-level genomic model). Stages: masked bases → regulatory and gene tracks → NTv3
   quantitative tracks. Then backbone v2/v2.1 (U-Net-like: base-resolution stem, pooled core, decoder), context ids
   (species, assembly, organelle), multi-genome pretraining (CHM13, naked mole-rat T2T, ten NTv3 species), JEPA
   variants, no-pad packing, split windows over two cards (1 Mb), and the v2.4 mito slice.
   On ClinVar by region (held-out chromosomes), the stage-2 splice delta near splice sites reached 0.923 against a
   distance baseline of 0.839. That is on 174 pathogenic variants, so small, but it is the model's best variant
   signal so far. The masked-base marginal alone is at chance on non-coding variants.
4. **Code audit, 18 Nov:** 17 findings fixed. None changed a reported number (`reports/audit-2026-11-18-response.md`).
5. **Longer cooldown** (branch `lr-cooldown`): `--lr-total` / `--cooldown-from` extend a run's learning-rate decay
   without a rewarm. Tested, not used yet.
6. **T2T CHM13 functional tracks:** a pipeline reproducing NTv3's 36 human tracks on CHM13, so the regions hg38
   lacks get labels. Running: alignment 37 of 138 units done, no failures. It must first reproduce the hg38 tracks
   (Pearson ≥ 0.90) before the CHM13 export.

## 4. Training now

| run | what | step / plan | held-out bits/base | mito | track Pearson (round 3) |
|---|---|---|---|---|---|
| A-best | v2, 8 layers, multi-genome, NTv3 tracks, 524 kb | 216k / 400k | 1.428 | 0.966 | 0.500 (134k) |
| D-jepa | A's recipe + block JEPA, two cards | 147k / 400k | 1.436 | 0.944 | 0.520 (91k) |
| E-control | 524 kb control, no NTv3 head | 372k / 400k | 1.367 | 0.779 | - |
| F (v2.4) | v2, 10 layers, D's recipe, 1 Mb windows, mito slice | 58k / 380k | 1.466 | 1.035 | 0.431 (22k, v2.3) |

Held-out and mito are means of each run's last 6 evaluations, 18 Nov at about 11:00. Single evaluations swing by
±0.07. Track Pearson is log1p at 128 bp over 96 held-out windows, from eval round 3 (17 Nov).

What the runs say:

- **Held-out bits have flattened** in every run, within ±0.01 over the last 30k steps. Mito still improves slowly.
- **Steps are not comparable across runs.** F's step is a 1 Mb window, twice A's, and F started from an earlier 1 Mb
  run's checkpoint at step 31k. At equal bases seen (about 59G), F sits on A's curve.
- **E leads on bits** but has run 1.7 times as many steps as A and has no track head. On the fixed masked-base
  benchmark, the 120k control (1.205) still leads every arm.
- **Longer context helps tracks a little:** +0.01 Pearson from 131 kb to 524 kb, almost nothing from 524 kb to 1 Mb.
  It does not change masked-base bits.
- **The track head costs masked-base bits:** 9 tracks at weight 1.0 cost about 1.7 %. A track-focused stage lifted
  34-track Pearson from 0.480 to 0.574.

## 5. Why we are behind AlphaGenome 1

| | AlphaGenome 1 | kron-dna now |
|---|---|---|
| Labelled tracks | thousands of human and mouse tracks (ENCODE, GTEx, FANTOM, splicing, contacts) | 155 (36 human bigwigs) |
| Splicing outputs | splice sites, splice-site usage, junctions | none (only a splice-distance signal through the gene tracks) |
| Output resolution | base level for most heads | base-level stem, track head at the decoder's resolution |
| Model size | much larger (not measured here) | 47-48M |
| Context | 1 Mb | 524 kb-1 Mb (works, small gain) |

The masked-base objective is not the bottleneck for AlphaGenome's tasks. Track Pearson and variant effects are learned
from labels, and labels are where the gap is widest. Our own runs show it: the track stage moves track Pearson by
about +0.09, while context and JEPA variants move it by about 0.01-0.03.

## 6. What would improve our chances, ranked

Each item says what it costs and what it needs from Anton.

1. **Many more track targets, by distillation.** Two sources:
   - AlphaGenome predictions (approved for the pilot; the API is rate-limited).
   - `NTv3_tracks`: 711M parameters, post-trained on 16,262 tracks across 15 species, with 7,362 output channels at
     base level. Covering hg38 takes about 8,000 windows of 1 Mb on one A100 (32 GB per pass). It is gated, so it
     needs **Anton's access request**. Its licence is non-commercial, it covers the outputs, and a model distilled
     from it must be non-commercial and released under the same licence. That is **Anton's call**.
2. **Real measured tracks in bulk.** Download the hg38 bigwigs behind NTv3's track list (mostly ENCODE experiments)
   and GTEx RNA-seq, and process them with the T2T pipeline's recipes. This means real labels and no licence on
   outputs, but per-experiment download and processing. It needs the track list (`NTv3_tracks`'
   `track_metadata.json`, gated, or the paper's supplement). Disk: hundreds of GB.
3. **Splicing heads.** Splice site, splice-site usage and junction heads, trained from GENCODE annotation and RNA-seq
   junctions, as new heads on the existing backbone. This is where AlphaGenome's ClinVar lead comes from, and our
   stage-2 splice delta (0.923 near splice sites) suggests the backbone already carries part of the signal.
4. **Reuse F's or A's weights.** `genome-train --init` loads every matching weight. Brand-new heads start from random
   weights; a changed track count in an existing head is refused, so each new output goes in as its own head. New
   heads would need a short warmup, or a lower rate for the reused layers at first, because the optimizer state
   starts fresh.
5. **A bigger model** once the labels are there. At 47-48M parameters the masked-base plateau is probably capacity.
   A bigger model cannot reuse current weights directly; the current model would serve as a teacher. Cost: days per
   run on two cards; the decision depends on items 1-3 showing gains.
6. **A longer cooldown** (built, branch `lr-cooldown`). E's finish (about 7 h from this writing, its learning rate
   now near zero) will show what the final decay alone buys. Then decide whether to stretch A and D to about 600k
   steps, about 2 days of GPU time each.
7. **One shared benchmark with AlphaGenome:** its predictions on our 96 held-out chr8/9 windows, scored by the same
   track evaluation, and on the same ClinVar sample by region. Without it every comparison is indirect.
8. **More sequence** (pangenome haplotypes, more mammals): modest gain on masked-base bits, best saved for the bigger
   model.

Suggested order: 7 (cheap, makes everything measurable) → 1 and 2 in parallel → 3 → 5. Item 6 is decided after E's
finish.

## 7. Open decisions for Anton

- Request access to the gated NTv3 models (needed for items 1 and 2), and decide whether NTv3's licence terms are
  acceptable for distillation (item 1).
- After E's finish: stretch A and D with the longer cooldown, or not.
- Disk and GPU budget for bulk real tracks (item 2) and for a bigger model (item 5).
