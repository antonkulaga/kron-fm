# Multi-track KronLSTM: a design note (2026-11-11, updated 2026-11-12)

The owner asked whether KronLSTM can read genome sequence and predict chromatin, expression and other tracks in
parallel, so that the tracks teach each other and the model learns more biology. This note is the design and the
data inventory; nothing here has run. The owner said yes to the pilot on 2026-11-12.

## Why

On ClinVar the codon model (kron-dna) is well below ESM-2 on missense and below chance on synonymous variants, because
pathogenic synonymous variants act through splicing, outside the coding sequence (README, kron-dna section). The
models that score them well read the genome around the exon: Evo 2 (0.86 AUROC on synonymous) and AlphaGenome's
splicing scores (0.94). Track models (Enformer, Borzoi, AlphaGenome) learn from many measured tracks at once; a
KronLSTM version would add what our backbone does cheaply: its recurrence runs in linear time, so windows of
100 kb and more cost little more than short ones.

## Shape

- **Input:** the reference genome, base by base, in windows around genes (Enformer used 196,608 bp, Borzoi
  524,288 bp); reverse-complement and shift augmentation as they did.
- **Backbone:** KronLSTM, bidirectional, a stride-down stem (bases to 32 or 128 bp bins) before the recurrent
  layers, as track models bin their outputs.
- **Heads, trained together:**
  - binned tracks: DNase and ATAC accessibility, histone ChIP, CAGE, RNA-seq coverage (Poisson or multinomial
    loss per track, as Borzoi);
  - splice-site usage, where the track set has it;
  - masked-base prediction on the sequence (self-supervised, keeps the language-model view);
  - the codon head on coding stretches, so the existing variant scores still apply.
- **Variant effect:** reference against alternate on every head (AlphaGenome's recipe), and the masked score.

## Data inventory (public; not downloaded)

| set | what | where | licence | size |
|---|---|---|---|---|
| Enformer intervals | 38,171 human / 33,521 mouse windows with their train, valid, test fold | Hugging Face `Genentech/enformer-data` | MIT | 1.1 MB (intervals only) |
| Borzoi intervals | 104.9K human and mouse windows, extended to 524,288 bp | Hugging Face `Genentech/borzoi-data` | MIT | 1.7 MB (intervals only) |
| Basenji2 / Enformer targets | 5,313 human tracks, binned coverage | the Basenji2 bucket on Google Cloud (Enformer's training notebook) | per source (ENCODE, FANTOM5, Roadmap) | not measured; several hundred GB expected |
| Borzoi targets | 7,611 human tracks incl. RNA-seq | the calico/borzoi repository's data instructions | per source | not measured |

The intervals keep each test window out of training the same way the published models did, so a pilot can be
scored against their published per-track correlations on the same test windows.

## What the AlphaGenome papers suggest

From AlphaGenome 1 (Nature 2026, doi 10.1038/s41586-025-10014-0) and AlphaGenome 2 (DeepMind, 2026-11-11,
https://deepmind.google/blog/alphagenome-2-deciphering-how-cells-read-dna-to-better-understand-disease/):

- A U-Net shape: base-level convolutions, a coarse long-range core (KronLSTM's place), skip connections back up to
  base-level heads. Base resolution matters for splicing and accessibility; histone marks are fine at 128 bp.
- Splicing at three levels: site, site usage, and donor-acceptor junctions (AG1's junction scorer alone beat
  earlier methods on most splicing benchmarks).
- Conditioning instead of thousands of parallel heads (AG2): one condition (cell type, tissue, state) and a species
  identifier enter every stage through adaptive normalization; conditioning only the output layers gained little.
  For kron-dna a species identifier fits the SynCodonLM corpus of 34k species.
- Self-distillation: several teacher models of the same design, then one student trained on their averaged
  predictions over randomly mutated inputs (AG1 and AG2 both report gains in variant-effect prediction).
- Long training windows help (AG1: 1 Mb beat 32 kb even when both were evaluated at 1 Mb).

**AlphaGenome as a teacher** (the owner's call, 2026-11-12): AG2 predictions through the API on the Enformer windows
can serve as training targets, for chosen conditions, where measured tracks cost a paid download.

## The pilot, if approved

1. A few hundred tracks (DNase, CAGE, a handful of RNA-seq) of the Enformer human set, 131 kb windows.
2. A small KronLSTM (about the current 68.6M) on one A100 card for about 12 hours.
3. Score: Pearson correlation per track on Enformer's test windows, against Enformer's published numbers for the
   same tracks; then ClinVar synonymous AUROC from ref-vs-alt on the splice and RNA heads.
4. Go on only if the pilot is within reach of Enformer on its tracks.

## Open questions for the owner

- The pilot's GPU time (about 12 hours of one card) and the data download (several hundred GB).
- Whether to start from the codon model's weights or train the base-level model from scratch.
