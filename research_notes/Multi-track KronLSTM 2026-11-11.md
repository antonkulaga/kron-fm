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

**Targets** (the owner, 2026-11-12: "why predict with alphagenome as teacher if we can download regulatory info from
ensembl?"): measured and annotated data first. The Ensembl Regulatory Build (promoters, enhancers, CTCF sites, open
chromatin, and their activity per epigenome), GENCODE exons and splice sites, and a set of free ENCODE coverage tracks
(DNase, CAGE, RNA-seq). AlphaGenome 2 predictions only where no measured data exists, or as an extra teacher if the
measured-only model plateaus. Genomic formats read with polars-bio.

## Context length (owner, 2026-11-12: "can we make it way longer ... can it get full chromosome?")

KronLSTM's cost is linear in length. The pilot takes 1 Mb (AlphaGenome's choice), binned to 128 bp before the
long-range core (8,192 steps), single-base heads after it. A whole chromosome (chr1: 248 Mb, ~1.9M steps at 128 bp)
streams at inference in constant memory; training uses 1 Mb chunks with the recurrent state carried along the
chromosome (truncated backpropagation). Test: the same model at 1 Mb versus with chromosome-long carried state, on the
same held-out windows. Prior expectation small (99% of validated enhancer-gene pairs within 1 Mb), but unmeasured.
The unit for carried state is the gap-free contig, not the chromosome (owner: "or at least full contig?"): the state
resets at each run of N. Contigs up to a few Mb train in one piece (8 Mb = 65k steps at 128 bp; memory to be measured
at 1, 4 and 8 Mb before the pilot), longer ones in carried-state chunks.

## Hybrid with protein embeddings (owner, 2026-11-12: "a hybrid model giving latest esm embedding as additional input
for coding regions, can it reuse the shared backbone?")

Yes. Each base in a coding exon gets the frozen per-residue ESM embedding of its amino acid (shared by the codon's
three bases, zero elsewhere), projected and added at the stem, so the encoder, KronLSTM core and heads stay shared.
Embeddings for all human proteins are precomputed once (~25 GB bf16). The same addition works per codon in kron-dna.
Evidence it pays: the gene-split ensemble of ESM-2 and kron-dna beats ESM-2 alone (0.925 vs 0.906). Stage 2b.

## Compression before the core (owner, 2026-11-12: "is having convolutions with 128 bins the most modern approach?")

Fixed convolutional downsampling to 128 bp is what AlphaGenome 1 and 2 and Borzoi use, so it is the baseline. Two
newer options are measured against it: no compression (single-base core; KronLSTM is linear, so it is affordable,
only slower), and learned dynamic chunking in the style of H-Net (2025), where the model learns its own boundaries.
Order: fixed bins first, then the H-Net-style chunker on the same held-out contigs.

## The pilot, if approved

1. A few hundred tracks (DNase, CAGE, a handful of RNA-seq) of the Enformer human set, 131 kb windows.
2. A small KronLSTM (about the current 68.6M) on one A100 card for about 12 hours.
3. Score: Pearson correlation per track on Enformer's test windows, against Enformer's published numbers for the
   same tracks; then ClinVar synonymous AUROC from ref-vs-alt on the splice and RNA heads.
4. Go on only if the pilot is within reach of Enformer on its tracks.

## Open questions for the owner

- The pilot's GPU time (about 12 hours of one card) and the data download (several hundred GB).
- Whether to start from the codon model's weights or train the base-level model from scratch.
