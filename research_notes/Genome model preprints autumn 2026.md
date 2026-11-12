# Preprints for the KronLSTM genome models, September to mid-November 2026

Requested by the owner on 2026-11-12 ("check 2026 preprints, esp autumn ones ... also ideas outside genomics").
Each item was confirmed against its server's own record (bioRxiv/medRxiv/arXiv API or page); only abstracts were
read, and numbers are quoted from them. Coverage: bioRxiv bioinformatics and genomics swept fully for 2026-09-01 to
11-12, genetics partly; medRxiv genetic medicine thinly.

## What changes our plan

1. **Cerberus** (Kelley, 2026-10-09, https://doi.org/10.64898/2026.10.02.756377): Borzoi with its transformer blocks
   replaced by Hydra (bidirectional Mamba-2); "5.7-fold faster on 8,192-token inputs", "0.692 versus 0.668 mean eQTL
   AUPRC across 48 tissues"; at 32-bp resolution the U-Net decoder can go. Our closest published analogue: test 32-bp
   bins without the decoder against 128-bp with it, and report on its released GTEx eQTL/sQTL/paQTL sets.
2. **Length via carried state** (Buitrago and Gu, ICML 2025, https://arxiv.org/abs/2507.02782; StateFlow, Zhao,
   https://arxiv.org/abs/2608.06838): truncated backpropagation with state passed between chunks; StateFlow splits one
   sequence across devices ("2.22x throughput", "2.45x memory reduction"). Add a test that chunked inference with
   carried state equals one full pass: Evo 2's chunked prefill was found "silently incorrect" (Patsakis,
   https://doi.org/10.64898/2026.08.28.747902). Test beyond training length as in Double-Take
   (https://arxiv.org/abs/2610.13073: trained at 32 kb, "0.927 AUROC at 131 kb and 0.856 near 1 Mb").
3. **Variant evaluation and splice targets.** CodonBench (Liang, https://doi.org/10.64898/2026.08.12.744371): with genes
   held out, codon-model representations are near chance on synonymous variants ("the best at 0.58"), while distance
   to the nearest exon junction alone reaches "0.84": use gene-held-out splits, that baseline, and alt-minus-ref
   embedding scores. SpliceAI2 (Jaganathan, https://doi.org/10.64898/2026.10.08.757673): quantitative splice sites,
   junctions and transcript usage across mammals; our stage-2 splice heads predict usage, not just GENCODE labels, and
   SpliceAI2 is the splice baseline. Reading-frame checks from Gupta (https://doi.org/10.64898/2026.11.03.763976).

## Also worth taking

- Codon models add to protein models on missense under gene-held-out CV (Chen, https://doi.org/10.1101/2025.03.12.642937),
  as our ensemble found; GLM-Missense (Su, https://doi.org/10.64898/2026.05.06.723362) supports DNA+ESM inputs.
- Bolero (Liu, https://doi.org/10.64898/2026.10.19.760026): parameters conditioned on cell-state embeddings, six
  mammals; condition stage-2 heads on a track or cell embedding.
- MirrorDNA (Datta, https://doi.org/10.64898/2026.11.04.764207): models "change 25-31% of their predictions" under
  reverse complement; learned chunk boundaries must be strand-equivariant; add an RC-consistency metric.
- hEVO (Xu, https://doi.org/10.64898/2026.10.19.760276) and Mendel (Salman, https://doi.org/10.64898/2026.09.28.755183):
  population haplotypes as augmentation; up-weight masking at polymorphic sites.
- Le (https://doi.org/10.64898/2026.11.09.765190): long-context gain fades with evolutionary distance; long-range
  targets from mammals only in multi-species training.
- ShiftCache (https://arxiv.org/abs/2611.11976): indels shift the binning grid; cache recurrent states either side.
- Pruning DNA Sequence (https://arxiv.org/abs/2611.10386): sample stage-1 windows by loss.
- General ML: xLSTM-class gating holds up against Mamba-2 and Gated DeltaNet (https://arxiv.org/abs/2606.12364); gate and
  readout ablations (Muir, https://arxiv.org/abs/2611.08376); a low-to-high compression curriculum for learned chunking
  (https://arxiv.org/abs/2605.30080); distillation into xLSTM (https://arxiv.org/abs/2603.15590); xLSTM scaling laws
  (https://arxiv.org/abs/2510.02228).
- Benchmarks to report against: Cavus (https://doi.org/10.64898/2026.10.11.758362), UKBBGym
  (https://doi.org/10.64898/2026.10.01.755962), and Tao on ClinVar circularity (https://doi.org/10.64898/2026.10.24.761512).
