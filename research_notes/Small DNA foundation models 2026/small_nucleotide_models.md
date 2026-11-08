# Small and mid-sized nucleotide-level DNA foundation models, 2025–2026

Scope: general-purpose DNA language models of about 1M–1B parameters (focus on 500M or fewer), released or substantially updated between January 2025 and November 2026, that can be trained or fine-tuned on a few A100s. Evo-2 7B/40B-scale models appear only as reference points. Codon models are out of scope.

Provenance tags used throughout:
- **[paper]**: read in the paper itself (arXiv HTML or abstract page)
- **[card]**: read on a Hugging Face model card, the Hub API or a GitHub README
- **[blog]**: read in a vendor blog
- **[summary]**: taken from a search-engine summary of the paper or abstract, with the full text not read; medium confidence
- **[self-reported]**: the authors' own claim, not reproduced independently

Research date: 2026-11-09.

## Q1. Which small/mid-sized DNA foundation models of 2025–2026 matter, and what are their identity, architecture, size, context, tokenization, data and objective?

### Takeaway
The 2025–2026 field splits into four groups:
- **Tiny bidirectional models (1–10M) trained on hg38 that beat 500M transformers on the NT benchmark and Genomic Benchmarks:** JanusDNA, HAD (a Gated-DeltaNet student), DNA-xLSTM, Caduceus and NTv3-8M.
- **Mid-size (100–700M) multi-species models with clearer recipes:** Omni-DNA, MergeDNA, ModernGENA, GENERanno, PlantCAD2, gLM2, GPN-Star and NTv3-100M/650M.
- **Long-context generative families at about 1–3B:** GENERator v1/v2 and Evo 2 1B.
- **A 2025–2026 wave of learned-tokenization models:** dnaHNet, MergeDNA and DNAChunker.

Two models matter most for kron-dna. HAD is a bidirectional Gated DeltaNet DNA model, the same family as KronLSTM. DNA-xLSTM is an LSTM-family DNA model.

### Cited Findings

#### Quick-reference table (details and sources in the per-model entries below)

| Model (date) | Arch | Released sizes | Context | Tokens | Objective | Data | License |
|---|---|---|---|---|---|---|---|
| NTv3 (Dec 2025) | Conv U-Net + Transformer core | 8M, 100M, 650M (+5-downsample variant) | up to 1 Mb | single nt | MLM, then supervised post-training; MDLM generative variant | ~8–9T bp pretraining; 16k tracks / 24 species post-training | Non-commercial |
| GENERator v1 (Feb 2025) / v2 (Nov 2025, report Jan 2026) | Llama decoder | 1.2B, 3B (euk; v2 also prok) | 98k nt | 6-mer | Autoregressive (AR); v2 adds factorized nucleotide supervision | 386B nt eukaryotic (v1) | MIT |
| GENERanno (Feb/Jun 2025) | Encoder | ~0.5B base (prok, euk); 1.2B CDS-annotator preview (Feb 2026) | 8k bp (euk card) | single nt | MLM | 386B bp eukaryotic (euk card) | MIT |
| Omni-DNA (Feb 2025, NeurIPS 2025) | OLMo decoder | 20M, 60M, 116M, 300M, 700M, 1B | 250 tokens | BPE-4096 | AR + multi-task SFT | 300B nt (30B unique), NCBI multi-species | MIT |
| JanusDNA (May 2025, NeurIPS 2025) | 2× unidirectional Mamba+MoE stacks + attention fusion | 0.4M–7.7M activated (1.8M–35.5M total) | 1k / 131k (claims 1 Mb) | single nt | "Janus" bidirectional AR-style | hg38 | code on GitHub; license not stated |
| HAD (May 2025) | Bidirectional Gated DeltaNet + 1 attention layer | ~1.1M | 1,026 | single nt | Distillation from NTv2-500M + masked recon | hg38 | no code found |
| ModernGENA (ICLR 2026 workshop; weights Apr 2026) | ModernBERT encoder | base (377M per card text), large | "long" (not stated) | BPE-32k | MLM | 353.6B bp, 443 vertebrates, TSS-centred | not stated (GENATATOR fine-tunes Apache-2.0) |
| PlantCAD2 (Aug 2025) | Caduceus (BiMamba, RC-equivariant) | 88M, 311M, 694M | 8,192 bp | single nt | MLM | 65 angiosperm genomes | Apache-2.0 |
| MergeDNA (Nov 2025, AAAI 2026) | Hierarchical token-merging + Transformer | 380M | 4,096 bp | learned merging | Merged-token recon + adaptive MTM | Multi-species RefSeq | no code found |
| dnaHNet (Feb 2026) | H-Net: Mamba/attention enc-dec + Transformer main | 10M–1B trained (218M highlighted) | 8,192 nt | tokenizer-free, learned chunks | Next-nt + compression loss | 144B nt prokaryotic (GTDB) | no weights found |
| gLM2 (2024; card updated Feb 2026) | Transformer encoder | 150M, 650M | 4,096 tokens | AA for CDS + nt for intergenic | MLM (30%) | OMG, 315B tokens | Apache-2.0 |
| GPN-Star (Sep 2025) | Phylogeny-aware attention over WGA | 25M, 85M, 200M (many species) | — | single nt + alignment | MLM | whole-genome alignments | MIT |
| PhyloGPN (Mar 2025, RECOMB 2025) | CNN → F81 rates | 83.2M | 481 bp receptive field | single nt | phylogenetic likelihood | Zoonomia alignment | "cc" |
| DNA-xLSTM (Nov 2024 / ICLR 2025) | sLSTM / mLSTM | 500k, 2M, 4M | 1k / 32k | single nt | CLM and MLM | GRCh38 | — |
| Evo 2 1B base (Feb 2025; Nature 2026) | StripedHyena 2 | 1B | 8K | single nt | AR | OpenGenome2 (8.8T tokens total dataset) | Apache-2.0 |

#### NTv3 (Nucleotide Transformer v3), InstaDeep
- Paper: Boshar et al., "A foundational model for joint sequence-function multi-species modeling at scale for long-range genomic prediction", bioRxiv Dec 2025 (DOI 10.64898/2025.12.22.695963). The blog post is dated 18 Feb 2026 — [blog](https://instadeep.com/2026/02/modelling-the-genome-with-ntv3/); [research page](https://instadeep.com/research/paper/a-foundational-model-for-joint-sequence-function-multi-species-modeling-at-scale-for-long-range-genomic-prediction/); [PDF](https://instadeep.com/wp-content/uploads/2025/12/NT_v3.pdf).
- A later Nature Methods publication is claimed only in a LinkedIn post. **Unverified** — [summary](https://instadeep.com/2026/02/modelling-the-genome-with-ntv3/).
- Architecture [card]: a U-Net-style conv tower, then a Transformer stack, then a deconv tower, then the LM head. The 8M checkpoint has 7.69M parameters, 2 transformer layers, 7 downsample stages, hidden dimension 256, FFN 1024 and 8 heads. Input length must be a multiple of 128. The tokenizer is character-level with vocabulary 11 — [NTv3_8M_pre card](https://huggingface.co/InstaDeepAI/NTv3_8M_pre).
- Released checkpoints [card]:
  - NTv3_8M_pre, NTv3_100M_pre, NTv3_650M_pre and NTv3_5downsample_pre, each also in an `_8kb` variant (9 Dec 2025)
  - post-trained `_post` and `_post_131kb` variants (10 Dec 2025)
  - NTv3_generative (MDLM, 8 Jan 2026)
  - NTv3_pretrained, NTv3_tracks_1mb, NTv3_annot_mammals and NTv3_annot_plants (23 Sep 2026)
  - Hub monthly downloads: 8M_pre 6.5K, 100M_pre 6.0K, 650M_pre 8.9K — [HF 650M_pre](https://huggingface.co/InstaDeepAI/NTv3_650M_pre); [HF 100M_post](https://huggingface.co/InstaDeepAI/NTv3_100M_post); [HF generative](https://huggingface.co/InstaDeepAI/NTv3_generative)
- The 8M card recommends the newer, gated NTv3_pretrained for new work — [card](https://huggingface.co/InstaDeepAI/NTv3_8M_pre).
- Context reaches 1 Mb at single-base resolution, trained with a length curriculum from kilobases to a megabase [blog].
- Pretraining data: the blog says "more than eight trillion nucleotides" and the card says "9 trillion base pairs". A search summary of the PDF names OpenGenome2 — [blog](https://instadeep.com/2026/02/modelling-the-genome-with-ntv3/); [card](https://huggingface.co/InstaDeepAI/NTv3_8M_pre); [summary](https://instadeep.com/wp-content/uploads/2025/12/NT_v3.pdf).
- Objectives [blog]:
  - MLM pretraining
  - post-training on about 16,000 functional tracks plus genome annotations from 24 species, combining supervised loss with continued MLM and using species-conditional normalisation
  - MDLM generative post-training from the 650M model
  - 1,000 designed enhancers validated with STARR-seq (number from a search summary) — [blog](https://instadeep.com/2026/02/modelling-the-genome-with-ntv3/)
- License: "other". The license summary limits use to non-commercial purposes, and outputs may not be used to train commercial third-party models — [card](https://huggingface.co/InstaDeepAI/NTv3_8M_pre).
- Training: JAX in bf16, ported to PyTorch. Compute is not reported — [card](https://huggingface.co/InstaDeepAI/NTv3_8M_pre).

#### Nucleotide Transformer v2 (reference point; formally published 2025)
- Dalla-Torre et al., Nature Methods 22:287–297 (Feb 2025). The family spans 50M–2.5B parameters, trained on 3,202 human genomes plus 850 species — [summary / Nature index](https://www.nature.com/nature-index/article/10.1038/s41592-024-02523-z); [InstaDeep](https://instadeep.com/research/paper/nucleotide-transformer-building-and-evaluating-robust-foundation-models-for-human-genomics/).
- NTv2-50M-multi-species has 55.9M parameters and a CC-BY-NC-SA-4.0 license. The Hub reports 446.5K downloads (see the note on download counts in Gaps) — [card](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-50m-multi-species).

#### GENERator v1 and v2, GenerTeam (Zhejiang University and Alibaba-affiliated authors; affiliations not verified)
- v1: Wu et al., arXiv 2502.07272 (v1 11 Feb 2025, v5 31 Jan 2026). Context 98k nt, pretrained on 386B nucleotides of eukaryotic DNA. Self-reported results include zero-shot variant-effect prediction competitive with alignment-based methods, and enhancers validated by UMI-STARR-seq — [paper](https://arxiv.org/abs/2502.07272); code at [GitHub](https://github.com/GenerTeam/GENERator).
- Weights [card]:
  - GENERator-eukaryote-1.2b/3b-base (Feb 2025)
  - GENERator-v2-eukaryote and -prokaryote at 1.2b and 3b (5 Nov 2025)
  - all MIT-licensed Llama-architecture decoders with a 6-mer tokenizer
  - the 1.2b card's metadata shows about 1B parameters — [v2 card](https://huggingface.co/GenerTeam/GENERator-v2-eukaryote-1.2b-base)
- Card warning: inputs that are not a multiple of 6 get an `<oov>` token and degenerate generations ("AAAAAA"). The fix is left-padding with A or left-truncating [card].
- v2 technical report, bioRxiv 2026.01.27.702015 [paper abstract]:
  - "Factorized Nucleotide Supervision" combines k-mer tokens with single-nucleotide likelihoods by marginalisation
  - "Genome Compression Pretraining" reshapes the training distribution towards gene-centric and regulatory regions
  - contexts reach 98 kb for eukaryotes and prokaryotes
  - claims to match or exceed SOTA with more efficient inference — [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.01.27.702015v1)

#### GENERanno, GenerTeam
- Li et al., "GENERanno: A Genomic Foundation Model for Metagenomic Annotation", bioRxiv Jun 2025. It is a compact encoder with single-nucleotide tokens, which the authors argue is needed for gene annotation. Self-reported, it beats GeneMarkS2, Prodigal and GeneLM, including zero-shot on archaea — [summary of bioRxiv](https://www.biorxiv.org/content/10.1101/2025.06.04.656517.full.pdf).
- Weights [card]:
  - GENERanno-prokaryote-0.5b-base and GENERanno-eukaryote-0.5b-base (fill-mask, MIT, Feb 2025)
  - GENERanno-prokaryote-0.5b-cds-annotator (Sep 2025)
  - GENERanno-eukaryote-1.2b-cds-annotator-preview (Feb 2026) — [HF prok](https://huggingface.co/GenerTeam/GENERanno-prokaryote-0.5b-base); [HF euk](https://huggingface.co/GenerTeam/GENERanno-eukaryote-0.5b-base); [HF 1.2b preview](https://huggingface.co/GenerTeam/GENERanno-eukaryote-1.2b-cds-annotator-preview)
- About 493M parameters for the prokaryote base. The eukaryote card describes an 8k bp context, 500M parameters, training on 386B bp of eukaryotic DNA, and "developmental phase" status. This comes from mirror listings, so treat it as medium confidence — [summary](https://hf-mirror.492719920.workers.dev/GenerTeam/GENERanno-prokaryote-0.5b-base).

#### Omni-DNA, Microsoft Research and Cambridge (Zehui Li et al.)
- arXiv 2502.03499 (Feb 2025). A search summary and the Microsoft Research page list it at NeurIPS 2025 — [paper](https://arxiv.org/html/2502.03499v1); [MSR](https://www.microsoft.com/en-us/research/?p=1153590).
- Architecture: an OLMo-based autoregressive transformer with RoPE (ALiBi converged more slowly). Layers/heads/hidden dimension by size [paper]:
  - 20M: 8/8/256
  - 60M: 8/8/512
  - 116M: 12/12/768
  - 300M: 16/16/1024
  - 700M: 16/16/1536
  - 1B: 16/16/2048
- Tokenizer: BPE with a 4,096-token vocabulary. Pretraining sequence length is 250 tokens [paper].
- Data: NCBI multi-species genomes. Exact deduplication cut about 170B nt to 30B unique nt, and models trained on 300B nt over multiple epochs. The objective is next-token prediction, then multi-task instruction-style SFT with an expanded vocabulary [paper].
- Weights: zehui127/Omni-DNA-{20M,60M,116M,300M,700M,1B}, plus Multitask, DNA2Function and DNA2Image variants, all MIT (Feb 2025) — [HF](https://huggingface.co/zehui127/Omni-DNA-116M).

#### JanusDNA, BIH/Charité, MPI, FU Berlin and others (Duan et al.)
- arXiv 2505.17257 (v4 28 Oct 2025), NeurIPS 2025 poster — [paper](https://arxiv.org/html/2505.17257v4); [NeurIPS](https://neurips.cc/virtual/2025/poster/119543); code at [GitHub](https://github.com/Qihao-Duan/JanusDNA).
- Architecture [paper]:
  - two independent unidirectional encoders (forward and backward) of 8 layers each, every layer a Mamba block plus an FFN block
  - MoE replaces the FFN at a 0.5 ratio with 16 experts
  - a FlexAttention fusion layer with a custom mask lets each position see full bidirectional context
  - an optional "mid-attention" variant swaps one Mamba block for attention
  - reverse-complement embeddings are averaged at inference
- Pretrained sizes (activated / total) [paper]:
  - width 32: 0.42M / 1.80M
  - width 72: 1.98M / 8.95M
  - width 144: 7.66M / 35.5M
- Data: human reference genome only. The text says hg38 but a reference cites GRCh37. Context 1,024 and 131,072 bp, single-nucleotide tokens [paper].
- Objective: "Janus modeling", where every token is predicted from full bidirectional context. All tokens contribute to the loss, compared with 15% in MLM. The authors claim it trains almost 2× faster than masked models [paper].
- No weights link appears in the paper [paper].

#### HAD (Hybrid Architecture Distillation), CAS Institute of Automation / UCAS (Yang et al.)
- arXiv 2505.20836 (27 May 2025), not peer-reviewed — [paper](https://arxiv.org/html/2505.20836v1).
- **The student is a bidirectional Gated DeltaNet**: 4 GDN blocks of dimension 128, with forward and reverse passes combined, followed by 1 FlashAttention self-attention layer and a gated MLP. It has about 1.1M parameters; the introduction's "11M" is likely a typo [paper].
- Teacher: NTv2-500M (498.3M parameters, 6-mer tokens) [paper].
- Losses [paper]:
  - MSE feature alignment to the teacher at visible positions, after pooling and a projection
  - masked-nucleotide reconstruction through a cross-attention decoder
  - "grouping" masking that masks 15% of 6-mers for the teacher and maps them to character positions for the student
- Data: GRCh38 with the HyenaDNA/Caduceus split, sequence length 1,026. Hardware and GPU-hours are not reported, and no code link is given [paper].

#### ModernGENA, AIRI (Aspidova, Kuratov, Shadskiy, Burtsev, Fishman)
- "Back to BERT in 2026: ModernGENA as a Strong, Efficient Baseline for DNA Foundation Models", poster at the ICLR 2026 FM4Science workshop. The authors release the full implementation and configuration as an open baseline — [ICLR](https://iclr.cc/virtual/2026/10014529).
- Architecture [card]: a ModernBERT encoder with RoPE, hybrid local/global attention, pre-norm, GeGLU and unpadding, using the 32k BPE tokenizer from GENA-LM.
- Data [card]: 443 vertebrate assemblies, 353,574,093,776 bp, forward and reverse-complement, in TSS-centred windows from −16 kb to +8 kb.
- Size [card]: 377M per the card description, but the safetensors metadata says 0.1B (a conflict).
- Weights (Apr 2026) [card]: AIRI-Institute/moderngena-base and -large — [card](https://huggingface.co/AIRI-Institute/moderngena-base); [large](https://huggingface.co/AIRI-Institute/moderngena-large).
- Fine-tuned "GENATATOR" annotation variants (Jun 2026) are Apache-2.0 — [HF](https://huggingface.co/AIRI-Institute/genatator-moderngena-base-multispecies-region-model).
- Predecessor GENA-LM (gena-lm-bert-base-t2t) has 135.3M parameters — [card](https://huggingface.co/AIRI-Institute/gena-lm-bert-base-t2t).

#### Other ModernBERT/ELECTRA-style DNA encoders (2025)
- **NucEL** (Ding, Parker, Wen; arXiv 2508.13191, Aug 2025) [paper abstract]:
  - ELECTRA replaced-token detection at single-nucleotide resolution
  - ModernBERT hybrid local/global attention and flash attention
  - trained on the human genome
  - claims to rival NT models "25x larger"
  - parameter count and code not found — [paper](https://arxiv.org/abs/2508.13191)
- **BMFM-DNA** (IBM; arXiv 2507.05265, Jul 2025): two ModernBERT models, one trained on reference sequence and one SNP-aware model that encodes variation — [summary](https://arxiv.org/abs/2507.05265).
- **ModernBERT-DNA-v1-37M-hg38** (Raphaël Mourad): 37M parameters, pretrained on 10 kb hg38 sequences — [summary of card](https://huggingface.co/RaphaelMourad/ModernBERT-DNA-v1-37M-hg38).
- **Mistral-DNA-v1-17M-hg38** (same author): 16.8M parameters, Mixtral architecture, causal LM, Apache-2.0 — [card](https://huggingface.co/RaphaelMourad/Mistral-DNA-v1-17M-hg38).

#### PlantCAD2 (successor of PlantCaduceus), Kuleshov group, Cornell / Buckler lab and others
- bioRxiv 10.1101/2025.08.27.672609 (Aug 2025). A co-author's LinkedIn post says it was later published in Cell Genomics. **Unverified** — [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.08.27.672609.full.pdf); [GitHub](https://github.com/plantcad/plantcad).
- Architecture: Caduceus (bidirectional Mamba with reverse-complement equivariance), trained with MLM on 65 angiosperm genomes with an 8,192 bp window. That is 16× the 512 bp window of PlantCaduceus v1 [summary of paper].
- Sizes: S 88M, M 311M, L 694M (the abstract says 676M) [summary of paper].
- Weights (Mar 2025): kuleshov-group/PlantCAD2-Small-l24-d0768, -Medium-l48-d1024 and -Large-l48-d1536, all Apache-2.0 — [HF S](https://huggingface.co/kuleshov-group/PlantCAD2-Small-l24-d0768); [HF L](https://huggingface.co/kuleshov-group/PlantCAD2-Large-l48-d1536).
- GeneCAD, a plant genome annotator, builds on PlantCAD2 — [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.10.31.685877.full.pdf).
- PlantCaduceus v1 was trained on 16 angiosperm genomes. Checkpoints PlantCaduceus_l20/l24/l28/l32 (May 2024) are Apache-2.0 — [CSHL](https://repository.cshl.edu/id/eprint/41592/); [HF](https://huggingface.co/kuleshov-group/PlantCaduceus_l32).

#### Caduceus (reference baseline, ICML 2024; card updated Oct 2025)
- caduceus-ph_seqlen-131k_d_model-256_n_layer-16 has 7.7M parameters and an Apache-2.0 license. The Hub details endpoint reports 415.9K downloads. 1k-context variants (4 layers, d=118 or d=256) are also published — [card](https://huggingface.co/kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16); [PS variant](https://huggingface.co/kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16).

#### dnaHNet, University of Toronto/Vector, Arc, Cartesia and CMU (Shah, …, Sukjun Hwang, Albert Gu, Patrick Hsu, Hani Goodarzi)
- arXiv 2602.10603 (v2 14 Feb 2026) — [paper](https://arxiv.org/html/2602.10603v2).
- Architecture: tokenizer-free H-Net-style dynamic chunking with a recursive two-stage hierarchy. Encoder and decoder are each 4 Mamba layers plus 1 Transformer layer, and the main network is a Transformer. Compression is 3×2 for XL and 2×2 for M [paper].
- Models from 10M to 1B were trained with 8,192-nt context [paper].
- Data: a GTDB-derived prokaryotic corpus of 144B nt from 85,205 organisms [paper].
- Objective: next-nucleotide NLL plus a compression ("ratio") loss, with α between 0.01 and 0.05. No code or weights links were found [paper].

#### MergeDNA, Westlake University and others (Siyuan Li, …, Stan Z. Li)
- arXiv 2511.14806 (Nov 2025), AAAI 2026 oral — [abs](https://arxiv.org/abs/2511.14806); [paper](https://arxiv.org/html/2511.14806v1).
- Size: 380M parameters, split as local encoder 51M, latent encoder 253M, latent decoder 51M and local decoder 25M [paper].
- Architecture: dimension 1024. The local encoder has 4 ToMe-attention blocks with a window of 16, the latent encoder has 20 Transformer blocks, and the decoders have 4 and 2 blocks. Maximum length is 4,096 bases [paper].
- Data: Multi-Species Genomes from NCBI RefSeq [paper].
- Objectives: merged-token reconstruction plus adaptive masked-token modeling. No code link was found [paper].

#### DNAChunker (learnable tokenization)
- Kim et al., arXiv 2601.03019 (Jan 2026; ICML 2026 camera-ready). It is a masked DNA LM with a learnable segmentation module trained on the human reference genome. Parameter count and code were not found — [abs](https://arxiv.org/abs/2601.03019).

#### Bio-xLSTM / DNA-xLSTM (Schmidinger et al., JKU/NXAI; arXiv Nov 2024, ICLR 2025)
- Sizes: DNA-xLSTM-500k (sLSTM, 5 blocks, d=128, 1,024 context), DNA-xLSTM-2M (sLSTM, 6 blocks, d=256, 1,024 context) and DNA-xLSTM-4M (mLSTM, 9 blocks, d=256, 32,768 context, RoPE) [summary].
- Trained with both CLM and MLM on GRCh38. RC-equivariant xLSTM-PH and xLSTM-PS variants exist [summary].
- Sources: [arXiv](https://arxiv.org/pdf/2411.04165); [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/60cefc59610f8f59ea10099f99a36726-Abstract-Conference.html).

#### ConvNova (Bo, Mao et al.; ICLR 2025, arXiv 2502.18538)
- A CNN with dilated convolutions, gated convolutions and a dual-branch gating framework. Code: aim-uofa/ConvNova — [arXiv](https://arxiv.org/abs/2502.18538v1); [ICLR](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5e0da5da69b71349ae0bd7ad716e4bc9-Abstract-Conference.html).

#### Lyra (Ramesh, Siddiqui, Gu, Mitzenmacher, Sabeti; arXiv 2503.16351, Mar 2025)
- SSMs for global dependencies plus projected gated convolutions for local features, evaluated on 100+ biological tasks. It reports "up to 120,000-fold" fewer parameters, and every task trains on 2 or fewer GPUs in under 2 hours. Not DNA-pretrained in the foundation-model sense — [summary of abstract](https://arxiv.org/abs/2503.16351v1).

#### HybriDNA, Microsoft Research, CMU and Renmin University
- arXiv 2502.10807, ICLR 2025 MLGenX workshop [summary]:
  - a decoder that alternates Transformer and Mamba2 blocks
  - 131 kb single-nucleotide context
  - sizes 300M, 3B and 7B
  - about 160B nt from 845 species, next-token prediction
  - weights availability not verified
- Sources: [arXiv](https://arxiv.org/abs/2502.10807v2); [MSR](https://www.microsoft.com/en-us/research/?p=1142049).

#### GenomeOcean, DOE JGI / Northwestern
- The 4B model was trained on more than 600 Gbp of metagenomic-assembly contigs, with a 4,096-token BPE vocabulary and 10,240 tokens of context (about 50 kbp) — [summary of HF/preprint](https://huggingface.co/pGenomeOcean/GenomeOcean-4B); [JGI](https://jgi.doe.gov/user-science/science-stories/genomeocean-leverages-ai-decode-natures-secret-language).
- GenomeOcean-100M now lives at DOEJGI/GenomeOcean-100M: 119.6M parameters, Mistral architecture, BSD license, 3.4M downloads reported — [card](https://huggingface.co/DOEJGI/GenomeOcean-100M).
- A 500M model exists, and the May 2026 TaxDistill paper uses it as a distillation teacher — [summary](https://www.alphaxiv.org/abs/2605.28868).

#### gLM2, Tatta Bio
- Two scales, 150M and 650M (670.6M parameters) — [card](https://huggingface.co/tattabio/gLM2_650M).
- Mixed-modality MLM transformer: amino-acid tokens for CDS, nucleotide tokens for intergenic regions, plus strand tokens. A 09/2024 update replaced BPE with per-nucleotide intergenic tokens [card].
- Context 4,096 tokens. Training: 315B tokens of the OMG corpus after SemDeDup removed 49%, 30% masking. Apache-2.0. Card updated Feb 2026. Paper: bioRxiv 10.1101/2024.08.14.607850 [card].

#### AIDO.DNA → GB.DNA (GenBio AI)
- The Hub repo is now genbio-ai/GB.DNA-300M: 300.3M parameters under the "GenBio AI Community License", with backbone id `aido_dna_300m` in the code. The card gives no training details and defers to GB.DNA-7B — [card](https://huggingface.co/genbio-ai/GB.DNA-300M).

#### Evo 2 small checkpoint (Arc Institute)
- evo2_1b_base has 1B parameters and 8K context. **It requires FP8 via Transformer Engine and a Hopper GPU** (the 7B models run in bf16) [card].
- The OpenGenome2 dataset is 8.8T tokens. License Apache-2.0. Paper in Nature (2026), DOI 10.1038/s41586-026-10176-5 [card].
- No distilled checkpoints are listed — [GitHub README](https://github.com/ArcInstitute/evo2); [HF](https://huggingface.co/arcinstitute/evo2_1b_base).
- A third-party distillation from Evo2-1B (HelixNano-mRNA, arXiv 2604.08574) aligns student layers with the teacher's 12th and 25th layers, on mRNA inputs only — [paper](https://arxiv.org/html/2604.08574).

#### Phylogeny-aware small models (Song lab, UC Berkeley)
- **GPN-Star** (bioRxiv 2025.09.21.677619): phylogeny-aware attention over whole-genome alignments and species trees — [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.09.21.677619.full.pdf).
  - Checkpoints (Sep 2025, MIT) come at 25M, 85M and 200M for hg38 (v100 vertebrate, m447 mammal, p243 primate) and at 25M/85M for mm39, galGal6, dm6, ce11 and tair10 — [HF](https://huggingface.co/songlab/gpn-star-hg38-v100-200m).
- **PhyloGPN** (arXiv 2503.03773, RECOMB 2025): a CNN that outputs F81 substitution-rate parameters — [card](https://huggingface.co/songlab/PhyloGPN); [arXiv](https://arxiv.org/abs/2503.03773).
  - Trained on the Zoonomia alignment, but needs no alignment at inference.
  - 83.2M parameters, receptive field 481 bp, license "cc".
- **songlab/mlm-baseline-hg38** (Aug 2026): a 0.3B reverse-complement-equivariant CNN MLM with 1,549 bp context and MIT license. It is a partial run released at step 93,750, with no paper yet — [card](https://huggingface.co/songlab/mlm-baseline-hg38).

#### Domain-specific small SSMs
- **BarcodeMamba** (Gao & Taylor, Guelph/Vector; arXiv 2412.11084): a Mamba-2 model for DNA barcodes — [summary](https://arxiv.org/html/2412.11084v1).
- **BarcodeMamba+** (arXiv 2512.15931, NeurIPS 2025 Imageomics workshop): fungal barcodes, scaled up to 49.2M parameters, code under bioscan-ml — [summary](https://arxiv.org/abs/2512.15931v1).

#### Other 2026 items found but not characterised
- **BioToken/BioFM** (M42 Health; ICML 2026): tokens that encode variants and annotations directly. Claims 100-fold less compute than large genomic models and beats Enformer on pathogenicity and expression prediction. No size given — [ICML](https://icml.cc/virtual/2026/poster/63585).
- **LDARNet** (June 2026): learnable tokenization. Only the title was seen — [HF papers](https://huggingface.co/papers/2606.04552.md).

### Inferences
- **HAD is the closest published prior art for kron-dna.** It is a bidirectional Gated DeltaNet plus one attention layer at about 1.1M parameters on hg38. kron-dna should cite it and compare against it head-to-head. JanusDNA (bidirectional Mamba stacks fused by attention) and DNA-xLSTM (sLSTM/mLSTM) are the other recurrent-family baselines at matching scale.
- Strong, cheap, permissively licensed baselines to run at the same parameter count:
  - Caduceus-Ph/PS (7.7M, Apache-2.0)
  - Omni-DNA-20M/60M/116M (MIT)
  - Mistral-DNA-17M (Apache-2.0)
  - PlantCAD2-S (88M, Apache-2.0) for plants
  - GENERanno-0.5B (MIT) for annotation-style per-nucleotide tasks
  - NTv3-8M/100M are excellent technical baselines but non-commercial
- Recipe ideas worth borrowing:
  - RC-equivariance or RC averaging (Caduceus, PlantCAD2, JanusDNA, mlm-baseline)
  - an all-token bidirectional loss instead of 15% MLM (JanusDNA)
  - teacher feature-alignment plus masked reconstruction with k-mer-to-character mask mapping (HAD)
  - a length curriculum up to 1 Mb and conv down/up-sampling around the sequence mixer (NTv3)
  - factorized nucleotide supervision and gene/regulatory-centric data reweighting (GENERator-v2)
  - TSS-centred multi-species windows (ModernGENA)
  - learned chunking (dnaHNet, MergeDNA, DNAChunker)
- Context-length bands: most small models train at 1k–8k. 131k is used by Caduceus, JanusDNA and HybriDNA, and 1 Mb by NTv3 and JanusDNA (claimed). A linear-recurrent kron-dna is naturally placed to compete at 131k–1M, where NTv3's U-Net and JanusDNA are the main small competitors.

### Gaps
- Not verified in this session:
  - PlantCaduceus v1 sizes (believed to be about 20M–225M)
  - GENA-LM's 2025 NAR publication
  - gLM2/OMG at ICLR 2025
  - Omni-DNA's NeurIPS 2025 acceptance (search summary only)
  - NTv3's Nature Methods publication
  - PlantCAD2's Cell Genomics publication
- Missing details:
  - DNAChunker, NucEL, BMFM-DNA and BioFM: parameter counts and weights
  - HybriDNA-300M and dnaHNet: weights availability
  - ModernGENA: exact sizes and context (the card text and metadata conflict)
  - JanusDNA, MergeDNA and HAD: code licenses
- Not re-checked in 2025–2026 sources: METAGENE-1 (7B metagenomic, beyond the size scope), GROVER (2024, HF PoetschLab/GROVER) and DNABERT-S successors.
- Two other leads were not followed: an AIRI "GENATATOR" paper and an Evo-2-style model "dnaGrinder" (2024).
- Download counts disagree between Hub endpoints. The details endpoint shows caduceus-ph at 415.9K and PlantCAD2-S at 131.4K; the search endpoint shows 3.2K and 4.7K. The first is probably all-time and the second the last 30 days, which is not confirmed.

## Q2. What training data scale and compute are reported, and what is feasible on a handful of A100s?

### Takeaway
Most small DNA models that report compute fit easily into an academic A100 budget:
- JanusDNA: about 3 hours on 8 GPUs at 1k context; 9.3 hours at 131k
- Omni-DNA: 8×A100-40GB for 1–7 days
- MergeDNA (380M): 8×A100-80GB for about 5 days
- dnaHNet: 4 GPUs per run

Large-vendor models (NTv3, GENERator, Evo 2) do not report GPU-hours.

### Cited Findings
- **JanusDNA** (all [paper](https://arxiv.org/html/2505.17257v4)):
  - pretraining ran 10k steps for 1,024-length models and 50k steps for the 131k model
  - wall-clock was about 3h–3h08m for 1k models and 9h17m for the width-144 131k model on H800 (Table 9)
  - about 27 min per 1,000 steps at 131k
  - GPUs listed: 80GB H100, A100 and A800
  - eQTL fine-tuning used 8× 80GB GPUs
  - the paper conflicts on batch size: the text says 128 and 1 across 8 GPUs, Table 9 says 1024 and 8
- **Omni-DNA** (all [paper](https://arxiv.org/html/2502.03499v1)):
  - trained on NVIDIA A100 40GB
  - Table 7 gives ranges from "8 GPUs × 1 day" to "8 GPUs × 7 days" without per-size GPU-hours
  - 300B nt seen from 30B unique nt
- **MergeDNA** (all [paper](https://arxiv.org/html/2511.14806v1)):
  - 8× A100-80G for "nearly 5 days"
  - 100K iterations, global batch 256 (per-GPU 8 × gradient accumulation 16), AdamW at lr 1e-4, cosine schedule, 10K warmup
  - roughly 960 A100-hours (inference)
- **dnaHNet** (all [paper](https://arxiv.org/html/2602.10603v2)):
  - A100/H100-class GPUs, 4 per run
  - token budgets swept from 5B to more than 200B
  - the compute-optimal model at 8×10¹⁹ FLOPs trains on 140B tokens
  - total GPU-hours not reported
- **HAD**: hardware and GPU-hours are not reported. The student is about 1.1M parameters at sequence length 1,026, so cost should be small (inference) — [paper](https://arxiv.org/html/2505.20836v1).
- **Lyra**: every task in the study trains and runs "on two or fewer GPUs in under two hours" — [summary](https://arxiv.org/abs/2503.16351v1).
- **NTv3**: compute not stated. Pretraining covers about 8–9T bp with a kb→Mb curriculum, in JAX bf16 — [blog](https://instadeep.com/2026/02/modelling-the-genome-with-ntv3/); [card](https://huggingface.co/InstaDeepAI/NTv3_8M_pre).
- **Data-scale reference points**:
  - GENERator v1: 386B nt — [paper](https://arxiv.org/abs/2502.07272)
  - gLM2: 315B tokens — [card](https://huggingface.co/tattabio/gLM2_650M)
  - ModernGENA: 353.6B bp of TSS-centred windows — [card](https://huggingface.co/AIRI-Institute/moderngena-base)
  - HybriDNA: about 160B nt / 845 species — [summary](https://arxiv.org/abs/2502.10807v2)
  - dnaHNet: 144B nt prokaryotes — [paper](https://arxiv.org/html/2602.10603v2)
  - GenomeOcean-4B: more than 600 Gbp — [summary](https://huggingface.co/pGenomeOcean/GenomeOcean-4B)
  - Evo 2: OpenGenome2 8.8T tokens — [GitHub](https://github.com/ArcInstitute/evo2)
  - Caduceus, JanusDNA, HAD, DNA-xLSTM, NucEL and DNAChunker train on a single human reference genome (about 3 Gbp)
- **gLM2 hyperparameters**: 30% masking, lr 1e-3, AdamW betas (0.9, 0.95), weight decay 0.1, bf16 — [card](https://huggingface.co/tattabio/gLM2_650M).
- **Hardware constraint**: evo2_1b_base requires FP8 and Hopper, so it is not listed as runnable on A100. The 7B models run in bf16 without Transformer Engine — [GitHub](https://github.com/ArcInstitute/evo2).

### Inferences
- At kron-dna's budget (up to about 6 concurrent single-A100 jobs for days to weeks), the following are all feasible:
  - a full reproduction of the JanusDNA, HAD, Caduceus and DNA-xLSTM-scale experiments (1–10M parameters, hg38, 1k–131k)
  - Omni-DNA-116M-scale multi-species AR pretraining (tens to hundreds of A100-days)
- A MergeDNA-scale 380M run costs about 960 A100-hours, which is about one week on 6 GPUs if parallelised.
- Evo 2 1B is a poor distillation teacher on A100s because of the FP8/Hopper requirement. Better teacher choices on A100:
  - GENERator 1.2B (Llama architecture, MIT)
  - NTv2-500M (HAD's teacher; license CC-BY-NC-SA)
  - NTv3-100M/650M (non-commercial)
  - Evo 2 7B in bf16 (heavy)

### Gaps
- No GPU-hour figures were found for NTv3, GENERator v1/v2, GENERanno, PlantCAD2, ModernGENA, GPN-Star, gLM2 or HybriDNA.
- HAD's distillation cost (teacher forward passes) is not reported.

## Q3. What do benchmarks show, and where do small models match or beat much larger ones?

### Takeaway
On the saturated short-sequence suites (NT benchmark, 18 tasks; Genomic Benchmarks, 8 tasks; GUE), many 2025 papers report 1–120M models matching or beating NTv2-500M, NT-2.5B, DNABERT-2 and GENERator-1.2B. Examples are HAD (1.1M), JanusDNA (2M), Omni-DNA-116M, NTv3-8M and MergeDNA. The margins are often a point or two, and each paper uses its own fine-tuning protocol. Long-range (DNALongBench) and variant-effect (TraitGym) results are less favourable to general DNA language models.

### Cited Findings
- **HAD** (all [paper](https://arxiv.org/html/2505.20836v1)):
  - with about 1.1M parameters it ranks first on 11 of 18 NT-benchmark tasks
  - it beats its 498M NTv2 teacher on histone tasks, e.g. H3K4me3 0.643 vs 0.410
  - it trails the teacher on splice sites (All: 0.911 vs 0.983) and slightly on promoters (0.968 vs 0.976)
  - Genomic Benchmarks average: HAD 0.875, Caduceus 0.866, HyenaDNA 0.853, Mamba 0.846, CNN 0.803
- **JanusDNA** (all [paper](https://arxiv.org/html/2505.17257v4)):
  - NT benchmark: the text claims SOTA on 12 of 18 but the table bolds 11, e.g. H3 MCC 0.835 vs Caduceus-Ph 0.815 and H3K4me3 0.688 vs ConvNova 0.566. Caduceus-Ph wins splice "All" (0.983 vs 0.967)
  - Genomic Benchmarks: the text claims 3 of 8 but the table bolds 2
  - DNALongBench eQTL (9 tissues, AUROC): JanusDNA-MLP beats Enformer and Caduceus-Ph on all nine, e.g. Artery Tibial 0.852 vs 0.741 vs 0.690
  - the abstract claim of beating "models with 250x more activated parameters" likely refers to NTv2-500M vs about 2M (inference from the paper's numbers)
- **Omni-DNA** (all [paper](https://arxiv.org/html/2502.03499v1)):
  - NT-benchmark averages: Omni-DNA-1B 0.767 and 116M 0.755, vs NT-2.5B 0.709, Caduceus-Ph 0.715, HyenaDNA 0.707, DNABERT-2 0.679
  - Genomic Benchmarks: 116M 0.879 vs Caduceus-Ph 0.866, DNABERT-2 0.859, HyenaDNA 0.853
  - the 20M (0.662) and 60M (0.690) models trail several baselines
  - scaling is non-monotonic: 300M scores 0.741, below 116M
- **MergeDNA (380M)** (all [paper](https://arxiv.org/html/2511.14806v1)):
  - Genomic Benchmarks average: 90.87 vs GENERator 90.71, NT-500M 89.26, MxDNA 89.12, DNABERT-2 87.30, HyenaDNA 87.07, Caduceus 85.89
  - NT benchmark: 78.39 vs MxDNA 78.14, GENERator 72.84, Caduceus-PS 72.50, NTv2-500M 71.13, DNABERT-2 69.74
  - GUE (24 tasks): 77.11 vs HybriDNA-7B 76.42, ConvNova 68.91, NT-multi 67.23, DNABERT-2 66.43
- **NTv3-8M** [blog, self-reported]:
  - described as competitive with much larger models, and outperforms Evo2-1B and NTv2-500M
  - NTv3 claims SOTA on its own 106-task, 7-species NTv3 Benchmark
  - beats SegmentNT, AUGUSTUS and SpliceAI on gene-structure annotation
  - beats AgroNT-1B on expression and protein abundance from promoter-proximal sequence
  - the 650M model is "order-of-magnitude faster" at 1 Mb than HyenaDNA and Caduceus — [blog](https://instadeep.com/2026/02/modelling-the-genome-with-ntv3/)
- **ConvNova** (CNN): SOTA on 12 of 18 NT tasks, with a 10.5% gain over the second-best model on H3K4me3, usually with fewer parameters and faster compute — [summary](https://arxiv.org/abs/2502.18538v1).
- **PlantCAD2**:
  - beats Evo2-7B on 10 of 12 zero-shot conservation tasks
  - with parameter-efficient fine-tuning, beats AgroNT-1B on 7 cross-species tasks
  - the 8 kb context raises maize ACR AUPRC from 0.587 to 0.711 — [summary of bioRxiv](https://www.biorxiv.org/content/10.1101/2025.08.27.672609.full.pdf)
- **dnaHNet vs StripedHyena2** (FLOP-matched, not parameter-matched) (all [paper](https://arxiv.org/html/2602.10603v2)):
  - perplexity scaling exponent: dnaHNet 0.06, SH2 0.04, Transformer 0.01
  - at 10⁶ nt, the 218M dnaHNet needs 3.89× fewer inference FLOPs than a 166M SH2
  - zero-shot protein VEP Spearman: dnaHNet 0.3266 vs SH2 0.3110 vs Transformer 0.1555
  - zero-shot gene-essentiality AUROC: dnaHNet 0.6050 vs SH2 0.5759
- **NucEL** claims to rival NT models about 25× larger on regulatory, TF-binding, chromatin and histone tasks — [paper abstract](https://arxiv.org/abs/2508.13191).
- **DNAChunker** reports surpassing Caduceus on all 5 DNALongBench tasks. It also reports beating the task-specific expert models on enhancer-target and eQTL with linear probing only (self-reported) — [summary](https://arxiv.org/pdf/2601.03019).
- **BarcodeMamba**:
  - matches BarcodeBERT on Canadian invertebrate species classification with 8.3% of its parameters
  - 99.2% species-level linear-probe accuracy — [summary](https://arxiv.org/html/2412.11084v1)
  - BarcodeMamba+ reaches 81.7% species accuracy on MycoAI vs 72.6% for a CNN, and 88.9% at 49.2M parameters — [summary](https://arxiv.org/abs/2512.15931v1)
- **PhyloGPN** (83M CNN) reports SOTA on 5 of 7 BEND tasks — [summary](https://arxiv.org/html/2503.03773v1).
- **GPN-Star** reports SOTA on coding and non-coding variant-interpretation benchmarks. Its vertebrate model beats GPN-MSA on the same alignment — [summary of bioRxiv](https://www.biorxiv.org/content/10.1101/2025.09.21.677619.full.pdf).
- **HybriDNA** reports SOTA on 33 BEND/GUE/LRB datasets, with gains from 300M→3B→7B. 300M-specific numbers were not retrieved — [summary](https://arxiv.org/abs/2502.10807v2).
- **ModernGENA** "ranks among the top-performing models" on the NT benchmark with a "strong efficiency–quality trade-off". No numbers were available on the pages read — [ICLR](https://iclr.cc/virtual/2026/10014529).

### Inferences
- A 1–10M bidirectional linear-recurrent model (kron-dna) can credibly target the NT-benchmark and Genomic-Benchmarks tables. HAD (GDN, 1.1M) and JanusDNA (Mamba, 2M) are the published bars. Splice-site tasks are where tiny students still lose to large 6-mer transformers.
- The strongest "small beats large" evidence uses either distillation (HAD) or bidirectional all-token objectives (JanusDNA), not plain MLM. Both are cheap to try with KronLSTM layers.
- Reporting long-range results (DNALongBench eQTL, enhancer-target) and zero-shot VEP would differentiate kron-dna more than another NT-benchmark row. Those are the axes where recurrent/linear models have a structural advantage and where NTv3, JanusDNA and DNAChunker compete.

### Gaps
- No independent reproduction was found of the headline numbers for HAD, JanusDNA, Omni-DNA, MergeDNA or NTv3-8M.
- No HybriDNA-300M per-task numbers were retrieved.
- NTv3's 106-task numbers were not retrieved.
- No GenomicLRB (InstaDeep long-range benchmark) tables for 2025–2026 small models were retrieved.

## Q4. What weaknesses, independent re-evaluations and criticisms apply?

### Takeaway
Independent 2025–2026 evaluations consistently find five things:
- randomly initialised models are strong baselines
- character-level models do as well as larger k-mer/BPE models
- leaderboard rankings are unstable and scale helps only modestly
- expert supervised models beat DNA language models on long-range tasks
- alignment-based methods beat alignment-free language models on causal-variant prediction

Self-reported tables also contain internal inconsistencies.

### Cited Findings
- **Vishniakov et al.**, "Genomic Foundationless Models: Pretraining Does Not Promise Performance" (bioRxiv Dec 2024), presented at ICLR 2026 as "Tokenization to Transfer: Do Genomic Foundation Models Learn Good Representations?" (arXiv 2601.05531) — [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.12.18.628606.full.pdf); [ICLR](https://www.iclr.cc/virtual/2026/poster/10011551); [arXiv](https://arxiv.org/abs/2601.05531v1) [summary]:
  - covers 7 genomic foundation models (GFMs) across 52 tasks, each against a randomly initialised counterpart
  - random initialisation gives "surprisingly strong baselines"
  - character-token models often match or beat larger pretrained k-mer/BPE models
  - larger embedding dimensions in random models often beat the pretrained ones
  - the GFMs show limited sensitivity to clinically relevant variants
  - code: m42-health/gfm-random-eval
- **GENEB** (Ledneva, Nuridinov, Kuznetsov; arXiv 2606.04525, ICML 2026) [paper abstract]:
  - probes 40 GFMs on 100 tasks in 13 categories with frozen-representation probing, including few-shot
  - aggregate leaderboards are unstable and rankings vary sharply by category
  - "scale yields only modest and inconsistent gains"
  - architecture and pretraining alignment often matter more than parameter count — [arXiv](https://arxiv.org/abs/2606.04525)
- **Feng et al.**, "Benchmarking DNA foundation models for genomic and genetic tasks" (Nature Communications 2025; bioRxiv 2024) [summary]:
  - zero-shot embeddings from DNABERT-2, NTv2, HyenaDNA, Caduceus-Ph and GROVER on 57 datasets
  - mean pooling beats summary-token and max pooling
  - general models are competitive on pathogenic-variant identification but weaker on gene-expression prediction
  - specialised models win on causal-QTL tasks — [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12663285/)
- **DNALongBench** (Cheng et al., Jian Ma lab; Nature Communications 2025): five tasks spanning up to 1 Mb. Expert models consistently outperform the fine-tuned DNA foundation models tested (HyenaDNA, Caduceus-Ph, Caduceus-PS) on all five — [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.01.06.631595.full.pdf); [Nat Comm record](https://ideas.repec.org/a/nat/natcom/v16y2025i1d10.1038_s41467-025-65077-4.html).
- **TraitGym** (Benegas, Eraslan, Song; bioRxiv 2025) — [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.02.11.637758v2) [summary]:
  - CADD and GPN-MSA are best on Mendelian and complex-disease causal variants
  - Enformer and Borzoi are best on complex non-disease traits
  - Evo2 improves with scale but trails alignment-based models and is weakest on enhancer variants
  - ensembles help
  - the paper introduced GPN-Promoter
- **Paper-internal inconsistencies**:
  - JanusDNA's text claims are higher than its own bolding: 12 vs 11 NT tasks, 3 vs 2 Genomic-Benchmarks tasks, and "8 of 9" eQTL tissues where all 9 are bolded
  - it gives conflicting attention-head counts (8 vs 4), batch sizes and mid-attention placement
  - its hg38 vs GRCh37 reference is ambiguous — [paper](https://arxiv.org/html/2505.17257v4)
  - HAD's introduction says 11M parameters, the rest of the paper 1.1M — [paper](https://arxiv.org/html/2505.20836v1)
  - ModernGENA's card text says 377M, the metadata 0.1B — [card](https://huggingface.co/AIRI-Institute/moderngena-base)
- **Limited baselines**: dnaHNet compares only against StripedHyena2 and Transformers. Caduceus, HyenaDNA and NT are not benchmarked, and the downstream margins are modest — [paper](https://arxiv.org/html/2602.10603v2).
- **Reproducibility of baselines**: the DNABERT-2 appendix reports that the official HyenaDNA checkpoint did not reproduce its published EMP results — [summary](https://arxiv.org/pdf/2306.15006).
- **Practical and licensing caveats**:
  - NTv3 is non-commercial and forbids training commercial models on its outputs — [card](https://huggingface.co/InstaDeepAI/NTv3_8M_pre)
  - NTv2 is CC-BY-NC-SA-4.0 — [card](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-50m-multi-species)
  - Evo2-1B needs Hopper/FP8 — [GitHub](https://github.com/ArcInstitute/evo2)
  - GENERator's 6-mer tokenizer breaks on lengths that are not a multiple of 6 — [card](https://huggingface.co/GenerTeam/GENERator-v2-eukaryote-1.2b-base)
  - Omni-DNA's 20M/60M models trail baselines and its scaling is non-monotonic — [paper](https://arxiv.org/html/2502.03499v1)
  - HAD loses to its teacher on splice sites — [paper](https://arxiv.org/html/2505.20836v1)

### Inferences
- For credibility, kron-dna's evaluation should include:
  - a randomly initialised KronLSTM of the same size on every downstream task (the Vishniakov control)
  - per-category reporting rather than one average (GENEB)
  - at least one long-range suite (DNALongBench) and one causal-variant suite (TraitGym or the ClinVar/BEND VEP tasks)
  - re-running the key baselines (Caduceus-Ph, HAD-style GDN, JanusDNA, Omni-DNA-116M, NTv2-50M or NTv3-8M) under one fine-tuning protocol instead of copying numbers across papers
- Single-nucleotide tokenization is well supported by independent work (Vishniakov) and by 2025–2026 model choices (NTv3, GENERanno, JanusDNA, HAD). It fits a recurrent model with linear cost in sequence length.

### Gaps
- The exact numeric tables were not read for Vishniakov/Tokenization-to-Transfer, GENEB, Feng et al., DNALongBench or TraitGym. Only abstracts and summaries were seen.
- No independent re-evaluation was found that specifically includes NTv3, JanusDNA, HAD, MergeDNA or Omni-DNA. GENEB's model list (40 GFMs) was not retrieved, so it may include some of them.
