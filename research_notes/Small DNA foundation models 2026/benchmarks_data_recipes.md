# Benchmarks, pretraining data, scaling evidence and training recipes for small DNA foundation models (2025–2026)

Scope note: written for "kron-dna" (small KronLSTM-based DNA LM, likely coding-DNA focus, about 6 single-A100 jobs for days to weeks). Source labels: **[paper]** = read in the primary paper (full text or abstract); **[dataset card]** / **[model card]** = Hugging Face card; **[secondary]** = third-party summary or search snippet, not checked against the primary source; **[my arithmetic]** = derived by me from cited numbers. Research cut-off: 2026-11-09.

## Q1. Which benchmark suites are current and respected in 2025–2026, what do they measure, which are saturated or leaky, and which suit coding-DNA models?

### Takeaway
No single suite is the accepted standard in 2026. The credible pattern is a battery: the **revised** NT benchmark (18 human tasks, chromosome held out) plus GUE for comparability; GENEB-style frozen-embedding probing with few-shot regimes; at least one counterfactual or variant task (DART-Eval, TraitGym, ClinVar/BRCA1); and for coding DNA, zero-shot DMS fitness from DNA, codon/synonymous tasks, and gene-level tasks. Promoter, coding/non-coding, species classification, CpG methylation, BRCA1/2 classification and Genomic Benchmarks are close to saturated or barely discriminative.

### Cited Findings

**Classic short-sequence classification suites**
- **NT benchmark, revised version:** 18 tasks (promoters ×3, enhancers ×2, splice sites ×3, 10 ENCODE K562 histone marks), 300–1,000 bp inputs. After peer review the authors made all data human, replaced randomly generated negatives with real genome chunks that lack the element, and gave every dataset a chromosome-held-out test set. The card for the original version tells users to move to the revised one. Sources: GENCODE V44 (splice sites), EPD (promoters), ENCODE SCREEN (enhancers) **[dataset card]** — [NT revised card](https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks_revised); [original card warning](https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks)
- The common NT protocol is MCC with 10-fold cross-validation. ModernGENA (bioRxiv, April 2026) argues that absolute differences on the NT benchmark are often modest, that small gaps are hard to read given fine-tuning variance, and that the benchmark needs longer-range tasks alongside it. Pretraining versus random initialisation of the same 377M model raised mean MCC from 0.420 to 0.671, with gains on all 18 tasks **[paper]** — [ModernGENA](https://www.biorxiv.org/content/10.64898/2026.04.21.719816v1.full)
- **GUE (from DNABERT-2):** 28 datasets in 7 task families (promoters, human and mouse TF binding, splice sites, yeast epigenetic marks (EMP), virus/COVID classification) **[paper]** — [Karpinsky et al. 2026](https://arxiv.org/html/2606.30140v1); copy on the Hub: [leannmlindsey/GUE](https://huggingface.co/datasets/leannmlindsey/GUE) **[dataset card]**. "GUE+" could not be characterised (see Gaps).
- **Genomic Benchmarks:** 8 classification tasks **[secondary, Bio-xLSTM repo README via search]** — [Bio-xLSTM](https://arxiv.org/pdf/2411.04165). The JanusDNA authors describe it as quite saturated, with models close together **[secondary, search snippet]** — [JanusDNA](https://arxiv.org/pdf/2505.17257). Hub copy: [katielink/genomic-benchmarks](https://huggingface.co/datasets/katielink/genomic-benchmarks), Apache-2.0 **[dataset card]**
- **BEND (ICLR 2024):** biologically meaningful human tasks (gene finding, enhancer annotation, chromatin, histone, CpG methylation, variant effects). Headline finding: LM embeddings approach expert methods on some tasks but carry little long-range information **[paper, 2023 reference]** — [BEND](https://arxiv.org/abs/2311.12570)
- **BEND 2.0 (reportedly arXiv, 2 Nov 2026, Bjerregaard et al.):** 8 tasks on 5 experimental datasets that test whether predictions can guide sequence design. Reported findings: linear readouts on frozen representations match or beat models' native outputs and transfer across assays and genes; strong average-effect prediction can hide weak context-specific performance; good prediction of combined mutations does not imply good prediction of their interactions **[secondary, unverified; arXiv listing not found]** — [aiweekly summary](https://aiweekly.co/alerts/bend-20-benchmark-tests-if-dna-foundation-models-can-guide-design)

**2025–2026 meta-benchmarks**
- **GENEB (ICML 2026, arXiv 2606.04525, v4 Aug 2026):** frozen representations of 40 genomic FMs, 100 tasks, 13 categories. Protocol: logistic regression on mean-pooled embeddings, MCC, 1-shot / 10-shot / full data, 5 seeds. Findings: aggregate leaderboards are unstable; rankings change sharply by category; few-shot evaluation changes the best model in 8 of 13 categories **[paper]** — [GENEB](https://arxiv.org/html/2606.04525v4)
- GENEB difficulty: promoters, coding/non-coding and species classification are "consistently easier" (coding/non-coding reaches 0.90–0.91 MCC with full supervision). DNA methylation (best 0.440), lncRNA, virus/phage (1-shot MCC 0.027) and regulatory tasks are hard; 28 of 100 tasks have mean MCC below 0.35, and the authors suspect some are noisy **[paper]** — [GENEB](https://arxiv.org/html/2606.04525v4)
- **Genomic Touchstone (bioRxiv, June 2025):** 36 tasks, 88 datasets, 5.34 Gbp, across DNA, RNA and protein. Protein tasks are recast as DNA by using each protein's CDS. 34 models from 3.3M to 2.5B parameters, all fine-tuned. No model dominates (best average rank 7.3). CpG methylation (top F1 0.957) and BRCA1/2 classification (top F1 0.893) are called saturated and poorly discriminative; epigenetic marks are hard (top F1 0.380) **[paper]** — [Genomic Touchstone](https://www.biorxiv.org/content/10.1101/2025.06.25.661622v1.full)
- **Feng et al., Nat. Commun., 28 Nov 2025:** zero-shot embeddings of DNABERT-2, NT-v2, HyenaDNA, Caduceus-Ph and GROVER on classification, gene expression, variant effect and TAD tasks. Mean token pooling "consistently and significantly" beats other pooling. The models are competitive for pathogenic-variant identification but weaker than specialised models for gene expression and causal QTLs **[paper abstract]** — [Feng et al. 2025](https://www.nature.com/articles/s41467-025-65823-8)

**Regulatory and long-range suites**
- **DART-Eval (NeurIPS 2024 D&B; PMC Aug 2025):** five regulatory task families: cCRE vs dinucleotide-shuffled background, motif sensitivity (1,443 HOCOMOCO motifs), cell-type-specific ATAC elements, DNase signal regression, and caQTL/dsQTL variant effects. Each is run zero-shot, probed and LoRA fine-tuned, with compositionally matched negatives (dinucleotide shuffles, GC-matched backgrounds) **[paper]** — [DART-Eval PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12340898/); [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/71998bfc3217ffe1cca1ee084dfadadd-Paper-Datasets_and_Benchmarks_Track.pdf)
- **Genomics Long-Range Benchmark (InstaDeep, 2024):** variant effect (eQTL, ClinVar), gene expression (CAGE, bulk RNA), regulatory elements and chromatin features. License CC-BY-NC-SA-4.0 **[dataset card]** — [GLRB](https://huggingface.co/datasets/InstaDeepAI/genomics-long-range-benchmark). At 12 kbp fine-tuning, Caduceus scored VEP-eQTL AUROC 0.68 and VEP-ClinVar 0.75, against 0.72 and 0.75 for NT-v2 **[paper]** — [Leveraging SSMs in long-range genomics](https://arxiv.org/html/2504.06304v1)
- **DNALongBench (Nat. Commun., 18 Nov 2025):** five tasks with dependencies up to 1 Mb (enhancer–target gene, eQTL, 3D contacts, regulatory activity, transcription initiation). Fine-tuned HyenaDNA and Caduceus still trail expert models **[paper abstract]** — [DNALongBench](https://www.nature.com/articles/s41467-025-65077-4)

**Variant-effect sets**
- **TraitGym (Benegas, Eraslan, Song; bioRxiv Feb 2025):** 338 causal non-coding variants for 113 Mendelian traits (OMIM) and 1,140 fine-mapped non-coding variants for 83 complex traits (UK Biobank), each with 9 matched controls (chromosome, consequence, TSS distance; plus MAF and LD for complex traits). Metric is AUPRC. Hub data under MIT licence, with a leaderboard Space **[dataset card + secondary]** — [TraitGym dataset](https://huggingface.co/datasets/songlab/TraitGym); [preprint](https://www.biorxiv.org/content/10.1101/2025.02.11.637758.full.pdf)
- TraitGym results: alignment-based models (CADD, GPN-MSA) do well on Mendelian and complex-disease traits; Enformer and Borzoi do better on complex non-disease traits; Evo 2 improves with scale but struggles on enhancer variants **[secondary, third-party review summaries; not checked against paper tables]** — [Moonlight review](https://www.themoonlight.io/review/biorxiv/benchmarking-dna-sequence-models-for-causal-regulatory-variant-prediction-in-human-genetics)
- **ClinVar / BRCA1 in Evo 2 (Feb 2025):** on ClinVar coding SNVs (N=14,319), Evo 2 40B and 7B rank 4th and 5th, behind AlphaMissense, ESM-1b and GPN-MSA. Evo 2 leads on non-SNV coding variants, noncoding SNVs (N=34,761) and SpliceVarDB. BRCA1 SGE has 2,077 coding and 1,125 noncoding SNVs; a supervised probe on embeddings reaches AUROC 0.94 on coding SNVs **[paper]** — [Evo 2 preprint](https://www.biorxiv.org/content/10.1101/2025.02.18.638918v1.full)
- The Benegas review warns of circularity: ClinVar and HGMD labels are partly derived from evidence that predictors also use, so clinical benchmarks can overstate performance **[paper]** — [Benegas et al. review](https://arxiv.org/html/2407.11435v2)

**Coding-DNA and gene-level tasks (most relevant to kron-dna)**
- **Zero-shot DMS fitness from DNA:** Evo 2 is "competitive with state-of-the-art autoregressive protein language models" on bacterial and human protein DMS **[paper]** — [Evo 2](https://www.biorxiv.org/content/10.1101/2025.02.18.638918v1.full). dnaHNet (Feb 2026) gives **small-scale reference points** on MaveDB, zero-shot Spearman: StripedHyena2 0.264 / 0.292 / 0.308 / 0.311 at 1e19 / 2e19 / 4e19 / 7.1e19 FLOPs; Transformer 0.135–0.156 over 8e18–8e19 FLOPs; dnaHNet 0.260–0.327 **[paper]** — [dnaHNet](https://arxiv.org/html/2602.10603v2)
- **Gene essentiality (DEG), zero-shot AUROC at small scale:** StripedHyena2 0.538–0.576 and dnaHNet 0.554–0.605 over about 8e18–7e19 FLOPs **[paper]** — [dnaHNet](https://arxiv.org/html/2602.10603v2)
- **Exon classification across 8 held-out species:** Evo 2 embedding classifier AUROC 0.82–0.99, ahead of NT and Evo 1 **[paper]** — [Evo 2](https://www.biorxiv.org/content/10.1101/2025.02.18.638918v1.full)
- **Boshar et al. (Bioinformatics 2024):** curated 5 datasets that pair proteins with their true CDS. gLMs trained mostly on non-coding DNA are competitive with, and sometimes beat, protein LMs. True CDS work better than sampled codons. A 3-mer NT-50M beats its 6-mer counterpart on protein tasks while keeping genomics performance **[secondary summary of paper]** — [InstaDeep page](https://instadeep.com/research/paper/exploring-genomic-language-models-on-protein-downstream-tasks/)
- **Codon-level tasks:** CodonFM (Arc + NVIDIA, Encodon 80M/600M/1B, 131M CDS from 22,000 species) is evaluated on synonymous and missense variant classification, mRNA translation efficiency and protein abundance, and reports accuracy rising with model and data size **[secondary, press/blog]** — [Arc news](https://arcinstitute.org/news/codonfm-foundation-model-for-codons); [NVIDIA blog](https://developer.nvidia.com/blog/introducing-the-codonfm-open-model-for-rna-design-and-analysis). CodonMamba (bioRxiv Aug 2026) uses a 12-task mRNA benchmark and finds protein LMs clearly worse on codon-sensitive tasks **[paper abstract via search]** — [CodonMamba](https://www.biorxiv.org/content/10.64898/2026.08.24.746601v1.full). A 2026 comparison of CaLM, EnCodon and CodonTransformer on 9 phenotype tasks found no consistent winner, with **simple sequence features competitive** in several settings **[secondary, search snippet]** — [Codon LM assessment](https://www.biorxiv.org/content/10.64898/2026.08.11.744178v1.full). In COMET, DNA and RNA models came close to codon-specific models on some tasks **[secondary]** — [COMET](https://arxiv.org/pdf/2412.10347)
- **Gene-type and taxonomic classification:** Gener Tasks (6 gene types plus non-gene controls across 6 eukaryotic groups; MIT licence) **[dataset card]** — [GenerTeam/gener-tasks](https://huggingface.co/datasets/GenerTeam/gener-tasks)
- In GENEB, coding/non-coding leaders are GENERator-Eukaryote-3B (0.904), LucaOne (0.901) and MutBERT-86M (0.894); MutBERT is best at 10-shot (0.748). The task is near ceiling **[paper]** — [GENEB](https://arxiv.org/html/2606.04525v4)

### Inferences
- For a coding-DNA model, the tasks that discriminate between models are (a) zero-shot DMS or MaveDB Spearman from DNA likelihoods, (b) ClinVar coding SNVs **with ESM-2/ESM-1b LLR and AlphaMissense/PhyloP as reference lines**, (c) codon- and synonymous-sensitive tasks (translation efficiency, protein abundance, synonymous variant effects), where protein LMs are structurally blind and a DNA model can earn a real win, (d) splice sites and exon detection, and (e) gene essentiality. Coding/non-coding classification is near ceiling and only works as a sanity check.
- dnaHNet's numbers are the most useful "what to expect" anchor for a sub-300M model trained with roughly 1e19–1e20 FLOPs on prokaryotic DNA: MaveDB Spearman about 0.26–0.33 and DEG AUROC about 0.55–0.60. A KronLSTM at matched FLOPs could be reported against these curves (same data: a GTDB-style OpenGenome subset).
- For NT revised, GUE and Genomic Benchmarks, report mean ± SD over seeds or folds and a random-init same-architecture control. ModernGENA and Karpinsky et al. both show gaps that sit within fine-tuning noise.

### Gaps
- "GUE+" and any official "ProteinGym-DNA" or CDS-level ProteinGym track: I found no primary source describing either. DMS-from-DNA evaluations appear to be ad hoc, per paper (Evo 2's set, dnaHNet's MaveDB set).
- BEND 2.0: primary paper and arXiv ID not found; only one secondary write-up.
- TraitGym exact AUPRC tables not retrieved; the CADD/GPN-MSA vs gLM ranking comes from secondary summaries.
- No source directly tests whether GUE or NT tasks can be solved from GC content or k-mer counts alone. DART-Eval only notes that regulatory sequences are GC-rich and controls for it.
- GENEB's mapping of its 100 tasks to source benchmarks (its Appendix B) was not read.

## Q2. Do DNA language models beat simple baselines (one-hot CNNs, k-mer models, supervised models) in independent evaluations, and what should a small model target?

### Takeaway
Independent 2024–2026 evaluations agree that on regulatory and epigenomic tasks, pretrained DNA LMs (frozen or fine-tuned) rarely beat well-tuned one-hot CNNs trained from scratch, and are clearly worse on counterfactual variant tasks. Pretraining helps transformers a lot relative to their own random init, but much less, if at all, relative to strong CNNs. A small model should therefore target zero-shot likelihood tasks (variant/DMS fitness, coding constraint), sample efficiency (few-shot probing), and codon-level signal, and should always report a one-hot CNN and a random-init twin.

### Cited Findings
- **Tang, Somia, Yu, Koo (Genome Biology, 14 July 2025):** across six functional-genomics tasks, probing pretrained gLM representations gives no substantial advantage over conventional ML on one-hot sequences. Highly tuned supervised one-hot models are competitive or better **[paper abstract]** — [Europe PMC record / Genome Biol.](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-025-03674-8). A CNN on Sei embeddings did best on lentiMPRA **[secondary, search snippet]** — same link
- **DART-Eval:** ab initio CNNs match or beat most DNALMs on cell-type-specific elements. Fine-tuned DNALMs only roughly match ChromBPNet on DNase regression. On caQTL/dsQTL variants ChromBPNet reaches AUROC about 0.77 / 0.89 against about 0.65 / 0.67 for the best fine-tuned DNALM. Embedding-based zero-shot motif detection is near chance (about 40–60%), while likelihood scoring detects most motifs. The authors conclude DNALMs "perform particularly poorly on counterfactual prediction tasks" despite far more compute **[paper]** — [DART-Eval](https://pmc.ncbi.nlm.nih.gov/articles/PMC12340898/)
- **Karpinsky, Mozziconacci, Delcey (arXiv, 29 June 2026), GUE only:** mean MCC×100 across 28 datasets: DNABERT-2 66.88 pretrained vs 45.97 random-init. U-Net 57.82 vs 57.45; ConvNova 61.08 vs 61.30, so pretraining makes no difference for CNNs. Pretraining helps DNABERT-2 most on human TF binding (−37 points without it), splice sites (−29) and promoters (−23), and least on virus classification (−3). They recommend always comparing against a random-init same-architecture model and against CNNs. Single runs, no variance reported **[paper]** — [Karpinsky et al.](https://arxiv.org/html/2606.30140v1)
- **Feng et al. 2025:** foundation models generally underperform a simple baseline CNN on multispecies and human/multispecies epigenetic tasks, and show clear gains on some promoter tasks and yeast epigenetic marks **[secondary, search snippet of paper]** — [Feng et al.](https://www.nature.com/articles/s41467-025-65823-8)
- **Genomic Touchstone:** CNN and LSTM baselines are competitive on several annotation tasks; an LSTM is best on mean ribosome load (Spearman 0.811) **[paper]** — [Touchstone](https://www.biorxiv.org/content/10.1101/2025.06.25.661622v1.full)
- **ConvNova (arXiv Feb 2025):** a purpose-built CNN beats transformer and SSM DNA FMs on 5 Genomic Benchmarks tasks, and the authors argue CNNs have been under-compared **[secondary, search snippet]** — [Revisiting convolution](https://arxiv.org/html/2502.18538)
- **Frozen vs fine-tuned (Datta et al., arXiv 5 Aug 2026):** on promoter tasks, frozen probes recover 95–100% of fine-tuned performance; on splice sites only 60–88%. Frozen embeddings are competitive on coding-region and species tasks in Genomic Benchmarks. Local signal is present but "not always accessible through final pooled embeddings" **[paper abstract]** — [Frozen but not always accessible](https://arxiv.org/abs/2608.05329)
- **Benegas et al. review ("Genomic language models: opportunities and challenges", arXiv v2 Sept 2024):** cites two studies in which gLMs with frozen embeddings "generally did not outperform non-gLM baselines" on human tasks. Asks for full fine-tuning evaluation, standardised extensible benchmarks, and treating gLMs as "another useful modeling tool" rather than a breakthrough **[paper]** — [Benegas et al.](https://arxiv.org/html/2407.11435v2)
- **Where gLMs do win:** zero-shot variant scoring of coding and splice variants (Evo 2 leads ClinVar non-SNV coding, noncoding SNVs and SpliceVarDB) **[paper]** — [Evo 2](https://www.biorxiv.org/content/10.1101/2025.02.18.638918v1.full). Feng et al. find FMs competitive at pathogenic-variant identification **[paper abstract]** — [Feng et al.](https://www.nature.com/articles/s41467-025-65823-8)
- **Data relevance over size:** a 2025 study on evolutionarily implausible sequences reports that GENERanno matched Evo2-7B with 14× fewer parameters and 12× less pretraining data, and argues functional generalisation depends more on data relevance and quality than on size **[secondary, search snippet]** — [arXiv 2506.10271](https://arxiv.org/html/2506.10271v3)
- **Small models are competitive in GENEB:** MutBERT (86M) is the top sub-100M model in 8 of 13 categories (macro-MCC 0.529) and beats eccDNAMamba (537M) by +0.110 **[paper]** — [GENEB](https://arxiv.org/html/2606.04525v4)

### Inferences
- Honest baselines for kron-dna, in rough order of importance:
  1. One-hot CNN trained from scratch per task (ConvNova or a ChromBPNet-like CNN for regulatory tasks; a small dilated CNN for classification).
  2. Random-init KronLSTM with the same fine-tuning budget, which isolates the value of pretraining.
  3. k-mer or codon-usage logistic regression or GBM, especially for codon tasks, where "simple sequence features" are known to compete.
  4. Size-matched public DNA LMs: NT-v2-50M/100M, MutBERT-86M, Caduceus-PS (~7.7M), DNA-xLSTM (0.5–4M), ModernGENA-135M, Evo 2 1B as an upper reference.
  5. For coding tasks, ESM-2 (8M–150M) on the translated protein and a codon LM (CaLM or Encodon-80M).
- Claims a small model can credibly aim for: (a) beating size-matched DNA LMs at equal FLOPs on zero-shot coding variant and DMS tasks; (b) better few-shot (1- and 10-shot) probing than one-hot CNNs, where pretraining should matter most; (c) beating protein LMs on synonymous and codon-sensitive tasks; (d) efficiency (throughput or long context per FLOP), a fair axis for a linear-recurrent model. Claiming wins over supervised CNNs on regulatory or epigenomic tasks is unlikely and would invite scrutiny.
- Report zero-shot (likelihood), frozen probing (linear, mean-pooled plus best layer) **and** full or LoRA fine-tuning. The literature shows the three can give different rankings.

### Gaps
- Tang & Koo full numbers and per-task tables not retrieved (abstract only).
- I found no independent evaluation designed specifically around coding-DNA LMs vs one-hot CNNs. The codon-LM comparisons are mostly by model authors, plus one 2026 independent comparison seen only as a snippet.
- The "GENERanno vs Evo2-7B" claim is from a search snippet; the paper's tables were not checked.

## Q3. Which pretraining datasets exist (sizes, licences, Hugging Face availability), and what is the evidence on data composition?

### Takeaway
Plenty of open data exists for a small model. OpenGenome2 (8.8T bp, Apache-2.0) ships with explicit per-subset token counts and Evo 2's sampling weights, and its prokaryotic and mRNA subsets alone hold hundreds of billions of mostly coding nucleotides. Composition evidence consistently favours (i) multi-species over human-only, (ii) gene- or function-centric windows over whole genomes, even though whole-genome data gives lower pretraining loss, and (iii) down-weighting repeats. All of these suit a coding-DNA focus.

### Cited Findings

**Large general corpora**
- **OpenGenome2** (Arc, used for Evo 2): about 8.8T bp, 5.52 TB, FASTA plus JSONL formats, Apache-2.0. Phase-1 subsets with tokens and Evo 2 dataloader weight:

  | Subset | Tokens | Weight |
  |---|---|---|
  | Metagenomics (MGD) | 854B | 24% |
  | Eukaryotic 5 kb windows, stitched | 405B | 35% |
  | GTDB v220 + IMG/PR | 351B | 18% |
  | Eukaryotic mRNAs, stitched | 99B | 9% |
  | Eukaryotic mRNAs (Ensembl/NCBI) | 89B | 9% |
  | IMG/VR | 34B | 3% |
  | ncRNA | 19B | 2% |
  | Organelles | 3B | 0.5% |
  | EPDnew promoters | 0.11B | 0.02% |

  Phase 2 adds whole NCBI eukaryote genomes (for example Animalia 4,907B tokens at 36%) **[dataset card]** — [arcinstitute/opengenome2](https://huggingface.co/datasets/arcinstitute/opengenome2)
- Evo 2 methods: 113,379 prokaryotic genomes (357B nt); 15,032 eukaryotic genomes (6.98T nt); 41,253 metagenomes and MAGs (854B nt); 32,240 organelle genomes (2.82B nt); eukaryotic genic subsets 602B nt. Genic windows are ±5 kb around exons; 50% of GTDB, transcript and genic sequences are reverse-complemented; eukaryote-infecting viruses are excluded (high perplexity on them confirmed the exclusion) **[paper]** — [Evo 2](https://www.biorxiv.org/content/10.1101/2025.02.18.638918v1.full)
- **OpenGenome (Evo 1):** about 300B nt from GTDB v214.1, IMG/VR v4 and IMG/PR; eukaryotic viruses excluded "for safety considerations" **[paper, 2024 reference]** — [Evo](https://www.biorxiv.org/content/10.1101/2024.02.27.582234v2.full). On the Hub as stage 1 (8k context) and stage 2 (131k), Apache-2.0 **[dataset card]** — [LongSafari/open-genome](https://huggingface.co/datasets/LongSafari/open-genome)
- **dnaHNet's GTDB subset:** 17.6M sequences, 144B nt, 85,205 prokaryotes, ≤8,192 nt each, filtered following OpenGenome **[paper]** — [dnaHNet](https://arxiv.org/html/2602.10603v2)
- **NT multispecies:** about 850 NCBI genomes, 174B nt (about 29B 6-mer tokens); 50 species each held out for validation and test; 6 kbp and 12 kbp chunk configurations **[dataset card + secondary]** — [InstaDeepAI/multi_species_genomes](https://huggingface.co/datasets/InstaDeepAI/multi_species_genomes); [NT model card / InfoQ summary](https://huggingface.co/InstaDeepAI/nucleotide-transformer-2.5b-multi-species)
- **Human reference:** GRCh38.p14, chunked **[dataset card]** — [InstaDeepAI/human_reference_genome](https://huggingface.co/datasets/InstaDeepAI/human_reference_genome). I found no 1000 Genomes pretraining set on the Hub (the `InstaDeepAI/1000_genomes` ID does not exist).
- **Plants:** 48 species, 2 each held out for validation and test **[dataset card]** — [InstaDeepAI/plant-multi-species-genomes](https://huggingface.co/datasets/InstaDeepAI/plant-multi-species-genomes)
- **Metagenomic OMG:** 3.1T bp from MGnify and IMG, stored mixed-modality (amino acids for CDS, nucleotides for intergenic). OG subset with taxonomy, OMG_prot50 subset. CC-BY-SA-4.0 **[dataset card]** — [tattabio/OMG](https://huggingface.co/datasets/tattabio/OMG)
- **Gene-centric eukaryotic corpus (GENERator):** 386B nt of RefSeq gene regions (gene bodies, introns, UTRs, nearby regulatory DNA) vs an all-sequence corpus of about 2T nt **[paper]** — [GENERator v5](https://arxiv.org/html/2502.07272v5)
- **ModernGENA corpus:** 443 vertebrate RefSeq assemblies (353.6 Gbp), trained on windows from −16 kb to +8 kb around every gene and pseudogene TSS, both strands, chromosome-held-out validation (human chr8, 20, 21) **[paper]** — [ModernGENA](https://www.biorxiv.org/content/10.64898/2026.04.21.719816v1.full)
- **DNABERT-2 corpus:** about 32.5B nt from 135 species **[paper]** — [Karpinsky et al.](https://arxiv.org/html/2606.30140v1)

**CDS / codon corpora (coding-DNA focus)**
- CodonFM: 131M protein-coding sequences from 22,000 species (NCBI RefSeq) **[secondary, press]** — [Arc news](https://arcinstitute.org/news/codonfm-foundation-model-for-codons). Predecessor cdsFM: more than 60M CDS from 5,000 species **[secondary]** — [cdsFM preprint](https://www.biorxiv.org/content/10.1101/2024.10.10.617568.full.pdf)
- SynCodonLM: more than 66M CDS from NCBI RefSeq, trained with synonymous-codon-constrained masking **[secondary, search snippet]** — [SynCodonLM](https://www.biorxiv.org/content/10.1101/2025.08.19.671089v1.full.pdf)
- Hub CDS datasets are small or niche: CodonTransformer (1,001,197 DNA–protein pairs, 164 organisms, CC-BY-4.0) **[dataset card]** — [adibvafa/CodonTransformer](https://huggingface.co/datasets/adibvafa/CodonTransformer); CodonTranslator-data (CC-BY-4.0, Apr 2026) **[dataset card]** — [alegendaryfish/CodonTranslator-data](https://huggingface.co/datasets/alegendaryfish/CodonTranslator-data). I found no large, ready-made RefSeq-wide CDS pretraining set on the Hub.

**Composition evidence**
- **Function-centric beats whole-genome:** in GENERator, all-sequence training gives *lower* pretraining loss, but gene-centric training performs better downstream **[paper; numbers in Fig. S6, not retrieved]** — [GENERator](https://arxiv.org/html/2502.07272v5)
- Evo 2 focuses pretraining weighting on genic windows to learn functional elements, and adds whole genomes only at midtraining. The ablations are in its Appendix B.1/B.2 **[paper; ablation results not retrieved]** — [Evo 2](https://www.biorxiv.org/content/10.1101/2025.02.18.638918v1.full)
- **Repeats:** Evo 2 down-weights repetitive positions in the loss by 0.1 and used lowercase (repeat-masked) input for the first 3T tokens **[paper]** — [Evo 2](https://www.biorxiv.org/content/10.1101/2025.02.18.638918v1.full). The Benegas review notes about 50% of the human genome is repetitive and only about 3.3% is significantly constrained. It recommends down-weighting or down-sampling repeats, reporting perplexity separately for repetitive and non-repetitive regions, deduplicating, and base-level loss weighting by functional evidence **[paper]** — [Benegas et al.](https://arxiv.org/html/2407.11435v2)
- **Species diversity (GENEB controlled pairs):** multi-species vs human-only gives +0.018 macro-MCC on average (chromatin accessibility +0.076, splice sites +0.065; virus/phage favours human-only, −0.025). Multi-species vs microbial-only gives +0.084 (splice sites +0.222). Eukaryotic-gene pretraining (GENERator) vs broad multi-species gives +0.063, from a single pair **[paper]** — [GENEB](https://arxiv.org/html/2606.04525v4)
- Genomic Touchstone: more diverse pretraining data consistently improves performance; NT-2.5B multi-species (850 species) consistently beats its human-only variant **[paper]** — [Touchstone](https://www.biorxiv.org/content/10.1101/2025.06.25.661622v1.full)
- NT (Nature Methods 2024/25): the 2.5B multispecies model matches or beats the 2.5B 1000-Genomes model on human-assay tasks **[secondary summary of paper]** — [NT paper PDF](https://people.stat.sc.edu/hoyen/STAT718/Papers/Nucleotide_Transformer.pdf); [InfoQ](https://www.infoq.com/news/2024/12/instadeep-nucleotide-transformer/)
- DART-Eval recommends balancing training across functional element classes, or training on functionally related subsets of the genome **[paper]** — [DART-Eval](https://pmc.ncbi.nlm.nih.gov/articles/PMC12340898/)

### Inferences
- For a coding-DNA KronLSTM, a cheap and well-motivated mix is: OpenGenome2's GTDB + IMG/PR (351B tokens, prokaryotic, about 85–90% coding by typical genome density, an assumption not taken from these sources), plus eukaryotic mRNAs (89–99B tokens), optionally the 405B-token eukaryotic genic windows, and optionally RefSeq CDS extracted from annotations (a CodonFM-like corpus of about 131M CDS ≈ 130–150 Gbp, assuming about 1 kb per CDS; my estimate). Any of these exceeds what a ≤300M model can consume in a few weeks on a few A100s, so data quantity is not the constraint; quality, deduplication and mixture are.
- Hold out by **species or clade** (NT style) and by **gene family or protein-identity cluster** for coding tasks. Otherwise orthologs and paralogs leak, which Benegas et al. call "quite challenging" to avoid.
- Report validation perplexity or bits-per-base separately for CDS, UTR/intron and repeats (following Benegas et al.), and exclude eukaryotic viruses, following Evo practice.
- Licence: OpenGenome and OpenGenome2 are Apache-2.0; OMG is CC-BY-SA-4.0 (share-alike); GLRB is NC (evaluation only). That points to OpenGenome2 subsets as the cleanest base for an open release.

### Gaps
- Evo 2's data-composition and loss-weighting ablation results (Appendices B.1/B.2) and GENERator's Fig. S6 numbers were not retrieved, so the size of the "functional regions help" effect is unquantified here.
- Licence of the NT multispecies dataset is not stated on the card overview.
- GPN's repeat handling (believed to down-weight repeats in the loss) could not be verified.
- No 2025–2026 evidence found on 1000 Genomes or pangenome variant augmentation beyond the NT comparison.
- No primary source found for coding fraction per OpenGenome2 subset.

## Q4. What do scaling laws and compute-optimal sizing look like for DNA LMs at small scale (10M–300M parameters)?

### Takeaway
The only careful DNA scaling sweeps (Evo 1 in 2024, dnaHNet in Feb 2026) sit almost exactly in kron-dna's budget: 4e18–8e19 FLOPs and 6M–1B parameters, which is roughly 1–8 A100-days per run. They show architecture strongly changes the scaling slope (attention at single-nucleotide resolution scales worst) and that compute-optimal DNA training uses far more tokens per parameter than Chinchilla's ~20. Downstream gains from scale are real but "modest and inconsistent" (GENEB), and data relevance often matters more.

### Cited Findings
- **Evo 1 scaling study:** more than 300 models across Transformer++, Mamba, Hyena and StripedHyena; about 6M–1B parameters; budgets of 8e18, 2e19, 4e19 and 8e19 FLOPs; 8,192-token context; OpenGenome. Compute-optimal size per budget came from a quadratic fit of perplexity vs log(model size) (IsoFLOP). Transformer++ had "significantly worse perplexity" at every budget, attributed to inefficiency at byte resolution. Hyena and StripedHyena scaled best. Transformer++ and Mamba were numerically unstable off the frontier **[paper, 2024 reference]** — [Evo](https://www.biorxiv.org/content/10.1101/2024.02.27.582234v2.full)
- Evo 1 estimated **250B tokens** as compute-optimal for 7B at its budget, trained at a 17% offset from optimal size, about 340B tokens and about 2e22 FLOPs in total **[paper]** — [Evo](https://www.biorxiv.org/content/10.1101/2024.02.27.582234v2.full). That is about 36 tokens/param at 7B **[my arithmetic]**
- **dnaHNet (arXiv Feb 2026):** more than 100 models, 10M–1B parameters, 4e18–8e19 FLOPs, 5B–200B pretraining tokens on 144B nt of GTDB. Fit PPL = A·C^−α with α = 0.06 (dnaHNet), 0.04 (StripedHyena2), 0.01 (Transformer). At 8e19 FLOPs the compute-optimal dnaHNet is 218M parameters trained on 140B tokens; StripedHyena2 uses 68B nucleotides, and its comparison model at that scale is 166M **[paper]** — [dnaHNet](https://arxiv.org/html/2602.10603v2). That implies about 400 nucleotides per parameter for StripedHyena2 at 8e19 FLOPs **[my arithmetic; the paper says the SH2 optimum is "proportionally larger" without stating the exact size]**
- **Token budgets used by real small DNA models:** NT-v2-50M multispecies trained on 300B tokens, 1M-token batches **[model card via search]** — [NT-v2-50M](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-50m-multi-species). Evo 2 1B base used 1T tokens at 8,192 context **[paper]** — [Evo 2](https://www.biorxiv.org/content/10.1101/2025.02.18.638918v1.full). ModernGENA-base (135M) used 1,510B BPE tokens (158 epochs) and ModernGENA-large (377M) used 1,320B (138 epochs), at about 1.33M and 0.486M tokens/s on 8×A100-80GB **[paper]** — [ModernGENA](https://www.biorxiv.org/content/10.64898/2026.04.21.719816v1.full). GENERator 1.2B used 6 epochs of a 386B-nt corpus, 2M-token batches **[paper]** — [GENERator](https://arxiv.org/html/2502.07272v5)
- **Downstream scaling is weak (GENEB):** Spearman ρ between log(params) and macro-MCC is 0.573 (0.694 without Evo-1-131k). Scaling is significant in 11 of 13 categories, but not for species classification or chromatin accessibility. "Architecture and pretraining alignment often matter more than parameter count" **[paper]** — [GENEB](https://arxiv.org/html/2606.04525v4)
- Genomic Touchstone: clear scaling trends in the NT family when size and corpus diversity grow together, but they "break down at extreme scales and in other architectures"; MistralDNA-1.6B scored low on several tasks **[paper]** — [Touchstone](https://www.biorxiv.org/content/10.1101/2025.06.25.661622v1.full)
- Evo 2's own scaling: StripedHyena 2 improves loss scaling on DNA versus Transformers; validation perplexity improves with size (7B→40B) and context **[paper]** — [Evo 2](https://www.biorxiv.org/content/10.1101/2025.02.18.638918v1.full)
- The Benegas review asks "how far the scaling hypothesis holds for gLMs", noting protein-LM scaling helped structure but not most function tasks **[paper]** — [Benegas et al.](https://arxiv.org/html/2407.11435v2)
- Pretraining matters a lot at small scale for transformer encoders: ModernGENA's random init vs pretrained gap is 0.420 → 0.671 MCC **[paper]** — [ModernGENA](https://www.biorxiv.org/content/10.64898/2026.04.21.719816v1.full). For CNNs it is about zero **[paper]** — [Karpinsky et al.](https://arxiv.org/html/2606.30140v1)

### Inferences
- **Compute arithmetic for kron-dna (assumptions: A100 bf16 dense peak ≈ 312 TFLOP/s, 30–45% MFU, so about 1e19 FLOPs per A100-day; FLOPs ≈ 6·N·D):**
  - 30M parameters × 30B tokens ≈ 5.4e18 FLOPs ≈ 0.5 GPU-day
  - 100M × 100B ≈ 6e19 FLOPs ≈ 6 GPU-days
  - 300M × 100B ≈ 1.8e20 FLOPs ≈ 18 GPU-days
  - 6 GPUs × 14 days ≈ 8e20 FLOPs in total, i.e. about 10 runs at Evo/dnaHNet's top budget.
  - So kron-dna can reproduce an Evo/dnaHNet-style IsoFLOP sweep (8e18–8e19) at a few GPU-days per point, and compare KronLSTM's α and frontier directly against published StripedHyena2 (α=0.04) and Transformer (α=0.01) numbers, ideally on the same GTDB/OpenGenome data. That is a credible, publishable small-scale contribution.
- **Tokens per parameter:** for single-nucleotide models at 1e19–1e20 FLOPs, the dnaHNet-implied optimum (about 400 nt/param for SH2) and practice (NT-v2-50M about 6,000 tokens/param; Evo 2 1B about 1,000; ModernGENA about 11,000 BPE tokens/param over 158 epochs) all lie far beyond Chinchilla's ~20. For a 50–150M model, plan on roughly 50–200B nucleotides; going further (overtraining) is cheap insurance for a model meant to be used, not just studied.
- Multi-epoch reuse looks safe at these scales: ModernGENA trained 138–158 epochs, GENERator 6.
- Expect only modest downstream gains from 50M→300M (GENEB). Spend compute on data relevance (coding/genic windows) and a clean evaluation sweep rather than on maximum size.

### Gaps
- Neither Evo 1 nor dnaHNet publishes an explicit fitted N_opt(C) and D_opt(C) or tokens/param table in the parts retrieved (Evo's Table S1 and Fig. S5 were not rendered). The ~400 nt/param figure is my derivation from point values.
- No DNA scaling-law study found for MLM objectives, for linear-recurrent or delta-rule layers specifically, or on coding-only corpora.
- NTv3 (late 2025, ~128,000 species) scaling results were not retrieved.
- The A100 MFU assumption is not sourced; KronLSTM kernel efficiency may be lower than for attention or convolution.

## Q5. Which training-recipe details matter at small scale (tokenization, objective, context, packing, learning rates, RC augmentation, context curriculum, evaluation protocol)?

### Takeaway
Single-nucleotide (or learned, codon-aligned) tokenization is the safest choice for variant scoring and coding work. BPE is unreliable for DNA, especially below transformer scale. k-mers help autoregressive transformers but need offset augmentation. MLM tends to give better representations and CLM easier indel/generation scoring; no controlled DNA comparison settles it. Strong recipes share: short-to-long context staging, sequence packing, AdamW (β2 0.95–0.98) with warmup plus cosine or WSD, 50% RC augmentation or RC equivariance, repeat down-weighting, and chromosome- or species-held-out validation. Evaluate with zero-shot likelihood, mean-pooled linear probes (several layers, few-shot), and fine-tuning with seeds.

### Cited Findings

**Tokenization**
- **GENEB:** no global tokenizer ordering. Under human pretraining, single-nucleotide MutBERT beats BPE baselines (+0.033 vs GENA-LM, +0.038 vs GROVER); BPE beats k-mer in one pair (+0.032); BPE ≈ k-mer for multi-species transformer encoders (+0.006 across 5 pairs) **[paper]** — [GENEB](https://arxiv.org/html/2606.04525v4)
- **GENERator (causal LM, sequence-recovery ablation):** 6-mer is best; 1-mer and 8-mer fall short; BPE (vocab 512–8,192) stays below 0.30 accuracy against a 0.25 random baseline; Mamba-2 single-nucleotide (0.378) ≈ 1-mer transformer (0.381). Random 0–5 nt offset augmentation counters k-mer frame artefacts **[paper]** — [GENERator](https://arxiv.org/html/2502.07272v5)
- BPE hurts CNNs on GUE: U-Net 57.82 → 34.45 and ConvNova 61.08 → 49.68 mean MCC **[paper]** — [Karpinsky et al.](https://arxiv.org/html/2606.30140v1)
- **Compression view (DNAGPT2, bioRxiv June 2026):** 10 GPT-2-small models differing only in BPE vocab; a 32-token vocab compresses best (1.47 bits/base on T2T human, 4th on a compression leaderboard), which the authors read as evidence that NLP-style tokenization may be suboptimal for DNA **[paper abstract]** — [Macala & Simecek](https://www.biorxiv.org/content/10.64898/2026.06.10.731316v1)
- For protein-coding tasks, 3-mer NT beats 6-mer NT **[secondary]** — [Boshar et al.](https://instadeep.com/research/paper/exploring-genomic-language-models-on-protein-downstream-tasks/). dnaHNet's learned chunking uses a codon-motivated first-stage compression R1=3 (R2=2, 6× total) **[paper]** — [dnaHNet](https://arxiv.org/html/2602.10603v2)
- Benegas review: single-nucleotide tokens simplify interpretation and attribution and handle variants better; k-mer and BPE buy context length at the cost of artificial vocabularies **[paper]** — [Benegas et al.](https://arxiv.org/html/2407.11435v2)
- Other 2025 results conflict: one July 2025 study found BPE beats fixed k-mers on most tasks; DNAMotifTokenizer (Dec 2025) reports beating BPE with motif-informed tokens **[secondary, search snippets]** — [arXiv 2507.15087](https://arxiv.org/html/2507.15087v1); [DNAMotifTokenizer](https://arxiv.org/abs/2512.17126v1)

**Objective**
- MLM "tends to achieve better representations and transfer learning" (evidence from proteins); CLM is natural for generation. For variant scoring, MLM needs one forward pass per SNP but expensive pseudo-likelihood for multi-base edits; CLM needs two passes but handles indels and multiple substitutions naturally **[paper]** — [Benegas et al.](https://arxiv.org/html/2407.11435v2)
- Masked diffusion is an emerging middle ground: D3LM (NT-v2 architecture trained with masked diffusion; MLGenX 2026) improves on NT-v2 at equal size and generates regulatory elements (SFID 10.92 vs 29.16 for the prior best AR model; real DNA 7.85) **[paper abstract]** — [D3LM](https://arxiv.org/abs/2603.01780). MDLM DNA models match or beat BERT-style training on downstream classification while adding generation **[secondary]** — [MDLM](https://arxiv.org/html/2406.07524v2)
- Bio-xLSTM trained its DNA-xLSTM models (0.5M–4M parameters, 1k and 32k contexts, GRCh38) with **both** CLM and MLM **[secondary, search summary of paper]** — [Bio-xLSTM](https://arxiv.org/pdf/2411.04165)
- I found no controlled same-architecture, same-data MLM vs CLM comparison on DNA variant effects; a 12-model likelihood benchmark notes that scoring rules differ between causal and masked models **[secondary, search]** — (search result summary; see Gaps)

**Context length and curriculum**
- Evo 2 40B: 1,024-token context for 6.6T tokens, then 8,192 for 1.1T, then multi-stage midtraining to 1M (positional interpolation plus a 10× RoPE base increase per doubling; 600B tokens) with more whole genomes. Evo 2 1B: 8,192 context, 1T tokens **[paper]** — [Evo 2](https://www.biorxiv.org/content/10.1101/2025.02.18.638918v1.full)
- Evo 1: 8k stage, then 131k extension via linear RoPE interpolation **[paper]** — [Evo](https://www.biorxiv.org/content/10.1101/2024.02.27.582234v2.full)
- ModernGENA samples training lengths uniformly from 10 to 1,024 BPE tokens (mean about 700) **[paper]** — [ModernGENA](https://www.biorxiv.org/content/10.64898/2026.04.21.719816v1.full)
- Genomic Touchstone: longer inputs consistently help; for species classification Caduceus and Hyena gain beyond 8,192 bp **[paper]** — [Touchstone](https://www.biorxiv.org/content/10.1101/2025.06.25.661622v1.full). SSMs such as Caduceus extrapolate zero-shot 10–100× beyond training context (12 kbp → 120 kbp) on GLRB **[paper]** — [arXiv 2504.06304](https://arxiv.org/html/2504.06304v1)

**Packing, optimiser, learning rate, precision**
- Evo 2: sequence packing; AdamW β1=0.9, β2=0.95, cosine decay; FP8 linear layers **[paper]** — [Evo 2](https://www.biorxiv.org/content/10.1101/2025.02.18.638918v1.full)
- ModernGENA: dynamic packing, global batch 4,096 sequences; decoupled AdamW lr 4e-4, β2 0.98, ε 1e-6, wd 1e-5 (none on biases or norms); warmup over 3e9 tokens then constant (warmup-stable) **[paper]** — [ModernGENA](https://www.biorxiv.org/content/10.64898/2026.04.21.719816v1.full)
- dnaHNet (4 A100/H100 GPUs per run, DeepSpeed ZeRO-2, BF16): AdamW wd 0.05, clip 1.0, linear warmup plus cosine; base lr 6.5e-4 to 8e-4; about 94k–135k steps **[paper]** — [dnaHNet](https://arxiv.org/html/2602.10603v2)
- NT-v2-50M: 1M-token effective batch **[model card via search]** — [NT-v2-50M](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-50m-multi-species)

**Reverse complement and repeats**
- Evo 2 reverse-complements 50% of pretraining sequences **[paper]** — [Evo 2](https://www.biorxiv.org/content/10.1101/2025.02.18.638918v1.full). ModernGENA includes both strands **[paper]** — [ModernGENA](https://www.biorxiv.org/content/10.64898/2026.04.21.719816v1.full)
- Bio-xLSTM has RC-equivariant variants (parameter sharing "PS" and post-hoc conjoining "PH"); xLSTM-PH wins 5 of 8 Genomic Benchmarks tasks against Mamba-DNA-PH **[secondary, search summary]** — [Bio-xLSTM](https://arxiv.org/pdf/2411.04165)
- Repeats: 0.1 loss weight, plus lowercase-masked inputs for the first 3T tokens in Evo 2 **[paper]** — [Evo 2](https://www.biorxiv.org/content/10.1101/2025.02.18.638918v1.full)

**Evaluation protocol**
- GENEB: logistic regression (max_iter 1000) on mean-pooled frozen embeddings; MCC; 1-shot, 10-shot and full data; 5 seeds; tasks over 1e5 sequences subsampled; macro-MCC as the aggregate **[paper]** — [GENEB](https://arxiv.org/html/2606.04525v4)
- Mean token pooling beats other pooling for frozen-embedding classification **[paper abstract]** — [Feng et al.](https://www.nature.com/articles/s41467-025-65823-8)
- Frozen probes recover 95–100% of fine-tuned performance on promoters but only 60–88% on splice sites; final pooled embeddings may hide signal that layer-wise probing finds **[paper abstract]** — [Datta et al.](https://arxiv.org/abs/2608.05329)
- DART-Eval's three-tier protocol: zero-shot embeddings and likelihoods, a CNN probe on frozen final-layer embeddings, and LoRA fine-tuning on all linear and conv layers **[paper]** — [DART-Eval](https://pmc.ncbi.nlm.nih.gov/articles/PMC12340898/)
- NT-benchmark fine-tuning grid in ModernGENA: lr {1e-5, 3e-5, 5e-5}, wd {1e-4, 1e-3, 1e-2}, batch {32, 64}, head dropout 0.1, 10-fold CV, no RC augmentation **[paper]** — [ModernGENA](https://www.biorxiv.org/content/10.64898/2026.04.21.719816v1.full)
- BEND 2.0: linear readouts on frozen representations can match or beat native model outputs **[secondary, unverified]** — [aiweekly](https://aiweekly.co/alerts/bend-20-benchmark-tests-if-dna-foundation-models-can-guide-design)

### Inferences
- **Recommended kron-dna default recipe** (synthesis, not a single source):
  - **Tokenization and objective:** single-nucleotide tokens; CLM (natural for a recurrent model and for indel and DMS scoring). Optionally add a small bidirectional or MLM probe model, or run KronLSTM in both directions for embeddings.
  - **Context:** pretrain at 1–8k context with a short-context first phase (Evo 2-style 1k → 8k), then a brief long-context phase if the layer extrapolates.
  - **Data handling:** packing with document-boundary resets of the recurrent state; 50% RC augmentation, or RC-equivariant conjoining at evaluation; repeat loss weight about 0.1 for eukaryotic data.
  - **Optimiser:** AdamW β2 0.95–0.98, peak lr around 5e-4 to 8e-4 for 50–200M models with warmup plus cosine or WSD. WSD makes it easy to branch checkpoints for scaling sweeps.
  - **Codon awareness:** if you try k-mers, use 3-mers aligned to frame only for CDS-only corpora, with offset augmentation otherwise; Boshar et al. and dnaHNet's R1=3 both point to codon-scale units helping coding tasks.
- **Evaluation protocol to report:**
  1. Bits/base on held-out species, split by CDS / non-CDS / repeat.
  2. Zero-shot LLR on ClinVar coding, BRCA1 SGE and DMS/MaveDB, alongside ESM-2 and size-matched DNA LMs.
  3. Frozen mean-pooled linear probes at the best layer, 1/10/full-shot, 5 seeds (GENEB protocol, maybe a GENEB subset).
  4. Full fine-tuning on NT revised and GUE against a random-init twin and a one-hot CNN.
  5. One counterfactual regulatory task (DART-Eval variant task or TraitGym) as an honest out-of-domain check.

### Gaps
- No controlled same-architecture, same-data MLM vs CLM comparison on DNA variant effects was found.
- The HyenaDNA (2023) sequence-length warm-up curriculum result is a recollection, not re-verified here. Evo 2's 1k → 8k staging is the verified example of a context curriculum.
- No ablation found quantifying the effect of RC augmentation vs RC equivariance at small scale (Caduceus/Bio-xLSTM numbers only via secondary summaries).
- Evo 2's learning rate and batch tables (Tables 3–4) and the Evo 1 pretraining hyperparameters (Table S2) were not retrieved.
- No source found on state-reset or document-boundary handling for packed sequences in recurrent DNA LMs.
