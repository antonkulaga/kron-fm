# Coding-sequence (CDS) / codon-level language models, 2024–2026

Scope note: this covers models whose *primary* unit is the coding sequence (codon tokens, or nucleotide tokens with explicit frame/codon structure). General genome-wide nucleotide models (NT, HyenaDNA, Caduceus, GENERator) appear only where they are used as baselines on CDS tasks. Evidence labels used below: **[paper]** = stated in the preprint/journal article; **[card]** = Hugging Face / NGC / GitHub model card or README; **[unverified]** = seen only in a search snippet or secondary source and not confirmed against a primary source.

---

## Q1. Which codon-level / CDS models exist, and what are their facts?

### Takeaway

The field has consolidated around two camps: **encoder MLMs over 64-codon vocabularies** (CaLM → CodonBERT → cdsFM/EnCodon → SynCodonLM → NVIDIA CodonFM) and **generative/seq2seq codon decoders for design** (CodonTransformer, DeCodon, Trias, codonGPT, CodonMamba, mRNA-GPT). The decisive 2025–2026 releases for a new entrant are **NVIDIA CodonFM/EnCodon** (80M–1B, 130M CDS, Apache-2.0 code + open weights + full training recipe on GitHub), **SynCodonLM** (102M, NAR 2026, best documented small-model recipe, Apache-2.0), and **CodonMamba** (71M bidirectional Mamba — the closest existing analogue to a linear-recurrent kron-dna, trained on 4×A100 for ~7 days).

### Cited Findings — encoder / MLM family

**CaLM (Outeiral & Deane, Nature Machine Intelligence 2024)** — the field's reference point.
- Published as "Codon language embeddings provide strong signals for use in protein engineering", *Nat. Mach. Intell.* **6**, 170–179 (2024) — [Nature](https://www.nature.com/articles/s42256-024-00791-0); cited as ref. 46 in the CodonFM preprint [paper].
- Architecture: 12 transformer encoder layers plus a dense head; masked-codon MLM [unverified — from HF card summary, not read from the paper itself] — [HF multimolecule/calm card](https://huggingface.co/multimolecule/calm).
- Data: coding DNA from the **European Nucleotide Archive (ENA)**, April 2022 snapshot, 114,214,475 sequences collected, restricted to assembly dataclass CON; split by species and clustered **at the protein level**; the training set is ~9M sequences with a ~4,000-sequence heldout from seven organisms spanning all kingdoms, excluding anything ≥40% identity to training [card/unverified] — [HF multimolecule/calm card](https://huggingface.co/multimolecule/calm). Note the multimolecule checkpoint is explicitly an **unofficial** reimplementation; official code is `oxpig/CaLM`.
- CodonMamba independently reports reusing "the processed CaLM corpus of **8,771,938 coding sequences from 1,544 organisms**, drawn from ENA (April 2022 snapshot), clustered at **40% amino-acid identity with CD-HIT**" [paper] — [CodonMamba, bioRxiv 2026.08.24.746601](https://www.biorxiv.org/content/10.64898/2026.08.24.746601v1.full). This is the most concrete published description of CaLM's actual training corpus and is the number to cite.

**CodonBERT (Sanofi; Li et al., Genome Research 34, 1027–1035, 2024)**
- Architecture: **12 bidirectional transformer encoder layers, 12 attention heads, hidden size 768**. Parameter count is not stated in the paper [paper] — [bioRxiv 2023.09.09.556981v2](https://www.biorxiv.org/content/10.1101/2023.09.09.556981v2.full).
- Tokenization: codons built from the 5-letter alphabet {A,U,G,C,N} at each of three positions → **125 codon tokens** plus 5 specials ([CLS],[SEP],[UNK],[PAD],[MASK]) = **130-token vocabulary** [paper]. (Prediction is still over the 64 real codons.)
- Objectives: **MLM at 15% masking** + **Homologous Sequence Prediction (HSP)** — an NSP-style binary task on concatenated sequence pairs, half same-category/half different, read out of [CLS]; total loss = MLM + HSP [paper].
- Data: **~10 million mRNA coding sequences from NCBI** — mammalian RefSeq, *E. coli* RefSeq, and complete nucleotide sequences of human-infecting viruses, labelled hierarchically into **13 leaf categories**. Filters: length a multiple of 3, AUG start, stop-codon end, alphabet restricted to {A,U,G,C,N} [paper].
- Context: **512 codons per sequence**, <1024 tokens for the concatenated pair; longer sequences split into ≤512-codon fragments [paper].
- Compute: **batch size 128, ~7 epochs on 4× A10G GPUs (96 GB total GPU memory), roughly two weeks** [paper]. This is the single most directly relevant compute data point for an academic budget.
- License/links: HF `Sanofi-Public/CodonBERT`; the `yarongef/CodonBERT` and `hgbrian/CaLM` repo IDs do **not** exist on the Hub (checked, 2026-11-09) — [HF API check].

**cdsFM — EnCodon + DeCodon (Goodarzi lab; bioRxiv 2024.10.10.617568; published in *Nucleic Acids Research*, Feb 2026 per a search snippet [unverified])**
- Models: **EnCodon 80M / 620M / 1B** (encoder MLM) and **DeCodon 200M / 1B** (causal decoder); "Ada" suffix = eukaryote-adapted continued pretraining [paper] — [bioRxiv full text via PMC11482952](https://pmc.ncbi.nlm.nih.gov/articles/PMC11482952/).
- Architecture: both are **RoFormer-based** transformers with **rotary position embeddings (RoPE)**, Sub-LayerNorm and scaled initialisation; DeCodon adds **XPOS** extrapolatable position embeddings [paper].
- Tokenization: EnCodon vocabulary = **69 tokens** (64 codons + CLS/SEP/PAD/MASK/UNK). DeCodon vocabulary = **5,124 tokens** (64 codons + 5 specials + **5,055 organism tokens**), with the organism token **prepended as the first token** and used as the conditioning signal for generation. No nucleotide-level information reaches the model [paper].
- Data: **~60 million CDS from >5,000 species** from NCBI Genomes with latest GTF annotations; **98.7% (~59.4M) bacterial**. Eukaryotic adaptation set: **567,281 CDS from 227 eukaryotic species** (human, mouse, fly, zebrafish) [paper]. Token count and deduplication procedure are **not reported** [paper].
- Context: **2,048 codons**, chosen to fit >99.5% of sequences [paper].
- Compute: **>1,000,000 optimisation steps**; **2× NVIDIA H100 for ~2 weeks** per Methods [paper]. Batch size and LR schedule not given in the main text.
- Caveat the authors' own numbers conflict between text and captions: ClinVar SNVs 48,283 (text) vs 48,042 (caption); Ribo-seq ORFs 7,264 (Methods) vs 7,624 (Fig. 4) [paper].
- HF: `goodarzilab/decodon-200M` is listed with **158.0M actual parameters**, `AutoModelForCausalLM`, custom architecture class `decodon`, 3.0K downloads, updated 2025-07-20, **no license declared** [card] — [HF](https://hf.co/goodarzilab/decodon-200M). Code: [github.com/goodarzilab/cdsFM](https://github.com/goodarzilab/cdsFM).

**NVIDIA CodonFM / EnCodon series (Darabi, Cao, Naghipourfar et al., NVIDIA + Arc Institute + UCSF/UC Berkeley/Stanford; preprint Oct 2025)** — *the most completely documented codon model, and a separate artefact from cdsFM despite sharing the EnCodon name and Hani Goodarzi as co-author.*
- Title "Learning the Language of Codon Translation with CodonFM"; correspondence Laksshman Sundaram (NVIDIA) and Hani Goodarzi (Arc) — [NVIDIA preprint PDF](https://research.nvidia.com/labs/dbr/assets/data/manuscripts/nv-codonfm-preprint.pdf) [paper].
- Data: **>130 million coding sequences spanning >22,000 species from the NCBI RefSeq/Genomes database**, covering bacteria (the clear majority of the pie chart), archaea, fungi, plants, protozoa and metazoans. **Human-affecting pathogen sequences were excluded for biosafety** [paper]. Dedup/clustering is not described in the main text, but the GitHub data pipeline includes `allseq_clustering_for_splits.ipynb`, which produces `allSeqClusterIdx.npy` used to build train/val/test splits — i.e. **cluster-aware splitting rather than random splitting** [card] — [GitHub NVIDIA-BioNeMo/CodonFM](https://github.com/NVIDIA-BioNeMo/CodonFM).
- Architecture, from the README table [card]:

  | Model | Masking | Hidden | Layers | Heads | Intermediate |
  |---|---|---|---|---|---|
  | EnCodon 80M | Random MLM p=0.15 | 1024 | 6 | 8 | 4096 |
  | EnCodon 600M | Random MLM p=0.15 | 2048 | 12 | 16 | 8192 |
  | EnCodon 1B | Random MLM p=0.15 | 2048 | 18 | 16 | 8192 |
  | EnCodon 1B-CDSWT | Codon-frequency-weighted MLM | 2048 | 18 | 16 | 8192 |

  Note the unusual **very wide, very shallow** shape (1B params in only 18 layers at d=2048).
- Tokenization: codon-level within ORFs, sequence framed by `<CLS>` … `<SEP>`; **RoPE + self-attention** blocks [paper].
- Context: **2,046 codons = 6,138 nucleotides**; the paper shows a CDS-length histogram in which most CDS are under 1,000 codons, so "nearly all sequences are completely represented within the 2,046-codon context" [paper].
- Objective: masked-codon prediction. The fourth variant uses a **Codon-frequency Weighted Masking Strategy (CDWT/CDSWT)** explicitly inspired by distributional token weighting in NLP (word2vec, ELECTRA refs), designed to **oversample underrepresented synonymous codons** [paper].
- Compute: **not stated** in the preprint or README; the README's pretraining command only exposes `--num_gpus`/`--num_nodes` placeholders, with the real values inside `experiment_scripts/pretraining/encodon_filtered/*.sh` [card]. This is a genuine gap.
- License: **source code Apache-2.0; weights under the NVIDIA Open Model License** [card]. HF `nvidia/NV-CodonFM-Encodon-1B-v1` shows `license:other`, 1.2K downloads, updated 2025-10-26 [card] — [HF](https://hf.co/nvidia/NV-CodonFM-Encodon-1B-v1). There are also `-TE-` (TransformerEngine-accelerated) variants, e.g. `nvidia/NV-CodonFM-Encodon-TE-80M-v1`. Also distributed via NGC and a BioNeMo recipe (`codonfm_ptl_te`).
- Arc Institute's write-up describes a **causal, next-codon variant** of CodonFM aimed at generation, so the family may extend beyond the MLM checkpoints [unverified — Arc news page, not confirmed in the preprint] — [Arc Institute](https://arcinstitute.org/news/codonfm-foundation-model-for-codons).

**SynCodonLM (Heuschkel et al., Boehringer Ingelheim; *Nucleic Acids Research* 54(5), gkag166, 2026)** — *the best-documented recipe at academic scale; read this one first.*
- Architecture: **DeBERTaV2** (disentangled attention, relative position embeddings), **12 layers, hidden 768, intermediate 3072, 12 heads, max length 1024**, **~102M parameters** [paper] — [NAR gkag166](https://academic.oup.com/nar/article/54/5/gkag166/8496864). HF confirms **102.6M params, `deberta-v2`, Apache-2.0** [card] — [HF jheuschkel/SynCodonLM-V2](https://hf.co/jheuschkel/SynCodonLM-V2).
- Tokenization: 64 codons + 5 specials; [CLS] prepended, [SEP] appended. **Species are encoded as token-type IDs** (capped at 501), with species grouped by **k-means clustering of RSCU vectors** and one ID reserved for "unknown" [paper]. V2 on HF is described as a replicate trained **without** token-type IDs [card].
- Objective: **synonymous-codon-constrained masking** — ~15% of codons masked, and a [V,V] codon→synonym matrix is added to the logits before softmax so that **logits for non-synonymous codons are set to −∞**; the loss is computed only over synonymous options [paper].
- Data: NCBI RefSeq CDS across nine organismal groups; **43,526,014 CDS from 35,618 species**; **MMseqs2 clustering per species at the nucleotide level, 90% identity / 80% coverage**; 98/2 train/test split (42,655,495 / 870,519); stratified so no group's codon count exceeds that of mammalian vertebrates [paper]. (The earlier bioRxiv version reports >66M CDS — the two figures differ, so cite the NAR number and the dataset card [paper vs preprint conflict] — [bioRxiv 2025.08.19.671089](https://www.biorxiv.org/content/10.1101/2025.08.19.671089.full.pdf).)
- Compute: **12× NVIDIA L40S, 16-bit mixed precision, effective batch size 1560, ~8 days, 3 epochs, 82,028 steps**; average **407 codons/sequence**, **~100.1 billion tokens processed**. AdamW, peak LR 2e-4, 10% warmup, cosine decay to 0, dropout 0.1, weight decay 0.01 [paper].
- Links: [GitHub Boehringer-Ingelheim/SynCodonLM](https://github.com/Boehringer-Ingelheim/SynCodonLM); dataset [jheuschkel/clustered-cds-dataset](https://huggingface.co/datasets/jheuschkel/clustered-cds-dataset); Zenodo archive doi:10.5281/zenodo.18262433. HF license **Apache-2.0** [card]; the article text I read did not state a license [paper gap].

**CodonMamba (Lang et al., Univ. of Macau + Hangzhou Institute of Medicine/CAS; bioRxiv 2026.08.24.746601, ~24 Aug 2026)** — *the single closest architectural precedent for kron-dna.*
- Architecture: **8 stacked bidirectional Mamba (BiMamba) blocks, hidden 768, ~71M trainable parameters**; bidirectional design follows **Caduceus**, with projection layers shared between directions [paper] — [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.08.24.746601v1.full).
- Tokenization: non-overlapping codon triplets; 64 codons + `<cls> <eos> <pad> <unk> <mask>` → 768-d learned embeddings [paper].
- Objective: MLM, **15% of positions selected, of which 75% → `<mask>`, 15% → random token, 10% unchanged**; loss only at selected positions [paper]. (Note this is a non-standard 80/10/10 → 75/15/10 variant.)
- Data: the **processed CaLM corpus — 8,771,938 CDS from 1,544 organisms, ENA April-2022 snapshot, CD-HIT at 40% amino-acid identity** [paper].
- Context: **1,024 codon tokens**, longer truncated [paper].
- Compute: **AdamW, peak LR 4e-4, weight decay 0.01, 47 epochs, ~7 days on 4× NVIDIA A100** [paper]. **This is almost exactly the kron-dna budget.**
- Links: [HF langmei/CodonMamba](https://huggingface.co/langmei/CodonMamba), [GitHub meilanglang/CodonMamba](https://github.com/meilanglang/CodonMamba); **no software license stated** in the excerpt [paper gap].
- Status: preprint, not peer-reviewed; **no experimental validation** — the authors state the work demonstrates computational control, not improved expression or stability [paper].

### Cited Findings — generative / design family

**CodonTransformer (Fallahpour, Gureghian et al.; *Nature Communications* 16, s41467-025-58588-7, 2025)**
- Encoder-only **BigBird**; HF reports **89.6M parameters**, `AutoModelForMaskedLM`, **Apache-2.0**, 2.2M downloads (by far the most-used codon model), updated 2025-10-31 [card] — [HF adibvafa/CodonTransformer](https://hf.co/adibvafa/CodonTransformer).
- Training strategy **STREAM** (Shared Token Representation and Encoding with Aligned Multi-masking): several mask tokens are unmasked at once, with organism-specific embeddings added to condition predictions [unverified — from project site/README summaries].
- Tokenization is **amino-acid–codon paired tokens** (organism + amino acid + codon encodings merged), not bare codons; ambiguous codons map to 'UNK' [unverified/card].
- Data: **>1 million DNA–protein pairs from 164 organisms** spanning all domains of life [unverified — Nat. Commun. abstract via search; not read from the article].

**Trias (Faizi, Sakharova & Lareau; bioRxiv 2025.05.13.653614)**
- **BART encoder–decoder** (bidirectional encoder + autoregressive decoder) for reverse translation; HF reports **46.6M parameters, MIT license, 163.6K downloads**, updated 2026-04-21 [card] — [HF lareaulab/Trias](https://hf.co/lareaulab/Trias). This is the smallest competitive codon model on the Hub.
- Data: **full-length coding regions of 10 million vertebrate mRNA sequences**; handles proteins up to **2,046 aa**, with codon output for **>600 (README: 640) vertebrate species** [unverified/card] — [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.05.13.653614v3.full).
- Decoding: greedy, beam and nucleus sampling [card]. Authors report it outperforms commercial codon-optimisation tools at generating sequences resembling high-expression variants [unverified — authors' claim, preprint].

**codonGPT (Nucleic Acids Research 53(22), gkaf1345, 2025)** — generative codon LM with **RL fine-tuning** under biological constraints; vocabulary = 64 nucleotide triplets + specials; evaluated against native codon usage in housekeeping genes (HLA-A, ACTB) [unverified — search snippets + NAR listing] — [NAR gkaf1345](https://academic.oup.com/nar/article/53/22/gkaf1345/8384118). It is the top scorer on the *E. coli* Proteins task in the CodonMamba comparison [paper] — [CodonMamba](https://www.biorxiv.org/content/10.64898/2026.08.24.746601v1.full).

**mRNA-GPT (bioRxiv 2025.12.22.695962, Dec 2025)** — GPT-2-style generative model, **302M parameters**, pretrained on coding sequences from NCBI RefSeq, for CDS generation and design [unverified — search snippet only] — [bioRxiv PDF](https://www.biorxiv.org/content/10.64898/2025.12.22.695962.full.pdf).

**ARCADE (bioRxiv 2025.08.19.668819)** — "Controllable Codon Design from Foundation Models via Activation Engineering": steers an existing codon foundation model's activations rather than retraining [unverified — title/snippet only] — [bioRxiv PDF](https://www.biorxiv.org/content/10.1101/2025.08.19.668819.full.pdf).

**HalluCodon (bioRxiv 2026.03.31.715573)** — species-specific codon optimisation in plants using multimodal language models [unverified — snippet] — [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.03.31.715573v1).

### Cited Findings — adjacent mRNA/multi-omic models that report CDS results

- **HELM** (Yazdani-Jahromi, Prakash, Mansi, Moskalev, Liao; arXiv:2410.12459, v1 Oct 2024, v2 Mar 2025, CC BY 4.0): ~50M-param GPT-2-style transformer (10 layers, hidden 640) plus ~50M Hyena and Mamba variants and a 100M transformer (14 layers, hidden 768); trained on **15.3M curated antibody mRNA sequences from the Observed Antibody Space (OAS)** [paper] — [arXiv HTML](https://arxiv.org/html/2410.12459v2). Full details in Q4.
- **Life-Code** (arXiv:2502.07299): tokenizes at the codon level but pretrains with **both nucleotide reconstruction and CDS→amino-acid translation objectives** [unverified — snippet] — [arXiv PDF](https://arxiv.org/pdf/2502.07299).
- **CodonMoE** (Bioinformatics 42(6), btag285, 2026; arXiv 2025): an **adapter** that converts an existing *nucleotide* DNA LM into an mRNA/codon-aware analyser rather than a new pretrained model — directly relevant if kron-dna wants to retrofit a nucleotide backbone — [Bioinformatics](https://academic.oup.com/bioinformatics/article/42/6/btag285/8675376), [PubMed 40799804](https://pubmed.ncbi.nlm.nih.gov/40799804/).
- **Helix-mRNA** (Helical; arXiv:2502.13785, ICLR 2025 workshop): hybrid state-space + attention over **full-length mRNA including UTRs**, **single-nucleotide tokenization with explicit codon-structure encoding**; authors claim 6× longer sequences at ~10% of the parameters of existing foundation models, and report **5.19M parameters** in one listing; two-stage pretraining (broad RefSeq mRNA across kingdoms + viruses, then higher-quality specialisation). Open weights on [HF helical-ai/helix-mRNA](https://huggingface.co/helical-ai/helix-mRNA) [unverified — abstract + snippets] — [arXiv](https://arxiv.org/html/2502.13785v2).
- **Orthrus** (bioRxiv 2024.10.10.617658, NeurIPS 2024): **Mamba encoder** trained with a **self-supervised contrastive objective** pairing splice isoforms from 10 model organisms with orthologous transcripts from **>400 mammalian species (Zoonomia)**; mature-RNA, not CDS-only; reported to fine-tune a half-life predictor from as few as **30 labelled examples** — [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.10.10.617658.full.pdf), [PMC11482885](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11482885/). The contrastive-over-orthologues objective is the most distinctive alternative to MLM in this space.
- **mRNA-FM** (RNA-FM project): BERT-style, trained on **45 million mRNA coding sequences**; used as a baseline throughout the CodonFM benchmarks [unverified for the 45M figure — snippet] — [HF multimolecule/mrnafm](https://huggingface.co/multimolecule/mrnafm), [GitHub ml4bio/RNA-FM](https://github.com/ml4bio/RNA-FM).

### Inferences

- The published parameter counts and the real ones diverge: DeCodon-200M is **158M** on the Hub, and the NVIDIA "600M" is a 12-layer/d=2048 model. Treat all advertised sizes as nominal.
- The two "EnCodon" lineages must be kept distinct in any comparison table: **cdsFM EnCodon** (60M CDS, 5,000 species, 98.7% bacterial, 2×H100×2wk) and **NVIDIA CodonFM EnCodon** (130M CDS, 22,000 species, pathogens excluded). They share an author and a name but not a corpus.
- The architectural frontier in late 2026 is explicitly **linear-recurrent**: CodonMamba (BiMamba, Caduceus-style), Orthrus (Mamba), Helix-mRNA (SSM+attention hybrid), and HELM's Hyena/Mamba ablations. A KronLSTM/Gated-DeltaNet CDS model is a natural, uncrowded next step — but note HELM found **transformers beat Hyena and Mamba on 5 of 7 datasets at matched ~50M scale** [paper], so the burden of proof is on the recurrent side.

### Gaps

- **NVIDIA CodonFM training compute is unpublished.** Neither the preprint nor the README gives GPU count, batch size or token budget; they live in shell scripts in the repo that I did not read. This is the most useful missing number for budgeting.
- CaLM's own paper was not read directly (bioRxiv returned HTTP 403, Nature article not fetched). CaLM's corpus facts here are reconstructed from the CodonMamba methods and the unofficial multimolecule card.
- CodonTransformer's Nature Communications methods (exact tokenizer vocabulary, species list, compute) were not read.
- A model called **"NUWA"** (claimed 115M multi-species CDS, Feb 2026) appeared in one search summary but **could not be confirmed by any primary source** — treat as nonexistent until seen.
- **GenSLM, mRNABERT, PlantRNA-FM, cdsBERT, Mistral-Codon** appear only as baselines in others' benchmark tables here; their own facts were not collected. cdsBERT, cdsBERT-plus, Mistral-117M and Mistral-16M are named in the CodonBench panel and would be worth a follow-up.
- No evidence found of a Ginkgo Bioworks codon model, or of an Evo-derived CDS-specific model, in this window.

---

## Q2. Downstream tasks and benchmarks for coding DNA

### Takeaway

There are now three distinct, reusable benchmark families: **the CodonBERT seven-task suite** (the de facto standard, reused by nearly everyone), **the SynCodonLM seven-dataset expression/abundance suite** (DNA-level-sensitive by construction), and **CodonBench** (2026, leakage-controlled, with gene-held-out splits and a memorisation canary). Clinical variant tasks (ClinVar missense/synonymous, DDD, ASD, cancer hotspots) are the other major axis, and codon models lose to ESM-2 on missense but win on synonymous.

### Cited Findings

**The CodonBERT suite (Li et al. 2024)** — seven tasks, reused as "the" benchmark by CodonBench, HELM, SynCodonLM and others [paper] — [bioRxiv 2023.09.09.556981v2](https://www.biorxiv.org/content/10.1101/2023.09.09.556981v2.full):
1. mRFP expression (*E. coli* gene variants) — regression, Spearman
2. Fungal expression (CDS >150 bp) — regression
3. *E. coli* proteins — 3-class expression classification (low/medium/high), with **2,308 / 2,067 / 1,973 proteins** per class
4. mRNA stability (human, mouse, frog, fish)
5. Tc-riboswitches — switching factor of riboswitch dimers upstream of GFP
6. SARS-CoV-2 mRNA vaccine degradation
7. Flu hemagglutinin expression in HeLa cells (new dataset introduced by the paper)
   Splits 0.70/0.15/0.15. **CodonBERT was best on 4 of 7 and second on 2**, usually by a small margin; a plain codon-based **TextCNN beat it on SARS-CoV-2 degradation**; it was ~10% better than the runner-up on flu and ~2% behind the top model on SARS-CoV-2; **mRNA stability was its clear weak spot**, which the authors attribute to structure dependence and the codon-only design [paper].

**CodonBench (Liang, Zhu, Pan, Liang — SJTU + Fudan; bioRxiv 2026.08.12.744371, ~Aug 2026)** — the first leakage-controlled suite [paper] — [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.08.12.744371v1.full):
- Classification datasets: **MisPath** (ClinVar missense, downloaded 2024-10-28, balanced to 5,000) and **SynPath** (1,420 pathogenic synonymous variants, 1:1 balanced → 2,840).
- Regression datasets, taken from the CodonBERT benchmark: **mRFPExpr n=1,459; EcoliExpr n=3,000; mRNAStab n=5,000; FungalExpr n=7,089**.
- Splits: random (5-fold CV or 80/20), **leave-one-gene-out CV (LOGO-CV, 240 valid folds)**, and gene-stratified splits for LoRA.
- Panel of 21: 9 codon cLMs (CodonBERT, CodonBERT-HF, EnCodon-80M, EnCodon-620M, CodonTransformer, CaLM, mRNABERT, Mistral-117M, Mistral-16M), 2 character-level CDS LMs (cdsBERT, cdsBERT-plus), 2 protein LMs (ESM-2-650M, ESM-1b-650M), 2 DNA LMs (NT-v2-500M, NT-v2-50M), and 6 non-neural baselines (onehot_pos memorisation canary, one-hot without position, k-mer frequency, codon-usage bias, GC content, AlphaMissense).
- Ships **gene-held-out splits, a probing ladder, validation-based early stopping, a matched PyTorch MLP baseline, and the `onehot_pos` canary as defaults** [paper].

**The SynCodonLM suite (NAR 2026)** — seven datasets deliberately chosen so that **codon usage varies while the protein sequence is held constant**, which isolates DNA-level signal [paper] — [NAR gkag166](https://academic.oup.com/nar/article/54/5/gkag166/8496864):
mRNA Toxicity 1 and 2 (*E. coli* GFP variants); GFP Expression (*E. coli*); mRFP Expression (*E. coli*); GFP mRNA Abundance 1 and 2 (*S. cerevisiae*); TDH3 mRNA Abundance (*S. cerevisiae*).
Protocol: **frozen encoders, mean pooling, linear regression head, five-fold CV repeated over 20–50 seeds, two-sided paired t-tests (α=0.05)**. Datasets at [github.com/Boehringer-Ingelheim/SynCodonLM/tree/master/benchmarking-datasets](https://github.com/Boehringer-Ingelheim/SynCodonLM/tree/master/benchmarking-datasets).

**The Roche therapeutic-optimisation benchmark (Toneyan, De Donno, Payne et al., Roche; bioRxiv 2026.08.11.744178, v. 16 Aug 2026)** — nine linear-probing tasks with 5-fold CV [paper] — [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.08.11.744178v1.full):
mean ribosomal load (9,933 seqs); protein abundance (11,790); transcript abundance (5,485); mRFP abundance in *E. coli* (1,455); MLOS flu-vaccine hemagglutinin expression (167); mRNA half-life (12,969); Nanoluc abundance (24); GFP fluorescence from Mordstein et al. (182/184/185 across three conditions).

**The CodonFM / EnCodon clinical-variant benchmarks** [paper] — [NVIDIA preprint](https://research.nvidia.com/labs/dbr/assets/data/manuscripts/nv-codonfm-preprint.pdf):
- *De novo* variants in the **Deciphering Developmental Disorders (DDD)** and **Autism Spectrum Disorder (ASD)** cohorts (Zhou et al., *Nat. Genet.* 54, 1305–1319, 2022), scored by Mann–Whitney U case-vs-control −log10 p.
- **ClinVar missense** and **cancer-hotspot somatic missense**, scored by AUROC.
- **ClinVar synonymous variants** with **50 stratified subsampling iterations** matching each pathogenic variant to benign variants on reference codon, alternate codon, position in gene, gene-level pLI, and local mutation rate — a strong covariate control worth copying.
- mRNA design: translation efficiency from Zheng et al. (bioRxiv 2024.08.11.607362) and mRFP protein expression from Nieuwkoop et al. (*NAR* 51, 2363–2376, 2023), both via **random-forest regression on frozen embeddings**, reporting 10-fold CV R² and Spearman respectively.

**The cdsFM task set** [paper] — [PMC11482952](https://pmc.ncbi.nlm.nih.gov/articles/PMC11482952/):
ClinVar zero-shot pathogenicity (~48k SNVs); **197 human deep mutational scanning studies across 72 genes** (~58,262 mutations) plus 7 other organisms (~16,403); **human Ribo-seq ORF classification (~7.3k ORFs)**; mRNA stability (26,725 sequences — 12,981 human, 13,744 mouse, from 39 human and 27 mouse studies); mRFP expression (LoRA fine-tuning); **1,054 synonymous variants across SLC22A1, GPR68, KCNJ2** for abundance/surface expression, with 513 SLC22A1 variants held out; **3,365 COSMIC synonymous variants** scored for pathogenicity, with 4 nominated and **2 experimentally confirmed (AGPAT2 c.702C>T, RAC2 c.501C>T)**.

**HELM's seven property tasks** [paper] — [arXiv HTML](https://arxiv.org/html/2410.12459v2): Ab1 and Ab2 (antibody mRNA protein expression, two platforms), MLOS (viral mRNA, HeLa expression), iCodon (thermostability), Tc-riboswitches (switching factor), mRFP (*E. coli* protein production), COVID-19 vaccine (degradation) — plus an antibody region annotation task.

**CodonMamba's 12 benchmarks**: five expression, four stability/degradation, three regulatory; frozen backbones with a lightweight ResNet-style head, three seeds per task. **CodonMamba first on 10 of 12, second on the other two**; codonGPT won *E. coli* Proteins; **CaLM won Subcellular Localization** [paper] — [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.08.24.746601v1.full).

### Does a codon model beat a protein LM of similar size?

- **On missense variants: no.** On ClinVar missense and cancer-hotspot somatic missense, "the EnCodon models are highly performant and are **second best to the ESM2 models** trained on amino acid sequences, while outperforming all other RNA-based baselines." The discussion repeats that EnCodon is "marginally outperformed by the ESM family", attributing the gap to protein LMs' explicit encoding of structure and amino-acid substitution patterns [paper] — [NVIDIA preprint](https://research.nvidia.com/labs/dbr/assets/data/manuscripts/nv-codonfm-preprint.pdf).
- **On *de novo* case/control cohorts: yes.** On DDD and ASD, EnCodon shows the strongest case-vs-control separation of all models tested, ahead of ESM-2 (3B and 650M), mRNA-FM, CodonBERT, Nucleotide Transformer, CaLM, RNA-BERT, RiNALMo, RNAErnie and AIDO.RNA [paper, Fig. 3A–B].
- **On synonymous variants: yes, decisively, within that paper's protocol.** "EnCodon models consistently outperform all RNA- and mRNA-based baselines, with the 1B-CDWT variant achieving the highest median significance across replicates" on matched ClinVar synonymous subsampling [paper, Fig. 4B].
- **Fine-tuned, codon models reach supervised-protein-model territory.** EnCodon 1B fine-tuned on gnomAD missense variants (EnCodon 1B-FT) is "similarly performant or better than AlphaMissense… even in the absence of explicit structural priors" on DDD and ASD [paper, Fig. 3F–G].
- **Independent 2026 work finds a small but real codon-specific signal on missense.** A study testing whether CaLM scores add information beyond protein LMs reports "a modest but reproducible component of variant-effect information not fully captured by protein-level models" [unverified — reported via a GitHub issue summarising a bioRxiv preprint dated 2026-09-14] — [GitHub delalamo/SKM issue 861](https://github.com/delalamo/SKM/issues/861).
- **A 2026 geometry study** finds CodonBERT and ESM-2 embedding spaces "align in a way that can be linearly decoded, with the synonymous-codon signal specific to codon-aware architectures" [unverified — snippet] — [bioRxiv 2026.06.30.735115](https://www.biorxiv.org/content/10.64898/2026.06.30.735115v1).

### Inferences

- The division of labour is clean and reproducible across independent groups: **protein LMs own amino-acid-changing variation; codon LMs own synonymous variation and quantitative expression/translation phenotypes.** A CDS model should be pitched and benchmarked on the second set, not the first.
- Because almost everyone reuses the CodonBERT seven tasks, kron-dna gets comparability nearly free by running them — but see Q3, because those tasks' standard protocol is exactly what CodonBench shows to be unsound.

### Gaps

- Per-dataset R²/Spearman numbers for SynCodonLM are in Figure 4b, which was not reproducible from the fetched text.
- CodonBERT's own Table 2 (exact per-task scores) and Table 1 (dataset sizes) were not retrieved; only the *E. coli* class counts are in the body text.
- No single maintained leaderboard exists. CodonBench is the closest thing, and it is two months old and not peer-reviewed.

---

## Q3. Does codon-level tokenization beat nucleotide-level? Does frame awareness matter? Small vs large?

### Takeaway

The strongest direct, matched-architecture evidence (**HELM**) says codon tokenization wins on 4 of 5 datasets, and that making the *loss* codon-aware helps more than the tokenizer alone. But the strongest *methodological* evidence (**CodonBench**, 2026) says that on synonymous-variant prediction the measured codon advantage does not survive gene-held-out evaluation — a result a new codon model must engage with directly rather than ignore.

### Cited Findings — tokenization

- **HELM ran the clean ablation**: same ~50M GPT-2-style setup, three tokenizations, Spearman on five datasets (nucleotide / codon / 6-mer) [paper] — [arXiv HTML](https://arxiv.org/html/2410.12459v2):

  | Dataset | nucleotide | codon | 6-mer |
  |---|---|---|---|
  | Ab1 | 0.699 | **0.747** | 0.733 |
  | Ab2 | 0.569 | **0.599** | 0.582 |
  | MLOS | 0.528 | **0.654** | 0.625 |
  | Tc-Riboswitches | 0.533 | **0.570** | 0.562 |
  | mRFP | 0.820 | 0.754 | **0.857** |

  Codon tokenization wins on 4 of 5; the paper's prose ("codon-level tokenization consistently outperforms other strategies") **overstates its own table**, since 6-mer wins on mRFP. The nucleotide-vs-codon gap is largest on MLOS (+12.6 points) and smallest on Ab2 (+3.0).
- **HELM's main contribution is the loss, not the tokenizer**: a **hierarchical cross-entropy (HXE)** over a codon tree (root → coding/non-coding; coding codons grouped by amino acid; non-coding split into start AUG and stop codons), with each level weighted by **λ(C) = exp(−α·h(C))**, h(C) the node height, so synonymous confusions are penalised less than cross-amino-acid ones. **α = 0.2 was best** among {0.2, 0.4, 0.6}. It applies to both MLM and CLM and **needs no architecture or tokenizer change** [paper].
- HELM vs plain cross-entropy at matched size (Spearman, transformer, MLM→HELM-MLM / CLM→HELM-CLM) [paper]: Ab1 0.748→0.767 / 0.752→0.760; Ab2 0.599→0.609 / 0.597→0.614; MLOS 0.653→**0.701** / 0.611→0.592; iCodon 0.503→0.525 / 0.498→0.529; Tc-Riboswitches 0.569→**0.626** / 0.531→**0.619**; mRFP 0.753→**0.822** / 0.815→0.849; COVID-19 vaccine 0.801→0.833 / 0.787→0.789. **HELM improved MLM on all 7 and CLM on 5 of 7** (small drops on MLOS and COVID-19).
- **SynCodonLM is the other synonymy-aware-objective data point**, and it works: restricting the softmax to synonymous codons made a 102M model beat CaLM, CodonBERT, CodonTransformer, cdsBERT and Mistral-Codon on **6 of 7** DNA-level-sensitive datasets (P<.05), ranking second on the seventh while still beating 4 of 5 [paper] — [NAR gkag166](https://academic.oup.com/nar/article/54/5/gkag166/8496864).
- **Counter-evidence from CodonBench's own tokenization ablation** (from-scratch BERTs v1/v3a/v3b/v4, codon vs character tokenizer): under **random splits** the codon advantage under MLP probing is **+2.3 to +14.3 points** — comparable to HELM's numbers — but under **LOGO-CV it shrinks to +1.6 to +2.2 points for v1/v3a/v3b and reverses to −5.8 points in favour of character tokenization for v4** (paired sign test p=0.03) [paper] — [CodonBench](https://www.biorxiv.org/content/10.64898/2026.08.12.744371v1.full).
- **COMET** reports that nucleotide models "have the potential to implicitly learn codon patterns", i.e. codon tokenization is not strictly necessary for every task [unverified — snippet] — [arXiv PDF](https://arxiv.org/pdf/2412.10347).
- **GENERator** reports that for causal genomic models **k-mer tokenization outperformed both single-nucleotide and BPE**, and that single-nucleotide tokenization costs effective context length under a fixed token budget [unverified — snippet] — [arXiv PDF](https://arxiv.org/pdf/2502.07272).
- **Helix-mRNA takes the hybrid route**: single-nucleotide tokenization *with explicit codon-structure encoding*, which lets one model span UTRs and CDS without a frame break [unverified] — [arXiv](https://arxiv.org/html/2502.13785v2).
- **Life-Code** tokenizes at codon level but adds a **CDS→amino-acid translation objective** alongside nucleotide reconstruction [unverified] — [arXiv PDF](https://arxiv.org/pdf/2502.07299).

### Cited Findings — the leakage result (read this before designing evaluations)

- The `onehot_pos` **memorisation canary** — encoding only (position, codon) identity, no model at all — reaches **SynPath LR AUC 0.891** and **MLP AUC 0.931** under random splits, **beating every neural model**, and falls to **0.739** under LOGO-CV [paper] — [CodonBench](https://www.biorxiv.org/content/10.64898/2026.08.12.744371v1.full).
- Under LOGO-CV the neural models range **0.507–0.564** — i.e. below the position-prior floor. **ESM-2's SynPath LR AUC falls from 0.685 (random) to 0.507 (LOGO-CV), near chance** [paper].
- **Gene overlap between train and test is 98.3% under an 80/20 split** and 0 of 240 folds under LOGO-CV [paper]. That one number explains most of the literature's synonymous-variant results.
- **LoRA fine-tuning reaches SynPath AUC 0.874–0.919 under random splits but drops 15.9–28.6 points under gene-stratified splits** [paper].
- **Zero-shot log-likelihood-ratio AUC on SynPath is 0.484–0.526** — chance [paper]. This directly contradicts the headline synonymous-variant claims of CodonFM/EnCodon, though the two use different ClinVar snapshots, different matching schemes and different metrics (Mann–Whitney U on matched subsamples vs AUROC on a balanced set), so they are not strictly comparable. **Flag this conflict explicitly.**
- **The "deeper probe helps" signature is an artifact**: six pretrained models gain **+11.7 to +19.3 points** from linear to MLP probing under random splits, but **−3.9 to +3.1** under LOGO-CV with a matched PyTorch MLP [paper].
- **A reported CodonBERT +4.7-point LOGO-CV gain existed only with scikit-learn's `MLPClassifier` defaults; with a matched PyTorch MLP it was −0.0.** Seed 42 alone gave +4.7; 20 other seeds ranged −2.3 to +3.1 [paper].
- **Regression best-epoch selection**: picking the epoch by *test* R² produces MLP-over-Ridge gains of **+6.4 to +23.7** points on mRFPExpr; **validation-based early stopping reverses them to −37.0 to −10.8** [paper].
- Residual information estimate: **I(σ;Y|A) ≈ 0.04 bits vs I(A;Y) ≈ 0.13 bits** — the authors say a thin codon-specific signal may exist but is not reliably detectable at n=2,840, and would need ~10× more synonymous variants, **codon-position-preserving pooling (e.g. attention pooling instead of mean pooling)**, or end-to-end fine-tuning [paper].
- Authors' own limitations: cross-model differences are observational and confounded by architecture and corpus; the scale comparison uses only two from-scratch points (20M and 110M); LOGO-CV does not control for gene families or GC strata [paper].

### Cited Findings — small vs large

- **Scaling is real and monotone within CodonFM**: validation loss, normalised MLM loss across taxonomic divisions, and KNN purity all improve 80M → 600M → 1B, and the 1B-CDWT model has the highest KNN purity of all. Larger models also show progressively **lower normalised synonymous-codon confusion scores** [paper] — [NVIDIA preprint](https://research.nvidia.com/labs/dbr/assets/data/manuscripts/nv-codonfm-preprint.pdf).
- But the downstream gains are modest in absolute terms: on mRNA translation efficiency (10-fold CV R²), EnCodon 1B ≈ 0.52, EnCodon 600M ≈ 0.49, mRNA-FM ≈ 0.485, EnCodon 1B-CDWT ≈ 0.48, **EnCodon 80M ≈ 0.44**, CodonBERT ≈ 0.41. On mRFP expression (Spearman): EnCodon 1B ≈ 0.73, 600M ≈ 0.71, **80M ≈ 0.655**, 1B-CDWT ≈ 0.65, CodonBERT ≈ 0.59, mRNA-FM ≈ 0.53 [paper, Fig. 5, read from the plotted bars — approximate].
  **So an 80M codon model already beats CodonBERT and mRNA-FM on both tasks, and sits ~0.08 R² / ~0.075 Spearman behind a 1B model.**
- **A 102M model is competitive with everything when the objective is right**: SynCodonLM (102M) beats CaLM, CodonBERT, CodonTransformer, cdsBERT and Mistral-Codon on 6/7 [paper].
- **A 71M model wins 10 of 12 benchmarks**: CodonMamba (71M) ranks first on 10 of 12 mRNA prediction tasks and second on the other two against models pretrained on DNA, ncRNA or protein [paper] — [CodonMamba](https://www.biorxiv.org/content/10.64898/2026.08.24.746601v1.full).
- **HELM states ~50M models "can outperform larger existing models"** (RNA-FM, SpliceBERT, CodonBERT, one-hot) with "up to 2× fewer model parameters", averaging ~8% better across the seven tasks; its 100M HELM also beat the 100M plain-XE model, so the objective's benefit persists with scale (MLOS: 0.664 at 100M-HELM vs 0.592 at 50M-HELM) [paper] — [arXiv HTML](https://arxiv.org/html/2410.12459v2).
- **A 46.6M model is the most-downloaded serious generative codon model** (Trias, 163.6K downloads) [card] — [HF](https://hf.co/lareaulab/Trias).

### Inferences

- For CDS tasks, **objective design dominates scale in the 50M–150M range**. The two biggest documented single-change wins — HELM's hierarchical loss (+~8% average, up to +6.9 Spearman points on mRFP) and SynCodonLM's synonymous-constrained softmax (6/7 wins at 102M) — are both *loss-function* changes, free in compute, and both exploit the same structure: the codon→amino-acid many-to-one map. Either is directly portable to a KronLSTM backbone.
- Frame awareness is the point of codon tokenization, and nothing in the literature argues for an unframed nucleotide model on CDS. But the HELM mRFP row and CodonBench's v4 reversal mean the advantage is **task-dependent and evaluation-dependent**, not categorical.
- A defensible 2026 claim for a new small CDS model is therefore *not* "we beat 1B models" but "at ~50–100M, with a synonymy-aware objective and a linear-recurrent backbone, we match transformer codon models at a fraction of the training cost **and** we report it under gene-held-out splits."

### Gaps

- No one has run HELM's hierarchical loss and SynCodonLM's constrained softmax **against each other** on the same corpus and backbone. That is an open, cheap, publishable comparison.
- CodonBench's leakage critique has been applied to **synonymous variant prediction and the four CodonBERT regression tasks only**. Whether the expression/translation-efficiency results (the CodonFM Fig. 5 tasks, the SynCodonLM suite, the Roche suite) survive gene-held-out evaluation is **untested**.
- No published matched-compute comparison of a linear-recurrent vs transformer codon model exists beyond HELM's single ~50M CLM-only ablation (which favoured the transformer) and CodonMamba's benchmark wins (which are not compute-matched against a transformer trained on the same corpus).

---

## Q4. Practical recipe lessons for training a new CDS model

### Takeaway

The convergent recipe across the three best-documented models is: RefSeq CDS, hard frame/start/stop filtering, **per-species redundancy reduction at 40–90% identity**, 64-codon vocabulary with 5 specials, 1,024–2,048 codon context, 15% masking, and a species conditioning signal. The compute numbers are reassuring — SynCodonLM's 102M model took 12× L40S for 8 days (~100B tokens), cdsFM trained 80M–1B on **2× H100 for 2 weeks**, CodonMamba's 71M took **4× A100 for 7 days** — all within a handful-of-A100s budget.

### Cited Findings — data curation

- **Frame and terminus filtering (CodonBERT, the clearest statement):** keep only sequences whose length is a multiple of 3, that **start with AUG**, **end with a stop codon**, and use only the {A,U,G,C,N} alphabet [paper] — [bioRxiv 2023.09.09.556981v2](https://www.biorxiv.org/content/10.1101/2023.09.09.556981v2.full).
- **Ambiguity handling** differs by model: CodonBERT **expands the vocabulary to 125 codons** over {A,U,G,C,N} so N-containing codons are representable; CodonTransformer maps ambiguous codons to `UNK`; cdsFM and CodonFM use the clean 64+5 vocabulary [paper / card].
- **Redundancy reduction, three published choices:**
  - CaLM: split by species, cluster **at the protein level**; CD-HIT at **40% amino-acid identity** per CodonMamba's description of the processed corpus; 114M raw ENA sequences → **~8.8M** training sequences — a **~13× reduction** [paper via CodonMamba; card for the raw figure].
  - SynCodonLM: **MMseqs2 per species, at the nucleotide level, 90% identity / 80% coverage**, plus stratified sampling so no organismal group's codon count exceeds that of mammalian vertebrates [paper].
  - NVIDIA CodonFM: cluster indices (`allSeqClusterIdx.npy`) are built specifically **to construct train/val/test splits**, i.e. clustering is used for leakage control rather than only for down-weighting [card] — [GitHub](https://github.com/NVIDIA-BioNeMo/CodonFM).
  The three differ by a factor of more than two in stringency (40% aa vs 90% nt). Nobody has ablated this.
- **Composition is the biggest silent design choice.** cdsFM's 60M CDS are **98.7% bacterial**, which is why it needed a separate eukaryotic adaptation stage on 567,281 CDS from 227 species [paper]. SynCodonLM instead **caps each group's codon budget at the mammalian-vertebrate level up front** [paper]. CodonFM's >130M CDS from >22,000 species is also bacteria-dominated by the sequence-group pie chart, but it trains in one stage [paper].
- **Biosafety filtering is now standard practice at the frontier**: CodonFM states "We exclude the human-affecting pathogen sequences for biosafety purposes", and the repo ships `taxids_to_remove_bac.json` to exclude bacterial taxa [paper + card].
- **Pseudogenes:** no paper in this set explicitly describes pseudogene removal. The closest mechanisms are using **GTF/RefSeq CDS annotations** (cdsFM: "NCBI Genomes using the latest GTF annotations") and the frame/start/stop filters, which exclude most pseudogene-derived sequences implicitly [paper]. **Treat explicit pseudogene filtering as an unaddressed design choice.**

### Cited Findings — sequence length and context

- **Most CDS are short.** CodonFM's length histogram shows "most CDS entries are under 1,000 codons, with a right-skewed length distribution", so a **2,046-codon context covers nearly all sequences** [paper].
- cdsFM chose **2,048 codons** to fit ">99.5%" of sequences (caption says 99.8%) [paper].
- **SynCodonLM's corpus averages 407 codons per sequence** at a 1,024 max length [paper] — the single most useful number for estimating a token budget.
- CodonMamba truncates at **1,024 codons**; CodonBERT at **512 codons** per sequence (1,024 for the concatenated HSP pair), splitting longer sequences into ≤512-codon fragments [paper].
- **Implication for kron-dna:** at ~400 codons average, a 2,048-token context wastes ~80% of positions unless sequences are packed. None of these papers describes sequence packing — this is an easy, uncontested efficiency win for a recurrent model.

### Cited Findings — masking strategies

- **Baseline is 15%**, used by CodonBERT, SynCodonLM, CodonMamba and all CodonFM random-masking variants [paper ×4].
- **CodonMamba's split within the masked 15% is non-standard: 75% → `<mask>`, 15% → random token, 10% unchanged** (vs BERT's 80/10/10); loss computed only at selected positions [paper].
- **Codon-frequency-weighted masking (CodonFM 1B-CDWT)** oversamples underrepresented synonymous codons. Results are **mixed, and the paper is honest about it**: 1B-CDWT gets the best DDD/ASD separation, the best ClinVar-synonymous median significance, and the highest KNN purity — but it is **worse than random-masking models on translation efficiency and mRFP expression**, because "the EnCodon-1B-CDWT embeddings are less influenced by simple sequence features such as GC content and therefore perform slightly worse… on tasks where these features have a stronger effect" [paper].
- **Synonymous-constrained masking (SynCodonLM)**: add a [V,V] codon→synonym matrix to the logits pre-softmax so non-synonymous codons get −∞ and the loss is over synonymous options only. This makes the task purely "which synonymous codon", given that the protein is known [paper].
- **Hierarchical loss (HELM)**: keep standard masking, reweight the *loss* by distance in the codon tree, λ(C)=exp(−α·h(C)), α=0.2 [paper].
- **Synonym-aware logit masking at inference** (CodonMamba) enforces protein identity during generation, with a soft **host codon-usage prior** (relative adaptiveness from Kazusa frequencies) added as a logit bias scaled by β. At β=2.0, human CAI rises to **0.97 vs 0.77 for natural CDS**, while the prior changes only **~11–14%** of the model's preferred codon choices; cross-host matched CAI: human 0.976, mouse 0.968, *E. coli* 0.881, yeast 0.980 [paper] — [CodonMamba](https://www.biorxiv.org/content/10.64898/2026.08.24.746601v1.full).

### Cited Findings — species conditioning (four distinct designs)

- **DeCodon**: 5,055 organism tokens in the vocabulary, prepended as the first token, used as the generation conditioning signal [paper].
- **SynCodonLM**: species as **token-type IDs** capped at 501, grouped by **k-means on RSCU vectors**, one ID for "unknown" [paper]. The V2 ablation trained **without** token-type IDs exists on the Hub [card].
- **CodonTransformer**: organism embeddings added to the sequence alongside amino-acid and codon encodings; taxonomy ID is a model input [unverified/card].
- **EnCodon (both lineages)**: **no species conditioning at all** — just `<CLS>` … `<SEP>` [paper].

### Cited Findings — optimisation hyperparameters and compute

| Model | Params | GPUs | Wall-clock | Batch | Steps/epochs | LR | Tokens |
|---|---|---|---|---|---|---|---|
| CodonBERT | 12L/768 | 4× A10G | ~2 weeks | 128 | ~7 epochs | — | — |
| cdsFM EnCodon/DeCodon | 80M–1B | 2× H100 | ~2 weeks | — | >1,000,000 steps | — | — |
| SynCodonLM | 102M | 12× L40S, bf16/fp16 | ~8 days | eff. 1560 | 3 epochs / 82,028 steps | 2e-4, 10% warmup, cosine→0 | ~100.1B |
| CodonMamba | 71M | 4× A100 | ~7 days | — | 47 epochs | 4e-4 | — |
| NVIDIA CodonFM | 80M–1B | not published | not published | not published | not published | not published | not published |

Sources: CodonBERT [paper]; cdsFM [paper]; SynCodonLM [paper]; CodonMamba [paper]; CodonFM [card — values exist only in `experiment_scripts/pretraining/encodon_filtered/*.sh`].
Common settings: **AdamW**, weight decay **0.01**, dropout **0.1** (SynCodonLM, CodonMamba) [paper ×2].

### Cited Findings — evaluation protocol lessons

- **Use gene-held-out splits.** 98.3% gene overlap under 80/20 is the default failure mode [paper] — [CodonBench](https://www.biorxiv.org/content/10.64898/2026.08.12.744371v1.full).
- **Ship a `onehot_pos` memorisation canary** alongside every result. If (position, codon) identity beats the model, the benchmark is measuring memorisation [paper].
- **Use validation-based early stopping, never test-R²-based epoch selection** — the sign of the result flips [paper].
- **Match probe implementations** (scikit-learn MLPClassifier defaults manufactured a +4.7-point effect that a matched PyTorch MLP did not reproduce) and **report across ≥20 seeds**, as SynCodonLM does (5-fold CV repeated over 20–50 seeds, paired t-tests) [paper ×2].
- **Consider attention pooling over mean pooling** for codon-position-sensitive tasks — CodonBench names mean pooling as a likely reason the thin synonymous signal is undetectable [paper].
- **Control covariates in variant benchmarks**: CodonFM's 50 stratified subsampling iterations matching on reference codon, alternate codon, position in gene, pLI and local mutation rate is the strongest published design [paper].

### Cited Findings — what the wet lab says (the only experimental validation in this set)

Roche tested 68 MLM-designed variants (17 each from CaLM, EnCodon, CodonTransformer, CodonTransformer_AA) plus 17 random synonymous variants, 4 commercial vendor sequences, a max-CAI construct and the reference, with a SEAP reporter in HEK293A [paper] — [bioRxiv 2026.08.11.744178](https://www.biorxiv.org/content/10.64898/2026.08.11.744178v1.full):
- **Transient expression:** MLM variants median **1.76** vs **−0.31** for random; **83.8% of MLM variants exceeded the reference vs 5.9% of random**. EnCodon 17/17, CodonTransformer 15/17, CaLM 14/17, CodonTransformer_AA 11/17.
- **Stable integration:** MLMs did **not** differ from the reference on average (p=0.73); only 50% exceeded it. EnCodon 15/17, CaLM 12/17, CodonTransformer 6/17, CodonTransformer_AA 1/17. **The transient→stable drop is the most important practical caveat in the field.**
- MLM variants beat the **max-CAI** construct in stable expression (p=1.5×10⁻⁵) but were **not significantly different from commercial vendor sequences** (p=0.41, n=4).
- Median specific productivity: MLMs **1.20**, random **0.47**, vendor **0.57**.
- Backtranslation fidelity across 2,012 proteins: **CaLM mean pa = 0.246 vs EnCodon 0.124**; CaLM higher in 97% (1,954/2,012) of proteins. So the model that reconstructs native codons best is **not** the one that designs best-expressing sequences.
- **Local context dominates:** masking reduced per-amino-acid fidelity below 20% of its unperturbed value at a window of **17 codons**, and distal context alone could not restore predictions.
- Zero-shot on the tiny Nanoluc set (n=24): pseudo-log-likelihood correlated **−0.46 to −0.52** with abundance, while **GC content (0.597) and CAI (0.423)** were also predictive.
- Authors' conclusion: "no single model performed best across all benchmark tasks", "simple sequence features remained competitive in several settings", the benchmarking is "not conclusive", and sampling from several models is advisable for library design [paper].
- On linear probing: MLM embeddings generally beat k-mer baselines (exception: GFP, where k-mer beat all MLMs); **k-mer + MLM embeddings was best on all tasks except protein abundance**; in **5 of 9 tasks simpler k-mer or GC-based models were about as predictive** [paper].

### Inferences

- **The 17-codon locality finding is the most actionable architectural result in this whole set for kron-dna.** If the predictive signal is overwhelmingly local (~17 codons ≈ 51 nt) and CDS average ~407 codons, then a matrix-state linear RNN is not making a sacrifice relative to full attention here — it is well matched to the task's actual receptive-field requirements. Worth citing as explicit motivation.
- **A concrete, defensible kron-dna v1 spec grounded entirely in the above:** RefSeq CDS; frame/AUG/stop/alphabet filters; MMseqs2 per-species at 90/80 (SynCodonLM) with group-balanced stratified sampling; 64+5 codon vocabulary; 1,024-codon context with sequence packing; 15% masking plus **either** HELM's hierarchical loss (α=0.2) **or** SynCodonLM's synonymous-constrained softmax; species as token-type IDs over RSCU k-means clusters; AdamW, LR 2e-4, cosine, wd 0.01, dropout 0.1; ~50–100M params; target ~100B tokens. SynCodonLM reached that on **12× L40S in 8 days**, so a handful of A100s for 1–2 weeks is the right order of magnitude. Evaluate on the CodonBERT seven tasks **plus** the SynCodonLM seven **under CodonBench's gene-held-out protocol with the onehot_pos canary**.
- **Add a k-mer/GC/CAI baseline to every table.** Two independent 2026 groups (Roche, CodonBench) found simple frequency features competitive or better; a paper that omits them in 2026 will be rejected.
- Publishing the compute numbers would itself be a differentiator — NVIDIA did not.

### Gaps

- **No ablation anywhere of redundancy-reduction stringency** (40% aa vs 90% nt clustering) on downstream performance.
- **No paper explicitly describes pseudogene removal** from a CDS training corpus.
- **No paper describes sequence packing** for codon LMs, despite the ~400-codon mean vs 1,024–2,048 context mismatch.
- **NVIDIA CodonFM's compute, batch size, sequence length and token budget are unpublished** and sit only in repo shell scripts.
- SynCodonLM's GPU-hours are not stated (only 12× L40S × ~8 days).
- cdsFM's batch size, LR schedule, token count and deduplication procedure are not reported.
- CodonMamba's batch size and software license are not stated.
- No data on whether a **gated-delta-rule / DeltaNet-family** recurrence has ever been tried on CDS — the closest are Mamba (CodonMamba, Orthrus, HELM ablation) and Hyena (HELM ablation). kron-dna appears to be genuinely first here.
