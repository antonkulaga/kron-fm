# Recurrent, state-space and linear-attention architectures for DNA (2024 to Nov 2026): analogues for a KronLSTM DNA model

Scope: non-transformer and hybrid DNA/genomic sequence models, read as analogues for "kron-dna" (a Gated-DeltaNet-family matrix-state delta-rule RNN with channel-wise decay, erase/write gates, short convolution and FLA chunkwise Triton kernels), to be trained on a few A100s for days to weeks. Pure-transformer DNA models appear only as comparison points.

Provenance tags used throughout: **[paper]** = stated in the paper (abstract or full text as fetched); **[code/README]** = GitHub README or repo; **[HF]** = Hugging Face model card or metadata; **[secondary]** = blog or summary site, not checked against the primary source; **[unverified]** = seen only in a search snippet or recalled, not confirmed in a primary source during this research.

Research date: 2026-11-09.

---

## 1. Which DNA models use Mamba/Mamba-2, Hyena/StripedHyena, xLSTM, RWKV, linear attention, DeltaNet/Gated DeltaNet, RetNet, S4/S5, or hybrids? (dates, links, sizes, context, tokenization, data, results vs transformers)

### Takeaway
Mamba-family models dominate the small (<10M parameter) bidirectional DNA encoders: Caduceus, JanusDNA and PlantCaduceus/PlantCAD2. Hyena/StripedHyena dominates large causal generative models: HyenaDNA, Evo and Evo 2. xLSTM (DNA-xLSTM) matches or slightly beats Mamba at 0.5 to 4M parameters. Two items use Gated DeltaNet on biological sequences: one small DNA model (HAD, 1.1M parameters, distilled) and one protein model (ProtSyntax). I found no RWKV or RetNet DNA model. Every recent model that scales well is a hybrid with a few attention layers (HybriDNA 7:1 Mamba-2:attention, Evo 2 with 5 MHA layers, HAD with one attention layer, JanusDNA with attention fusion). At 1 to 8M parameters and single-nucleotide tokens, these recurrent models match or beat transformers that are 50 to 500x larger on histone and enhancer tasks. They consistently lose on splice-site tasks.

### Cited Findings

#### Catalogue (the most relevant entries first)

**Gated DeltaNet / delta rule**
- **HAD (Hybrid Architecture Distillation), 27 May 2025, arXiv preprint (cs.LG / q-bio.GN).** Student: **4 bidirectional Gated DeltaNet blocks (d=128) plus one FlashAttention self-attention layer after the last GDN block**, followed by a gated MLP. About **1.1M parameters**; the paper also states 1M and 11M in places, so the count is inconsistent. Character-level tokens, GRCh38 (same data as HyenaDNA and Caduceus), sequence length 1026. Distilled from **NTv2-500M** (6-mer) with MLM, using a 6-mer-grouped masking strategy, MSE feature alignment on visible tokens, and cross-attention reconstruction of masked tokens. Bidirectionality: forward and reversed sequences go through **separate GDN modules**; the reversed output is flipped back and added. Results: Genomic Benchmarks average **0.875 vs Caduceus 0.866, HyenaDNA 0.853, Mamba 0.846, CNN 0.803**. On NT tasks it beats Caduceus (1.9M) on histone marks (H3K4me3 0.643 vs 0.544; H3K4me2 0.562 vs 0.487) and beats its own teacher on several histone tasks (H3K4me3 0.643 vs 0.410). It loses on splice sites (Splice-all 0.911 vs Caduceus 0.940 vs NT 0.983). No GPU or compute figures are reported. **[paper]** — [arXiv abs](https://arxiv.org/abs/2505.20836); [HTML](https://arxiv.org/html/2505.20836v1)
- **ProtSyntax, July 2026 bioRxiv (protein, not DNA).** A "Bi-Gated DeltaNet" module: **two parameter-independent Gated DeltaNet cores scan N→C and C→N**, and their states are fused by zero-parameter cross-gating, then concatenated and projected. The module uses 16 heads, per-head decay and write gates, and a rank-one gated delta update with normalized, SiLU-activated q/k. The model is about 4B parameters (1.0B active) with MoE, in a 33-layer hybrid with a **3:1 interleave of Bi-GDN and attention ("Geometric Gated Attention")**. In ablation, Bi-GDN adds **+4.3% MCC** on phosphorylation sites. Training uses 55-residue windows on a PTM-site corpus, so it is not a self-supervised foundation-model pretraining. **[paper]** — [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.07.18.739331v1.full)
- **LDARNet (DNA Adaptive Representation Network), 3 Jun 2026 (v2 4 Aug 2026), arXiv.** 110M parameters, MLM, H-Net-style learned chunking adapted to MLM, BiMamba-2 plus local attention with bidirectional routing. Fine-tuned on 27 NT and Genomic Benchmarks tasks: 15 of 18 wins among compact (<300M) models; best on 9 of 10 histone tasks. Learned boundaries give up to +14 pp on histone tasks over fixed-grid boundaries at the same FLOPs. **[paper, abstract]** One search-engine snippet listed Gated DeltaNet among LDARNet's methods; the abstract does not mention GDN, so GDN use in LDARNet is **[unverified]**. — [arXiv abs](https://arxiv.org/abs/2606.04552)
- I found no other DNA foundation model built on DeltaNet, Gated DeltaNet, Gated DeltaNet-2 or KDA, and no large-scale (>10M) GDN DNA pretraining (searches in this session).

**Mamba / Mamba-2 (incl. hybrids)**
- **Caduceus, ICML 2024 (arXiv Mar 2024).** BiMamba (forward plus reversed Mamba with **tied in/out projections**) and MambaDNA (RC-equivariant channel split). Variants: Caduceus-PH (post-hoc conjoining) and Caduceus-PS (parameter sharing). Character-level tokens; hg38 training split of about 35B tokens. MLM with 15% masking. Configurations: 1k context (4 layers, d=118/128/256), 32k (8 layers, d=256) and 131k (16 layers, d=256, 50k updates). Each batch holds 2^20 tokens. Sizes: about 470k parameters for Genomic Benchmarks and 1.9M for NT tasks. Caduceus-PH beats much larger attention models on 8 of 18 NT tasks (e.g. H3: 0.815 vs NT-v2 0.784). HyenaDNA is better on splice tasks. **[paper]** — [arXiv](https://arxiv.org/abs/2403.03234). Released checkpoints include `caduceus-ps_seqlen-131k_d_model-256_n_layer-16` (the most downloaded, ~20k) and `caduceus-ph_...` **[HF]** — [HF](https://hf.co/kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16)
- **JanusDNA, 22 May 2025, arXiv preprint (Charité/BIH, FU Berlin et al.).** Two independent unidirectional stacks, each with 8 layers of a Mamba block plus FFN. MoE with 16 experts replaces FFNs at a 0.5 ratio, and optional "mid-attention" replaces one Mamba block. The two directions are fused by FlexAttention over the concatenated 2T sequence with a leakage-preventing mask. **Every token is predicted from all others**, so every position contributes to the loss, unlike the ~15% of MLM. Sizes: 0.42M active / 1.8M total (width 32), 1.98M / 8.96M (width 72), 7.66M / 35.5M (width 144). Single-nucleotide tokens, hg38 only, trained at 1,024 and 131,072 context. Results: Genomic Benchmarks roughly on par with Caduceus (2 wins plus 1 tie of 8); claimed state of the art on 12 of 18 NT tasks (e.g. H3K14ac 0.685 vs Caduceus-PH 0.631 vs NT-v2 0.551). DNALongBench eQTL: beats Caduceus-PH (7.7M) and Enformer on 8 of 9 tissues. It claims 1M bp on one 80GB GPU, but experiments stop at 131k. **[paper]** — [arXiv HTML](https://arxiv.org/html/2505.17257v1)
- **HybriDNA, Feb 2025, arXiv (Microsoft Research AI for Science; co-authors include Tri Dao and Albert Gu).** Decoder-only, **Mamba-2 : Transformer = 7:1**, with attention at position 4 of every 8 layers and no positional embeddings. Sizes 300M, 3B and 7B. Single-nucleotide tokens. 845 species, ~160B training nucleotides (the paper also says ~200B tokens). Next-token prediction, with **echo embeddings** for understanding tasks. Context ramps 8k → 32k → 131k. In a 300M, 8k ablation, hybrid loss < pure Mamba-2 loss (figure only). BEND: 7B gives chromatin accessibility 0.84 vs Caduceus-PH 0.83. GUE results are close to NT-2.5B-MS. **[paper]** — [arXiv](https://arxiv.org/abs/2502.10807)
- **PlantCaduceus (2024) and PlantCAD2 (Aug 2025, bioRxiv; reportedly in Cell Genomics).** PlantCAD2 keeps the Caduceus architecture, single-nucleotide tokens and MLM, with **8,192 bp** windows (vs 512 in PlantCAD) on 65 angiosperm genomes. Sizes: Small 88M, Medium 311M, Large 694M. Maize accessible-chromatin AUPRC goes from 0.587 to 0.711 with the longer window. **[paper/README]** — [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.08.27.672609); [GitHub](https://github.com/plantcad/plantcad). The Cell Genomics publication is **[unverified]** (LinkedIn post only).
- **H-Net (dynamic chunking), Jul 2025.** DNA experiment on HG38 at d_model=512: isotropic Transformer 29M → PPL 2.769; isotropic Mamba-2 33M → 2.757; H-Net with Transformer main network 64M → 2.705; H-Net with Mamba-2 main network 66M → **2.697**. Encoder and decoder are mostly Mamba-2 (M3T1 / M4). H-Net reaches the isotropic models' perplexity with **~3.6x less data**. The authors say "Mamba layers are essential for effective byte-level sequence processing", and Mamba-2 beat a Transformer as the DNA main network. **[paper]** — [arXiv HTML](https://arxiv.org/html/2507.07955)
- **dnaHNet, Feb 2026 (Gu, Hsu, Goodarzi et al.).** Tokenizer-free autoregressive model with recursive dynamic chunking, trained on prokaryotic genomes. It claims better scaling and efficiency than StripedHyena 2 and >3x faster inference than Transformers. The abstract does not name the backbone. **[paper, abstract]** — [arXiv](https://arxiv.org/abs/2602.10603)
- **BarcodeMamba, Dec 2024 (NeurIPS 2024 FM4Science workshop).** Compares Mamba and Mamba-2 on DNA barcodes. It beats BarcodeBERT with 8.3% of its parameters; 99.2% species-level linear-probe accuracy. Tokenization and objective were ablated, but the abstract gives no details. **[paper, abstract]** — [arXiv](https://arxiv.org/abs/2412.11084)
- **eccDNAMamba, Jun 2025.** Bidirectional Mamba-2 encoder for circular DNA. **[unverified, snippet]** — [arXiv](https://arxiv.org/abs/2506.18940)
- **DGRNA, Oct 2024 (RNA).** 12 bidirectional Mamba-2 blocks plus a final attention layer. **[unverified, snippet]** — [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.10.31.621427v1.full)
- **GENATATOR Caduceus-PS multispecies (AIRI), 2026.** A 14.0M-parameter Caduceus-PS used for gene segmentation and transcript-type classification. **[HF metadata only]** — [HF](https://hf.co/AIRI-Institute/genatator-caduceus-ps-multispecies-segmentation)
- **Mamba-3 (Mar 2026)** is a general architecture ("matches strong baselines with half the decoding cost"). I found no DNA use. **[paper/blog]** — [arXiv](https://arxiv.org/html/2603.15569)

**Hyena / StripedHyena**
- **HyenaDNA, NeurIPS 2023.** Single-nucleotide tokens, up to **1M context**, "up to 160x faster than Transformer" training, and "first use of in-context learning in genomics". **[paper, abstract]** — [arXiv](https://arxiv.org/abs/2306.15794)
- **Evo 2 (bioRxiv Feb 2025; Nature 2026, Brixi et al., DOI 10.1038/s41586-026-10176-5).** StripedHyena 2 at 1B, 7B, 20B and 40B on OpenGenome2 (9T+ tokens), byte-level. Checkpoints: `evo2_1b_base` (8K), `evo2_7b_base` (8K), `evo2_7b` (1M), `evo2_7b_262k`, `evo2_20b` (1M, Feb 2026, "40B-level performance with double the speed"), `evo2_40b_base` (8K) and `evo2_40b` (1M). **[code/README]** — [GitHub](https://github.com/ArcInstitute/evo2); [HF evo2_20b](https://hf.co/arcinstitute/evo2_20b)
- **StripedHyena 2 systems paper (Ku et al., Feb 2025)**: see question 6. **[paper]** — [arXiv](https://arxiv.org/abs/2503.01868)

**xLSTM**
- **Bio-xLSTM / DNA-xLSTM, 6 Nov 2024, arXiv (JKU Linz).** DNA-xLSTM-500k (d=128, 5 sLSTM blocks, 1k context), DNA-xLSTM-2M (d=256, 6 sLSTM blocks, 1k) and DNA-xLSTM-4M (d=256, **9 mLSTM blocks**, RoPE, **32k** context). Character-level tokens, GRCh38. All three were trained with **both CLM and MLM**, for 10k steps × ~2^20 tokens (~10.5B tokens). Pretraining loss: 2M sLSTM beats Mamba, HyenaDNA and Transformer++ at 1k, with a larger gap for MLM. At 32k, the 4M mLSTM has the lowest loss, "comparable to Mamba (Caduceus-4M)". xLSTM-PS beats Caduceus-PS at 500k and 2M, and the gap widens with size. Downstream: best on 12 of 18 NT tasks among models <2M; wins 8 of 18 vs NT-v2 500M. On Genomic Benchmarks it is roughly on par with Mamba (5 of 8 wins vs Mamba-PH); e.g. Coding-vs-Intergenomic 0.931 (xLSTM-PH) vs 0.915 (Mamba-PH). Loss numbers appear only in figures. Compute: ~50k A100 GPU-hours across all Bio-xLSTM experiments (DNA, protein and chemistry together). **[paper]** — [arXiv](https://arxiv.org/abs/2411.04165); [code](https://github.com/ml-jku/DNA-xLSTM)

**Other linear RNN / SSM / linear attention**
- **"Leveraging State Space Models in Long Range Genomics", Apr 2025 (ICLR 2025 LMRL workshop spotlight).** Tests **Caduceus and Hawk (RG-LRU linear RNN)** against a 50M transformer trained the same way. The SSMs "match transformer performance", extrapolate zero-shot to contexts **10 to 100x longer** than trained, and process 1M tokens on one GPU. **[paper, abstract]** — [arXiv](https://arxiv.org/abs/2504.06304)
- **Lyra, Mar 2025 (Broad/MIT/Harvard).** S4D plus projected gated depthwise convolution, motivated by epistasis. Tested on 100+ biological tasks (protein fitness, RNA function, CRISPR guides). It claims up to 120,000x fewer parameters than foundation models and every task runnable on ≤2 GPUs in <2 h. **[paper, abstract]** — [arXiv](https://arxiv.org/abs/2503.16351)
- **M5 (2024).** Encoder-only transformer with linear attention for multi-megabase bacterial whole genomes. The approximation is tight only when the query-key dimension is low. **[paper snippet]** — [arXiv](https://arxiv.org/abs/2407.03392)
- **RWKV and RetNet:** targeted searches found **no RWKV- or RetNet-based DNA foundation model**. **[search result, absence]**

**Comparison points (non-recurrent)**
- **ConvNova (ICLR 2025).** A dilated, gated, dual-branch CNN of ~7M parameters that "significantly outperforms" HyenaDNA and Caduceus on more than half of the tasks. Example: BEND gene finding MCC 0.55 vs HyenaDNA 0.35; ~1.35x faster than HyenaDNA at 1M bp. Per-task numbers are **[secondary]**. — [arXiv](https://arxiv.org/abs/2502.18538); [ICLR](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5e0da5da69b71349ae0bd7ad716e4bc9-Abstract-Conference.html)
- **Learned-tokenizer transformers, 2025–26:** MergeDNA (AAAI 2026, token merging plus full-attention latent encoder) — [arXiv](https://arxiv.org/abs/2511.14806); DNACHUNKER (ICML 2026, MLM with learnable segmentation) — [arXiv](https://arxiv.org/abs/2601.03019); GeneZip (Feb 2026, H-Net-style routing at 137.6 bp per token, 70M/636M, 128K context trains on one A100 80GB) — [arXiv](https://arxiv.org/abs/2602.17739). **[paper, abstract]**
- **Tang & Koo (Genome Biology 2025):** within cis-regulatory elements, gLM representations gave "little to no advantage" over one-hot encodings. **[paper, snippet]** — [Springer](https://link.springer.com/article/10.1186/s13059-025-03674-8)

### Inferences
- **The closest published analogue to kron-dna is HAD.** It is small (4 × d128), bidirectional, Gated DeltaNet plus one attention layer, and character-level on hg38, and it beat Caduceus on Genomic Benchmarks and histone tasks. It was distilled from NTv2-500M, though, and never ablated GDN against Mamba. So it shows only that GDN is a viable DNA mixer, not that GDN is better.
- Across HAD, JanusDNA, Caduceus and DNA-xLSTM, models of 1 to 8M parameters reach a ceiling on the NT and Genomic Benchmarks suites where architecture differences are 0.01 to 0.03. Splice-site tasks remain the one place where large k-mer transformers win, and they test local motif sharpness. A KronLSTM model should be judged on long-range and zero-shot tasks, not only on NT and Genomic Benchmarks.
- The strong CNN result (ConvNova) and Tang & Koo's one-hot parity mean "recurrent beats transformer" claims on short benchmarks partly reflect weak baselines. A strong CNN baseline is cheap and should be included.

### Gaps
- No DNA paper compares Gated DeltaNet, Mamba-2, mLSTM and Hyena at matched parameters and data. Bio-xLSTM compares xLSTM, Mamba, Hyena and Transformer++, but not delta-rule models.
- No RWKV-7 or RetNet DNA models were found. RWKV-7 is itself a generalized delta rule, so its absence matters for novelty.
- I could not confirm whether LDARNet uses GDN.
- Orthrus (Mamba RNA model) and LC-PLM (BiMamba-S protein model, [arXiv 2411.08909](https://arxiv.org/abs/2411.08909)) are adjacent and were not fetched; treat them as **[unverified]**.

---

## 2. Reverse-complement handling: Caduceus RC-equivariance (BiMamba, MambaDNA), post-hoc conjoining, RC augmentation — what helped, at what cost, and how it applies to a delta-rule RNN

### Takeaway
Two recipes work, and neither dominates. **PH** (RC augmentation in pretraining plus averaging forward and RC predictions at inference) was best overall on Genomic Benchmarks in Caduceus, but costs 2x inference. **PS** (an architecturally equivariant channel split) gave lower MLM loss and better long-range variant-effect prediction, but has high variance on some tasks. Augmentation alone leaves about 10% of predictions strand-inconsistent unless outputs are averaged. The MambaDNA channel-split wrapper is operator-agnostic, so it can wrap a delta-rule layer unchanged; with a causal objective, it would leak future tokens.

### Cited Findings
- **MambaDNA construction.** Split channels into two halves. Half A goes through the sequence operator as is. Half B is reverse-complemented, run through **the same operator (shared weights)** and reverse-complemented back. Then concatenate. The channel count is not doubled (unlike RCPS from Shrikumar et al.). The PS embedding produces D/2 channels for the sequence and for its RC; the LM head sums one half's projection with the channel-flipped projection of the other. **[paper]** — [Caduceus](https://arxiv.org/html/2403.03234)
- **PH vs PS outcomes in Caduceus.** PH is pretrained with RC augmentation and averages over the sequence and its RC at inference, which costs two forward passes. PS is pretrained without augmentation. On Genomic Benchmarks, PH is best overall (5 of 8 wins); PS wins Mouse Enhancers (0.793), Human Enhancer Ensembl (0.900) and Human Regulatory (0.873). On NT tasks, PS has high variance: H3K14ac 0.541 ± 0.212, Donor 0.874 ± 0.289. In long-range eQTL variant-effect prediction, **PS beats NT-v2 500M as distance to the TSS grows, and beats Enformer beyond 100 kb** (figure only). RC equivariance lowered MLM pretraining loss (figure only). The paper does not quantify the compute cost of RC handling. **[paper]** — [Caduceus](https://arxiv.org/html/2403.03234)
- **The same construction transfers to xLSTM.** DNA-xLSTM-PS uses the identical split (half direct, half RC → block → RC). xLSTM-PS beats Caduceus-PS in pretraining loss at 500k and 2M parameters, and the gap widens with size. PH models used RC augmentation plus summed forward and RC outputs downstream. On Genomic Benchmarks (500k): xLSTM-PH vs xLSTM-PS is 0.931 vs 0.934 on Coding-vs-Intergenomic and 0.920 vs 0.902 on Enhancers Ensembl, so neither dominates. **[paper]** — [Bio-xLSTM](https://arxiv.org/abs/2411.04165)
- **How strand-inconsistent current models are (MirrorDNA, bioRxiv, posted 6 Nov 2026).** NT-v2, DNABERT-2 and HyenaDNA change **25–31% of predictions** when test sequences are reverse-complemented. A model trained with RC augmentation still changes **about 1 in 10** unless its outputs are averaged over both strands. BPE and k-mer tokenizers split **92–100% of TF motifs differently on the two strands**. MirrorDNA (2.3M parameters, strand-symmetric learned tokenizer) more than doubles the "correct on both strands" MCC of those models. It is more accurate than Caduceus trained on the same data, and stays accurate on inputs 16x longer than training. The backbone is not stated in the abstract. **[paper, abstract]** — [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.11.04.764207v1)
- **JanusDNA** feeds both the forward and the RC sequence to the same model and pools the representations; no RC ablation is reported. **[paper]** — [JanusDNA](https://arxiv.org/html/2505.17257v1)
- **TrinityDNA (AAAI 2026)** proposes a "Gated Reverse Complement (GRC)" module; the mechanism and numbers are not in the abstract. **[paper, abstract]** — [arXiv](https://arxiv.org/abs/2507.19229)
- **HAD** uses no explicit RC mechanism. **[paper]** — [HAD](https://arxiv.org/html/2505.20836v1)

### Inferences
- **Applying MambaDNA PS to a delta-rule layer.** The wrapper needs only that the inner operator maps (B, L, D/2) → (B, L, D/2). A KronLSTM/GDN layer (q, k, v, gates, short conv) can sit inside unchanged. Implementation: `y = cat(op(x[..., :D/2]), RC(op(RC(x[..., D/2:]))))`, where RC = flip along L plus a complement-consistent channel permutation (Caduceus uses a channel flip). The cost per layer is one operator call on a batch of 2 × (D/2) channels. That keeps FLOPs about equal to a width-D layer but halves the number of distinct parameters per width, i.e. less capacity per FLOP; this is my reading, not measured in the papers. The short causal convolution inside the layer is fine, because the RC'd half sees it flipped by construction.
- **The PS wrapper and causal LM do not mix.** The RC half processes the reversed sequence, so position t's output for that half depends on tokens >t. That leaks future tokens in next-token prediction. Every PS model I found (Caduceus, DNA-xLSTM-PS, PlantCAD) is MLM or bidirectional. For a causal KronLSTM, the choices are PH (RC augmentation plus averaging) or training on both strands as separate sequences. A PS wrapper used with a causal objective also acts as an implicit bidirectional layer.
- **Cost-benefit for a small budget.** PH costs nothing in training (augmentation) and 2x at inference. PS costs nothing extra at inference but has variance issues and is restricted to MLM. Given MirrorDNA's ~10% residual inconsistency with augmentation alone, averaging both strands at evaluation is cheap insurance in any case.
- Character-level tokenization (already the kron-dna default) keeps strand symmetry exact. Moving to BPE or k-mers would break it (92–100% of motifs split asymmetrically).

### Gaps
- No paper measures the wall-clock or FLOP cost of PS vs PH for an SSM or linear RNN.
- RC flip rates for Caduceus-PH and Caduceus-PS were not reported by MirrorDNA (abstract only).
- No study applies RC equivariance to a delta-rule or GDN layer specifically.

---

## 3. Bidirectionality for MLM with recurrent layers vs causal autoregressive training — which objective works better for small recurrent DNA models

### Takeaway
On classification benchmarks, the best small (<10M) recurrent DNA models are all bidirectional: Caduceus BiMamba, DNA-xLSTM-MLM, HAD bidirectional GDN and JanusDNA. Causal HyenaDNA trails them on most tasks except splice sites. Tying the forward and backward projections (Caduceus) beat untied stacks at half the depth. Three alternatives aim to get both understanding and generation: JanusDNA's all-tokens objective, HybriDNA's echo embeddings for causal models, and MDLM-style masked diffusion. No controlled study compares CLM and MLM for small recurrent DNA models at matched parameters and data.

### Cited Findings
- **Caduceus:** weight-tied BiMamba (shared in/out projections; the projections hold most of a Mamba block's parameters) **beats untied bidirectional models at half the depth**. Mamba beats HyenaDNA in MLM cross-entropy at matched size and length (figure only). **[paper]** — [Caduceus](https://arxiv.org/html/2403.03234)
- **Bio-xLSTM bidirectional mLSTM.** The default runs the same-weight mLSTM on x and on flip(x) and sums the outputs. Because mLSTM gates depend only on the current step, the backward cell and normalizer states can be accumulated in the same pass ("h = h⁺ + h⁻"); this is mathematically equivalent and uses less memory. **For long context, they use unidirectional cells with the direction alternating at each block.** At 2M/1k, the MLM gap over Mamba and Transformer is larger than the CLM gap. **[paper]** — [Bio-xLSTM](https://arxiv.org/abs/2411.04165)
- **HAD (GDN):** separate forward and reversed GDN modules, summed, plus one attention layer. Validation MLM loss: GDN without attention 1.0090, with attention 1.0033. The full model "significantly outperformed" the variants (figure only). There is no unidirectional ablation. **[paper]** — [HAD](https://arxiv.org/html/2505.20836v1)
- **ProtSyntax (GDN, protein):** two parameter-independent GDN directions with cross-gating, rather than summation. **[paper]** — [ProtSyntax](https://www.biorxiv.org/content/10.64898/2026.07.18.739331v1.full)
- **JanusDNA objective:** predicting each token from all others with two independent stacks and masked attention fusion gave higher last-token accuracy than MLM in a toy test. It ran in ~27 min per 1,000 steps, ~2x faster than the masked model (single-layer toy). Mid-attention helped at width 32, hurt at width 72 and was neutral at 128. MoE lowered perplexity. **[paper]** — [JanusDNA](https://arxiv.org/html/2505.17257v1)
- **HybriDNA (causal):** echo embeddings (repeat the sequence, mean-pool the second copy) supply bidirectional context to a causal model for understanding tasks. At 300M on BEND it scores below Caduceus-PH on chromatin accessibility (0.78 vs 0.83); 3B and 7B match or exceed it. **[paper]** — [HybriDNA](https://arxiv.org/abs/2502.10807)
- **MDLM (NeurIPS 2024):** DNA models pretrained with masked diffusion give "similar or higher downstream performance than classical BERT-style training" and can also generate. **[paper, snippet]** — [arXiv](https://arxiv.org/abs/2406.07524). D3LM (Mar 2026) is a discrete DNA diffusion LM for "bidirectional DNA understanding and generation". **[title only]** — [arXiv](https://arxiv.org/abs/2603.01780)
- **Downstream comparisons at matched size (Caduceus and HAD tables):** causal HyenaDNA (1.6M) is below Caduceus (1.9M) on most NT tasks but **better on splice sites** (Donor 0.949 vs PH 0.948 vs PS 0.874). On Genomic Benchmarks: HyenaDNA 0.853, Caduceus 0.866, HAD 0.875. **[paper]** — [Caduceus](https://arxiv.org/html/2403.03234); [HAD](https://arxiv.org/html/2505.20836v1)
- **Causal and generative evidence at scale:** Evo 2 and HybriDNA are causal; HybriDNA-300M beats HyenaDNA on regLM enhancer design (HepG2 top-1 activity 7.3 vs 5.5). **[paper]** — [HybriDNA](https://arxiv.org/abs/2502.10807)

### Inferences
- **Recommended default for a small KronLSTM encoder:** a weight-tied bidirectional block. Run the same delta-rule kernel on x and flip(x), sharing q/k/v/gate projections (Caduceus-style tying) and summing or gating the outputs. With FLA kernels this is just a second kernel call on a flipped tensor, so cost is about 2x mixer FLOPs and no extra projection parameters.
- **Unlike mLSTM, a delta-rule state cannot be accumulated in both directions within one scan.** The erase term (I − β k kᵀ) makes S_t depend nonlinearly on the order of updates, so the forward and backward states are genuinely different objects. The Bio-xLSTM single-pass trick therefore does not carry over, and the two scans are needed (my inference from the recurrence).
- **Cheapest bidirectional option for long context:** alternate the scan direction per block (Bio-xLSTM's long-context recipe). The cost equals a causal model, and the receptive field becomes bidirectional after two blocks. It suits 32k–131k pretraining on A100s.
- **Choosing the objective:** if the main targets are variant-effect and regulatory classification, MLM or JanusDNA-style bidirectional training wins at small scale. If generation and likelihood-based zero-shot VEP matter (Evo-style), use causal training plus echo embeddings, or train causal first and adapt later. The NLP result that causal pretraining then masked adaptation was best is a hint, but it is not DNA evidence.
- The JanusDNA objective is attractive for a recurrent model. It gives a loss on every token (more signal per token than 15% MLM) with two independent causal stacks, which map directly onto causal FLA kernels. The cost is one attention-fusion layer over 2T tokens, which becomes the bottleneck at 131k+.

### Gaps
- No controlled CLM vs MLM vs JanusDNA objective comparison exists for small recurrent DNA models on identical data. Bio-xLSTM trained both objectives but reports only pretraining losses, not a downstream CLM vs MLM comparison.
- No study compares bidirectional fusion methods (sum, concat plus projection, cross-gating, attention fusion) for linear RNNs on DNA.

---

## 4. Long-context behaviour: do recurrent DNA models actually use long context? Character-level tokenization effects

### Takeaway
The evidence is mixed and mostly negative for current DNA LMs. Expert supervised models (Enformer, Akita and similar) beat Caduceus and HyenaDNA on every DNALongBench task. Benchmarks find little or inconsistent gain from 6k→196k inputs for Caduceus, and Evo 2's sequence recovery saturates after ~1 kb of context. Gains on eQTL from 8k→131k are small (HybriDNA 300M: AUROC 0.71→0.74). On the positive side, the SSMs extrapolate to 10–100x their training length, Evo 2 perplexity keeps dropping out to 1M, and Caduceus-PS's VEP advantage grows with distance to the TSS. Character-level tokens are standard for recurrent DNA models and keep strand symmetry. The 2025–26 trend is learned, dynamic chunking (H-Net-style) for efficiency, with Mamba-2 layers at the byte level.

### Cited Findings
- **DNALongBench (Cheng et al., Nature Communications 2025):** five tasks with sequences up to 1M bp. DNA FMs "capture long-range dependencies to some extent", but **expert models outperform them across all tasks**. Official eQTL averages, as reported by GeneZip: expert 0.681, CNN 0.528, **HyenaDNA 0.514, Caduceus-PH 0.566, Caduceus-PS 0.538** AUROC. **[paper via secondary table]** — [GeneZip PDF](https://arxiv.org/pdf/2602.17739); [DNALongBench](https://www.nature.com/articles/s41467-025-65077-4)
  - JanusDNA reports much higher AUROCs on DNALongBench eQTL for the same baselines (e.g. Caduceus-PH Nerve Tibial 0.842; Enformer Artery Tibial 0.741). The eQTL numbers are therefore **not comparable across papers**; fine-tuning protocols differ. — [JanusDNA](https://arxiv.org/html/2505.17257v1)
- **Nature Communications 2025 benchmark of DNA FMs:** extending input length helped HyenaDNA significantly, but the gain for Caduceus-PH was not significant. The short Caduceus-PH beat the long version on pathogenic SNPs. 6k→~196k gains were "not uniformly substantial". **[paper, snippet]** — [Nature Comms](https://www.nature.com/articles/s41467-025-65823-8)
- **HybriDNA on Genomics LRB (300M):** causal eQTL AUROC 0.71 at 8k, 0.72 at 32k and 0.74 at 131k; baselines 0.68–0.72. OMIM zero-shot AUROC ~0.51 for all models (chance). **[paper]** — [HybriDNA](https://arxiv.org/abs/2502.10807)
- **Genomics Long-Range Benchmark (Kao, Trop, …, Kuleshov; ICLR 2024 MLGenX).** Tasks: CAGE prediction, bulk RNA expression (218 tissues) and variant effect on gene expression. Context extension was explored for NT, with better performance at larger context. **[paper/dataset card]** — [ICLR](https://iclr.cc/virtual/2024/23032); [HF dataset](https://huggingface.co/datasets/InstaDeepAI/genomics-long-range-benchmark). OpticalDNA (Feb 2026) reports 0.852 average eQTL AUROC over nine GTEx tissues. **[snippet]** — [arXiv](https://arxiv.org/abs/2602.02014)
- **Evo 2 context use (GCP preprint, Jan 2026):** Evo 2 shows "limited improvement in recovery accuracy once the input length exceeds approximately 1,024 base pairs", whereas the GENERator transformer keeps improving. Prior work is cited showing that SSM context exploitation "often saturates as sequence length increases". **[paper]** — [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.01.27.702015v1.full)
- **Evo 2 / SH2 context extension (7B):** validation perplexity 2.782 (32K) → 2.763 (65K) → 2.748 (131K) → 2.707 (262K) → 2.663 (524K) → **2.597 (1M)** with PI+ABF, and all models did in-context recall at the target length. **[paper]** — [SH2 HTML](https://arxiv.org/html/2503.01868v1). Evo 2 retrieves a 100 bp "needle" from a 1 Mbp random-DNA "haystack" with high accuracy (categorical-Jacobian retrieval score). **[secondary]** — [blog summary](https://mcneela.github.io/post/evo2)
- **SSM extrapolation:** Caduceus and Hawk match a 50M transformer on long-range genomics tasks and extrapolate zero-shot to 10–100x the training context. **[paper, abstract]** — [arXiv](https://arxiv.org/abs/2504.06304)
- **Caduceus-PS eQTL** improves relative to NT-v2 as distance to the TSS grows, and beats Enformer beyond 100 kb. **[paper]** — [Caduceus](https://arxiv.org/html/2403.03234). **PlantCAD2:** 512→8,192 bp lifted maize accessible-chromatin AUPRC from 0.587 to 0.711. **[paper]** — [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.08.27.672609)
- **Tokenization:**
  - Bioinformatics 2025 compared 3 attention and 3 state-space gLMs (SSMs pretrained on hg38 at 4k context) on 44 tasks, including a controlled 4-layer, d256, L4096 Mamba with character vs BPE tokens. Tokenizer choice "can significantly impact" performance, especially on motif-detection tasks. **[paper]** — [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12453675/)
  - GENERator (a transformer) found 6-mer > single-nucleotide and BPE for causal DNA LM. **[paper, snippet]** — [arXiv](https://arxiv.org/abs/2502.07272)
  - Bio-xLSTM notes character-level tokens may limit scaling to larger contexts. **[paper]** — [Bio-xLSTM](https://arxiv.org/abs/2411.04165)
  - H-Net reaches the same perplexity with ~3.6x less data on DNA using Mamba-2 byte-level encoders. **[paper]** — [H-Net](https://arxiv.org/html/2507.07955)
  - LDARNet's learned boundaries give up to +14 pp on histone tasks over a fixed grid at matched FLOPs. **[paper]** — [arXiv](https://arxiv.org/abs/2606.04552)
  - GeneZip compresses to 137.6 bp per token, allocating more tokens to repeats and cis-regulatory regions. **[paper]** — [arXiv](https://arxiv.org/abs/2602.17739)

### Inferences
- With a few A100s, the evidence favours pretraining mostly at **8k–32k** with a short extension phase to 131k, rather than investing early in 1M context. Measured gains above ~32k are small (eQTL +0.02–0.03 AUROC) and model-dependent. HyenaDNA-style species classification and Evo-scale perplexity are the exceptions.
- Recurrent models extrapolate past their training length (Caduceus and Hawk at 10–100x), so train short and evaluate long. A delta-rule model with data-dependent decay should extrapolate at least as well as Hawk. That needs testing, because GDN's S-NIAH retrieval degrades past 4k (question 5).
- Character-level tokens remain right for kron-dna: RC symmetry, motif precision and no tokenizer to tune. A later H-Net-style chunking stage (a Mamba-2 or KronLSTM byte encoder, then a coarser main network) is the most promising 2025–26 route to cheaper long context.
- Long-range task results disagree heavily between papers (DNALongBench eQTL 0.5 vs 0.8 AUROC for the same baselines). Use one fixed evaluation harness and rerun Caduceus as a baseline yourself.

### Gaps
- I did not retrieve the Genomics LRB paper's numeric context-extension results, and found no perturbation or "effective receptive field" study for Caduceus, HyenaDNA or xLSTM that measures how far back the models actually attend.
- No study measures long-context use for delta-rule or GDN DNA models.
- The Evo 2 needle-in-haystack numbers come from a secondary summary.

---

## 5. Evidence on Gated DeltaNet / DeltaNet / delta-rule models on biological sequences, and relevance of state tracking and associative recall to genomics

### Takeaway
Direct evidence is thin: one small DNA model (HAD) and one protein model (ProtSyntax), both bidirectional GDN with some attention, both positive. Neither isolates GDN against Mamba. From general LM work, GDN improves single-needle retrieval over Mamba-2 and DeltaNet at 2–4k, but still collapses at 8k on harder retrieval. Gated DeltaNet-2 raises those numbers modestly. This supports keeping a few attention or sliding-window layers, which every scaled DNA hybrid already does. The tie between associative recall and genomics is argued only through needle-in-haystack and in-context recall tests. No paper links synthetic recall ability to genomic task performance.

### Cited Findings
- **HAD (DNA):** 4 bidirectional GDN blocks plus one attention layer, 1.1M parameters; Genomic Benchmarks average 0.875, beating Caduceus 0.866 (details in question 1). Adding attention slightly lowered MLM validation loss (1.0090 → 1.0033). **[paper]** — [HAD](https://arxiv.org/html/2505.20836v1)
- **ProtSyntax (protein):** Bi-GDN in a 3:1 hybrid with attention, +4.3% MCC in ablation. **[paper]** — [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.07.18.739331v1.full)
- **Gated DeltaNet (ICLR 2025), 1.3B, zero-shot S-NIAH accuracy (%):**
  - S-NIAH-1 at 1K/2K/4K/8K: DeltaNet 97.4/96.8/99.0/98.8; Mamba-2 99.2/98.8/65.4/30.4; GDN 98.4/88.4/91.4/91.8.
  - S-NIAH-2: DeltaNet 98.4/45.6/18.6/14.4; Mamba-2 99.4/98.8/56.2/17.0; **GDN 100/99.8/92.2/29.6**.
  - S-NIAH-3 (1K/2K/4K): GDN 86.6/84.2/27.6.

  **[paper]** — [arXiv](https://arxiv.org/html/2412.06464)
- **Gated DeltaNet-2 (NVIDIA, 21 May 2026):** S_t = (I − k_t (b_t ⊙ k_t)ᵀ) Diag(α_t) S_{t−1} + k_t (w_t ⊙ v_t)ᵀ, with a channel-wise erase gate b (key axis), a channel-wise write gate w (value axis) and channel-wise decay α computed in fp32. Tying both gates to one scalar recovers KDA, and also tying the decay recovers GDN. At 1.3B and 100B tokens:
  - RULER S-NIAH-2 at 8K: **39.2 vs 32.0** for GDN; MK-NIAH-1 at 1K: 72.6 vs 58.0.
  - The [0,2] erase-gate (negative-eigenvalue) option gave no consistent gain.
  - Retrieval-suite average: erase-only 29.51, write-only 28.92, full 29.88.
  - No biological experiments.

  **[paper]** — [arXiv](https://arxiv.org/html/2605.22791v1)
- **Genomic recall evidence:** in the SH2 context extension, "all models did in-context recall at the target length" up to 1M **[paper]** — [SH2](https://arxiv.org/html/2503.01868v1). Evo 2's DNA needle-in-haystack test: 100 bp needle in 1 Mbp **[secondary]** — [summary](https://mcneela.github.io/post/evo2). HyenaDNA claims the first use of in-context learning in genomics **[paper]** — [arXiv](https://arxiv.org/abs/2306.15794). The Bio-xLSTM title includes "in-context learning", applied to protein and chemical sequences with homology context **[paper]** — [arXiv](https://arxiv.org/abs/2411.04165).
- The GCP preprint argues that as context grows, more of the sequence must be compressed into fixed-size states, limiting long-range understanding in SSMs. **[paper]** — [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.01.27.702015v1.full)
- **Attention pockets in scaled DNA hybrids:**
  - HybriDNA: 1 attention layer per 8; the hybrid beats pure Mamba-2 in loss. — [HybriDNA](https://arxiv.org/abs/2502.10807)
  - StripedHyena 2: 5 MHA operators interleaved with the convolution blocks in 32 layers. — [SH2](https://arxiv.org/html/2503.01868v1)
  - HAD: 1 attention layer. — [HAD](https://arxiv.org/html/2505.20836v1)
  - JanusDNA: attention fusion; mid-attention helps only at width 32. — [JanusDNA](https://arxiv.org/html/2505.17257v1)

  **[paper]**

### Inferences
- **Why recall matters for DNA:** genomes are full of exact and near-exact repeats (tandem repeats, transposable elements, segmental duplications, paralogous motifs). A causal model gets large loss reductions from copying, which is the induction-head and associative-recall mechanism. GeneZip spending more tokens on TE-derived and tandem repeats points the same way. The delta rule's targeted overwrite (erase then write) suits copying with a finite state better than additive linear attention or Mamba-2. The S-NIAH-1 and S-NIAH-2 results suggest this, but on DNA it is unmeasured.
- **Watch for repeat-copying inflating perplexity.** A delta-rule model may reach low DNA perplexity mostly by copying repeats, which helps little on regulatory tasks. Report loss on repeat-masked positions separately (RepeatMasker soft-mask in hg38), or downweight repeats.
- Because GDN retrieval degrades at 4–8k and DNA needs 8k+, include 1 attention or sliding-window layer per 4–8 KronLSTM layers, or evaluate whether GDN-2's channel-wise gates remove the need. A small DNA needle-in-haystack and copy test (random DNA plus a planted repeat at distance d) is a cheap diagnostic to build.
- **Channel-wise erase and write gates (GDN-2 style)** could also help strand- and position-specific motif memory, since different channels can hold different motif families at different decay rates. This is speculative.

### Gaps
- No DNA study pits GDN against Mamba-2, mLSTM or Hyena at matched size.
- No DNA MQAR, copy or state-tracking benchmark exists.
- No paper reports GDN or delta-rule results on RNA, beyond the protein ProtSyntax.
- No paper relates synthetic recall scores to genomic downstream performance.

---

## 6. Evo 2's StripedHyena 2 and small checkpoints / ablations: what was learned about multi-hybrid designs at small scale

### Takeaway
The only public operator-layout ablation is at **7B / 400B tokens**: mixing short, medium and long Hyena operators (SE-MR-LI, PPL 2.83) beat all-long-convolution (2.87) and all-attention (3.09) blocks. Short explicit filters can replace long ones with little loss (SE-SE-LI 2.88) and run faster. No sub-1B ablations are published. The smallest checkpoint, `evo2_1b_base` (8K context), officially needs FP8 on Hopper GPUs, so it is not a drop-in A100 baseline. The 7B checkpoints run in bf16.

### Cited Findings
- **Operators:**
  - Hyena-SE: explicit filters of length 4–7 (filters <14 train fine with plain explicit parametrization).
  - Hyena-MR: explicit filters "in the hundreds" (128 in the layout experiments) with an exponential-decay regularizer.
  - Hyena-LI: implicit filters as long as the sequence, built from real exponentials, with a recurrent form available.

  **[paper]** — [SH2 HTML](https://arxiv.org/html/2503.01868v1)
- **Layout ablation (7B, 400B OpenGenome2 tokens, validation PPL):** MHA-MHA-MHA 3.09; LI-LI-LI 2.87; SE-SE-LI 2.88; **SE-MR-LI 2.83**. Blocks repeat to depth 32, with 5 MHA operators interleaved. **[paper]** — [SH2 HTML](https://arxiv.org/html/2503.01868v1)
- **Systems:** 1.2–2.9x faster end-to-end training than optimized Transformers at 7B and 40B (H100, FP8), and 1.1–1.4x faster than previous-generation hybrids. At operator level (width 4096), ~2x throughput over linear attention and SSMs. Also overlap-add blocked kernels and all-to-all or point-to-point context parallelism. The 7B base trained on 2T tokens at 8,192 context. **[paper]** — [arXiv](https://arxiv.org/abs/2503.01868)
- **Context extension 32K→1M:** see question 4 (PPL 2.782 → 2.597). **[paper]**
- **Checkpoints and hardware:** the 1B, 20B and 40B models need Hopper GPUs with FP8 via Transformer Engine "for numerical accuracy". The 7B models run in bf16 without Transformer Engine. The 20B (Feb 2026) gives "40B-level performance with double the speed". Training uses Savanna and inference uses Vortex (optional Triton kernels). Published in Nature (2026). **[code/README]** — [GitHub](https://github.com/ArcInstitute/evo2); `savanna_evo2_1b_base` is on HF — [HF](https://hf.co/arcinstitute/savanna_evo2_1b_base)
- **Practical inference cost:** in the GCP preprint's sequence-recovery run (30k sequences, 6,144 bp, +50 nt generated), Evo2-1B took **7.21 h** vs 0.37 h for the GENERator-v2-1B transformer (Evo2-7B 27.12 h). The final three layers of Evo2-1B have activations of ~1e10 magnitude, so the authors probed layer 21. **[paper]** — [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.01.27.702015v1.full)
- **dnaHNet (Feb 2026)** claims better scaling and efficiency than StripedHyena 2 on prokaryotic genomes. **[paper, abstract]** — [arXiv](https://arxiv.org/abs/2602.10603)

### Inferences
- The transferable SH2 lesson is **multi-scale mixing**: very short explicit convolution (motif scale, ~4–7 bp), medium regularized filters (~100s of bp, nucleosome and TF-cluster scale), long global mixing and a few attention layers. KronLSTM already has a short conv (SE-like) and a long recurrent state (LI-like). Adding a medium-range component (a decayed medium conv, or heads with fast decay initialization in the channel-wise α) and 1 attention layer per ~6 blocks would mirror SE-MR-LI+MHA.
- Evo 2's measured throughput advantage depends on H100 FP8 and custom kernels. On A100 in bf16, an FLA-based delta-rule model is probably simpler to run fast than SH2 at small scale. The Evo2-1B FP8 requirement and its slow generation in third-party use both favour building your own small baselines over using Evo2-1B.
- The 1e10 activation magnitudes in Evo2-1B's last layers point to numerical-stability risk in deep byte-level hybrids. Keep the decay log-space and the state in fp32 (as GDN-2 does), and use norms on q/k.

### Gaps
- No public SH2 ablations below 7B, and no published SH2 vs Mamba-2 vs Transformer++ DNA loss numbers in text (Figure 1 only).
- No needle-in-haystack numbers for the 1B or 7B base models.
- Whether `evo2_1b_base` runs acceptably in bf16 on A100 is unverified (the README says FP8 is required).

---

## 7. Throughput / efficiency numbers relevant to an A100 budget (tokens/s, memory at 8k–1M context)

### Takeaway
Published numbers are sparse and mostly from H100s. GDN-family kernels run at ~36–38K tokens/s for a 1.3B hybrid on one H100, nearly flat from 2K to 16K context. GDN is ~2–3K tokens/s slower than Mamba-2 at 1.3B. Mamba-2 hybrids give ~3.4x transformer throughput at ~49k context on A100, where FlashAttention-2 transformers run out of memory at ~65k. Small recurrent DNA models (≤10M) train at 131k context on 8 GPUs in hours, and several claim 1M-token inference on one 80GB GPU. Pretraining token budgets for the small published models are 10–50B tokens.

### Cited Findings
- **Gated DeltaNet (1.3B, single H100, sequence length × batch from 2K×16 to 16K×2):** GDN and DeltaNet are "slightly slower than Mamba2 (2–3K tokens/sec)". Gating adds marginal overhead over DeltaNet. Transformer++ is fastest at 2K, and GDN-H1 stays competitive at all lengths. **[paper]** — [arXiv](https://arxiv.org/html/2412.06464)
- **Gated DeltaNet-2 (1.3B hybrid, single H100, fixed token budget):** **38.0 → 36.1 Kt/s** as sequence length grows. The Transformer degrades "sharply". Fused Triton kernels, chunk size 64, UT/WY transform. Per-layer state is H·d_k·d_v = 262,144 floats per batch element, kept in fp32 for decoding. **[paper]** — [arXiv](https://arxiv.org/html/2605.22791v1)
- **HybriDNA 300M vs Transformer (4× A100 80GB, ZeRO-1, bf16, FlashAttention-2 baseline):** **~3.4x throughput at ~49k context**, with the gap widening past 32k. The Transformer runs out of memory at ~65k. 300M trained on 8× MI300X for ~300 h; 3B on 8× H100 for ~500 h; 0.5M tokens per batch, 500k steps. **[paper]** — [HybriDNA](https://arxiv.org/abs/2502.10807)
- **JanusDNA:** pretraining on 8 GPUs (80GB H100/A100/A800), 2^20 tokens per batch, 131k context with batch 1 per GPU. Small-model runs took **3 h 2 min – 9 h 17 min on H800**. It claims 1M bp on one 80GB GPU, which is not validated in the experiments. **[paper]** — [JanusDNA](https://arxiv.org/html/2505.17257v1)
- **GeneZip:** the 128K-context and 636M variants train on **one A100 80GB**; eQTL fine-tuning is 50.4x faster than JanusDNA (50 vs 2,520 min). **[paper, abstract]** — [arXiv](https://arxiv.org/abs/2602.17739)
- **Caduceus:** 2^20 tokens per batch (1,024 × 1k or 8 × 131k), 10k updates at 1k and 50k updates at 131k, on a mix of 3090/A5000/A6000/V100/A100 GPUs. Total time is not reported. **[paper]** — [Caduceus](https://arxiv.org/html/2403.03234)
- **DNA-xLSTM:** 10k steps × ~2^20 tokens (~10.5B tokens); 32 × 32,768 batch for the 4M model. About 50k A100 GPU-hours across all Bio-xLSTM experiments, not DNA alone. **[paper]** — [Bio-xLSTM](https://arxiv.org/abs/2411.04165)
- **Caduceus and Hawk:** 1M-token sequences on a single GPU. **[paper, abstract]** — [arXiv](https://arxiv.org/abs/2504.06304)
- **HyenaDNA:** "up to 160x faster than Transformer" training at long sequence lengths; 1M context. **[paper, abstract]** — [arXiv](https://arxiv.org/abs/2306.15794)
- **StripedHyena 2:** 1.2–2.9x faster than optimized Transformers (H100 FP8, 7B/40B); ~2x operator throughput vs linear attention and SSMs at width 4096. **[paper]** — [arXiv](https://arxiv.org/abs/2503.01868)
- **ConvNova:** ~1.35x faster than HyenaDNA at 1M bp. **[secondary]** — [arXiv](https://arxiv.org/abs/2502.18538)
- **Lyra:** each of its 100+ tasks ran on ≤2 GPUs in <2 h. **[paper, abstract]** — [arXiv](https://arxiv.org/abs/2503.16351)
- **Inference generation cost:** Evo2-1B 7.21 h vs GENERator-v2-1B 0.37 h for the same recovery job (30k × 6,144 bp, +50 nt). **[paper]** — [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.01.27.702015v1.full)

### Inferences
- **Order-of-magnitude budget (my estimate, not from a source).** A 10–50M-parameter KronLSTM at 8k–32k context is dominated by projection and MLP FLOPs, not the recurrence. A100 bf16 dense throughput is roughly 1/3 of H100. The ~36K tokens/s for a 1.3B GDN-2 hybrid on H100 therefore suggests a 30M model should reach several hundred thousand tokens/s per A100, but kernel launch and chunk overheads dominate at small width, so measure it.
  - At 300k tokens/s per GPU on 4 A100s, 35B tokens (one hg38 pass at Caduceus scale) takes about 8 hours. A multi-species 100–200B-token corpus (HybriDNA scale) takes 1–2 days.
  - That makes a "days to weeks" budget enough for several 10–100M-parameter ablation runs (GDN vs Mamba-2 vs attention-hybrid; MLM vs CLM; PS vs PH), and one ~100–300M-parameter main run.
- **Memory:** recurrent state memory is constant in L, but chunkwise training stores activations linear in L. 131k × d256 × 16 layers × bf16 fits easily on 80GB at batch 1–8 (Caduceus did 8 × 131k per step across GPUs). 1M-token training needs activation checkpointing or sequence parallelism; 1M-token inference is feasible per the Caduceus, Hawk and JanusDNA claims.
- Bidirectional (two-scan) and PS wrappers each roughly double mixer FLOPs. PH doubles inference only. Alternating the direction per block keeps the causal-model cost.

### Gaps
- No published A100 throughput for FLA GDN or GDN-2 kernels at small widths (d=256–768) or at 32k–131k context.
- No measured tokens/s for Caduceus, DNA-xLSTM or JanusDNA training.
- No memory curves (GB vs context) for any recurrent DNA model.
- Exact SH2 per-operator latencies appear only in figures.
