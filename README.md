# kron-fm

A family of foundation models built on **KronLSTM**, NX-AI's Kronecker delta-rule linear-attention
recurrence ([Hartl et al., arXiv:2610.26834](https://arxiv.org/abs/2610.26834),
[NX-AI/KronLSTM](https://github.com/NX-AI/KronLSTM), MIT). Each head keeps a `K x V` matrix memory
written along one rank-one key per step, so a step costs the same at any position. Training runs
NX-AI's chunkwise Triton kernels, which are vendored here.

| Package | What it is | Status |
|---|---|---|
| `kron_core` | The vendored KronLSTM layer and kernels, the causal residual block and state cache, a bidirectional block and a document-masked attention block, packed-sequence helpers, a CPU stand-in for tests | working |
| `krontime` | Time-series models. First member: the glucose forecaster from [glucose_forecasting_metabonet](../glucose_forecasting_metabonet) | model ported; trainer still in the glucose repo |
| `kron_dna` | A masked-language model over coding DNA, tokenized by codon | skeleton: data, losses, model and trainer run end to end; nothing pretrained yet |
| `kron_ppg` | A causal, streaming KronLSTM encoder over raw PPG, with a PPG aging clock and an ONNX streaming graph for a phone | run 1 on 2,000 VitalDB people: test MAE 7.3 y (r 0.78); the age gap does not yet predict outcomes |

## Setup

```bash
uv sync --group dev
```

Python 3.12 (as in the glucose repo) and the same CUDA 12.8 torch wheels. The KronLSTM kernels need a
CUDA GPU: on a machine without one, the models build but only the CPU stand-in runs.

```bash
uv run pytest -q
```

Each model test runs twice. The CPU run uses `kron_core.testing.CausalMixer` in place of the layer.
The GPU run uses the real kernels and skips without CUDA.

## krontime

`krontime.glucose.KronLSTMForecaster` is the glucose repo's `kronlstm/model.py` (commit `6e9bbbe91`),
ported with module and parameter names unchanged. A forecaster built in the glucose repo loads here
strictly and gives bit-identical forecasts on the GPU (checked 2026-11-09). These stay in the glucose
repo, because they are wired into its MetaboNet trainer, data pipeline and scorer: record streaming
(`stream.py`), the trainer (`stream_train.py`) and the family spec. The streaming ONNX bundle export is
ported (`krontime.onnx_export.export_stream_bundle`); a CLI for it waits for krontime's own trainer.

Next steps toward a time-series foundation model: let the input channels be general rather than glucose,
basal, bolus and carbs, and pretrain on more than one dataset.

## kron-dna

The design follows the research report [reports/Small DNA foundation models 2026.md](reports/Small%20DNA%20foundation%20models%202026.md):

- **Tokens:** 5 special tokens plus the 64 codons (69, fits `uint8`).
- **Data:** CDS filters for frame, start, stop, internal stops and alphabet. Rows of 1,024 tokens are
  packed with several documents each, and the recurrence restarts at every document.
- **Model:** bidirectional KronLSTM blocks (separate forward and backward scans, since the delta rule's
  state depends on order), with a document-masked attention block every 6 recurrent blocks. The default
  has 68.6M parameters, or 51.9M with `--tie-directions`. There is an optional species embedding.
- **Objectives:** plain masked LM (`mlm`), SynCodonLM's synonymous-constrained softmax (`synonymous`),
  and HELM's hierarchical codon loss (`hierarchical`, `--alpha 0.2`). These two synonymy-aware losses
  have never been compared on the same corpus and backbone.

```bash
uv run kron-data download ecoli-k12                    # data/input/cds/ecoli-k12 (1.5 MB)
uv run kron-dna prepare ecoli-k12                      # -> data/interim/cds/ecoli-k12/packed_1024
uv run kron-dna train ecoli-k12 --objective hierarchical --steps 2000   # -> data/output/runs/kron-dna/...
uv run kron-dna export-onnx data/output/runs/kron-dna/<run> --seq-len 256   # -> <run>/onnx/kron_dna.onnx
uv run kron-dna prepare-stream syncodonlm --workers 6  # -> data/interim/cds/syncodonlm/packed_1024/{train,test}
uv run kron-dna train <dir>/train --val-data <dir>/test --max-hours 20   # rerun with --out <run> to resume
uv run kron-dna evaluate data/output/runs/kron-dna/<run> yeast-s288c <dir>/test   # -> <run>/eval/*.json
uv run kron-dna info --d-model 768 --n-layers 12       # parameter count of a config
```

The ONNX graph reads a fixed number of codon tokens, one sequence per row, through
`kron_core.onnx.ChunkedKronMixer`: the KronLSTM recurrence in plain PyTorch, 16 steps at a time with
matrix products (it matches the step-by-step recurrence to about 1e-6). The export checks ONNX Runtime
against PyTorch and, when CUDA is present, PyTorch against the Triton kernels, and records both in
`onnx_meta.json`.

`prepare` holds every row in memory, which suits a genome. `prepare-stream` is for the 17 GB SynCodonLM
corpus: it reads the CSV line by line, checks and encodes sequences with numpy lookup tables in worker
processes (the same verdicts and tokens as `check_cds` and `encode`, which the tests check), and writes each
value of the corpus's own `Set` column as its own directory of `uint8` shards (`tokens_00000.npy`, 256 MB
each) with a `meta.json` listing the shards, the counts, the filter rejects and the species and group
tallies. `--holdout-species 0.01` sends that fraction of species (a fixed hash of the name) to a third
directory, `species_holdout`, whatever their `Set`. `PackedRows` reads either layout by memory map. The corpus is already MMseqs2-clustered per
species at 90% identity by its authors, so it is not clustered again.

`train` uses AdamW, a 10% linear warmup, cosine decay and bf16 autocast. It writes `checkpoint.pt` (with
the optimizer state), `config.json` and `metrics.csv`; `val_bits_per_codon` is the plain cross-entropy
whatever the objective. A run directory that holds a checkpoint is resumed, and `--max-hours` stops a run
cleanly before a job's time limit, and `--snapshot-every N` keeps the weights every N steps under
`snapshots/` for later probing. By default validation is a random split of packed rows, which shares
genes with training and only tracks the loss; `--val-data` validates on another prepared directory
instead: the corpus's `Set=test` rows, another species' genome, or the genes `prepare --holdout-genes 0.05`
set aside (a fixed hash of the `[gene=...]` name, so all isoforms of a gene fall on one side).

`evaluate` scores a run on prepared sets with the same masking for every run. It splits bits per masked
codon into amino-acid bits and synonymous bits (the codon given its amino acid), because a model trained
with the `synonymous` loss never learns the amino acid: only its synonymous bits compare with the others.

`probe` is the first piece of the evaluation harness: frozen features, a ridge head whose penalty is
chosen on the task's validation split, and Spearman's rho on its test split, for a run's mean hidden
state, its untrained twin and the model-free baselines (GC, codon usage, CAI, nucleotide k-mers and
CodonBench's `onehot_pos` canary). On CodonBERT's mRFP expression task (1,459 synonymous variants of one
gene; its own 1,021/219/219 split; CAI against E. coli K-12), the baselines score test rho 0.30 (GC),
0.47 (CAI), 0.59 (codon usage), 0.61 (k-mers) and **0.78 (`onehot_pos`)**: the bar a model has to clear.

**The three objectives, matched** (default 68.6M model, human GRCh38 CDS with 5% of genes held out, 15,000
steps of 32x1024 tokens each, ~3.3 h on one A100; one seed). `evaluate` on 4,096 rows of each set, bits per
masked codon split into amino-acid (aa) and synonymous (syn) bits; mRFP is the probe's test rho:

| objective | held-out human genes: total / aa / syn | yeast: total / syn | SynCodonLM test: total / syn | mRFP rho |
|---|---|---|---|---|
| mlm | **4.784** / **3.526** / 1.258 | **5.342** / 1.492 | **5.153** / 1.394 | **0.716** |
| hierarchical (alpha 0.2) | 4.846 / 3.591 / **1.255** | 5.412 / **1.485** | 5.222 / **1.390** | 0.617 |
| synonymous | 5.562 / 4.289 / 1.273 | 5.885 / 1.501 | 5.726 / 1.408 | 0.629 |
| untrained twin | | | | 0.496 |

At this budget plain `mlm` is best: lowest total bits everywhere, synonymous bits within 0.01 of the best,
and the best probe point estimate, though its interval (0.632-0.780) overlaps those of hierarchical (0.511-0.693)
and synonymous (0.548-0.698). The synonymous loss does not win even on synonymous bits, its own target. No model
reaches the `onehot_pos` canary (0.78) on mRFP, a task of one gene's variants where a per-position lookup
is hard to beat. The GPU export of the hierarchical run matched ONNX Runtime to PyTorch within 4.8e-6
(logits) and PyTorch to the Triton kernels within 3.3e-3 (max, fp32; mean 2.5e-4).

```bash
uv run kron-data download ecoli-k12                    # CAI's reference codon usage
uv run kron-dna probe mRFP_Expression.csv --run data/output/runs/kron-dna/<run>   # -> data/output/eval/<task>/probe.json
```

**ClinVar, zero-shot.** `kron-dna clinvar` reads the ClinVar GRCh38 VCF and RefSeq's `cds_from_genomic` with
polars-bio. It keeps reviewed SNVs labelled pathogenic / likely pathogenic or benign / likely benign, then places each
on a codon of its gene's longest valid CDS, strand-aware: 974,270 SNVs land on a codon, with 0 reference mismatches.
Start and stop codons are dropped. The score is the masked marginal, log p(alt codon) - log p(ref codon) with the
codon masked in a 256-codon window, and the metric is the AUROC of -score with pathogenic as the positive class.
The test draws 5,000 variants per (consequence, label) by seed; ClinVar has only 756 pathogenic synonymous SNVs on a
codon, so all of those are kept. "Held-out" means genes of the human runs' 5% gene hold-out. The human runs were scored on CPU through the export's
plain-PyTorch graph, the SynCodonLM snapshots on GPU (about a minute each). Every ClinVar gene is in their training
corpus (it saw no ClinVar labels):

| score | missense, all genes | missense, held-out genes | synonymous, all genes |
|---|---|---|---|
| mlm (human, 15k steps) | 0.616 (0.604-0.627) | 0.614 (0.561-0.662) | 0.435 (0.413-0.458) |
| hierarchical (human, 15k steps) | 0.620 (0.609-0.632) | 0.631 (0.584-0.679) | 0.431 (0.410-0.453) |
| synonymous (human, 15k steps) | 0.537 (0.525-0.548) | 0.618 (0.568-0.666) | 0.433 (0.411-0.455) |
| mlm (SynCodonLM, 25k steps) | 0.649 (0.638-0.659) | 0.623 (0.571-0.672) | 0.427 (0.406-0.449) |
| mlm (SynCodonLM, 50k steps) | 0.705 (0.695-0.716) | 0.658 (0.610-0.706) | 0.423 (0.402-0.445) |
| mlm (SynCodonLM, 75k steps, 1,000-codon window) | 0.737 (0.727-0.747) | 0.670 (0.623-0.715) | 0.429 (0.407-0.451) |
| mlm (SynCodonLM, 100k steps, 1,000-codon window) | 0.762 (0.752-0.771) | 0.685 (0.640-0.730) | 0.443 (0.422-0.465) |
| mlm (SynCodonLM, 125k steps, 1,000-codon window) | 0.776 (0.766-0.784) | 0.707 (0.663-0.751) | 0.437 (0.416-0.459) |
| mlm (SynCodonLM, 150k steps, 1,000-codon window) | **0.790** (0.782-0.799) | **0.723** (0.678-0.769) | 0.439 (0.418-0.461) |
| codon-usage delta | 0.539 (0.528-0.550) | 0.593 (0.545-0.642) | 0.533 (0.510-0.556) |
| GC delta | 0.467 (0.457-0.476) | 0.418 (0.373-0.465) | **0.602** (0.586-0.618) |

In brackets: 95% bootstrap intervals over the variants (1,000 resamples). Missense: 5,000 P / 5,000 B overall, of
which 263 P / 213 B are in held-out genes. Synonymous: 756 P / 5,000 B. Over all missense variants the
SynCodonLM model at 50k steps is above every other score, its interval clear of theirs. It also improved from 25k to 50k.
On the 476 held-out-gene variants every model's interval overlaps every other's. The models separate missense variants only weakly: ESM-2 reaches
0.906 on the same variants (the next table). All five models sit
below 0.5 on synonymous variants, their intervals included: pathogenic synonymous SNVs look *more* likely to them
than benign ones. Pathogenic synonymous variants act mostly through splicing, which a CDS-only model sees at best weakly
(Encodon 1B reaches 0.62, below), and the GC change of the substitution alone does better (0.60). The CPU and GPU paths gave the same AUROCs to within
0.001. Context helps a little. With a 1,000-codon window instead of 256, the 50k model goes from 0.705 to 0.711 on
missense, a gain of +0.005 with a paired bootstrap interval of 0.003-0.008 over the same variants. Synonymous does
not move (+0.003, interval -0.000 to 0.006).

**Against the state of the art, on the same variants.** Each row is AUROC (95% bootstrap interval). "Held-out" is
the human runs' 5% gene hold-out (263 P / 213 B missense). A score covers only what it can score: AlphaMissense has
9,705 of the 10,000 missense variants, and AlphaGenome 5,733 of the 5,756 synonymous ones.

| scorer | what it sees | missense, all genes | missense, held-out genes | synonymous, all genes |
|---|---|---|---|---|
| kron-dna SynCodonLM 50k, 1,000-codon window | the CDS's codons | 0.711 (0.701-0.721) | 0.661 (0.614-0.709) | 0.426 (0.404-0.448) |
| kron-dna SynCodonLM 75k, 1,000-codon window | the CDS's codons | 0.737 (0.727-0.747) | 0.670 (0.623-0.715) | 0.429 (0.407-0.451) |
| ESM-2 650M, masked marginal | the protein | 0.906 (0.900-0.912) | 0.834 (0.800-0.870) | - |
| AlphaMissense | protein, structure, population frequency | **0.966** (0.963-0.969) | **0.950** (0.933-0.967) | - |
| AlphaGenome, splice junctions | 100 kb of genome | 0.594 (0.582-0.605) | 0.540 (0.494-0.594) | **0.939** (0.926-0.951) |
| AlphaGenome, RNA-seq | 100 kb of genome | 0.530 (0.519-0.541) | 0.489 (0.434-0.541) | 0.810 (0.793-0.830) |
| GC delta | the substitution | 0.467 (0.457-0.476) | 0.418 (0.373-0.465) | 0.602 (0.586-0.618) |

The codon model is far behind on both classes. On missense, a protein language model of a similar size (ESM-2,
650M) is near 0.91 zero-shot. AlphaMissense is higher still, but it is tuned with population frequency, and much of
ClinVar's benign class is common variants, so its number is not a clean zero-shot comparison. On synonymous variants
the splicing scores of AlphaGenome reach 0.94, which confirms that pathogenic synonymous variants are mostly splicing
variants: a model that reads only the CDS scores them poorly (kron-dna below chance; Encodon 1B 0.62, below). ESM-2 runs in fp32: in bf16 the same model drops to
0.70, measured on 1,000 of these variants. AlphaGenome was called through its API with the scorers its client
recommends, taking the largest absolute effect over the tracks.

**Evo 2 7B**, on a fixed subsample of the same variants (2,000 per class; all 756 pathogenic synonymous): the change
in mean log-likelihood of a 4,096-base genomic window, alternate minus reference (`scripts/kron_dna/evo2/`, in its
own environment). On the subsample, each scorer's AUROC (95% interval):

| scorer | missense (2,000 P / 2,000 B) | synonymous (756 P / 2,000 B) |
|---|---|---|
| kron-dna SynCodonLM 75k | 0.732 (0.717-0.747) | 0.423 (0.401-0.446) |
| Evo 2 7B | 0.868 (0.856-0.879) | 0.859 (0.841-0.877) |
| ESM-2 650M | 0.910 (0.899-0.919) | - |
| AlphaMissense | **0.971** (0.966-0.975) | - |
| AlphaGenome, splice junctions | 0.590 (0.571-0.607) | **0.939** (0.925-0.951) |

Evo 2 reads the genome around the variant, introns and splice sites included, so it scores both classes. It is above
the codon model by 0.136 on missense (paired interval 0.119-0.153) and by 0.436 on synonymous. On each class it is
below the specialist: ESM-2 or AlphaMissense on missense, AlphaGenome's splicing score on synonymous.

**NVIDIA's CodonFM (Encodon), run here.** It is the closest published peer: masked codon models of 80M and 1B
parameters, trained on about 130M RefSeq coding sequences from over 22,000 species. It was run with NVIDIA's own
mutation recipe (the CDS's first 2,046 codons, the variant's codon masked), from its v1.1 release in a scratch
environment. That environment's xformers has no CUDA build, so its attention call was routed through PyTorch's own
kernel, checked equal to the explicit formula at start-up. Its context covers 9,339 of the 10,000 missense and
5,183 of the 5,756 synonymous variants; all three models are compared on those (AUROC, 95% interval):

| model | missense (4,790 P / 4,549 B) | synonymous (704 P / 4,479 B) |
|---|---|---|
| kron-dna, 68.6M, 100k steps | 0.766 (0.756-0.774) | 0.443 (0.418-0.465) |
| Encodon 80M | 0.870 (0.862-0.876) | 0.475 (0.451-0.498) |
| Encodon 1B | **0.907** (0.900-0.912) | **0.624** (0.603-0.643) |

Encodon 80M, near kron-dna's size, is ahead by 0.104 on missense (paired interval 0.095-0.113), so most of the gap
is training data and length, not the codon view itself. Encodon 1B matches ESM-2 on missense (0.906 here), and is
the only coding-sequence model above chance on synonymous variants (0.62), still far below AlphaGenome's splicing
score (0.94).

Does the codon model add anything to the protein model? A logistic regression on the two scores, cross-validated
in 5 folds by gene (3,948 genes, so no gene is in both training and test), gives 0.911 on missense against 0.906 for
ESM-2 alone. The gain is +0.005, with a paired bootstrap interval of 0.003-0.006: small but real. The two scores
correlate only 0.45 (Spearman). Adding the GC and codon-usage deltas does not move it (0.911).

**A first supervised ensemble** (gradient-boosted trees, 5 folds split by gene so no gene is in both training and
test, on the same 15,756 variants). The inputs are scores already computed: kron-dna 75k, ESM-2, AlphaGenome's four
scorers, the GC and codon-usage deltas, and the consequence. AUROC (95% interval):

| inputs | missense | synonymous |
|---|---|---|
| kron-dna alone | 0.711 (0.700-0.720) | 0.449 (0.430-0.471) |
| ESM-2 alone | 0.902 (0.896-0.908) | 0.475 (0.456-0.498) |
| everything but kron-dna | 0.915 (0.910-0.921) | 0.933 (0.921-0.946) |
| everything | **0.925** (0.920-0.931) | **0.938** (0.926-0.951) |

kron-dna adds +0.010 on missense (paired interval 0.008-0.013) and +0.005 on synonymous (0.001-0.009) to the rest.
AlphaMissense alone is at 0.966 on missense, with population frequency behind it. Not in yet: conservation, the
models' embeddings, and a check on a set other than ClinVar.

```bash
uv sync --inexact --all-groups                          # polars-bio in `genomics`; fair-esm, alphagenome in `baselines`
uv run kron-dna clinvar --run data/output/runs/kron-dna/<run>   # -> data/output/eval/clinvar/zero_shot.{json,parquet}
uv run kron-dna clinvar --esm esm2_t33_650M_UR50D --alphamissense AlphaMissense_hg38.tsv.gz --alphagenome-key KEYFILE
```

**SynCodonLM pretraining (running).** `mlm` on the whole corpus (`packed_1024_sp01`: 1% of species held out),
same 68.6M model and batch, up to 300k steps. Its snapshots, by `evaluate` on 4,096 rows of each set
(total / aa / syn bits per masked codon):

| set | step 25k | step 50k | step 75k | step 100k | step 125k | step 150k | human `mlm`, 15k steps |
|---|---|---|---|---|---|---|---|
| held-out species | 4.879 / 3.606 / 1.273 | 4.735 / 3.474 / 1.261 | 4.642 / 3.387 / 1.256 | 4.568 / 3.316 / 1.252 | 4.504 / 3.255 / 1.249 | 4.449 / 3.203 / 1.246 | |
| SynCodonLM test (species seen in training) | 4.892 / 3.622 / 1.270 | 4.757 / 3.498 / 1.259 | 4.668 / 3.414 / 1.254 | 4.596 / 3.346 / 1.250 | 4.533 / 3.286 / 1.246 | 4.479 / 3.236 / 1.243 | |
| SynCodonLM test, the step-1 sample | 4.882 / 3.610 / 1.272 | 4.747 / 3.486 / 1.261 | 4.657 / 3.402 / 1.256 | 4.585 / 3.333 / 1.252 | 4.522 / 3.274 / 1.249 | 4.468 / 3.223 / 1.245 | 5.153 / - / 1.394 |
| yeast | 5.099 / 3.727 / 1.372 | 5.018 / 3.650 / 1.368 | 4.977 / 3.610 / 1.367 | 4.939 / 3.577 / 1.363 | 4.905 / 3.545 / 1.360 | 4.882 / 3.522 / 1.359 | 5.342 / - / 1.492 |
| held-out human genes | 4.788 / 3.528 / 1.260 | 4.645 / 3.401 / 1.245 | 4.542 / 3.306 / 1.236 | 4.458 / 3.228 / 1.230 | 4.381 / 3.155 / 1.226 | 4.317 / 3.095 / 1.221 | 4.784 / 3.526 / 1.258 |

After 25k steps the corpus model already beats the human-only model on yeast and on SynCodonLM; after 50k it
also beats it on human genes (4.645 against 4.784), the human-only model's own domain. Human is ~0.1% of
SynCodonLM, and its genes there are not split by the human hold-out, so the "held-out human genes" row is not
held out for this model. On the mRFP probe the snapshots reach test rho 0.695 (0.612-0.757) at 25k, 0.642 (0.545-0.715) at 50k and 0.648
(0.549-0.725) at 75k 0.644 (0.555-0.720) at 100k, 0.674 (0.580-0.752) at 125k and 0.622 (0.525-0.709) at 150k, against 0.716
(0.632-0.780) for the human `mlm` model. These are 95% bootstrap intervals over the 219 test variants, and they all
overlap: one gene's split cannot rank these models.

**Not built yet:**
- clustering of a genome's paralogs (`prepare --holdout-genes`
  keeps every isoform of a gene on one side, but paralogs under other gene names still cross);
- the rest of the evaluation harness: the other CodonBERT and SynCodonLM tasks, leave-one-gene-out
  splits for the multi-gene tasks, cross-validation over seeds (`probe` scores one task on its own split);
- multi-GPU training and gradient checkpointing;
- the finished SynCodonLM model (training, with `mlm`; snapshot numbers above).

## kron-ppg

A causal, streaming KronLSTM encoder over raw photoplethysmography (PPG), with a PPG aging clock as its
first task. The design and its sources are in
[reports/PPG foundation models and aging clocks 2026.md](reports/PPG%20foundation%20models%20and%20aging%20clocks%202026.md).
It follows the earlier PPG work in [pulse-exploration](../pulse-exploration) (`docs/ppg-age.md`).

- **Input:** PPG at 25 Hz, the rate of the watch's no-setup tier. It is band-passed to 0.5–8 Hz,
  robust-scaled per 30 s window, and cut into 0.2 s patches; each patch is one token.
- **Encoder:** `kron_core.blocks.KronLSTMBlock` layers with `LayerStates`, so a stream read in chunks
  equals the stream read whole. The default has 5.85M parameters (d_model 256, 6 layers);
  `uv run kron-ppg info` prints the count of any configuration.
- **Pretraining:** `next` predicts the next 1 s of patches from every step (GPT-PPG's objective, with
  an L1 loss). `next+subject` adds a contrast in which windows of the same person are positives
  (Apple's PPG foundation model).
- **Age clock:** an MLP reads the mean hidden state. It is scored per person, with a bias-corrected
  age gap, and that gap's AUC is reported for every outcome flag the source carries (VitalDB:
  hypertension, diabetes, ASA ≥ 3, in-hospital death). Outcome evidence, not MAE, is what makes a
  clock worth having.
- **ONNX:** `stream.onnx` is one graph. It takes a 5 s chunk, every layer's memory and the running
  pool, and returns the age so far with the updated state. It runs through
  `kron_core.onnx.PlainKronMixer`. The export checks it against PyTorch in ONNX Runtime and, on a
  GPU, against the Triton kernels.

```bash
uv run kron-data download bidmc galaxyppg vitaldb   # vitaldb: 200 cases (KRON_PPG_VITALDB_CASES), ~0.9 GB
uv run kron-ppg download vitaldb --cases 1000       # more VitalDB, one case per person
uv run kron-ppg prepare bidmc vitaldb galaxyppg     # -> data/interim/ppg/<source>/fs25-w30s
uv run kron-ppg pretrain vitaldb galaxyppg bidmc --objective next+subject --steps 5000
uv run kron-ppg clock vitaldb --encoder data/output/runs/kron-ppg/<pretrain run>/checkpoint.pt
uv run kron-ppg evaluate data/output/runs/kron-ppg/<clock run> bidmc
uv run kron-ppg export data/output/runs/kron-ppg/<clock run>   # -> <run>/onnx/stream.onnx + stream.json
```

| Source | What | Licence |
|---|---|---|
| `vitaldb` | surgical finger PPG at 500 Hz, with age and outcome flags; 6,157 cases, about 4 MB each | CC BY 4.0 (VitalDB data use agreement) |
| `bidmc` | 53 ICU recordings of 8 min, finger PPG at 125 Hz, with ages | ODC-By 1.0 |
| `galaxyppg` | Galaxy Watch5 PPG at 25 Hz from 24 people aged 20–29; linked from pulse-exploration when present | CC BY 4.0 |

A PPG age is a within-person trend, never a biological age. It must be shown with the run's error
band.

Status: run 2 (5,887 VitalDB people, one A100) reached a held-out MAE of 6.8 y (r 0.82, 897 test people)
with the pretrained encoder (6.4 y after four times the pretraining, runs 3-4), against 7.75 y from scratch and 11.7 y for predicting the median age; run 1
(2,000 people) reached 7.3 y. It does not transfer to BIDMC (no better than the median age), and its
bias-corrected age gap still does not separate hypertension, diabetes or ASA ≥ 3 (AUC 0.49–0.55, every
95% interval containing 0.5, against 0.62–0.73 for age alone). So it is an age regressor, not yet an
aging clock. A multi-task trait head on the same encoder is where the first outcome evidence is: added
to age, PPG raises the AUC for ASA ≥ 3 by 0.11–0.12 (95% interval 0.03–0.20) and for being healthy by
0.03 (0.004–0.05), though not for hypertension or diabetes, and it follows blood pressure within a
person (r 0.65 per 30 s window). The prepared windows are public as
[`GlucoseDao/ppg-windows`](https://huggingface.co/datasets/GlucoseDao/ppg-windows) (`kron-ppg hub-export`).
Details are in the report. Next: foundation-model baselines on the same split, the trait head on
wrist data, and outcome evidence outside one hospital.

## Data and the cluster

Data lives in `data/input` (downloads as published, each with a `SOURCE.json`), `data/interim` (derived,
rebuildable) and `data/output` (runs, ONNX graphs); see [data/README.md](data/README.md). `$KRON_FM_DATA`
moves the tree. `uv run kron-data list` shows every dataset and `uv run kron-data status` what is on disk.

Large downloads and training belong on the SLURM cluster. `slurm/submit.sh gpu|cpu NAME COMMAND...` runs
one kron-fm command as a job (`uv run --no-sync`). The GPU nodes there may have no GPU GRES, so a GPU job
picks the first card with no compute process (or `CUDA_DEVICE`). The partition, account and node names
go in `slurm/env.local.sh`, which is git-ignored because this repository is public; start it from
`slurm/env.local.example.sh`. Put the checkout, `UV_CACHE_DIR` and `UV_PYTHON_INSTALL_DIR` on the large
scratch filesystem, not a small home quota.

```bash
slurm/submit.sh gpu dna-ecoli kron-dna train ecoli-k12 --steps 20000
slurm/submit.sh cpu prep-human kron-dna prepare human-grch38
```

## Research

- [reports/Small DNA foundation models 2026.md](reports/Small%20DNA%20foundation%20models%202026.md):
  which small 2025–2026 DNA models matter for kron-dna, and the recommended setup.
- [reports/PPG foundation models and aging clocks 2026.md](reports/PPG%20foundation%20models%20and%20aging%20clocks%202026.md):
  PPG foundation models, published PPG age clocks, open datasets with ages, and kron-ppg's design.
- [research_notes/](research_notes/): the notes behind it, with each claim tagged by its source.

## Vendored code

`src/kron_core/vendor/fla` is the import closure of `fla.layers.kronlstm` from NX-AI/KronLSTM at
commit `3d800c9a`, with `fla.` rewritten to `kron_core.vendor.fla.`; see its `LICENSE` (MIT). Regenerate
it with `scripts/vendor_kronlstm.py`; do not edit it by hand. The tree is identical, apart from the
namespace, to the glucose repo's `kronlstm.vendor.fla`.
