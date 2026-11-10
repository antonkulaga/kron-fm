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
and the best probe. The synonymous loss does not win even on synonymous bits, its own target. No model
reaches the `onehot_pos` canary (0.78) on mRFP, a task of one gene's variants where a per-position lookup
is hard to beat. The GPU export of the hierarchical run matched ONNX Runtime to PyTorch within 4.8e-6
(logits) and PyTorch to the Triton kernels within 3.3e-3 (max, fp32; mean 2.5e-4).

```bash
uv run kron-data download ecoli-k12                    # CAI's reference codon usage
uv run kron-dna probe mRFP_Expression.csv --run data/output/runs/kron-dna/<run>   # -> data/output/eval/<task>/probe.json
```

**Not built yet:**
- clustering of a genome's paralogs (`prepare --holdout-genes`
  keeps every isoform of a gene on one side, but paralogs under other gene names still cross);
- the rest of the evaluation harness: the other CodonBERT and SynCodonLM tasks, leave-one-gene-out
  splits for the multi-gene tasks, cross-validation over seeds (`probe` scores one task on its own split);
- multi-GPU training and gradient checkpointing;
- a model pretrained on the whole SynCodonLM corpus (training, with `mlm`; no numbers yet).

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
