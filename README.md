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
(`stream.py`), the trainer (`stream_train.py`), the family spec and the ONNX export.

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
uv run kron-dna prepare cds.fa.gz --out data/packed/refseq_demo           # --alt-starts for bacteria
uv run kron-dna train data/packed/refseq_demo --out data/runs/demo --objective hierarchical --steps 2000
uv run kron-dna info --d-model 768 --n-layers 12                          # parameter count of a config
```

`train` uses AdamW, a 10% linear warmup, cosine decay and bf16 autocast. It writes `checkpoint.pt`,
`config.json` and `metrics.csv`; `val_bits_per_codon` is the plain cross-entropy whatever the objective.
Its validation split is random over packed rows, so it shares genes with training and only tracks the
loss.

**Not built yet:**
- the RefSeq/SynCodonLM corpus download, MMseqs2 clustering and sharded `prepare` for tens of millions
  of sequences (the current `prepare` holds rows in memory);
- gene- and species-held-out splits;
- the evaluation harness: CodonBERT and SynCodonLM tasks under leave-one-gene-out, the `onehot_pos`
  canary, k-mer/GC/CAI baselines, and a random-init twin;
- multi-GPU training and gradient checkpointing.

## Research

- [reports/Small DNA foundation models 2026.md](reports/Small%20DNA%20foundation%20models%202026.md):
  which small 2025–2026 DNA models matter for kron-dna, and the recommended setup.
- [research_notes/](research_notes/): the notes behind it, with each claim tagged by its source.

## Vendored code

`src/kron_core/vendor/fla` is the import closure of `fla.layers.kronlstm` from NX-AI/KronLSTM at
commit `3d800c9a`, with `fla.` rewritten to `kron_core.vendor.fla.`; see its `LICENSE` (MIT). Regenerate
it with `scripts/vendor_kronlstm.py`; do not edit it by hand. The tree is identical, apart from the
namespace, to the glucose repo's `kronlstm.vendor.fla`.
