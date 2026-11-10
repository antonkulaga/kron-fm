# Training kron-ppg on whole streams instead of windows (2026-11-10)

KronLSTM is a streaming model: its recurrent memory carries a person's signal from one chunk to the next, at
the same cost per step at any depth. Runs 1-4 nonetheless trained it on 30 s windows, each starting from an
empty memory, with at most 240 windows per recording (at most 2 h of a multi-hour VitalDB surgery). This note
records the first runs without windows: what was built, what it reads, and how it compares.

## What was built

- `src/kron_ppg/streams.py`: every recording of a source written whole at 25 Hz, filtered as for windows
  (gaps up to 0.5 s filled, 0.5-8 Hz band-pass) but with no windows, no cap, no quality drop and no scaling; a
  validity mask; per-10 s block robust scale, pulse periodicity and valid fraction as labels. Wrist readers give
  whole recordings, split only where no sample comes for over an hour (a stray clock once made one watch's
  four weeks a three-year grid).
- `src/kron_ppg/stream_train.py`: truncated backpropagation through time. 64 lanes read 2 min chunks side by
  side and carry the KronLSTM memory from chunk to chunk (reset only on a new stream); causal scaling (median
  scale of the past 30 good blocks), samples beyond 20 masked; next-patch L1, masked and weighted by block
  periodicity, plus the participant InfoNCE over same-person lane pairs. The stream clock reads the running pool
  of a person's hidden states over their streams read whole (optionally weighted by pulse periodicity, a soft
  quality gate). Splits are the window trainer's, so the test people are runs 1-4's.

## What the streams hold (valid hours)

| Source | Streams | People | Valid hours | As 30 s windows (cap 240) |
|---|---|---|---|---|
| VitalDB | 5,887 | 5,887 | 17,839 | 9,963 h (1,195,506 windows) |
| Galaxy Watch Active 2 | 1,671 | 49 | 17,767 | 2,424 h |
| Vilnius AF | 368 | 35 | 4,070 | 1,088 h |
| Multisite-PPG, WildPPG, PPG-Sport, PPG-DaLiA, WESAD, UTSA, WF-PPG, GalaxyPPG, BIDMC | | | 717 | |

## Does the memory help? (same chunks, cold vs warm)

Each validation chunk (64 lanes x 15 chunks of 2 min) is scored twice on identical samples: from an empty
memory and with the memory carried in from the chunks before it (next-patch L1, lower is better).

| Encoder | Warm minus cold, chunks 1-14 | Relative |
|---|---|---|
| s1, trained on streams (all sources) | -0.010 to -0.015 at every context length | memory helps, -2.4 % |
| s3, trained on streams (VitalDB + GalaxyPPG), on those two sources | -0.0065 to -0.0077 at every context length | memory helps, -4.2 % |
| run 4, trained on 30 s windows | +0.060 to +0.083 | memory hurts, +8.4 % |

A KronLSTM trained on windows has never used more than 30 s of memory, and streaming it makes it worse; trained
on streams it uses its memory at every context length measured (2-28 min).

## Age clocks (VitalDB test people, MAE in years / r; median-age baseline 11.7)

| Encoder (pretraining) | Window probe | Stream probe, first 2 h | Stream probe, whole record, quality-weighted | Finetune |
|---|---|---|---|---|
| run 4: windows, VitalDB + GalaxyPPG, ~170k h | **6.38 / 0.84** (6.41 +- 0.03, 3 seeds) | 7.14 / 0.78 | 7.19 / 0.78 | |
| run 2: windows, VitalDB + GalaxyPPG, ~43k h | 6.82 / 0.81 | | | 6.77 / 0.82 |
| s1: streams, all 11 sources, 82k h (VitalDB 34 %) | 8.80 / 0.68 | 8.04 / 0.72 | 8.07 / 0.71 | 7.81 / 0.73 |
| s3: streams, VitalDB + GalaxyPPG, 42k h | 8.82 / 0.68 | 8.39 / 0.69 (whole record) | | |

Readings:

- The encoder decides; the clock does not. Switching the clock moves an encoder by under a year; switching the
  encoder moves every clock by 1.5-2.4 years.
- Quality weighting is not the window clock's advantage: weighting the pool by pulse periodicity over whole
  records changes nothing for either encoder (7.14 -> 7.19, 8.04 -> 8.07).
- The data mix is not it either: s3, trained on streams of run 2's own data at run 2's budget, is as far behind
  run 2 as s1 is.
- So the stream pretraining as built learns a person-level representation that is worse for age than window
  pretraining, while it learns to use its memory (above).

## What differs between the two pretrainings, and what to try next

1. **Scaling.** Windows are robust-scaled each on its own: every window has the same spread, so the encoder
   sees only shape. Streams are scaled by the median of the past five minutes, so amplitude drifts within and
   between people (perfusion, sensor contact): the next-patch loss can be lowered by tracking amplitude, which
   says little about age. Next: per-chunk or short-horizon scaling, or a scale-free target.
2. **What the contrast compares.** Window InfoNCE pairs two 30 s windows of a person drawn anywhere in the
   record; stream pairs are two lanes of one person pooled over a 2 min chunk while each carries its own memory.
3. **Clock training batches.** A stream-clock step sees 32 people (lanes in pairs, 15 chunks each), a window step
   up to 128: fewer people per gradient. This affects the stream probes, not the window probes of stream
   encoders, which trail too.

Not done: seeds for the stream runs (single seed each); Vilnius clocks (35 patients give a 4-person validation
split, too small to read).
