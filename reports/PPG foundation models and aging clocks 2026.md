# PPG foundation models and aging clocks, 2026: what kron-ppg builds on

Written 2026-11-09 for `kron_ppg`. It builds on the user's earlier PPG work in
`~/sources/pulse-exploration`:

- `docs/ppg-age.md`, the AI-PPG Age trial on watch PPG;
- `research_notes/PPG health and aging measures/`, a survey of models, datasets and licences as of 2026-10-24;
- the 25 Hz HRV and pulse-morphology benchmarks.

It does not repeat that survey: it adds what a KronLSTM PPG model needs, together with a fresh check
(2026-11-09) of which datasets with an age per subject can be downloaded without credentials.

## What the earlier sessions decided or found

- **Two device tiers.**
  - The no-setup tier is the Galaxy Watch's `hr_raw` stream: about 24.9 Hz, read through plain
    Android sensors.
  - The "all" tier uses the Samsung Health Sensor SDK, for 100 Hz on-demand PPG and ECG.
  - A model that serves both must read **25 Hz**. 25 Hz does not limit pulse shape: average beats
    rebuilt from 25 Hz match those rebuilt from 64 Hz (`benchmarks/morph25`).
- **Aging clocks are judged by outcome evidence, not MAE.** This is the user's explicit view. A gap
  between predicted and calendar age is worth something only if it carries disease or death risk.
- **AI-PPG Age** (MIT; Hugging Face `Ngks03/PPG-VascularAge`) is the only open PPG clock with outcome
  evidence.
  - In UK Biobank, a gap above 9 y carries a MACCE hazard ratio of 2.37.
  - Its UK Biobank MAE is 7.57 y; zero-shot on VitalDB it is 11.5 y, and its predictions collapse into
    38–67 y.
  - On the user's watch the beat was near-sinusoidal and the wrist moved, so the reading was
    **unreliable**.
  - Any PPG age is a within-person trend, shown with its error band and a warning. It is never a
    biological age.
- **Open clocks have an error of 6–10 y**, larger than the 6–9 y gaps that carry risk. Apple's PpgAge
  reaches 2.4–3.2 y, but only with 214k people and their whole-life wrist data.
- **Repeated spot checks.** The OPPO heart-age protocol is at least 10 seated 30 s checks per quarter,
  reporting the quarter's median.
- **GalaxyPPG** (Zenodo 14635823, CC BY 4.0) is already downloaded in pulse-exploration. kron-fm links
  to that copy rather than downloading it again.

## PPG foundation models, in brief

| Model | Open? | Input | Objective | Relevance here |
|---|---|---|---|---|
| Apple PPG FM (ICLR 2024, arXiv 2312.05409) | no | 4-ch wrist green, 60 s, 64 Hz | contrastive, **positives = two segments of the same participant**, plus KoLeo | its age linear probe gives MAE 3.19 y; the participant-positive idea is kron-ppg's second objective |
| Apple PpgAge (Nat Commun 16:9264, 2025) | no | wrist PPG, 213,593 people | supervised age on the FM | MAE 2.43 y (healthy), 3.18 y (all); a 6 y gap gives HR 1.40 for cardiac events |
| GPT-PPG (arXiv 2503.08015) | weights promised, not found | 30 s, patches | **autoregressive next patch**, logit-Laplace likelihood | the objective a causal stream model reads exactly as served: kron-ppg's first objective |
| PaPaGei (ICLR 2025) | BSD-3 code; weights with no licence | 10 s, 125 Hz finger | contrastive with morphology-aware positives | the finger-domain baseline |
| Pulse-PPG (IMWUT 2025) | MIT | 4 min, 50 Hz wrist field data | relative contrastive | the only open wrist field model; best open age result in the Brag benchmark |
| AnyPPG (KDD 2026) | no licence | 10 s, 125 Hz | PPG–ECG alignment | age MAE 6.08 (WenXinWuYang) |
| HiMAE, MMR, xMAE (Samsung) | no | 10 s, 100 Hz Galaxy Watch green | masked reconstruction | show that Galaxy hardware carries hypertension, PVC and lab signals; nothing runnable |
| SiamQuality | code only, no licence | 30 s pairs | Siamese on quality pairs | – |
| `lowdown-labs/fela-ppg` (HF, July 2026) | licence "other" | on-device | gated linear attention plus a Fourier operator, for AF | the closest architecture to kron-ppg (a linear RNN on PPG); worth a baseline run |

Independent age benchmark: Brag, arXiv 2603.14030, on 906 PulseDB surgical patients.

| Method | MAE |
|---|---|
| AI-PPG Age | 9.72 y |
| Pulse-PPG + ridge | 9.28 y |
| HR/HRV + demographics | 9.59 y |
| Pulse-PPG + demographics | 8.22 y |

Its learning curve had not plateaued, so sample size is the binding limit.

**No open-weight PPG model is a causal recurrent encoder with a carried state.**
- Every open model reads fixed windows; Pulse-PPG reads 4 minutes.
- A streaming KronLSTM can read a person's day at constant cost per step.
- It keeps one memory per person on the phone.
- It averages evidence over as many beats as it is given, which is what Apple's participant-level
  averaging does offline.

That is the case for kron-ppg.

## Datasets with an age per subject that download without credentials

Checked 2026-11-09 by fetching the files.

| Source | What | Age | Size | Licence | In kron-ppg |
|---|---|---|---|---|---|
| **VitalDB** API (`api.vitaldb.net`) | surgical finger PPG `SNUADC/PLETH`, 500 Hz | age, sex, ASA, `preop_htn`, `preop_dm`, `death_inhosp` per case; `subjectid` links one person's repeat surgeries | 6,157 cases with PLETH; about 4.3 MB gzipped per case (about 3 h at 500 Hz) | CC BY 4.0, through the API (vitaldb.net data use agreement, read 2026-11-09: commercial use and redistribution allowed with credit, a licence link and a note of changes; no re-identification) and as the PhysioNet copy (95 GB of `.vital` files) | `vitaldb`: the training set |
| **BIDMC** (PhysioNet `bidmc`) | 53 ICU recordings × 8 min, finger PPG 125 Hz | `Age:` in `bidmc_NN_Fix.txt`; three say `90+`, one says `NaN` | 147 MB (CSV + Fix files) | ODC-By 1.0 | `bidmc`: external check and smoke test |
| **GalaxyPPG** (Zenodo 14635823) | Galaxy Watch5 25 Hz green PPG, E4 64 Hz, Polar H10 | ages 20–29 only | 482 MB | CC BY 4.0 | `galaxyppg`: wrist pretraining and device domain, not a clock set |
| BUT PPG v2 (PhysioNet) | smartphone fingertip PPG, 30 Hz, 10 s × 3,888 | ages 19–76, 50 subjects | 203 MB | CC BY 4.0 | not yet (WFDB reader, 10 s windows): the closest public match to the user's fingertip camera |
| PPG-BP (figshare) | finger PPG, 1 kHz, 2.1 s × 3 per subject | 219 subjects | 1.5 MB | CC0 | not yet (2.1 s segments are too short for a 30 s window; age is in xlsx) |
| PPG-DaLiA (UCI 495) | wrist E4 BVP 64 Hz, 15 subjects | in a pickle's questionnaire | about 2.7 GB | CC BY 4.0 | not yet (the age is in a pickle, which we do not load from a download) |
| MIMIC-IV Waveform 0.1.0 | ICU, 198 patients | **no**: age needs credentialed MIMIC-IV | 12.8 GB | ODbL | no |
| Autonomic Aging (PhysioNet) | ECG + finger NIBP waveform, 1,121 healthy | 15 age bins only | 5.9 GB | ODbL | no: no PPG channel |
| Charlton PWDB (Zenodo 3275625) | simulated pulse waves, 4,374 virtual subjects aged 25–75 | yes | 243 MB CSV | ODC-PDDL | not yet: useful as a sanity probe, not training |
| UK Biobank, MC-MED, MESA/CFS | the large age- and outcome-labelled sets | yes | – | paid / credentialed / DUA | no |

**Only VitalDB is large enough to train a clock.** All of it is surgical finger PPG from Korean adults
under anaesthesia, mostly middle-aged and older. A clock trained there must be checked on wrist data
before it means anything on a watch. GalaxyPPG's 20–29-year-olds cannot do that check; Aurora-BP
(wrist, ages 21–85, access unclear) or the user's own cohort could.

## Design choices in `kron_ppg`

- **One rate for every tier: 25 Hz, 5-sample patches (0.2 s per token).** A 30 s window is 150 steps;
  an hour is 18,000. `--sample-rate 50` builds a 50 Hz member for the SDK tier.
- **Preprocessing**, shared by every source and by a phone:
  - fill gaps of 0.5 s or less;
  - band-pass 0.5–8 Hz (PaPaGei, AnyPPG, HiMAE);
  - polyphase resample;
  - 30 s windows, each robust-scaled (median, IQR/1.349);
  - a quality gate: finite, not flat, not clipped, and autocorrelation ≥ 0.35 at a lag of 30–200 bpm.

  Each recording keeps at most 240 windows, spread over its length, so a 3 h surgery does not outweigh
  an 8 min ICU record.
- **Objectives:**
  - **`next`**: predict the next 5 patches (1 s) from every step, with an L1 loss (a fixed-scale
    Laplace). It is GPT-PPG's objective, minus the predicted scale.
  - **`next+subject`** adds an InfoNCE term in which other windows of the same person are the
    positives. This is Apple's participant-positive contrast, applied to the pooled embedding that
    the clock reads.

  Masked reconstruction (HiMAE, MMR) was not chosen: it trains a bidirectional reader, while a phone
  serves a causal one.
- **Clock:**
  - the age head reads the mean of the final hidden states over all valid patches;
  - the head is an MLP that predicts a deviation from the training mean, in units of the age SD;
  - training draws windows weighted by 1/(the subject's window count);
  - scoring is per subject (the mean over up to 20 windows);
  - the bias correction is fitted on validation subjects (see run 1 for why not training ones), and the
    gap is the residual of prediction on age;
  - for every outcome flag present, the gap's AUC is reported on validation and test subjects;
  - splits are by subject, from a hash, so they stay put as data is added.
- **Streaming and ONNX:**
  - `PPGAgeClock.stream` carries the `LayerStates` and a running pool.
  - `stream.onnx` takes a chunk of samples (default 25 patches = 5 s), a per-patch valid mask, the pool
    and every layer's memory, and returns the age so far with the updated state.
  - It reads the trained weights through `kron_core.onnx.PlainKronMixer`, as krontime does.

## First results: run 1 (2026-11-09, one A100)

**Data.** 2,000 VitalDB cases, one per person, gave 405,242 windows of 30 s from 1,996 people. Four
cases had no window that passed the gate. GalaxyPPG gave 1,847 windows and BIDMC 820. Subjects were
split by hash: 1,384 train, 304 validation and 308 test people with an age. Training ages ran 7–89 y,
mean 58.1, SD 14.5.

**Pretraining.**
- Encoder: 5.85M parameters, `next+subject`, 20,000 steps × 128 windows on VitalDB + GalaxyPPG;
  BIDMC was held out as a clean external check.
- It took 28 min at about 1,600 windows/s.
- Validation next-patch L1 fell from 0.263 to 0.197.

**Clock.** Three clocks, each trained for 6,000 steps on VitalDB:
- *scratch*: no pretraining;
- *finetune*: from the pretrained encoder;
- *probe*: the pretrained encoder frozen, only the head trained.

Scores are per person, as the mean over up to 20 windows.

| | test MAE | test r | val MAE | BIDMC MAE (52 people) | BIDMC r |
|---|---|---|---|---|---|
| predicting the median age | 11.33 | – | 11.17 | 14.31 | – |
| scratch | 7.79 | 0.75 | 8.31 | 12.58 | 0.65 |
| finetune | 7.45 | 0.77 | 7.67 | 13.60 | 0.65 |
| probe (frozen encoder) | **7.33** | **0.78** | 7.78 | 15.29 | 0.68 |

- **Pretraining helps a little.** The frozen pretrained encoder with a small head beats the same model
  trained end to end from scratch: test MAE 7.33 against 7.79. Fine-tuning overfits: its training MAE
  is 2.8 y and its test MAE 7.4 y.
- **Against published numbers**, 7.3 y on held-out VitalDB people sits at the better end of the open
  results:
  - AI-PPG Age zero-shot on VitalDB: 11.5 y;
  - Brag benchmark on PulseDB: 8.2–9.7 y;
  - AnyPPG: 6.1 y, on another cohort.

  It is far from Apple's 2.4–3.2 y with 214k people. VitalDB is one hospital and one device, so this
  is in-domain.
- **Transfer is weak.** On BIDMC (ICU, mostly elderly, another hospital and monitor) every clock is only
  slightly better than predicting the median age, and r stays at 0.65–0.68.
- **The age gap carries no outcome signal yet.** The gap was bias-corrected on validation people and
  read on the 308 test people (87 hypertensive, 21 diabetic, 29 with ASA ≥ 3; one in-hospital death):

  | outcome | AUC of gap (scratch / finetune / probe) | AUC of age alone |
  |---|---|---|
  | hypertension | 0.52 / 0.51 / 0.54 | 0.75 |
  | diabetes | 0.56 / 0.54 / 0.54 | 0.76 |
  | ASA ≥ 3 | 0.51 / 0.50 / 0.51 | 0.68 |

  With these counts the AUCs carry about ±0.07 of sampling noise, so none differs from 0.5. By the
  standard this project uses, run 1 is **not yet a useful aging clock**, only a fair age regressor.
- **A bias-correction bug, caught in this run.**
  - The first scoring fitted the correction on the training people. The overfit model's slope is near
    1 there, so the held-out gap stayed anti-correlated with age (r −0.3 to −0.6), and every age-linked
    outcome read as a *younger* gap (AUC 0.34–0.43).
  - `kron_ppg.clock` now fits the correction on validation people and reports age's own AUC beside the
    gap's.
  - The table above is recomputed from run 1's saved predictions.
- **ONNX:** `stream.onnx` for the finetuned clock (24 MB, 25-patch chunks).
  - It matches PyTorch in ONNX Runtime to 1e-5 y.
  - In float32 it matches the A100 kernels to 4e-6 y; under bfloat16 autocast the difference is 0.03 y.

Run folders: `data/output/runs/kron-ppg/r1-{pretrain,clock-scratch,clock-finetune,clock-probe}`.
Each holds `metrics.json`, `predictions.csv` and `bidmc.json`.

## Run 2 (2026-11-10, one A100)

**Data.** 5,887 VitalDB cases, one per person (every case with a PLETH track and an age), gave 1,195,506
VitalDB windows of 30 s at 25 Hz after the quality gate. People with an age were split by the same hash
as run 1: 4,071 train, 912 validation and 897 test. Run 1's 308 test people are among the 897, because
the split does not move when people are added. The same windows, and the same windows at 125 Hz, are
published as the dataset [`GlucoseDao/ppg-windows`](https://huggingface.co/datasets/GlucoseDao/ppg-windows)
(below).

**Pretraining.**
- The same encoder as run 1 (5.85M parameters, `next+subject`), on VitalDB + GalaxyPPG, with BIDMC held out.
- 40,000 steps × 128 windows, twice run 1's length, on three times the people. It took 49 min at
  about 1,740 windows/s.
- Validation next-patch L1 fell to 0.156 (run 1: 0.197).

**Clock.** The same three clocks as run 1, 8,000 steps each, scored per person on up to 20 windows.

| | test MAE | test r | val MAE | BIDMC MAE (52 people) | BIDMC r |
|---|---|---|---|---|---|
| predicting the median age | 11.74 | – | 11.46 | 14.31 | – |
| scratch | 7.75 | 0.76 | 8.11 | 12.54 | 0.54 |
| finetune | **6.77** | **0.82** | 7.12 | 16.62 | 0.70 |
| probe (frozen encoder) | 6.82 | 0.81 | 7.11 | 14.11 | 0.64 |
| *run 1, best (probe; 308 test people)* | *7.33* | *0.78* | *7.78* | *15.29* | *0.68* |

- **Pretraining now pays.** With three times the people, both clocks on the pretrained encoder beat the
  same model from scratch by about 1 y (6.8 against 7.75), and the gap to scratch is larger than in
  run 1 (0.5 y). Scratch itself barely moved (7.79 → 7.75), so the gain comes from the encoder, not
  from more labelled people for the head. Fine-tuning overfits less than in run 1 (train MAE 4.6 against
  2.8) and reaches its best validation score at step 5,500 of 8,000.
- **In-domain only.** On BIDMC no clock beats predicting the median age by much: the fine-tuned clock is
  worse than that (16.6 against 14.3) although its r is the highest (0.70), so it ranks BIDMC's people
  by age but sits at the wrong level for an ICU monitor's signal. Transfer to another hospital and
  device is not shown.
- **The age gap still carries no outcome signal.** Bias-corrected on the 912 validation people, read on
  the 897 test people (273 hypertensive, 101 diabetic, 87 with ASA ≥ 3, 3 in-hospital deaths). The
  intervals are 2,000 bootstrap resamples of the test people:

  | outcome | AUC of gap: scratch | finetune | probe | AUC of age alone |
  |---|---|---|---|---|
  | hypertension | 0.53 (0.49–0.57) | 0.53 (0.49–0.57) | 0.54 (0.49–0.57) | 0.73 |
  | diabetes | 0.49 (0.44–0.55) | 0.51 (0.45–0.57) | 0.52 (0.45–0.58) | 0.69 |
  | ASA ≥ 3 | 0.51 (0.45–0.58) | 0.55 (0.48–0.62) | 0.54 (0.47–0.61) | 0.62 |

  Every interval contains 0.5. Three deaths are too few to read. A better age regressor did not give a
  more informative gap: what the encoder learns about age it uses to predict age, and what is left over
  is not, so far, about these outcomes. By this project's standard run 2 is still **an age regressor,
  not an aging clock**.

**Seeds.** Each clock was trained again with other seeds (the head's initialisation and the order of
the batches; the encoder, the split and the test people stay the same). Test MAE: scratch 7.75 / 7.84 /
7.69 (mean 7.76, SD 0.08), probe 6.82 / 6.76 / 6.83 (6.80, SD 0.04), finetune 6.77 / 6.78. The
pretraining gain of about 1 y is some fifteen times the spread between seeds, and the gap's outcome
AUCs stay between 0.49 and 0.56 in every seed. Seeds show the results are not a training accident; they
do not shrink the sampling uncertainty of 897 test people, which the bootstrap intervals carry.

**Longer pretraining (run 3).** The run 2 encoder pretrained again for 80,000 steps instead of 40,000
(same data, objective and seed; 1 h 40 min) reached a validation next-patch L1 of 0.152 (run 2: 0.156).
Read through a frozen encoder, it gives a clock with test MAE 6.48 y (r 0.83), against 6.80 ± 0.04 for
run 2's probe, and a trait head whose gain over age for ASA ≥ 3 is +0.148 (0.074–0.223; run 2: +0.12)
and for `healthy` +0.037 (0.015–0.060); hypertension (+0.014, −0.001–0.029) and diabetes (+0.003) still
gain nothing, and within-person r for pressure stays at 0.62–0.64. The age gap still carries nothing
(AUC 0.53–0.55). Over three seeds the run 3 clock gives 6.48 / 6.49 / 6.57 (mean 6.51, SD 0.05),
against 6.80 ± 0.04 for run 2, so longer pretraining's gain on age, 0.3 y, is real. On the traits it is
not settled: a second seed of the run 3 head gives ASA ≥ 3 +0.131 (0.058–0.207) and `healthy` +0.030,
against +0.120 to +0.122 and +0.027 to +0.031 for run 2's three seeds, so the gain there is at most a
little and within the spread between seeds. Run folders `r3-pretrain`,
`r3-clock-probe` (and `-s1`, `-s2`), `r3-traits-probe` (and `-s1`).

**Longer still (run 4).** At 160,000 steps (3 h 15 min) the validation L1 reached 0.149. The frozen
clock gives 6.38 / 6.40 / 6.45 over three seeds (mean 6.41, SD 0.03), against 6.51 for run 3 and 6.80 for
run 2: each doubling of pretraining has bought less, 0.3 y and then 0.1 y. The trait head did not move
(ASA ≥ 3 +0.131, 0.053–0.209; `healthy` +0.032; hypertension +0.011; diabetes +0.007; within-person r
for pressure 0.64), and the gap still carries nothing (AUC 0.52–0.57). More steps on the same 5.85M-
parameter encoder and the same 4,000 training people are close to spent; the next gains, if any, should
come from a larger encoder, more people, or other data. Run folders `r4-pretrain`, `r4-clock-probe` (and
`-s1`, `-s2`), `r4-traits-probe`.

**Not done in this run: foundation-model baselines.** Pulse-PPG, PaPaGei-S and AnyPPG embeddings on the
same split were being computed by another session; its comparison was not finished when this was
written, so kron-ppg is not yet compared with them here.

**Traits from PPG directly (run 2a).** The gap is not the only way to ask whether PPG carries health
information. Run 2a trained one multi-task head (`kron-ppg traits`) on run 1's encoder for age, BMI,
hypertension, diabetes, ASA ≥ 3 and sex, on the 3,986 people downloaded by then (2,735 train, 637
validation, 614 test), and scored each trait on the test people against age alone and against a
logistic model of age plus the PPG score, fitted on validation people:

| trait (test positives / negatives) | AUC of PPG | AUC of age | AUC of age + PPG |
|---|---|---|---|
| hypertension (183 / 431) | 0.71–0.72 | 0.74 | 0.76 |
| diabetes (63 / 551) | 0.65–0.68 | 0.68 | 0.69–0.70 |
| ASA ≥ 3 (56 / 545) | 0.73 | 0.65 | 0.71–0.72 |
| male sex (298 / 316) | 0.75–0.76 | 0.60 | 0.76–0.78 |

Ranges span the frozen and fine-tuned encoder. PPG alone ranks ASA ≥ 3 and sex better than age does,
and hypertension and diabetes about as well; adding it to age gains 0.01–0.03, within the noise of
these counts (run 2a had no intervals; the code now bootstraps `delta_auc`). BMI: MAE 2.66 against 2.87
for the median, r 0.38. Both heads reached their best validation loss at step 500 of 6,000, so the
encoder's features were read almost as they are. These trait heads are the more promising route to
outcome evidence than the gap, and run 2's traits (every target, on the run 2 encoder, with the
window-level vitals) test it with intervals.

**Traits on the run 2 encoder (run 2b).** The same multi-task head, now on the run 2 encoder and all 5,887
people (4,071 train, 912 validation, 897 test), with every target the code has: the outcome flags, sex,
height, weight, BMI, four preoperative lab values, `healthy` (ASA 1 with neither hypertension nor
diabetes), and five vitals read per window (below). 8,000 steps, frozen and fine-tuned encoder; the two
agree within about 0.02 everywhere, and the table gives frozen / fine-tuned. `Δ` is the AUC of age + PPG
minus the AUC of age alone (both logistic fits on validation people), with a 95% bootstrap interval over
the test people.

| trait (test positives / negatives) | AUC of PPG | AUC of age | AUC of age + PPG | Δ (95% interval) |
|---|---|---|---|---|
| hypertension (273 / 624) | 0.71 / 0.71 | 0.73 | 0.74 | +0.009 (−0.006–0.025) / +0.007 (−0.005–0.018) |
| diabetes (101 / 796) | 0.67 / 0.67 | 0.69 | 0.68 | −0.006 (−0.05–0.04) / −0.008 (−0.05–0.04) |
| ASA ≥ 3 (87 / 794) | 0.76 / 0.74 | 0.62 | 0.75 / 0.73 | **+0.12 (0.045–0.20)** / **+0.11 (0.034–0.18)** |
| healthy (256 / 631) | 0.75 / 0.75 | 0.74 | 0.76 | **+0.027 (0.004–0.05)** / **+0.027 (0.006–0.049)** |
| male sex (448 / 449) | 0.77 / 0.77 | 0.59 | 0.78 | +0.19 (0.15–0.23) |

- **PPG adds to age for overall health, not for hypertension or diabetes.** For ASA ≥ 3 (severe systemic
  disease, as the anaesthetist graded it) PPG alone ranks people better than age does, and adding it to
  age gains about 0.11–0.12 AUC with an interval clear of zero. For `healthy` the gain is small (0.03) but
  also clear of zero. For hypertension and diabetes PPG alone is nearly as good as age, but it carries
  nothing age does not: the gain is 0.01 or less, with intervals across zero. That is the first outcome
  evidence kron-ppg has, and it comes from a trait head, not from the age gap. Two more seeds of the
  frozen head give the same: ASA ≥ 3 +0.120 (0.040–0.201) and +0.122 (0.041–0.202), `healthy` +0.029 and
  +0.031 (lower bounds 0.008), hypertension +0.009 to +0.010, diabetes −0.011 to −0.013; within-person r
  for pressure 0.61–0.65.
- **Body and blood.** Height (MAE 6.4 cm against 7.3 for the median, r 0.50), weight (8.0 kg against 9.3,
  r 0.55), BMI (r 0.39), haemoglobin (r 0.41) and albumin (r 0.41) are read weakly; glucose (r 0.28) and
  creatinine (r 0.16–0.19, MAE no better than the median) are not.
- **Age**, as a by-product of the multi-task head: MAE 7.7–7.8 y, worse than the dedicated clock (6.8).

Vitals, per 30 s window on the test people (window-level r across all windows, and *within-person* r,
where each person's mean is removed first, which is what a watch tracking one wearer needs):

| vital | windows (people) | r | within-person r |
|---|---|---|---|
| systolic pressure | 17,390 (897) | 0.58 / 0.59 | 0.65 / 0.67 |
| diastolic pressure | 17,389 (897) | 0.58 / 0.58 | 0.64 / 0.64 |
| RMSSD (ECG) | 8,124 (765) | 0.80 / 0.89 | 0.70 / 0.82 |
| pulse arrival time | 7,883 (758) | 0.87 / 0.90 | 0.76 / 0.80 |
| SpO2 | 17,876 (897) | 0.19 / 0.18 | 0.10 / 0.10 |

- **Blood pressure moves with the PPG within a person** (r 0.64–0.67), from a model never told who the
  person is. It is not heart rate in disguise: within a person, systolic pressure correlates with the
  PPG's own heart rate at r 0.13 and with the monitor's at 0.29 (all 178,828 test windows), so the head
  reads more of the pulse than its rate. Pressure here is the arterial line's where there is one, else the cuff's, which is held
  between readings, so part of each window's label is minutes old. SpO2 is not read: in surgery it sits
  at 97–100% with little to follow.
- **RMSSD and PAT are read closely, and that needs care.** PPG beat-to-beat intervals carry RMSSD, so that
  one is expected. PAT is not cleanly physiological here: VitalDB's PLETH lags its ECG by a device delay,
  the measured PAT is a median 0.83 of the R–R interval (0.6–0.8 s), and it correlates with the R–R
  interval (r 0.51 across windows, 0.37 within a person). The PPG's r of 0.87–0.90 is more than heart
  rate explains, but part of it may be the PPG recognising the monitor and its delay. PAT from this data
  should not be read as arterial stiffness.

**The healthy-only clock.** If a clock trained only on healthy people (1,103 train, 268 validation people
with ASA 1 and neither hypertension nor diabetes; frozen run 2 encoder) read the sick as older, its gap
would separate them. It does not: on all 897 test people, MAE 7.75 y (r 0.79), and gap AUCs of 0.50
(0.46–0.54) for hypertension, 0.48 (0.42–0.54) for diabetes and 0.50 (0.43–0.56) for ASA ≥ 3.

**A vitals bug, caught in this run.** The first vitals table had pulse arrival times piled at the two ends
of the 80–600 ms search (10% at 80 ms, over a quarter at 598 ms) and a median RMSSD of 106 ms, absurd
under anaesthesia. Two causes: the upstroke often lay beyond 600 ms (the device delay above), so an edge
maximum passed as a value; and in some recordings the R-peak detector took T waves for R peaks (R–R
alternating 320/640 ms, ECG rate twice the monitor's). `kron_ppg.vitals` now searches to the next beat
and drops edge maxima, drops R–R features that alternate or disagree with the PPG's pulse rate, and reads
HRV on normal-to-normal intervals. ECG features now cover 539,315 windows (from 1,038,202), median RMSSD
is 7.2 ms and the ECG rate agrees with the monitor's to a median 0.6 bpm. The first traits run
(`r2-traits-*`) used the broken table: its subject-level results match run 2b's, its PAT and RMSSD ones
are void.

Run folders: `data/output/runs/kron-ppg/r2-{pretrain,clock-scratch,clock-finetune,clock-probe}`,
`r2a-traits-{probe,finetune}`, `r2b-traits-{probe,finetune}`,
`r2-clock-healthy-probe`, and the seed repeats `r2-clock-{scratch,probe}-s{1,2}`, `r2-clock-finetune-s1`,
`r2b-traits-probe-s{1,2}`. Each clock holds `metrics.json`, `predictions.parquet` and `bidmc.json`.

**The dataset.** `GlucoseDao/ppg-windows` holds the prepared windows of VitalDB (25 and 125 Hz), BIDMC
(25 and 125 Hz) and GalaxyPPG (25 Hz), with the training split as the Hub's train/validation/test and a
subject table per source; each config was loaded back from the Hub with polars. VitalDB at 125 Hz has
183,618 test windows. The VitalDB configs (25 and 125 Hz) also carry the per-window vitals described
below (blood pressure, SpO2, temperature, ECG heart rate and variability, pulse arrival time), from the
corrected vitals table, with the device-delay caveat on pulse arrival time in the card.

## What is not done

- BUT PPG (phone fingertip, 30 Hz) and PPG-DaLiA readers.
- Baselines on the same splits:
  - age + sex alone;
  - pyPPG-style morphology features + ridge;
  - AI-PPG Age on averaged beats;
  - Pulse-PPG embeddings + ridge.

  Without them, a kron-ppg MAE cannot be called good or bad.
- A random-init twin at full scale, to show what pretraining buys (the `clock` command can do it:
  omit `--encoder`).
- A wrist validation of anything trained on VitalDB.
- A predicted scale for the next-patch likelihood (GPT-PPG's logit-Laplace) and a masked-patch
  variant for comparison.
- Calibration of the reported error band per device; until then, the app shows the band from the run's
  `metrics.json` with the warning from `docs/ppg-age.md`.

## Sources

- Earlier work: `~/sources/pulse-exploration/docs/ppg-age.md`;
  `research_notes/PPG health and aging measures/open_models_datasets.md` (the full citation list).
- [GPT-PPG, arXiv 2503.08015](https://arxiv.org/abs/2503.08015)
- [Apple PPG/ECG FM, arXiv 2312.05409](https://arxiv.org/abs/2312.05409)
- [AI-PPG Age, arXiv 2502.12990](https://arxiv.org/abs/2502.12990)
- [Brag age benchmark, arXiv 2603.14030](https://arxiv.org/abs/2603.14030)
- [VitalDB](https://vitaldb.net/dataset/); [VitalDB on PhysioNet](https://physionet.org/content/vitaldb/1.0.0/)
- [BIDMC](https://physionet.org/content/bidmc/1.0.0/); [BUT PPG](https://physionet.org/content/butppg/2.0.0/)
- [PPG-BP](https://doi.org/10.6084/m9.figshare.5459299); [PPG-DaLiA](https://archive.ics.uci.edu/dataset/495/ppg+dalia)
- [MIMIC-IV WDB](https://physionet.org/content/mimic4wdb/0.1.0/); [Autonomic Aging](https://physionet.org/content/autonomic-aging-cardiovascular/1.0.0/)
- [PWDB](https://zenodo.org/records/3275625); [GalaxyPPG](https://zenodo.org/records/14635823)
- [fela-ppg](https://huggingface.co/lowdown-labs/fela-ppg)
