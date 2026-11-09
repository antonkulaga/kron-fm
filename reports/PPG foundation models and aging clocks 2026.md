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
| **VitalDB** API (`api.vitaldb.net`) | surgical finger PPG `SNUADC/PLETH`, 500 Hz | age, sex, ASA, `preop_htn`, `preop_dm`, `death_inhosp` per case; `subjectid` links one person's repeat surgeries | 6,157 cases with PLETH; about 4.3 MB gzipped per case (about 3 h at 500 Hz) | CC BY-NC-SA 4.0 under the VitalDB data use agreement (research). The PhysioNet copy (95 GB of `.vital` files) is CC BY 4.0 | `vitaldb`: the training set |
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
  - the bias correction is fitted on training subjects, and the gap is the residual of prediction on
    age;
  - for every outcome flag present, the gap's AUC is reported on validation and test subjects;
  - splits are by subject, from a hash, so they stay put as data is added.
- **Streaming and ONNX:**
  - `PPGAgeClock.stream` carries the `LayerStates` and a running pool.
  - `stream.onnx` takes a chunk of samples (default 25 patches = 5 s), a per-patch valid mask, the pool
    and every layer's memory, and returns the age so far with the updated state.
  - It reads the trained weights through `kron_core.onnx.PlainKronMixer`, as krontime does.

## First results

None yet worth reporting.
- **Smoke runs** (30 VitalDB cases + BIDMC, a 1M-parameter encoder, 300 steps each) showed the whole
  path runs on the GPU, from pretraining through the clock, with metrics, predictions and outcome AUCs
  written. With about 12 test subjects their numbers carry no information.
- **ONNX:** the exported graph matches its PyTorch module to 3e-5 years in ONNX Runtime. In float32 it
  matches the Triton streaming path to 3e-5 years; under the bfloat16 autocast used in training the
  difference is 0.36 years.

The first real run needs several hundred to a few thousand VitalDB cases. A 500-case download was
under way on 2026-11-09.

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
