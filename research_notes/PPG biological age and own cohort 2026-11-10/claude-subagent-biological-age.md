# Separating chronological from biological age with kron-ppg: a critical proposal

Written 2026-11-10. Inputs read:
- `~/sources/kron-fm/reports/PPG foundation models and aging clocks 2026.md` (runs 1 to 3)
- `src/kron_ppg/{clock,traits,model,data,train,readers,cli}.py`
- `~/sources/pulse-exploration/docs/ppg-age.md`
- `~/sources/pulse-exploration/docs/galaxy-watch-ppg-foundation-models.md`, including the r0 frozen-encoder benchmark
- the VitalDB case table at `data/input/ppg/vitaldb/cases.csv.gz` (all 6,388 cases), which I counted directly

Nothing in the repository was changed.

---

## 0. Bottom line

1. **None of the three foundation models separates biological from chronological age.**
   - Pulse-PPG has no age task at all.
   - PaPaGei and AnyPPG report age only as a linear-probe MAE.
   - None reports an age gap, any bias correction, or any link between a gap and outcomes. The only
     outcome-validated PPG ages (Apple PpgAge, AI-PPG Age) are supervised age models validated by
     hazard ratios that are adjusted for age.
2. **The gap fails here for a predictable reason: dilution.**
   - With an MAE of 6.5 y, the bias-corrected gap has an SD of about 8 y. That SD is mostly estimation
     error.
   - At that SD, a disease that truly shifts pulse age by 2 y gives an AUC of only about 0.57. The
     same 2 y shift gives about 0.66 at Apple's error.
   - More important, the trait head shows that most of the health signal in the embedding is **not
     along the age direction**. A residual of an age regression can only carry health signal that
     lies on that direction.
3. **What the data can test.** The most informative next steps are cheap and need no new training:
   - re-test the trait head's ASA gain against nuisance covariates (BMI, heart rate, surgery type),
     which may explain it away;
   - measure how much of the health signal lies on the age axis;
   - fit a normative "unusual for your age" deviation score on frozen embeddings;
   - use the case table's timeline (incision, emergence) to read within-person responses that a
     streaming model is suited to.
4. **On the multi-task question.** The held-out-task test is right, but held-out tasks must be whole
   *families*, and the run needs compute-matched and shuffled-label controls. A pilot already exists
   (section 3): fine-tuning on six subject-level traits changed nothing on held-out targets, and it
   made wrist transfer worse.
5. **On the term.** Do not call anything here "biological age" until a score predicts outcomes beyond
   age, sex, BMI and heart rate, in people and outcomes it was never trained on.

---

## 1. Literature: did Pulse-PPG, PaPaGei and AnyPPG distinguish biological from chronological age?

**Short answer: no.** None of the three defines a biological age, an age gap or age acceleration, and
none relates a predicted age to any outcome.

### Pulse-PPG (Saha et al., IMWUT 2025; [arXiv 2502.01108](https://arxiv.org/abs/2502.01108), DOI 10.1145/3749494)
- **Age is not a downstream task.**
  - The downstream tasks (Table 1 and appendix Tables 6–10) are stress, activity, heart rate, sleep
    disturbance, systolic and diastolic BP, and hypertension.
  - "Age" appears only in cohort descriptions (participants with an "average age of 38 years").
  - I read the main text and the appendix.
- **No biological age, gap or outcome link.**
- The repository's statement that Pulse-PPG gives the "best open age result in the Brag benchmark"
  rests on **Brag's** own ridge probe (arXiv 2603.14030), not on the Pulse-PPG paper.

### PaPaGei (Pillai et al., ICLR 2025; [arXiv 2410.20542](https://arxiv.org/abs/2410.20542))
- **Age appears only in Appendix E** ("Using the PaPaGei features, we predict downstream demographics
  such as age and sex").
- **Table 16:**
  - age regression MAE 8.78 [8.47–8.09 as printed];
  - age classification AUROC 0.85;
  - sex classification AUROC 0.74;
  - all on its own pre-training sets (VitalDB, MESA, MIMIC-III).
- **The text and the table disagree.** The text says "7.78 MAE in age regression, 0.85 accuracy". The
  printed interval is also inverted.
- **Its comparison is a cited result, not a run.** It compares with Apple's PPG FM (Abbaspourazad et
  al. 2023: patient-level 3.19 MAE, segment-level 6.60). It notes that patient-level positives capture
  demographics better, which is kron-ppg's `next+subject` design.
- **No biological age, gap or outcome link.** "Vascular Response Index" is a morphology feature, not an
  age.

### AnyPPG (Peking University, KDD 2026; [arXiv 2511.01747](https://arxiv.org/abs/2511.01747), v4, DOI 10.1145/3770855.3818911)
- **"Age estimation" is one regression task**, on the proprietary consumer-device WenXinWuYang set
  (10,035 segments), scored as ridge linear-probe MAE.
- **Table 3:**

  | model | MAE |
  |---|---|
  | AnyPPG | 6.08 [5.89–6.26] |
  | BYOL | 6.08 |
  | SimCLR | 6.09 |
  | MOMENT and Chronos-2 | 6.19 |
  | Pulse-PPG | 6.27 |
  | PaPaGei-S / -P | 6.47 / 6.33 |

  The intervals overlap, so this is not a clear win, and the paper does not discuss the age task.
- **No gap or outcome link for age.** The paper cites AI-PPG Age and Apple's PpgAge only in its
  introduction.
- **A phenome-wide study comes closest to "beyond age".**
  - The paper says AnyPPG "outperforms the strongest demographic baseline (age, sex, and race) in 843 of
    1,468 (57.4%) phenotypes".
  - From the text I could read, it is **PPG against demographics, not demographics + PPG against
    demographics**. That is a weaker test than kron-ppg's ΔAUC.

### How the outcome-validated PPG and ECG ages define and validate "biological"

| Work | Training target and reference | Gap | Validation |
|---|---|---|---|
| **Apple PpgAge** (Miller et al., Nat Commun 16:9264, 2025; [PMC12537950](https://pmc.ncbi.nlm.nih.gov/articles/PMC12537950/)) | Self-supervised encoder on about 213k people with no age or disease labels. Then a **linear probe on chronological age in 5,355 healthy people**: self-reported no disease, no medications, never smoked; 1,373 held out | Raw gap, then a **spline regression of the gap on age**, keeping the residual | MAE 2.43 y healthy, about 3.1–3.3 general. Incident ASCVD composite: **HR 1.464 (1.36–1.57) per 6 y of gap**, adjusted for age, sex, BMI, VO2max, smoking, prior hypertension, cholesterol and diabetes (1,328 events, N 89,553). Within-person: median +3.56 y in pregnancy; +1.7 to 2.5 y around bypass, MI and valve surgery. The paper says it has no mortality data and limited understanding of the mechanism |
| **AI-PPG Age** (Nie et al., Commun Med 2025; [arXiv 2502.12990](https://arxiv.org/abs/2502.12990)) | Supervised age on a **development cohort of 98,672 UK Biobank people without circulatory disease, diabetes or dyslipidaemia** | **Raw gap, no bias correction**; every Cox model includes age | MAE 7.57 y (r 0.49). Gap > 9 y: MACCE **HR 2.37 (2.17–2.59)**, adjusted for age, sex, ethnicity and BMI; 2.24 with risk factors added. MIMIC-III in-hospital mortality: OR 1.02 per year in model 1, which **loses significance** in model 2 (1.01, p = 0.06) |
| **OPPO smartwatch heart age** ([arXiv 2608.01620](https://arxiv.org/abs/2608.01620)) | ECG-guided cross-modal pretraining with subject-aware learning; 581,804 people | Age-adjusted gap | MAE 4.3–5.9 y. Validated against **pulse-wave velocity** (partial r 0.26 after age) and prevalent hypertension (OR 1.72 per SD). Cross-sectional, no hazards. Repeated recordings improve stability |
| **AI-ECG age** ([EHJ Digital Health 6(6):1204](https://academic.oup.com/ehjdh/article/6/6/1204/8272814); search-result level only) | ECG age | Δ-age ≥ +8 y | 10-year mortality HR 1.45 in 48,950 patients |
| **AIRE** (Sau et al., Lancet Digit Health 2024; abstract level) | **Trained directly on survival** (deep network + discrete-time survival model), not on age | – | The outcome-anchored alternative to an age clock, on 1.16M ECGs. I did not read the full text |

**What the working clocks share, and VitalDB lacks:**
1. A **truly healthy reference** of thousands, defined by absence of disease and medication.
2. **Many repeated measurements per person** (Apple: long-term wrist data), which pushes the gap's noise
   SD down to about 3 y.
3. **Incident hard outcomes**, with tens of thousands to hundreds of thousands of people followed over
   time.
4. **Age always in the outcome model**, whether the gap is bias-corrected (Apple) or not (AI-PPG Age).

kron-ppg has a surgical "healthy" subset of about 1,100 people, about 20 windows from one session,
and only prevalent comorbidities plus 57 in-hospital deaths among 6,388 cases (48 people). That
mismatch explains most of the null result. It is not a property of the KronLSTM.

---

## 2. Why the residual gap fails while a trait head works

### 2.1 Dilution: the gap is mostly estimation error

- **The arithmetic.**
  - At MAE 6.5 y, the gap SD is about 7.5–8 y.
  - For a binary condition that shifts true pulse age by Δ years, AUC ≈ Φ(Δ / (σ√2)).
  - At σ = 7.8: Δ = 2 y gives AUC 0.57 and Δ = 1 y gives 0.54.
  - At Apple's σ of about 3.5, the same Δ = 2 y gives 0.66.
- **What run 2's intervals rule out.** Upper bounds of 0.57 (hypertension) and 0.62 (ASA ≥ 3) exclude
  shifts above about 2 y and 3.4 y.
- **Effect sizes in the literature.** Apple's per-6-year hazard ratio of about 1.46 implies gap
  differences of a few years between diseased and healthy groups.
- **Conclusion.** Run 2 was never powered to see an Apple-sized aging effect through the gap. Better
  MAE alone (6.5 → 5) would barely change that.

### 2.2 What the clock is optimised for: it learns to discount disease

- **The mechanism.**
  - An MSE age clock trained on a mixed cohort is rewarded for predicting *calendar* age.
  - If disease makes a young person's pulse look old, that costs loss.
  - So the more the encoder knows about disease (and it does: ASA is decodable), the more the head
    learns to *correct for* disease to hit the calendar.
- **The prediction.** A better encoder gives a better age MAE and a *less* informative gap. That is
  run 2's pattern exactly: MAE fell, gap AUC stayed at 0.5.
- **This is why aging clocks use a healthy reference.**

### 2.3 The healthy-only clock is a weak version of the right design

- **"Healthy" is not healthy.** It means ASA 1 with neither hypertension nor diabetes, on the day of
  surgery: 77% of cases are general surgery, much of it oncological.
- **The reference is small.** About 1,100 people.
- **It is younger than the cohort.** The bias line is fitted on healthy validation people and then
  *extrapolated* to the older, sicker test people.
- **It is noisier.** MAE 7.75 y, so σ is larger and dilution worse.
- **It is not Apple's design.** Apple used self-report of no disease, no medication and no smoking,
  plus long-term averaging.

### 2.4 Most of the health signal is not along the age axis

- **The geometry.**
  - The gap reads only the projection of health information onto the age direction of the embedding.
  - The trait head reads health from *any* direction: perfusion, heart rate, rhythm, body size, the
    shape changes of dialysis or obesity.
- **The pattern fits.**
  - ASA ≥ 3 gains +0.12 to +0.15 AUC from a trait head.
  - The CKD probe gains +0.156 and BMI +0.19 R² (r0 benchmark).
  - Hypertension and diabetes, the conditions most like "accelerated vascular aging", gain about 0.
- **This can be measured directly** (section 6): split the ASA probe into its component along the age
  probe and the orthogonal rest, and score each.

### 2.5 Label and outcome problems specific to VitalDB

- **ASA is a clinician's grade, not a biomarker.**
  - It encodes age implicitly.
  - It encodes **BMI** explicitly: obesity is ASA 2, BMI ≥ 40 is ASA 3.
  - It encodes ESRD and dialysis, AF, heart failure and emergency status, and it depends on the
    surgical service.
  - The trait head's ASA gain may therefore be BMI + kidney + heart rate + surgery type. Run 2b's "beyond
    age" model has **only age** as a covariate.
  - The r0 benchmark's heart-rate-only baseline gives −0.03 for ASA, which helps. But there is no BMI-,
    eGFR- or department-adjusted test yet.
- **Hypertension and diabetes are prevalent and mostly treated.** Antihypertensives and glucose
  control move exactly the physiology a pulse would show, so the label is a history, not a state.
- **Anaesthesia overrides vascular tone.**
  - The case table puts `anestart` a median **583 s before** the recording starts. Only 424 of 6,388
    cases have more than 60 s of recording before anaesthesia, and 262 more than 3 min.
  - So almost every window is under propofol or volatile vasodilation, often with ephedrine (3,211
    cases) or phenylephrine (844 cases).
  - Age-related stiffness survives that; subtler disease signal may not.
- **Per-person averaging mixes states.** Twenty windows spread over a whole case average across
  induction, incision, maintenance and emergence.
- **Hard outcomes are rare.**
  - In-hospital death: 57 cases (48 people), 3 in the test split.
  - ICU stay > 0: 1,204 cases, but it is largely planned by surgery type.
  - Length of stay: median 7 d, 90th percentile 19 d, and strongly driven by the operation.
- **Some targets are built from age.**
  - eGFR (CKD-EPI) and therefore CKD are computed *from age*, so their association with age is
    partly arithmetic.
  - ΔAUC over age is still valid, but "PPG adds to age for CKD" really means "PPG reads creatinine
    given age and sex".

### 2.6 Device and heart-rate shortcuts
- The PAT target is contaminated by monitor delay; the report already says so.
- Albumin and anaemia gains ride on heart rate.
- The r0 benchmark's *untrained* KronLSTM still gets albumin ΔR² +0.157. Every claim needs the
  untrained-encoder and heart-rate-only rows beside it.

---

## 3. Can multi-task training force the encoder to learn biology? (the user's question)

### 3.1 Existing pilot evidence, in the repository and the r0 benchmark

The r2a six-trait head (age, BMI, hypertension, diabetes, ASA, sex) was **fine-tuned end to end**.
Its encoder was then frozen and probed by `fm_baselines` alongside the self-supervised encoder, on
targets that were **not** in its training set:

| held-out target (Δ over age+sex) | SSL encoder | six-trait fine-tuned encoder |
|---|---|---|
| CKD, ΔAUC | +0.156 | +0.153 |
| eGFR, ΔR² | +0.039 | +0.042 |
| anaemia, ΔAUC | +0.090 | +0.103 |
| albumin, ΔR² | +0.188 | +0.184 |
| hemoglobin, ΔR² | +0.093 | +0.099 |
| GalaxyPPG wrist, mean predicted age (true 20–29) | 33.6 | **43.5** |

- **Held-out tasks: no change.** That is unsurprising: run 2a's head reached its best validation loss at
  step 500. With about 4k subject-level labels, the encoder barely moves.
- **Wrist transfer got about 10 y worse.** That is the signature of fitting the source domain (finger,
  anaesthesia, monitor), not physiology.
- **So multi-task training on *subject-level clinical labels* does not teach biology here.** If
  multi-task training is going to work, the gradient has to come from somewhere else.

### 3.2 Train the encoder, but on what: window-level physiology, not clinician labels

**Recommendation:** continued pretraining.
- **Loss:**
  - `L = L_next + λ_s·L_subject + λ_m·L_multitask`;
  - keep the self-supervised losses as an anchor against collapse onto the targets and against
    forgetting;
  - encoder learning rate about 1e-4; evaluate at a fixed budget.
- **Draw the multi-task gradient mainly from window-level physiological targets**, which are many and
  within-person:

  | target | coverage |
  |---|---|
  | SBP, DBP | about 1.2M windows (arterial line where present) |
  | RMSSD | about 0.54M windows after the vitals fix |
  | cardiac output and SVV | about 940 cases (EV1000 617, Vigileo 323) |
  | BIS | about 5,900 cases, better used as a nuisance than a target |

- **These targets are *upstream* of aging and disease.** Vascular load, autonomic tone and stroke volume
  are what aging changes. ASA is a downstream summary someone else wrote.

### 3.3 Loss designs, ranked by how well they resist shortcuts

1. **Within-person delta targets.**
   - Take two windows of the same person, a and b. Train `g(z_a − z_b) → (y_a − y_b)` for SBP, DBP,
     RMSSD and CO.
   - Everything stable about the person cancels: device, surgery type, demographics, BMI.
   - The only way to reduce the loss is to map *physiological change* to embedding change.
   - It is the strongest anti-shortcut design, and the existing `SubjectPairs` sampler already draws
     same-person pairs.
2. **Explain away heart rate.**
   - Give the head the window's heart rate (from `hr_bpm`) as an *input*, or residualise each target on
     heart rate within person.
   - Then the encoder gains nothing by re-deriving heart rate and must carry shape information.
3. **Adversaries on nuisances, not on age.**
   - Use gradient reversal against source or device (VitalDB, BIDMC, GalaxyPPG), department,
     emergency status and anaesthesia type.
   - These are things the representation should not know.
   - **Do not reverse age in the shared trunk.** Arterial stiffening *is* the age signal and is also
     health signal; removing it removes biology. Adversarial removal is also typically incomplete: the
     attribute stays decodable nonlinearly.
4. **Subject-level targets residualised on age + sex + BMI.**
   - This forces subject heads to explain what demographics do not.
   - Use it for ASA, eGFR, albumin and hemoglobin, with few heads and a small weight.
   - Note: a linear head given age as an input can already do this. Residualising changes what the
     *encoder's* gradient rewards, not what a linear probe can read.
5. **Task weighting.**
   - Uncertainty weighting (Kendall-style), GradNorm or PCGrad balance tasks with very different label
     counts and noise.
   - That is bookkeeping, not biology. Do it, but expect no conceptual gain from it.
   - Per-task normalised losses plus per-subject sampling for subject-level heads are enough to start.

### 3.4 How to tell "learned biology" from "learned the targets"

**I agree that held-out tasks are the key test, with five refinements:**

1. **Hold out families, not single targets.** Correlated targets leak:
   - SBP → hypertension;
   - glucose → diabetes;
   - creatinine or BUN → eGFR → CKD;
   - BMI, weight and height together;
   - ASA → everything.

   Possible splits:
   - train on hemodynamics (SBP, DBP, RMSSD, CO) and hold out all labs, all diagnoses and outcomes;
   - or the reverse.
2. **Controls on the same budget:**
   - (a) the SSL encoder continued for the same number of steps on the SSL loss only (compute-matched);
   - (b) the multi-task encoder trained on **labels shuffled across people within age-sex strata**. If
     held-out gains also appear here, they are regularisation, not biology;
   - (c) the untrained encoder and the heart-rate-only baseline.
3. **One probe protocol for every encoder.** Use the same people, windows and linear probe, with the
   penalty chosen on validation people, and Δ over **age + sex + BMI + heart rate** (not only age), with
   paired bootstrap intervals. `fm_baselines.probe` already implements this; it needs BMI and heart rate
   added as covariates.
4. **Held-out *within-person* physiology.** For example, train without CO, then test whether CO and SVV
   changes are tracked within person in the 940 cases. Within-person tracking cannot be a demographic
   shortcut.
5. **Held-out device and domain.**
   - BIDMC age and heart rate, and GalaxyPPG wrist heart rate and HRV.
   - Biology should transfer at least as well as SSL features. Run 2a's fine-tune transferred *worse*.
     Make "no worse on the wrist" a pass condition.

A multi-task encoder "learned biology" if it beats the compute-matched and shuffled controls on
held-out families, at a fixed probe protocol, and does not lose on the wrist.

### 3.5 Risks
- **Negative transfer.** The multi-task head's age MAE was 7.7 against 6.8 for the dedicated clock. It
  can hurt held-out tasks the same way.
- **Overfitting subject-level labels.** The run 1 fine-tune reached train MAE 2.8 against test 7.4.
- **Shortcuts:**
  - heart rate (albumin, anaemia);
  - monitor delay (PAT);
  - department or surgery type, which a pulse under a particular anaesthetic practice may reveal;
  - quality-gate survival, which differs with AF and low perfusion.
- **Label noise:**
  - cuff BP is held between readings, so part of each window's label is minutes old;
  - ASA is subjective;
  - diagnoses are treated.
- **Domain lock-in.** The more physiology is learned *under anaesthesia on a finger*, the less is
  guaranteed on an awake wrist. Watch with GalaxyPPG at every checkpoint.
- **Test reuse.** The test split has now been read in runs 1, 2, 2a, 2b, 3 and in seeds. Section 4.7
  covers this.

---

## 4. Approaches, ranked by information per unit of effort on data we have

### Approach 1. Normative age-conditional deviation ("how unusual is this pulse for your age")

**Method**
- Inputs: frozen run 3 encoder; per-person mean embedding of up to 24 windows, or phase-specific means
  (Approach 3); PCA to 16–32 dimensions on training people.
- **Reference:** healthy training people (ASA 1, no hypertension or diabetes, BMI 18.5–30, elective).
- **Fit:**
  - each PCA component's conditional mean and variance given age (spline) and sex, GAMLSS-like;
  - or a linear-Gaussian model `z ~ N(μ(age, sex), Σ(age))`.
- **Two scores per person:**
  - (a) the non-directional deviation: Mahalanobis distance, or the negative log-likelihood under the
    healthy-for-age model;
  - (b) a directional "looks older" score: the age you would have to be for the person's embedding to
    be most likely, minus the actual age. This is a likelihood-based age, not a regression, so it does
    **not** regress to the mean the way an MSE clock does.
- Optional: quantile age heads (pinball loss at 0.1, 0.5 and 0.9) to give a per-person percentile.

**Evaluation**
- Δ over age + sex + BMI + heart rate, on test people, for:
  - ASA ≥ 3, CKD, low albumin, hypertension, diabetes;
  - ICU admission and length of stay, both stratified by `optype`;
  - in-hospital death through **5-fold out-of-fold** scores over all 5,887 people (about 45 deaths:
    the only way to get usable counts).
- Success: positive Δ for some outcome, with the interval clear of zero, after the nuisance
  covariates, and larger than the untrained-encoder row.
- Also report the **ICC across windows and phases**: a trait score should be stable within one session.

**How it could fool us**
- Deviation from "healthy surgical patient of your age" is dominated by acute state: tachycardia,
  vasopressors, anaesthetic depth, low perfusion, surgery type.
- A sick person deviates for reasons unrelated to aging.
- The directional score depends on the healthy reference's age coverage, which is thin above 70.

**Cost:** CPU, minutes, on saved embeddings.

### Approach 2. Two-axis model: an age axis plus an age-orthogonal health axis, anchored on outcomes (PhenoAge-style)

**Method**
1. **Stage 1, the anchor, with no PPG.**
   - On training people, fit a low-dimensional model of a hard composite outcome from age and
     preoperative clinical measures.
   - Outcome: in-hospital death, or ICU stay ≥ 2 d not planned by the surgery type. Use department and
     optype fixed effects, or length of stay above the 90th percentile within optype.
   - Measures: albumin, creatinine or eGFR, hemoglobin, BUN, glucose, sodium, BMI.
   - Convert the risk into "phenotypic age": the age at which an age-only model gives the same risk.
     This is Levine's PhenoAge construction.
   - Full PhenoAge cannot be used: VitalDB lacks CRP, lymphocyte %, MCV, RDW, ALP and WBC.
   - Fallback when the outcome is too sparse: a **Klemera–Doubal** biological age from the same labs.
     It needs only the biomarkers' regressions on age in the healthy reference.
2. **Stage 2, the PPG model.**
   - A two-output head on the frozen encoder: `a(z)` trained on chronological age, and `h(z)` trained
     on (phenotypic age − chronological age).
   - Add a cross-covariance or HSIC penalty between `a` and `h`, so `h` is age-independent within the
     healthy reference.
   - "PPG biological age" = age + h(z). This is `TraitHead` with two targets and one extra penalty term.
3. **A note on residualising.** Residualising the embedding on age and then fitting a linear probe *with
   age as a covariate* is identical to probing the raw embedding with age (Frisch–Waugh–Lovell).
   - "Removing age" adds nothing for a linear Δ-over-age test.
   - It matters only for producing a reportable age-independent score and for nonlinear heads.
   - Do it for the score, not as a discovery method.

**Evaluation**
- `h` must predict **held-out** outcomes and targets beyond age + sex + BMI + heart rate *and beyond the
  lab anchor itself* (death out-of-fold, LOS within optype, CKD, anaemia).
- Otherwise the PPG is only a noisy proxy for labs that were in hand already. For a watch, a noisy lab
  proxy is still useful; report it as such.

**How it could fool us**
- The anchor's labs are partly read through heart rate (albumin, hemoglobin), so `h` can become an
  acute-illness score.
- 57 deaths make Stage 1 unstable; keep it to 3–5 terms with a ridge penalty, and bootstrap Stage 1
  into Stage 2's intervals.
- LOS and ICU encode the surgery and hospital practice.

**Cost:** CPU for Stage 1; one frozen-head GPU run (about 10–15 min) for Stage 2.

### Approach 3. Within-person dynamics: what a streaming recurrent model is suited to

**Background:** the case table gives a timeline per case.
- `opstart` (incision) is a median 34 min into the recording, and more than 5 min in for 6,380 of 6,388
  cases.
- `opend` and `aneend` (emergence) fall near the end. 2,481 cases have more than 5 min of recording
  after `aneend`.
- `Orchestra/PPF20_RATE` (propofol infusion rate) exists in 3,512 cases.
- The arterial waveform `SNUADC/ART` is in 3,645, giving beat-to-beat pressure for baroreflex sensitivity.
- BIS is in about 5,900, as a depth covariate.

Prepared windows carry `start_s`, so phase labels cost no new preprocessing.

**Pre-induction awake windows are rare** (424 cases with more than 60 s), so an induction-hypotension
response is **not** testable on most of VitalDB. Verify this against the PLETH tracks before relying on
the case-table times.

**Method**
1. **Phase embeddings and response vectors, from the existing windows:**
   - pre-incision (−10 to 0 min), post-incision (0 to +10 min), steady maintenance, post-emergence;
   - the response vector is Δz (post − pre);
   - the response magnitude and its time constant come from window-wise heart rate and embedding
     distance over time;
   - inputs are the existing windows plus `start_s` plus the case table.
2. **Streaming version:** run the encoder over the full raw PLETH around each event, using the carried
   state the clock already exposes (`PPGAgeClock.stream`). Read the trajectory of the pooled state at
   5 s resolution.
3. **Baroreflex sensitivity** (sequence method on ART systolic pressure and R–R) and **propofol
   dose-response** (change in pulse amplitude and embedding per change in the TCI rate). Both are
   physiological targets with known age dependence. Try them as *targets*, as in section 3.3, and as
   features.

**Evaluation**
- Δ over age + sex + BMI + baseline heart rate + BIS + propofol and remifentanil rates.
- Targets: diabetes first (autonomic neuropathy is the best physiological prior for diabetes, and
  static pulse shape gave nothing), then hypertension, ASA and CKD.
- A null on diabetes here is informative.
- Test-retest uses the **237 people with a second surgery**: 298 extra cases, currently not prepared
  or used.

**How it could fool us**
- Responses are dominated by drugs, analgesia depth and surgical stimulus (laparoscopic insufflation
  against open incision), all of which vary by `optype`.
- Use stratification and covariates, and permutation of event times as a negative control: a
  "response" at a random time should carry nothing.

**Transfer:** this is the approach that maps best onto a watch, because daily life has perturbations
too (section 7).

**Cost:** phase features on CPU in under an hour; full streaming and ART processing take a day or two
of engineering.

### Approach 4. Multi-task physiological continued pretraining (section 3)

This is the user's proposal, implemented as in section 3: within-person delta targets plus heart-rate
explain-away plus nuisance adversaries, with family-held-out evaluation and the two controls.

- **Cost:** 2–3 GPU runs of about 30–60 min each on the A100. It needs about 100 lines in `traits.py`:
  a pair sampler for delta targets, a heart-rate input, a GRL head, and keeping `next_patch_loss` in the
  sum.
- **How it could fool us:** see section 3.5.

### Approach 5 (lowest). Adversarial removal of age from the trunk, then probing health

- **Why it ranks last:**
  - It removes arterial stiffness, which is the biological age signal itself.
  - Removal is incomplete.
  - A linear Δ-over-age test already conditions on age.
- **When it is useful:** as a diagnostic.
  - Train a GRL-on-age encoder and check that held-out health probes *survive*.
  - If ASA's Δ survives age removal, its signal is not age-like (2.4).
  - If it collapses, health and age share directions.
- The cheap version of the same question is the projection test in section 6, step 3.

### Not recommended as a next step: another age-only clock

- Better age MAE does not fix dilution (2.1) and makes discounting worse (2.2).
- A healthy-reference clock is worth redoing only with:
  - a stricter reference (ASA 1, normal labs, BMI 18.5–30, elective, no vasopressors);
  - phase-restricted windows;
  - the gap tested with age in the model and with Apple's spline correction.

---

## 5. Statistical pitfalls of "biological age" claims, and the guard for each

| Pitfall | Guard |
|---|---|
| **Regression to the mean**: the raw gap correlates negatively with age | Fit the correction on held-out people (already done); report `gap_age_r` on test; always keep age in the outcome model; prefer Apple's spline correction to a linear one |
| **Dilution**: the gap is mostly error | Report the minimal detectable shift, Δ = σ√2·Φ⁻¹(AUC upper bound); use continuous scores in Δ-over-covariates models, never dichotomised gaps |
| **Thresholds chosen after the fact** (AI-PPG Age's ±9 y) | Pre-specify a per-SD effect; any threshold is fixed on validation |
| **Circularity**: validating on the label trained on (an ASA head "predicts" ASA) | Outcomes and target families never seen in training; out-of-fold scores for rare outcomes |
| **Confounding** by BMI, sex, heart rate, surgery type, emergency, anaesthesia, device, quality | Δ over age + sex + BMI + heart rate (+ department or optype fixed effects); untrained-encoder and heart-rate-only rows; **negative-control outcomes** (e.g. operation side, recording hour) and **negative-control exposures** (event-time permutation, shuffled labels) |
| **Targets built from age** (eGFR, CKD) | State it; test creatinine directly with age and sex as covariates |
| **Prevalent, treated disease read as aging** | Call these associations "cross-sectional"; prefer untreated or physiological targets (BP within person, CO, baroreflex) |
| **Reference selection**: a healthy reference younger than the cohort, so the bias line is extrapolated | Check the age overlap; restrict claims to ages the reference covers; reweight the reference by age |
| **Informative missingness**: sick people have longer cases, fewer gated windows, AF | Fix windows per person; quality as a covariate; report gate pass rate by outcome |
| **Garden of forking paths and test reuse**: many encoders, heads, seeds and targets, all read on the same 897 test people | **Pre-register** one primary endpoint per approach; choose on validation; treat the test split as already partly a development set; keep a **lockbox**. Options: the 237 people's second surgeries, BIDMC, and any newly prepared cohort, scored once |
| **Nested-model AUC tests have poor power and calibration** | Keep the paired bootstrap of ΔAUC (in the code), and add log-loss or deviance improvement and calibration |
| **Reliability** never checked | ICC within session (windows, phases) and across the 237 repeat surgeries. A biological-age score must be stable short-term and move with real perturbations (Apple's pregnancy jump) |
| **Seeds read as confirmation** | Seeds show training stability, not sampling uncertainty (the report already says so); intervals come from people |

---

## 6. Minimal experiment for tomorrow (one A100, existing code)

**Goal:** decide which way to go with four cheap tests, each with a pre-stated pass rule.

**Where:** run on the cluster, where the full prepared VitalDB windows and run folders live. This
laptop has only 30 smoke-test people and overheats under load.

| Step | What | Code | Time |
|---|---|---|---|
| 1 | **Per-window embeddings** of all VitalDB windows from the frozen r3 encoder, with `start_s` | `fm_baselines` embedding with the KronLSTM and `per_subject=240` (all windows), or a 30-line loop over `encoder.embedding(encoder.encode(...))` | about 10–20 min GPU (about 1.2M windows; forward only, about 5–10k windows/s); about 600 MB float16 |
| 2 | **Nuisance-adjusted ASA test.** Re-score the existing `r3-traits-probe` and `r2b-traits-probe` ASA ≥ 3 and `healthy` predictions with Δ over age + sex + BMI + mean heart rate + eGFR + `emop` + department, and separately over age + sex + BMI only | `predictions.parquet` + `subjects.parquet` + `cases.csv.gz` + `traits._beyond_age` / `fit_logistic` extended to more covariates | CPU, minutes |
| 3 | **Is health along the age axis?** On per-person mean embeddings (train people), fit ridge probes for age and for ASA ≥ 3, CKD, hypertension and diabetes. Split each health probe into its projection on the (whitened) age direction and the orthogonal rest; score each part's Δ over age on test. Report the cosine between probes | numpy and sklearn-style ridge on the step 1 output | CPU, minutes |
| 4 | **Normative deviation score** (Approach 1) from healthy training people: Mahalanobis distance and likelihood-age. Δ over age + sex + BMI + heart rate for ASA, CKD, hypertension, diabetes, LOS (within optype) and ICU; death out-of-fold | CPU on the step 1 output | under 30 min |
| 5 | **Phase features** (Approach 3, light): per-person mean embedding and heart rate in pre-incision, post-incision and post-emergence phases from `start_s` vs `opstart`/`aneend`; the response vector Δz. Probe for diabetes and hypertension, Δ over age + sex + BMI + baseline heart rate; event-time permutation control | CPU on the step 1 output | under 1 h |
| 6 (optional GPU) | **Family-held-out multi-task pilot.** Two runs of `kron-ppg traits` from the r3 encoder, fine-tuned (not frozen), 8,000 steps: (a) `--traits age,sbp,dbp,rmssd_ms` (hemodynamics only); (b) the same traits with each person's labels shuffled within age-sex strata (needs a 10-line flag). Embed both (step 1) and probe the *held-out* family (CKD, eGFR, albumin, hemoglobin, ASA, hypertension, diabetes) against the r3 SSL encoder; also GalaxyPPG mean predicted age | existing CLI plus one flag | about 2 × 20–25 min training + 2 × 15 min embedding |

**Total:** about 1 h of GPU (about 2 h with step 6) and 2–3 h of wall time including the CPU analyses.

**Pass rules, written before running:**
- **Step 2.**
  - If ASA's Δ stays ≥ +0.05 with its interval clear of zero after BMI, heart rate and surgery type,
    the trait-head evidence is real and Approaches 2 and 4 are worth building.
  - If it falls to about 0, the "first outcome evidence" was BMI, kidney or heart rate. Say so in the
    report.
- **Step 3.**
  - If more than half of each health probe's Δ comes from the orthogonal part, the "aging clock"
    framing is wrong for this signal. Report health scores, not a biological age.
  - If the age-parallel part carries it, a better healthy-reference clock (Apple's design) is worth
    another try.
- **Step 4.**
  - A Δ clear of zero for any outcome after nuisance covariates, with within-session ICC ≥ 0.7, makes
    the normative score the candidate "unusual for your age" readout.
- **Step 5.**
  - Any diabetes Δ clear of zero beats everything static so far.
  - The permutation control must sit at 0.
- **Step 6.**
  - The multi-task encoder "learned biology" only if it beats the shuffled-label twin on held-out
    families and is no worse on GalaxyPPG.

---

## 7. What would transfer to a Galaxy Watch and a self-collected cohort

**Does not transfer.**
- Any VitalDB-trained head: finger, anaesthesia, one monitor, Korean surgical patients.
- The evidence is already in the benchmark: GalaxyPPG people aged 20–29 get mean predicted ages of
  15–58 y across models, and fine-tuning made the KronLSTM's wrist age worse (43.5 against 33.6).
- VitalDB's outcome associations, even if steps 2–5 pass.

**Transfers as method.**
1. **Normative modelling on the user's own cohort.** Fit "healthy-for-age" from the cohort's own wrist
   embeddings by age decade and sex, then report a deviation percentile.
   - It needs a reference of hundreds across ages. A self-collected cohort of tens supports only
     within-person trends.
   - A public wrist encoder (Pulse-PPG, MIT) plus GalaxyPPG for domain pretraining are the starting
     points.
2. **Within-person dynamics with everyday perturbations.** These are the watch's version of incision and
   emergence, and suit a streaming encoder with a carried state:
   - orthostatic response (a standardised sit-to-stand in the app);
   - heart-rate recovery after exertion;
   - nocturnal heart-rate and pulse-shape dipping and sleep onset;
   - paced-breathing response.

   The protocol can be standardised (fixed posture, time of day, repeated spot checks, as in
   `ppg-age.md`'s OPPO protocol). These responses have known age and autonomic dependence.
3. **Anchors measurable at home instead of hard outcomes:**
   - home cuff BP (window-level, within person: the same delta-target design as section 3.3);
   - resting heart rate and HRV;
   - optional PWV or labs (eGFR, HbA1c, lipids) at enrolment.

   Validation is then: (a) cross-sectional Δ over age + sex + BMI for these anchors; (b) **within-person
   movement under known perturbations** (illness or fever episodes, a training block, alcohol or sleep
   loss), as Apple used pregnancy and cardiac events.
4. **Reliability first.** Measure an ICC over at least 10 spot checks per quarter before reading any
   trend.
5. **Evaluation discipline.** The Δ-over-covariates protocol, family-held-out tests, and a lockbox
   apply unchanged.

**The honest product claim for now:** a within-person trend of a pulse-health deviation score, with
its error band. Not a biological age.

---

## Sources read for this proposal
- Pulse-PPG: https://arxiv.org/abs/2502.01108 (HTML full text, main body and appendix)
- PaPaGei: https://arxiv.org/abs/2410.20542 (HTML full text incl. Appendix E, Table 16)
- AnyPPG: https://arxiv.org/abs/2511.01747 (v4 HTML; Table 1, Table 3, PheWAS paragraph)
- Apple PpgAge, Miller et al. Nat Commun 2025: https://pmc.ncbi.nlm.nih.gov/articles/PMC12537950/
- AI-PPG Age: https://arxiv.org/abs/2502.12990 (HTML full text)
- OPPO smartwatch heart age: https://arxiv.org/abs/2608.01620 (abstract)
- AI-ECG biological age, EHJ Digital Health: https://academic.oup.com/ehjdh/article/6/6/1204/8272814 (search-result summary only)
- AIRE, Sau et al., Lancet Digit Health 2024;6(11):e791–e802 (search-result summary only; full text not read)
- PhenoAge (Levine) and Klemera–Doubal are named as methods only; I did not re-read them for this note.
