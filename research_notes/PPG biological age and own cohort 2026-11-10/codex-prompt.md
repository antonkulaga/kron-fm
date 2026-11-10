You are advising on a research design. Read the code and results in this repository (kron-fm), then write a concrete proposal. Do not edit any files.

## Context to read first
- `reports/PPG foundation models and aging clocks 2026.md`: the whole kron-ppg story and its results through run 3, in sections "Run 1", "Run 2", "Longer pretraining (run 3)", "Traits", "The healthy-only clock".
- `src/kron_ppg/`: the model (`model.py` or similar), `clock.py`, `traits.py`, `vitals.py`, `pretrain*`, data and splits.
- `README.md`, the kron-ppg section.

## The facts, briefly
- kron-ppg is a causal, streaming KronLSTM encoder (5.85M parameters) over raw 25 Hz finger PPG in 30 s windows. It is pretrained on next-second prediction plus a same-person contrast.
- Data: VitalDB, 5,887 surgical patients, one hospital. The splits are 4,071 train, 912 validation and 897 test people. Per person it has age, sex, height, weight, BMI, ASA class, preop hypertension, diabetes, in-hospital death (only 3 deaths in test), preop labs (Hb, albumin, glucose, creatinine/eGFR, ...), and per-window vitals (arterial/cuff BP, SpO2, ECG HR/HRV, pulse arrival time).
- An age clock on the frozen encoder reaches test MAE 6.5 y (r 0.83). But the bias-corrected age gap carries **no** outcome signal: AUC 0.49–0.56 for hypertension, diabetes and ASA>=3, in every run and seed, and also for a clock trained only on healthy people.
- A multi-task trait head on the same encoder **does** add to age. Delta AUC (age + PPG vs age) is +0.12 to +0.15 for ASA>=3 and +0.03 for "healthy", with bootstrap intervals clear of zero. Hypertension and diabetes gain about 0. Within-person systolic pressure tracking is r 0.65.
- The eventual target is a Galaxy Watch (wrist, 25 Hz, motion). Nothing trained on VitalDB transfers to wrist data yet.

## The question
How can we build something on top of kron-ppg that separates **chronological age** from **biological (physiological) age**? We want a quantity that varies between people of the same age, predicts health outcomes beyond age, and is not just a noisy age estimate. Classic gap clocks failed here.

Please give:
1. A diagnosis: why does the residual gap of an age regressor carry nothing here, while a trait head does? Be specific to this data and setup (regression dilution, what the encoder is optimized for, label noise, cohort, and so on).
2. 3–6 candidate approaches ranked by expected value. For each, give the method in enough detail to implement on this codebase: the losses, the targets, which encoder layers or embeddings, and the training data split. Also give the evaluation that would show success or failure on the data we have, and the main way it could fool us. Consider at least:
   - outcome-anchored clocks (PhenoAge/GrimAge-style: train on mortality or morbidity or a composite, then express it in years);
   - adversarial or orthogonal removal of age from the embedding (gradient reversal, conditional independence, residualizing on age splines) and then probing for health;
   - latent-variable / factor models (an age axis plus a health axis; disentangled VAE-style heads; contrastive same-age-different-health pairs);
   - within-person dynamics (the response to anesthesia induction or a vasopressor as a stress test; beat-to-beat variability and recovery) as biological-age features that a streaming recurrent model is uniquely suited for;
   - quantile or distributional age ("how unusual is this pulse for your age") rather than a point estimate.
3. Statistical pitfalls specific to "biological age" claims (age-dependence of the residual, Simpson's paradox, treatment confounding in surgical patients, ASA as a clinician's judgment, too few deaths), and how to guard against each.
4. What would transfer to a watch and a self-collected cohort, and what would not.
5. A minimal experiment we could run tomorrow on one A100 with the existing code, with expected run time.

Be concrete and critical. Cite papers when you are sure of them. Keep it under about 1,800 words.
