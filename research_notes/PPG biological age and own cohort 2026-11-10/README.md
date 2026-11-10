# PPG biological age, multi-task training and our own cohort (2026-11-10)

These notes answer three questions about kron-ppg:
- How can we separate biological from chronological age on top of kron-ppg?
- Can multi-task training force the encoder to learn biology?
- What would our own finger-plus-watch cohort cost?

Two independent design answers were written from the same repository state:
- a Claude subagent, [claude-subagent-biological-age.md](claude-subagent-biological-age.md);
- Codex `gpt-6.1-sol` at `xhigh` reasoning, [codex-gpt-6.1-sol-biological-age.md](codex-gpt-6.1-sol-biological-age.md), from [codex-prompt.md](codex-prompt.md).

The device and budget research is in [own-cohort-devices-and-cost.md](own-cohort-devices-and-cost.md).

## Where the two design answers agree

- **Why the age gap fails while the trait head works.**
  - The age head squeezes the embedding to one number, so health directions that do not help predict age are discarded.
  - An age regressor trained on a mixed cohort is rewarded for correcting disease away.
  - At an MAE of 6.5 y the gap is mostly error: an effect of Apple's size would give an AUC of only about 0.57 here.
- **The ASA gain needs a harder test.**
  - It has been tested only against linear age.
  - ASA is a clinician's grade that also encodes BMI, kidney function and emergency status.
  - Re-test it against age splines + sex + BMI + heart rate + eGFR/creatinine + procedure.
- **Promising directions for a streaming model:**
  - an outcome-anchored health axis, possibly expressed in years afterwards;
  - an age-conditioned "how unusual is this pulse for your age" score;
  - within-session responses (induction, incision, emergence, vasopressors).
- **Last resort:** removing age adversarially from the embedding.
- **Expressing a score in years** adds no evidence of its own.

## Where they differ, and the call

- **Codex's suggested first experiment does not work as written.** It compares a probe on the raw embedding with a probe on an age-residualised embedding, with age in both models. With linear heads those two fits are identical (Frisch–Waugh–Lovell), as the subagent points out. Drop it, and keep the stronger-confounder re-test of ASA.
- **Ranking.**
  - The subagent puts the deviation score first; Codex puts the outcome-anchored score first.
  - They are cheap on the same frozen embeddings, so run both.
- **The test split is no longer fresh.** It has been read in about eight runs. The subagent proposes a lockbox: the 237 people with a second surgery, which no run has used. Keep it untouched until a method is fixed.

## Multi-task training

An existing pilot (the r0 foundation-model benchmark) argues against naive multi-task fine-tuning:
- Fine-tuning on six subject-level traits left the held-out targets unchanged (CKD +0.153 vs +0.156).
- It made the wrist age worse: GalaxyPPG's 20–29-year-olds went from 33.6 to 43.5 y.

The more promising design:
- Take gradients from window-level physiology (blood pressure, HRV; cardiac output where present).
- Use within-person change targets, which cancel stable shortcuts.
- Give the head heart rate as an input.
- Put adversaries on device and surgery type, not on age.
- Judge it by held-out target families, against a compute-matched control and a shuffled-label control.

## Combined plan for the next experiment (about 1–2 GPU-hours)

1. Embed every window with the frozen run 3 (or run 4) encoder.
2. Re-test the ASA and `healthy` gains against the stronger confounders, with paired bootstrap intervals, log loss and calibration.
3. Fit the age-conditioned deviation score and the outcome-anchored score. Score rare outcomes out of fold.
4. Compare embeddings around incision and emergence, with a shuffled-time control.
5. Leave the 237-person lockbox untouched.
6. Optionally, the held-out-family multi-task pilot with its shuffled-label twin.

Nothing trained on VitalDB carries over to the watch. The method does: a deviation score refit on our own cohort, everyday perturbations (standing up, recovery after exertion, night-time dipping), and home cuff readings as within-person anchors.

## Did Pulse-PPG, PaPaGei and AnyPPG separate biological from chronological age?

No, none of the three reports an age gap or its link to outcomes.
- **Pulse-PPG** has no age task.
- **PaPaGei** reports age only as a demographic probe (Appendix E).
- **AnyPPG** reports age MAE 6.08 and a phenome-wide comparison against age, sex and race, not PPG added to them.

The PPG ages with outcome evidence all train on healthy people and adjust for age in the outcome model:
- **Apple PpgAge:** hazard ratio 1.46 per 6 y of gap.
- **AI-PPG Age:** hazard ratio 2.37.
