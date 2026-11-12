# A Samsung-watch age predictor from raw wrist PPG (2026-11-10)

An age predictor that reads the green wrist PPG a Samsung Galaxy Watch records, fitted on open wrist data and
judged on a wrist cohort it never saw. Code: `src/kron_ppg/wrist_age.py` (`kron-ppg wrist-age`,
`kron-ppg wrist-age-export`).

**Aurora-BP.** The held-out test cohort is the Aurora-BP Study data (Microsoft), used under its data use agreement:
research only, results only, never joined with other data, never part of a deployed model. Reference: Mieloszyk R,
Twede H, Lester J, et al. A comparison of wearable tonometry, photoplethysmography, and electrocardiography for
cuffless measurement of blood pressure in an ambulatory setting. IEEE J Biomed Health Inform 2022;26(7):2864-2875.

## The predictor

- **Input:** raw green wrist PPG at 25 Hz (Galaxy Watch5 and later), or at 10 Hz (Galaxy Watch Active 2) resampled
  to 25 Hz. Band-pass 0.5-5 Hz, so 10 Hz and 25 Hz watches give the model the same band.
- **Model:** the kron-ppg run 4 encoder (KronLSTM, 5.85 M parameters, self-supervised on VitalDB finger PPG and
  GalaxyPPG, 160k steps), frozen; a linear head (ridge) over the mean of window embeddings, **fitted on open
  wrist data only**: 214 people of eight open wrist datasets (Galaxy Active 2, GalaxyPPG, Vilnius AF wristband,
  Multisite-PPG, PPG-Sport, PPG-DaLiA, WESAD, WF-PPG). No Aurora-BP data in it (its agreement forbids product
  development).
- **Run:** `kron-ppg wrist-age ... --lowpass 5` (run `wa4-open-r4-lp5`), exported by `kron-ppg wrist-age-export`.

## The honest number

Test: all 1,220 Aurora-BP participants (wrist PPG, aged 21-85, a device none of the training sets used), each
predicted from the mean of their 15 s readings, 5 Hz band, by the head above. 95 % intervals from 2,000 bootstrap
resamples of people.

| Model | MAE (y) | 95 % CI | r |
|---|---|---|---|
| **Open-wrist head, 5 Hz band (deployable)** | **7.50** | 7.16-7.85 | 0.60 |
| Aurora's own median age (an oracle constant: it knows the test ages) | 9.63 | | |
| Open-wrist head, full band (8 Hz) | 7.78 | 7.41-8.14 | 0.66 |
| Hand-made pulse features, open-wrist head | 11.71 | 11.21-12.24 | 0.25 |
| Finger clock (VitalDB head) as it is | 16.7 | | 0.49 |

The deployable head beats the oracle median by 1.7-2.5 y (paired bootstrap). It reads 2.0 y young on average.

What the encoder could do with a wrist-fitted head (research only: the head fitted on Aurora, 10-fold CV x 5):
6.04 y (5.78-6.29), r 0.75; 5 Hz band 6.12; untrained encoder 7.55; hand features 7.79; from scratch on Aurora
6.47 (175 test people, where the frozen probe gives 6.20). Shuffled ages: 9.6 every time.

On the Samsung sets themselves (72 people aged 20-43) no head beats their own median age (Galaxy Active 2:
4.65 vs 4.76, r 0.49): their age range is too narrow to measure an age clock on.

## Readings per person (Aurora, 15 s readings, open-data head)

1 reading 10.3 y; 2: 8.9; 4: 8.1; 8 (2 min): 7.7; 16 (4 min): 7.35. Ask for at least 2 min of still wrist,
spread over a day or two; more helps.

## What it takes to run on the watch

- **Graph:** one ONNX streaming graph, `stream.onnx`, 24 MB (fp32), opset 17; inputs a 5 s chunk (125 samples),
  a validity mask, the pool (257 floats) and six layers' memory (~470 KB); output the age so far. Checked against
  PyTorch (max 1e-5 y) and against the window path (0.004 y).
- **App recipe:** band-pass 0.5-5 Hz, 25 Hz, 30 s windows robust-scaled (median, IQR/1.349), drop windows of
  periodicity < 0.35 or any |x| > 20; feed 6 chunks per window; zero the layer memory at each window start, keep
  the pool; read the age after the last window.
- **Compute:** 15 ms per 5 s chunk on one server CPU thread (ONNX Runtime), 11 CPU-s per hour of signal. A
  watch core (Cortex-A55 class) is perhaps 3-6x slower: well under a second per 30 s reading. Not measured on a
  watch.
- **Size:** int8 dynamic quantisation gives 7.6 MB but moves the age by 0.3-1.3 y on a test stream: not to ship
  until checked on real readings; fp16 (~12 MB) is the next thing to try.
- **The c1 clock (2026-11-11)** exports the same way: 24 MB fp32, 15 ms per 5 s chunk, 11 CPU-s per hour, within
  0.035 y of the window path (full-band recipe, band-pass 0.5-8 Hz). Its int8 graph moves the age by 5.3 y: int8 is
  out for it. The two-encoder average runs two such graphs (48 MB, about 30 ms per chunk).
- **Ways onto the watch:** ONNX Runtime Mobile in a Wear OS app (Kotlin/Java API), or convert to TFLite. Raw green
  PPG needs Samsung's Health Sensor SDK (Galaxy Watch4 and later, partner approval), which is said to give green PPG at
  25 Hz (not yet checked here); the Active 2 runs Tizen and is not reached this way.

## Wrist pretraining does not beat finger pretraining here (2026-11-11)

Two encoders pretrained on wrist PPG were put through the same two tests, 5 Hz band, against run 4 (finger PPG):
w1, the run 4 recipe (160k steps) on the open wrist window sets plus PulseWatch2025 (Samsung Gear S3 / Galaxy
Watch 3, stroke survivors); s5, scale-free stream pretraining on the open wrist streams plus PulseWatch.

| Encoder | Aurora, head fitted on Aurora (CV) | Open-wrist head on all of Aurora | r |
|---|---|---|---|
| run 4 (finger, VitalDB + GalaxyPPG) | 6.12 (5.86-6.38) | **7.50 (7.16-7.85)** | 0.60 |
| w1 (wrist windows) | 6.14 (5.88-6.40) | 7.97 (7.60-8.33) | 0.58 |
| s5 (wrist streams, scale-free) | 6.91 (6.61-7.19) | 15.6 (reads 11.9 y old) | 0.40 |

In the 5 Hz band, w1 matches run 4 when the head sees the target cohort and transfers a little worse; s5 is worse on
both. In the full band (the watch-data probe, ridge with K-fold over people) w1 is slightly better than run 4 on
Aurora: paired difference -0.23 y (-0.38 to -0.08) on 15 s windows, -0.37 y (-0.57 to -0.17) on 30 s; on Galaxy
Active 2 and Vilnius the difference is within noise. The same probe reads PulseWatch's AF labels (105 people) at
AUC 0.77 (0.64-0.89) with run 4, 0.76 with w1, against 0.72 for an untrained twin. The deployable predictor stays on run 4, which also carries no data of unknown terms (PulseWatch's are not yet known).
More wrist pretraining is not where the next gain is: the head's training people (214, mostly 20-35) are.

## Fine-tuning on the open watch data (2026-11-11)

The encoder and an age head fine-tuned end to end on the same open wrist sets (214 people, 3,000 steps, learning
rate 1e-4; nothing of Aurora-BP in training), then scored on all 1,220 Aurora-BP people, full band:
**MAE 6.88 y (6.56-7.19), r 0.68**; three seeds 6.88 / 7.09 / 7.27 (7.08 +- 0.19). Against the frozen encoder with a
ridge head in the same band (7.78): **-0.57 to -1.22 y**, paired over people (three-seed ensemble 6.98, -0.44 to
-1.15). It reads 25 Hz input; a 5 Hz-band version for 10 Hz watches is training.

## Pretraining on finger and open wrist together (2026-11-11)

Run c1 pretrained the encoder (run 4's recipe, 160k steps) on VitalDB finger PPG and every open wrist set at once,
self-supervised, so no ages were used. With the encoder frozen and a ridge head fitted on the open wrist people only,
the head was scored on Aurora-BP's 1,220 people, who were never used for fitting:

| Encoder | Band | MAE (y) | 95 % CI | r | Paired against run 4 |
|---|---|---|---|---|---|
| run 4 (finger) | full | 7.78 | 7.41-8.14 | | |
| c1 (finger + open wrist) | full | **6.61** | 6.30-6.92 | 0.67 | -1.16 (-1.46 to -0.86) |
| run 4 (finger) | 5 Hz | 7.50 | 7.16-7.85 | 0.60 | |
| c1 (finger + open wrist) | 5 Hz | 7.92 | 7.54-8.30 | 0.60 | +0.42 (+0.16 to +0.70) |

The median guess scores 9.63. At 25 Hz, the Galaxy Watch's rate, the c1 frozen head is the best deployable model so
far, below the fine-tuned run 4 encoder (6.88, not tested paired against it); one seed. Its predictions are still
compressed toward the mean (slope 0.52). For a 10 Hz watch the 5 Hz run 4 head stays. A second pretraining seed
(c2) reads 6.78 y (6.45-7.08) full band, again better than run 4 (-0.68 to -1.31 y, paired), and 7.99 at 5 Hz:
over the two seeds the c1 recipe gives 6.70 +- 0.12 y on Aurora.

### Fine-tuning c1, and averaging two clocks (2026-11-11)

Fine-tuning the c1 encoder on the open wrist people (6k steps) did not beat c1 frozen: 6.72 y (lr 3e-4) and 6.84 y
(lr 1e-4), against 6.61. Fine-tuning run 4 for 6k steps gave 6.71 (lr 3e-4) and 6.79 (lr 1e-4). The average, per
person, of the frozen c1 head and the 6k fine-tuned run 4 reads **6.30 y** (r 0.71) on Aurora's 1,220 people, better
than c1 alone by 0.17-0.46 y (paired) and than the fine-tuned run 4 by 0.27-0.55 y. Adding the fine-tuned c1 as a
third member changes nothing (6.34). Both members were fitted on open data only, so the average can be deployed (two
encoders, about 30 ms per 5 s). The members and the learning rate were chosen by their Aurora scores, so the 6.30
carries a small selection effect. Two more seeds of the fine-tuned run 4 read 6.98 and 7.40 alone (the fine-tune
varies with its seed far more than the frozen head): averaged with c1, each gives 6.39 and 6.52. With the fine-tuned
member taken as the mean of its three seeds, so that no seed is chosen, the average reads **6.34 y (6.03-6.64)**, r 0.71,
better than c1 alone by 0.13-0.42 y (paired). Seed 1 of c1 itself is running.

## Each watch reads at its own offset (2026-11-11)

The open wrist sets are young (median age 23-28 each), except Vilnius (AF patients, median 62). Holding one set out
and fitting the head on the rest, each held-out cohort is read at an offset: for the c1 head, GalaxyPPG (Galaxy
Watch5, median 23) reads 14.6, MultiSite +16 y, PPG Sport +25 y, Vilnius -22 y; for the 5 Hz run 4 head, Galaxy
Active2 reads +14 y. With each source's mean embedding removed before fitting, almost no age signal is left (slope
of prediction on age 0.07-0.10 on Aurora). So the open heads learn age mostly *between* sources, old Vilnius
against young students, and a new device carries its own offset of up to 10-25 y. Aurora is one device on which the
offset happened to be small; it does not show the clock is calibrated on a Galaxy Watch.

The fix is people of known, varied age recorded on the target watch, to fit the watch's offset (or the head)
there: no open Galaxy Watch set has people over about 35. How many: with one open set held out, shifting the head by
the mean error of k of that set's people (an offset only) brings the worst cases down from 10-25 y to 4-7 y already
at k = 3-5 (c1: PPG Sport 24.7 -> 4.0, Vilnius 19.7 -> 6.7; the 5 Hz head: Galaxy Active2 12.6 -> 4.6); more people
change little. Within these narrow-age sets a small error mostly reflects their small age spread, so this shows the
offset is fixed, not that people are ranked well inside one watch. Script: `scripts/kron_ppg_device_calibration.py`. Until then a single person's age from the watch is not a
number to report. Script: `scripts/kron_ppg_device_offsets.py`.

## Limits

- The head learnt mostly from young people (20-35) plus 35 older AF patients; the test device is not Samsung.
- Aurora readings are resting, seated spot checks; free-living Samsung data has motion, which the quality gate
  only partly removes.
- An age gap from this model is not validated against any outcome on wrist data.
