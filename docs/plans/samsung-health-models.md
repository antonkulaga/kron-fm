# Plan: age from what Samsung Health stores (2026-11-11)

The maintainer, 2026-11-11: plan the training of models that take standard Samsung Health data, and search for the
datasets they need. A Samsung Health export (the phone app's "Download personal data") holds no PPG waveform: the
watch turns its optical signal into summaries on the watch. What it holds, and the head each stream gets:

| Export stream | Resolution | Head | Trained on (open now / by request) |
|---|---|---|---|
| Heart rate | 1-min bins (min/mean/max) over years; ~1 s in workouts | daytime HR head: diurnal HR profile, resting HR, HR reserve | Homekit2020 (5,034 people, Fitbit minute HR; Synapse request) · All of Us Fitbit (~15k; institutional agreement) |
| RR intervals in sleep, Samsung's HRV bins | per beat (recent nights) / 30 s SDNN-RMSSD | night HRV head: RMSSD, SDNN, spectral, nightly HR minimum | Autonomic Aging (1,121 healthy, 18-92; open; resting supine, not sleep) · NSRR SHHS / MESA / MrOS / CFS / HCHS-SOL (PSG ECG in sleep; data use agreement per cohort) |
| Sleep sessions and stages | 30 s stages | sleep architecture head: duration, efficiency, stage shares, fragmentation, timing | NSRR PSG stages (same cohorts; N1+N2 = light, N3 = deep) |
| Steps | 10-min bins | activity head: hourly profile over a week, intensity distribution, timing, regularity | NHANES 2011-2014 wrist monitor (~14k, 6-80; open; MIMS, not steps: profile features made unit-free) · UK Biobank (103k, 43-79; paid application) |
| SpO2, respiratory rate | overnight 1 Hz / per bin | night oxygenation and breathing features, folded into the night head | NSRR oximetry and respiration |
| ECG app | 30 s single lead, 500 Hz | ECG head (lead I as the watch's lead) | PTB-XL (18,869 patients, 0-95; open) · CODE-15% (233k patients; open, large) |
| Skin temperature, stress | per bin | none: no large cohort with ages measures them | |

## Stages

1. **Open data, now** (download job on the cluster, then one head each): activity (NHANES), resting HRV (Autonomic
   Aging), ECG (PTB-XL). Each: features computed exactly as the export's stream allows (e.g. HRV from RR in 5-min
   segments, the activity profile from hourly sums), a ridge or gradient-boosted head and, where data allow, a small
   network; per-person grouped cross-validation, bootstrap intervals over people, a constant (median age) baseline,
   a sex-only baseline, a permutation null. The kron-ppg `wrist_age` evaluation is the template.
2. **By request** (on the desk, with links and steps): NSRR (night head, the closest match to a watch's night), then
   Homekit2020 and All of Us (daytime HR head). UK Biobank only with a fee budget.
3. **Transfer to the watch**: each head on the stream as the export gives it (Samsung's units, gaps, its own HRV
   algorithm). The RR intervals in a few recent nights and years of minute HR are the strongest personal inputs.
   Where a cohort measures two streams (MESA: PSG and actigraphy), heads are fused late and the fusion is
   validated there.
4. **Personal estimates** (private, never in the repository): each head on the person's export, with its own error
   on its own validation data, under its own name ("activity age", "night HRV age"), never as the PPG clock.

## Reference errors (published, for the bar)

Apple's wearable behaviour model on hourly HealthKit summaries: MAE 3.67 y (with PPG 2.46 y), on 162k people;
UK Biobank accelerometry 3.7 y; resting HRV 5.6 y; PSG 5.8 y; 12-lead ECG 6.9 y. Public cohorts are smaller and
narrower: expect 5-7 y per head, better fused, and judge every number against its cohort's own median.

## Not covered

Skin temperature and the stress score: no cohort with ages. Apple Heart and Movement Study data: not shared.
LifeSnaps (71 people, Fitbit, age only under/over 30) serves to test the readers, not to train.

Dataset survey and access steps: kept beside the trainer's brief (they name contacts); summarised in the trainer's
status file.

## First results (2026-11-11; scripts in `scripts/kron_health/`)

Per person, held-out people (10-fold CV, or PTB-XL's patient-wise test fold), 95 % intervals from bootstrap over
people, against the training folds' median age; shuffled ages as the null.

| Head | Cohort | People, ages | MAE (y) | Median age | r |
|---|---|---|---|---|---|
| Activity (hourly rhythm, unit-free), boosting | NHANES 2011-2014 wrist monitor | 8,924, 18-80 | 11.88 (11.71-12.05) | 15.82 | 0.61 |
| Activity, with the daily total (device units) | same | | 11.41 | 15.82 | 0.65 |
| ECG, lead I 10 s, 1D ResNet | PTB-XL test fold | 1,861 patients, 18-89 | 9.37 (9.03-9.71) | 13.93 | 0.72 |
| ECG, CODE-15% + PTB-XL folds 1-8, strips band-passed 0.5-40 Hz | PTB-XL test fold / CODE-15% test patients | 1,861 / 8,997 | **9.03 (8.72-9.35)** / 9.33 (9.17-9.49) | 13.93 / 17.07 | 0.75 / 0.79 |
| Resting HRV (5-min features), boosting with sex | Autonomic Aging | 1,093, 18-92 (age groups) | 8.70 (8.20-9.20) | 9.66 | 0.54 |

Checks: activity's clock alignment (the least active 5 h falls at night for 96 %); HRV's beat detection (two ECG
leads agree: HR to 0.00 bpm median, RMSSD to 0.9 ms median, over 617 records). The HRV head barely beats the median
(ridge: -1.09 to -0.05 y), in a cohort that is 64 % under 30 and at supine rest, not asleep: the night head waits on
NSRR. The ECG head trained on one cohort loses ~2.5 y on the other (CODE-15% only: 9.16 on its own test patients, 11.82 on
PTB-XL's); band-passing the strips closes most of it (11.82 -> 9.67), and training on both cohorts gives the best
head (9.03 on PTB-XL). Band-passing is also what a watch needs: on one person, a watch SDK's raw ECG read 14.6 y
(unfiltered PTB-XL head) and 37.4 y (unfiltered combined head) above the same person's ECG-app recordings; with the
filtered combined head the gap is 6.6 y, of which 2.6 y is the time between the two sets of recordings (4 SDK
strips, a sanity check on one person, not a validation). All heads are far from the published bars (3.7, 6.9, 5.6 y); the next data are minute-level activity
(UK Biobank), CODE-15% for the ECG, and the NSRR nights.

**On a watch's data (2026-11-11, two exports, private):** the NHANES activity head does not transfer to step counts.
Step counts are zero in most hours where the monitor's activity never is, so quiet hours and day-to-day variation
fall at the 99th-100th percentile of the monitor's and the head gives adult "ages" of 11 and 20; `kron-health
estimate` now refuses an input outside a head's training range. The activity head needs a cohort with step data
and ages (Homekit2020, All of Us). The resting-HRV head reads sleep RR young (sleep raises HRV over supine rest)
and is not read as a person's age. The ECG head is the one summary head to read on a watch, filtered.

