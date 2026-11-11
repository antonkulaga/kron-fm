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
