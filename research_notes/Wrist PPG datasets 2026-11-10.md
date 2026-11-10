# Wrist and smartwatch PPG datasets (2026-11-10)

Verified on 2026-11-10 from the hosting pages and APIs unless marked *(notes)* (from earlier notes, not re-checked)
or *(search)* (from a search snippet, not opened). "Registry name" = `uv run kron-data download <name>` (see `src/kron_ppg/watch_sources.py`); each lands in
`data/input/ppg/<name>/` with a `SOURCE.json`. Readers are in `src/kron_ppg/watch_readers.py`.

Verdicts: **P** = pretraining (raw wrist volume), **B** = bridging (wrist with finger or ECG on the same seconds),
**A** = age (age per person over a wide range).

## Open, registered, downloaded or downloading

| Registry name | Dataset / link | Device, site | Wavelengths, rate | People, ages | Hours | Labels | Licence | Access | Size | P / B / A |
|---|---|---|---|---|---|---|---|---|---|---|
| `galaxy_active2` | Baigutanova et al. 2025, Sci Data, [figshare 10.6084/m9.figshare.28509740](https://doi.org/10.6084/m9.figshare.28509740) | **Samsung Galaxy Watch Active 2**, wrist | green, **10 Hz** (100 ms steps; verified in `ppg.csv.gz`) | 49, 21-43 (mean 28); age, sex, ISI/PHQ-9/GAD-7/MEQ in `survey.csv` | ~33,600 (4 weeks each, free-living) | sleep diaries, surveys, watch HR, HRV features, acc/gyro/light | CC BY 4.0 | open | 19.8 GB | **P: top** (largest raw wrist, Samsung); B: no; A: narrow |
| `vilnius_af` | Bacevičius, Pluščiauskaitė et al., [Zenodo 23921320](https://zenodo.org/records/23921320) (version 2026-11-03) | research wristband (KTU), wrist | green, 100 Hz; Holter ECG 500 Hz; acc | 39 patients with suspected AF, **aged 37-79 (median 62)** | 261 days ≈ 6,260 h recorded (much of daytime is motion artefact, per authors) | AF episodes (reviewed), age, sex, height, weight, comorbidities, medication | CC BY-NC-SA 4.0 | open | 62.7 GB (PPG ~36 GB) | **P: high**; **B: wrist + ECG**; **A: best open lead (37-79), but 39 cardiac patients** |
| `multisite_ppg` | [HF snowballlab/Multisite-PPG](https://huggingface.co/datasets/snowballlab/Multisite-PPG) | earring, ring, watch, necklace | green + IR, 100 Hz; Polar H10 ECG | 20, 20-51 (18 women; 2 over 45); skin type | ~27 h span per person per device (sessions over 2 days) | activity logs | CC BY-NC 4.0 | open (ungated) | raw_data ~10 GB (repo 16.9 GB incl. windowed copy, not taken) | P: mid; **B: ring (finger) + watch + ECG**; A: narrow |
| `wildppg` | Meier, Demirel & Holz, NeurIPS 2024, [siplab](https://siplab.org/projects/WildPPG) (polybox share) | forehead, sternum, wrist, ankle | green 530 / red 660 / IR 950 nm, 128 Hz (verified); acc; lead-I ECG | 16, no ages in the files | ~12 h per person at the wrist (5.68 M samples at 128 Hz for an0); ~216 total *(notes)* | ECG HR, activity, per-file notes on excluded spans | CC BY-NC-SA 4.0 | open | 19.6 GB (16 .mat) | **P: high** (long free-living); B: wrist + ECG; A: narrow |
| `wf_ppg` | WF-PPG, Sci Data 2025, [figshare 27011998](https://doi.org/10.6084/m9.figshare.27011998) | wrist + fingertip at once | wrist G/IR/R raw, finger G/IR/R band-passed; **~100 Hz with jittered timestamps** (not 128); ECG, SpO2, BP | 27, 20-32 | short sessions | contact pressure, BP, SpO2 | CC BY 4.0 | open | 384 MB | P: low; **B: top (wrist + finger same seconds)**; A: narrow |
| `utsa_ppg` | Xu et al., IEEE ICHI 2025, [GitHub](https://github.com/utsanpb2-234/UTSA-PPG) (Google Drive zip) | wrist green + fingertip red/IR | 100 Hz (filtered by the authors); acc; Holter RR | 12, no ages | ~40 min × 3 scenes each (sit, sleep, office), ~24 h total | ECG RR/HR, Fitbit HR, O2Ring | MIT (repo licence; no separate data licence) | open | 496 MB | P: low; **B: wrist + finger, incl. sleep**; A: unknown |
| `ptt_ppg` | Mehrgardt et al., [PhysioNet pulse-transit-time-ppg 1.1.0](https://physionet.org/content/pulse-transit-time-ppg/1.1.0/) | two finger sites (proximal/distal phalanx) | green, red, IR, 500 Hz; ECG | 22 | 66 short records | BP, SpO2, activity | ODbL 1.0 | open | 2.9 GB | P: no (finger); B: finger multi-wavelength + ECG; A: narrow |
| `ppg_dalia` | Reiss et al. 2019, [UCI 495](https://archive.ics.uci.edu/dataset/495/ppg+dalia) | Empatica E4 wrist + chest ECG | BVP 64 Hz | 15, **21-55** | ~37 (≈2.5 h each) | activities, ECG HR | CC BY 4.0 | open | ~2.7 GB | P: mid; B: wrist + ECG; A: narrow (20s-30s) — **pickles never unpickled** |
| `wesad` | Schmidt et al. 2018, [UCI 465](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection) (authors' sciebo link) | Empatica E4 wrist + RespiBAN chest | BVP 64 Hz; ECG 700 Hz | 15, **24-35** | ~1.5 h each | stress / affect | none stated in the archive readme (asks for citation); open via UCI | open | 2.25 GB | P: low; B: wrist + ECG; A: narrow — **pickles never unpickled** |
| `wrist_exercise` | Jarchi & Casson 2016, [PhysioNet wrist 1.0.0](https://physionet.org/content/wrist/1.0.0/) | wrist + chest ECG | 256 Hz | 8, 22-32 | ~1 | exercise type | ODC-By 1.0 | open | 50 MB | P: low; B: wrist + ECG; A: no |
| `ppg_sport` | PPG-Sport, IMWUT 10(2) 2026, [GitHub LaserHu/PPG-Sport](https://github.com/LaserHu/PPG-Sport) | both wrists + ECG | 64 Hz, processed by the authors; IMU | 30, **21-34** | ~48 (8 min per activity per wrist) | activities incl. badminton, table tennis, basketball; ECG HR | CC BY-NC-ND 4.0 *(search: publication record)*; GitHub states none | open | 0.4 GB | P: low; B: two wrists + ECG; A: to read |
| `galaxyppg` (existing) | Park, Zheng & Lee 2025, Zenodo 14635823 | Galaxy Watch5 + E4 + Polar H10 | green 25 Hz | 24, 20-29 | short protocol | ECG | CC BY 4.0 | open | 482 MB | device check; B: watch + ECG |

## Not downloaded: access needs an account, a token or an agreement

| Dataset | Link | What is needed | Why it is worth it |
|---|---|---|---|
| DeepBeat | Synapse syn21985690 | a Synapse login (anonymous access sees the project but no files) | wrist PPG, AF labels, large (≈500k windows per the paper) |
| PulseWatch (UMass) | [Synapse syn23565056](https://www.synapse.org/Synapse:syn23565056), folder `Pulsewatch2025` (syn64402376) | only a **valid Synapse login**: restriction level OPEN, no access requirement (verified anonymously 2026-11-10), but anonymous users cannot download | Samsung Gear S3 wrist PPG + acc at 50 Hz with ECG patch, **106 post-stroke patients, 14 days each** *(search)*; an earlier subset of 37 arrhythmia patients aged 50-91 *(search)*: **the wide-age Samsung set, thousands of hours** |
| DREAMT | [PhysioNet dreamt 2.2.0](https://physionet.org/content/dreamt/2.2.0/) | PhysioNet credentialed access + Restricted Health Data Use Agreement 1.5.0 (verified) | E4 BVP overnight in a sleep clinic, older and wider ages than lab sets |
| HEART-Watch | [arXiv 2512.03988](https://arxiv.org/abs/2512.03988) | data use agreement / on request *(search, brief)* | Pixel Watch PPG + ECG, 40 adults aged 19-75 *(notes)*: wide age |
| Aurora-BP (Microsoft) | *(notes)* | DUA; also forbids joining with other data | wrist PPG + BP, ages 18-80+ |
| Mol et al. 2020 (Radboud) | *(notes)* | probably a DUA | wrist + finger, 31 people over 70 |

## Seen, not taken
- **E4SelfLearning** (JMIR mHealth 2024): an aggregation of eleven open E4 sets (161 people) with a pipeline *(search)*; check which sources are not already here.
- **PerHeart pilot** (Zenodo 17459937): older heart-failure patients, but the wrist sensor is inertial only, no PPG.
- **IEEEPPG / TROIKA**: wrist PPG during running, 12 people 18-35; small, superseded by DaLiA.
- **Apple Heart and Movement**, **UK Biobank** (finger PPG): not open.

## Readers (kron-ppg prepare <name>, 25 Hz, 30 s windows)
| Name | Channel | Polarity | Recording unit | Windows (real data) |
|---|---|---|---|---|
| `wf_ppg` | wrist green raw, regridded to 100 Hz | inverted | subject/session | 1,185 from 27 people |
| `multisite_ppg` | watch green, 100 Hz | inverted | subject/watch.chunk (≤ 6 h) | 7,819 from 20 people |
| `galaxy_active2` | wrist green, 10 Hz (needs `python -m kron_ppg.watch_readers unpack galaxy_active2`) | inverted | device/UTC day | **303,911 from 49 people** (≈2,530 h of 30 s windows; 682.6 M samples ≈ 19,000 h unpacked) |
| `vilnius_af` | wrist green, 100 Hz | inverted | patient/segment.chunk (≤ 6 h) | **134,649 from 35 patients** (4 have no PPG) |
| `utsa_ppg` | wrist green, 100 Hz | inverted | subject/scene | 1,939 from 12 people |
| `ppg_dalia` | E4 BVP, 64 Hz | upright already | subject | 2,027 from 15 people |
| `wesad` | E4 BVP, 64 Hz | upright already | subject | 1,395 from 15 people |
| `wildppg` | wrist green, 128 Hz | upright already | subject/chunk (≤ 6 h) | 5,376 from 16 people |
| `ppg_sport` | each wrist, 64 Hz | inverted | subject/wrist/activity | 3,512 from 30 people |
**Outliers:** the long free-living sets carry windows the quality gate passes but robust scaling blows up
(a spike in a quiet window): non-finite after the float16 cast in vilnius_af 101, galaxy_active2 33; |x| > 50 in
vilnius_af 2,565, galaxy_active2 17,426, multisite_ppg 225, wildppg 243 (VitalDB: 0 non-finite). Drop them before
training; the gate fix belongs in `signal.py` (not changed here).

Not read: `ptt_ppg` (finger, not wrist) and `wrist_exercise` (WFDB, 8 people; small). Windows use the default cap
of 240 per recording; `--max-windows` takes more.

Polarity was measured, not assumed: median over quality-gated windows of log(upper / lower 2 % tail of the
derivative); positive = fast rise = systole up. It reads positive on BIDMC, GalaxyPPG and WF-PPG's own band-passed
channel (known upright); negative on the raw wrist green of WF-PPG, Multisite, Galaxy Active 2, Vilnius, UTSA and
PPG-Sport (the readers invert them); positive on WildPPG's green and the E4 BVP (read as they are).

## Ranked shortlist (2026-11-10)
1. `galaxy_active2`: the largest open raw wrist PPG, and Samsung: the pretraining core.
2. `vilnius_af`: thousands of wrist hours with ECG, patients with age and comorbidities: the best open lead on priority 3.
3. `wildppg`: long free-living multi-site with ECG.
4. `wf_ppg`, `utsa_ppg`, `multisite_ppg`: wrist with finger (or ring) on the same seconds, for bridging.
5. `ppg_dalia`, `wesad`, `wrist_exercise`, `ptt_ppg`: small, labelled, for checks.
Behind an account or agreement, by value: **PulseWatch** (a Synapse login is enough: OPEN, no agreement) and DeepBeat, DREAMT, HEART-Watch.
