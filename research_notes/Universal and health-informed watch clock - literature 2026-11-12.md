# A watch clock that holds across devices and reads health: literature and data (2026-11-12)

The question (from the maintainer): our wrist clock reads 6.7 y on an unseen cohort, but each watch reads at its own
offset of 10-25 y, and its age gap carries almost no recorded disease. What does recent work do about both, and
which datasets pair watch PPG with blood or methylation? Collected by a search agent and read for this note; items
marked (unverified) were not confirmed on the source.

## Health-informed targets

- **PpgAge** (Apple Heart and Movement Study, Nat Commun 2025, https://doi.org/10.1038/s41467-025-64275-4): 213,593
  people; a self-supervised encoder, then a ridge head on 30-day mean embeddings fitted on **5,355 healthy people
  only** (no disease, no medication, never smoked); gap residualised on age with a spline. MAE 2.4 y (healthy),
  3.1-3.3 y (all). Per 6 y of gap: incident ASCVD HR 1.46, hypertension HR 1.62. No mortality; no per-watch
  calibration described.
- **AI-PPG age** (UK Biobank finger PPG, n 212,231, https://arxiv.org/abs/2502.12990): MACCE HR 2.37 for a gap over
  9 y; mortality HR (unverified). An independent 2026 benchmark (https://arxiv.org/abs/2603.14030) finds it collapses
  out of domain (predictions 38-67 y on surgical patients), where Pulse-PPG + a linear probe gives 9.3 y.
- **Smartwatch PPG heart age taught by ECG** (OPPO watches, 581,804 people, https://arxiv.org/abs/2608.01620): MAE
  4.3-5.9 y; hypertension OR 1.72 per SD of gap; gap vs pulse-wave velocity partial r 0.26.
- **ECG age and mortality**: Lima et al. 2021 (https://doi.org/10.1038/s41467-021-25351-7), gap > 8 y HR 1.79;
  a 2026 meta-analysis pools HR 1.83 (1.45-2.32) (https://esc365.escardio.org/journal/90109).
- **Outcome-trained instead of age-trained**: AIRE (Lancet Digit Health 2024, https://doi.org/10.1016/S2589-7500(24)00172-9),
  a deep ECG model with a discrete-time survival head, mortality C-index 0.64-0.78 across cohorts.
- **A blood-composite target**: RetiPhenoAge (Age Ageing 2022, https://pubmed.ncbi.nlm.nih.gov/35363255/) trained
  retinal photographs to predict blood PhenoAge; its gap predicts mortality (HR 1.92) independent of measured
  PhenoAge. No PPG or ECG model trained on a PhenoAge target was found: an open niche.

## Device invariance

- Foundation models across devices: PaPaGei (https://arxiv.org/abs/2410.20542, open), **Pulse-PPG**
  (https://arxiv.org/abs/2502.01108, wrist field data, weights on Zenodo), Samsung's HiMAE on Galaxy Watch PPG
  (https://arxiv.org/abs/2510.25785, weights not released).
- Removing the device: a gradient-reversal device classifier; or ComBat-style harmonisation of embeddings fitted
  inside folds with age and sex protected (brain-age practice, https://pmc.ncbi.nlm.nih.gov/articles/PMC10805868/);
  success is "the device can no longer be predicted". No published PPG age clock does this yet.
- Calibrating on a new watch: no published recipe. PpgAge pools watch models. The practical one is a small set of
  people of known age on the new watch, a linear map fitted there, and the gap judged within that watch.

## Watch PPG with blood or methylation in the same people

| Dataset | PPG | Blood / omics | n | Access |
|---|---|---|---|---|
| BIG IDEAs Glycemic | Empatica E4 wrist, raw 64 Hz | HbA1c, glucose, lipids, CGM | 16 | open, https://physionet.org/content/big-ideas-glycemic-wearable/ |
| AI-READI | Garmin, summaries only | HbA1c, diabetes staging | ~2,280 | DUA, https://docs.aireadi.org/ |
| All of Us | Fitbit, minute HR only | EHR labs, genomes | >15k | workbench, https://support.researchallofus.org/hc/en-us/articles/20281023493908 |
| Human Phenotype Project | WatchPAT (finger probe), raw | blood, CGM, genetics, transcriptomics | ~7k | application, https://pubmed.ncbi.nlm.nih.gov/40665053/ |
| UK Biobank | finger pulse wave | biochemistry (PhenoAge), proteomics, methylation from 2026-27 | ~200k | application + fee, https://community.ukbiobank.ac.uk/hc/en-gb/articles/34007463768221 |
| MESA | PSG (pleth unverified), wrist actigraphy | monocyte methylation (450K) | 2,237 | NSRR + dbGaP, https://sleepdata.org/datasets/mesa |

No open dataset has raw wrist PPG with methylation; with blood labs, only BIG IDEAs (16 people).

## What follows for this repository

1. Now, on our data: a healthy-only head (as PpgAge), the gap residualised per dataset, judged leave-one-dataset-out;
   a device-removal step (gradient reversal or in-fold harmonisation) measured by whether the dataset can still be
   predicted; Pulse-PPG as a second backbone against ours.
2. With UK Biobank: a head trained to predict blood PhenoAge from PPG (the open niche), and a survival head on
   mortality; finger PPG, so its transfer to the wrist must be tested.
3. On the watch: a small set of people of known age per watch model to calibrate.
