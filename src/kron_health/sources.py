"""Open cohorts with ages for the summary-signal heads, downloadable by ``uv run kron-data download <name>``."""
from __future__ import annotations

from kron_core.download import RemoteFile, Source

DOMAIN = "health"
PHYSIONET = "https://physionet.org/static/published-projects"
NHANES = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public"

SOURCES: dict[str, Source] = {
    s.name: s for s in (
        Source("autonomic_aging", DOMAIN,
               "Autonomic Aging: resting ECG (1000 Hz) and continuous blood pressure, 8-45 min supine, from 1,121 "
               "healthy volunteers aged 18-92 (age in 15 groups): the resting HRV head",
               (RemoteFile(f"{PHYSIONET}/autonomic-aging-cardiovascular/autonomic-aging-a-dataset-to-quantify-changes-"
                           "of-cardiovascular-autonomic-function-during-healthy-aging-1.0.0.zip",
                           "autonomic-aging-1.0.0.zip", 5_578_641_955),),
               "ODbL 1.0 (PhysioNet open access)",
               "Schumann A, Baer KJ (2022) Autonomic aging: a dataset to quantify changes of cardiovascular autonomic "
               "function during healthy aging. Sci Data 9:95. PhysioNet, doi 10.13026/2hsy-t491.",
               "https://physionet.org/content/autonomic-aging-cardiovascular/1.0.0/",
               {"format": "zip of WFDB", "fs": 1000, "age": "15 groups (18-19 ... 85-92)", "approx_mb": 5_579}),
        Source("ptb_xl", DOMAIN,
               "PTB-XL: 21,799 10 s 12-lead ECGs (500 Hz) from 18,869 patients aged 0-95 with age and sex: the ECG "
               "head (lead I stands in for a watch's single lead)",
               (RemoteFile(f"{PHYSIONET}/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip",
                           "ptb-xl-1.0.3.zip", 1_839_504_686),),
               "CC BY 4.0",
               "Wagner P et al. (2020) PTB-XL, a large publicly available electrocardiography dataset. Sci Data 7:154. "
               "PhysioNet, doi 10.13026/kfzx-aw45.",
               "https://physionet.org/content/ptb-xl/1.0.3/",
               {"format": "zip of WFDB", "fs": 500, "approx_mb": 1_840}),
        Source("code15", DOMAIN,
               "CODE-15%: 345,779 12-lead ECGs (10 s, 400 Hz, HDF5) from 233,770 Brazilian primary-care patients with "
               "age and sex: the ECG head at scale (lead I as the watch's lead)",
               tuple(RemoteFile(f"https://zenodo.org/records/4916206/files/{name}", name, size) for name, size in (
                   ("exams.csv", 35500900),
                   ("exams_part0.zip", 2672786404),
                   ("exams_part1.zip", 2668177892),
                   ("exams_part10.zip", 2672404916),
                   ("exams_part11.zip", 2665653597),
                   ("exams_part12.zip", 2674194718),
                   ("exams_part13.zip", 2667409911),
                   ("exams_part14.zip", 2671739754),
                   ("exams_part15.zip", 2695359122),
                   ("exams_part16.zip", 2725249232),
                   ("exams_part17.zip", 782633664),
                   ("exams_part2.zip", 2675394813),
                   ("exams_part3.zip", 2676107654),
                   ("exams_part4.zip", 2673870870),
                   ("exams_part5.zip", 2673013182),
                   ("exams_part6.zip", 2667377774),
                   ("exams_part7.zip", 2671653394),
                   ("exams_part8.zip", 2666112083),
                   ("exams_part9.zip", 2668370881),
               )),
               "CC BY 4.0",
               "Ribeiro AH et al. (2021) CODE-15%: a large scale annotated dataset of 12-lead ECGs. Zenodo, "
               "doi 10.5281/zenodo.4916206. Lima EM et al. (2021) Nat Commun 12:5117.",
               "https://zenodo.org/records/4916206",
               {"format": "zip of HDF5", "fs": 400, "approx_mb": 46_300}),
        Source("nhanes_pam", DOMAIN,
               "NHANES 2011-2014 wrist activity monitor (ActiGraph GT3X+, 7 days): hourly and daily MIMS summaries with "
               "wear and sleep-wake flags, and demographics (age top-coded at 80) for ~14,000 people aged 6-80: the "
               "activity head (a watch's step bins summed to hours)",
               tuple(RemoteFile(f"{NHANES}/{year}/DataFiles/{name}.xpt", f"{name}.xpt", size) for year, name, size in (
                   (2011, "DEMO_G", 3_753_760), (2013, "DEMO_H", 3_833_200),
                   (2011, "PAXHD_G", 385_280), (2013, "PAXHD_H", None),
                   (2011, "PAXDAY_G", 6_486_720), (2013, "PAXDAY_H", 7_318_800),
                   (2011, "PAXHR_G", 128_186_160), (2013, "PAXHR_H", 144_753_200))),
               "public domain (US federal government work, NCHS)",
               "CDC/NCHS National Health and Nutrition Examination Survey 2011-2012 and 2013-2014, Physical Activity "
               "Monitor files.",
               "https://wwwn.cdc.gov/nchs/nhanes/",
               {"format": "SAS transport (xpt)", "approx_mb": 296}),
    )
}
