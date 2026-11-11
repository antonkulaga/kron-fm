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
