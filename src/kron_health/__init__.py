"""kron_health: age from what a smartwatch app stores (heart rate, HRV, sleep, activity, short ECG), not raw signals.

A Samsung Health export (read by :mod:`kron_ppg.samsung_health` and GlucoseDAO's samsung_health_extractor) holds
minute heart rate over years, nightly HRV and RR intervals, sleep stages, step bins, SpO2, skin temperature,
respiratory rate and 30 s ECG recordings, but no PPG waveform. Each stream gets its own head, trained on the open
or obtainable cohort that measures it with ages (the plan: ``docs/plans/samsung-health-models.md``).
"""
