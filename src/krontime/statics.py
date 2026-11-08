"""Per-person static inputs for KronLSTM (C79): the insulin delivery family, age and gender.

Anton, 2026-10-23: "do you think if we use (age, gender, insulin algorithm, device type) as static
covariates and fill it in for Livia it can improve performance?". C76 read the values these columns
take: the algorithm is recorded for nearly everyone and nearly follows the cohort in training, every
person in the annual input and the iLet pair is on the iLet with a Dexcom G6, so there only age and
gender vary, and weight and height are recorded in mixed units and change within most records, so
they are left out. Device and CGM add little beside the algorithm and are left out too.

A static model (``"static": true`` in ``--kronlstm-kwargs``) reads one vector per person in every step
of its forecast block, added through a zero-initialised projection, so a warm start forecasts exactly
as its parent until training moves it. A person with nothing recorded reads the all-zero vector, the
same as "unknown".

**Version 2** (``"static_version": 2``; asked for at the laptop on 2026-10-24: "we can add gender, age,
insulin type, sensor, weight") appends weight, the bolus insulin's speed and the sensor family to the
first version's vector, which stays its first ``STATIC_FEATURES`` entries, so version-1 models load and
read exactly as before. Weight is recorded in pounds in every study, not in mixed units: a check of every
person on 2026-10-24 found the adult studies at medians of 140 to 175 lb and PEDAP's young children at
37 lb, so it is read in pounds, turned into kilograms, and each person takes their median, since it
changes within a record. The insulin speed separates Fiasp and Lyumjev (ultra-rapid) from lispro, aspart
and glulisine (rapid) and regular human insulin; the sensor family separates the Dexcom G6 that every
scored person wears from the older Dexcoms, Medtronic's sensors and the Libre.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

#: The insulin delivery families: what the algorithm does overnight and between meals. Automated
#: delivery is split by system because each corrects on its own schedule; a pump without automation
#: (including low-glucose suspend) is one family, injections another.
ALGORITHM_FAMILIES = ("loop", "control_iq", "ilet", "medtronic_aid", "other_aid", "pump_manual", "injections")
#: Recorded algorithm names (lower case) and their family; "Basal-Bolus" and a missing name go by modality.
ALGORITHM_NAMES = {
    "loopalgorithm": "loop", "loop": "loop",
    "control-iq": "control_iq", "control iq": "control_iq",
    "ilet bionic pancreas": "ilet", "ilet": "ilet",
    "smartguard": "medtronic_aid", "780g advanced hcl": "medtronic_aid",
    "omnipod 5": "other_aid",
    "basal-iq": "pump_manual", "640g": "pump_manual",
    "mdi": "injections",
}
MODALITY_NAMES = {"sap": "pump_manual", "mdi": "injections", "aid": "other_aid"}
#: age / AGE_SCALE, so a child of 10 reads 0.25 and an adult of 40 reads 1.
AGE_SCALE = 40.0
STATIC_FEATURES = len(ALGORITHM_FAMILIES) + 4  # the family one-hot, age and its flag, female, male
STATIC_COLUMNS = ("insulin_delivery_algorithm", "insulin_delivery_modality", "age", "gender")
#: Version 2's additions: how fast the bolus insulin acts, and which family of sensor reads the glucose.
INSULIN_SPEEDS = ("ultra-rapid", "rapid", "regular")
SENSOR_FAMILIES = ("dexcom_g6", "dexcom_g4_g5", "medtronic", "libre")
#: weight in kg / WEIGHT_SCALE_KG, so 80 kg reads 1; MetaboNet records pounds.
WEIGHT_SCALE_KG = 80.0
KG_PER_POUND = 0.45359237
STATIC_FEATURES_V2 = STATIC_FEATURES + 2 + len(INSULIN_SPEEDS) + len(SENSOR_FAMILIES)
STATIC_COLUMNS_V2 = STATIC_COLUMNS + ("weight", "insulin_type_bolus", "cgm_device")
#: Recorded bolus insulin names (lower case) and their speed: every name MetaboNet, the iLet files and
#: Livia's datasets carry (C83 found no other in training or test).
INSULIN_SPEED_NAMES = {
    **dict.fromkeys(("fiasp", "fiasp (rapid aspart)", "novolog fiasp", "lyumjev"), "ultra-rapid"),
    **dict.fromkeys(("humalog", "humalog (lispro)", "novolog", "novolog (aspart)", "novalog", "novorapid",
                     "novorapid (aspart)", "apidra", "apidra (glulisine)", "apidra (glusine)", "admelog",
                     "humalog (lispro) or novolog (aspart)"), "rapid"),
    **dict.fromkeys(("novolin r", "humulin r", "regular insulin"), "regular"),
}
SENSOR_NAMES = {
    "dexcom g6": "dexcom_g6", "dexcom one": "dexcom_g6", "dexcom g4": "dexcom_g4_g5", "dexcom g5": "dexcom_g4_g5",
    "guardian 3": "medtronic", "medtronic guardian 4": "medtronic", "medtronic enlite": "medtronic",
    "freestyle libre 2": "libre", "abbott libre": "libre",
}


def static_features(version: int = 1) -> int:
    """How long a version's vector is."""
    if version not in (1, 2):
        raise ValueError(f"static version {version} is not 1 or 2")
    return STATIC_FEATURES if version == 1 else STATIC_FEATURES_V2


def static_version_of(features: int) -> int:
    """The version a static projection of ``features`` inputs reads."""
    return {STATIC_FEATURES: 1, STATIC_FEATURES_V2: 2}[int(features)]


def insulin_speed(name: str | None) -> str | None:
    """``"ultra-rapid"``, ``"rapid"`` or ``"regular"`` for a recorded name or a speed itself; ``None`` unknown."""
    key = (name or "").strip().lower()
    return key if key in INSULIN_SPEEDS else INSULIN_SPEED_NAMES.get(key)


def sensor_family(name: str | None) -> str | None:
    """The sensor family of a recorded ``cgm_device`` or a family itself; ``None`` unknown."""
    key = (name or "").strip().lower()
    return key if key in SENSOR_FAMILIES else SENSOR_NAMES.get(key)


def algorithm_family(algorithm: str | None, modality: str | None = None) -> str | None:
    """The family of a recorded algorithm, by its modality where the name does not say (``None`` unknown)."""
    name = (algorithm or "").strip().lower()
    if name in ALGORITHM_NAMES:
        return ALGORITHM_NAMES[name]
    if name in ALGORITHM_FAMILIES:
        return name
    return MODALITY_NAMES.get((modality or "").strip().lower())


def static_vector(*, family: str | None = None, age: float | None = None, gender: str | None = None,
                  weight_kg: float | None = None, insulin: str | None = None, sensor: str | None = None,
                  version: int = 1) -> np.ndarray:
    """``(static_features(version),)``: the family one-hot, age / 40 and whether it is known, female, male;
    version 2 then weight / 80 kg and whether it is known, the insulin speed one-hot and the sensor family
    one-hot. ``insulin`` and ``sensor`` take a recorded name or the speed or family itself."""
    if family is not None and family not in ALGORITHM_FAMILIES:
        raise ValueError(f"unknown algorithm family {family!r}; one of {ALGORITHM_FAMILIES}")
    out = np.zeros(static_features(version), dtype=np.float32)
    if version == 2:
        k = STATIC_FEATURES
        if weight_kg is not None and np.isfinite(weight_kg) and weight_kg > 0:
            out[k], out[k + 1] = min(float(weight_kg), 250.0) / WEIGHT_SCALE_KG, 1.0
        speed, family_of_sensor = insulin_speed(insulin), sensor_family(sensor)
        if speed is not None:
            out[k + 2 + INSULIN_SPEEDS.index(speed)] = 1.0
        if family_of_sensor is not None:
            out[k + 2 + len(INSULIN_SPEEDS) + SENSOR_FAMILIES.index(family_of_sensor)] = 1.0
    if family is not None:
        out[ALGORITHM_FAMILIES.index(family)] = 1.0
    k = len(ALGORITHM_FAMILIES)
    if age is not None and np.isfinite(age) and age > 0:
        out[k], out[k + 1] = min(float(age), 100.0) / AGE_SCALE, 1.0
    g = (gender or "").strip().lower()
    if g in ("female", "f"):
        out[k + 2] = 1.0
    elif g in ("male", "m"):
        out[k + 3] = 1.0
    return out


def person_statics(frame, *, key: str = "user_id", version: int = 1) -> dict[str, np.ndarray]:
    """One static vector per person of a MetaboNet-schema frame, keyed ``source_file:id`` as the trainer keys them.

    ``key="id"`` keys by ``id`` alone (a file of one cohort, such as the iLet pair). Each person takes
    the value they carry most often, and their median age and weight; a column the frame lacks reads as
    unknown.
    """
    import polars as pl

    numbers = ("age", "weight")
    lf = frame.lazy() if isinstance(frame, pl.DataFrame) else frame
    names = lf.collect_schema().names()
    wanted = [c for c in (STATIC_COLUMNS if version == 1 else STATIC_COLUMNS_V2) if c in names]
    cols = [pl.col(c).cast(pl.Float64) if c in numbers else pl.col(c).cast(pl.String) for c in wanted]
    ident = (pl.col("source_file").fill_null("unknown").cast(pl.String) + pl.lit(":") + pl.col("id").cast(pl.String)
             if key == "user_id" else pl.col("id").cast(pl.String))
    per = (lf.select(ident.alias("_person"), *cols).group_by("_person")
           .agg([(pl.col(c).drop_nulls().median() if c in numbers else pl.col(c).drop_nulls().mode().sort().first())
                 .alias(c) for c in wanted])
           .collect())
    out = {}
    for row in per.iter_rows(named=True):
        pounds = row.get("weight")
        out[str(row["_person"])] = static_vector(
            family=algorithm_family(row.get("insulin_delivery_algorithm"), row.get("insulin_delivery_modality")),
            age=row.get("age"), gender=row.get("gender"),
            weight_kg=None if pounds is None else float(pounds) * KG_PER_POUND,
            insulin=row.get("insulin_type_bolus"), sensor=row.get("cgm_device"), version=version)
    return out


def static_from_spec(spec: Mapping[str, Any], version: int = 1) -> np.ndarray:
    """A person's vector given by hand (a person pool, Livia): ``{"family"|"algorithm", "modality", "age",
    "gender"}``, and for version 2 ``"weight_kg"``, ``"insulin"`` (a name or a speed) and ``"sensor"`` (a
    name or a family)."""
    family = spec.get("family") or algorithm_family(spec.get("algorithm"), spec.get("modality"))
    return static_vector(family=family, age=spec.get("age"), gender=spec.get("gender"),
                         weight_kg=spec.get("weight_kg"), insulin=spec.get("insulin"), sensor=spec.get("sensor"),
                         version=version)


def family_of(vector: np.ndarray | None) -> str:
    """The family a vector names, ``"unknown"`` for none: the group the readings by family use."""
    if vector is None:
        return "unknown"
    hot = np.flatnonzero(np.asarray(vector)[:len(ALGORITHM_FAMILIES)] > 0.5)
    return ALGORITHM_FAMILIES[int(hot[0])] if len(hot) else "unknown"


def insulin_speed_of(vector: np.ndarray | None) -> str:
    """The insulin speed a version-2 vector names, ``"unknown"`` for none or a version-1 vector."""
    if vector is None or len(vector) != STATIC_FEATURES_V2:
        return "unknown"
    hot = np.flatnonzero(np.asarray(vector)[STATIC_FEATURES + 2:STATIC_FEATURES + 2 + len(INSULIN_SPEEDS)] > 0.5)
    return INSULIN_SPEEDS[int(hot[0])] if len(hot) else "unknown"


def age_band(vector: np.ndarray | None) -> str:
    """``"child"`` under 18, ``"adult"``, or ``"unknown"``, from a vector's age."""
    if vector is None or vector[len(ALGORITHM_FAMILIES) + 1] < 0.5:
        return "unknown"
    return "child" if vector[len(ALGORITHM_FAMILIES)] * AGE_SCALE < 18 else "adult"
