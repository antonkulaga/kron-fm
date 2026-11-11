"""The ECG head: age from a 10 s single-lead ECG (lead I, the lead a watch's two electrodes record).

Trained on PTB-XL (lead I of 12-lead clinical ECGs, patients aged 18-89; PTB-XL writes 300 for 90 and over, which
is dropped). Strips are 10 s at 250 Hz, each robust-scaled (median, IQR/1.349), so the units of a clinical
recorder and of a watch's ECG app need not agree. PTB-XL's own ``strat_fold`` keeps every patient in one fold:
folds 1-8 train, 9 validates, 10 tests. A watch's 30 s recording is three 10 s strips, predictions averaged.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

FS = 250
SECONDS = 10
LENGTH = FS * SECONDS


def robust(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float64)
    med = np.median(x, axis=-1, keepdims=True)
    q75, q25 = np.percentile(x, [75, 25], axis=-1, keepdims=True)
    return ((x - med) / np.maximum((q75 - q25) / 1.349, 1e-6)).astype(np.float32)


def band(x: np.ndarray, lo: float = 0.5, hi: float = 40.0) -> np.ndarray:
    """Each 250 Hz strip band-passed (zero-phase Butterworth, order 2) and robust-scaled again: baseline wander and the
    offset of an unfiltered recorder (a watch SDK's raw ECG) go, so a head reads the waveform, not the drift."""
    from scipy.signal import butter, sosfiltfilt

    sos = butter(2, [lo, hi], btype="band", fs=FS, output="sos")
    y = sosfiltfilt(sos, np.asarray(x, np.float64), axis=-1, padtype="odd", padlen=min(3 * FS, x.shape[-1] - 1))
    return robust(y)


def strips(ecg: np.ndarray, fs: float, filtered: bool = False) -> np.ndarray:
    """A recording at ``fs`` as robust-scaled 10 s strips at 250 Hz, ``(n, 2500)``; the tail shorter than 10 s is
    dropped. With ``filtered``, each strip is also band-passed 0.5-40 Hz (:func:`band`), as the filtered heads read."""
    from scipy.signal import resample_poly

    x = np.nan_to_num(np.asarray(ecg, float))
    if fs != FS:
        from math import gcd

        g = gcd(int(fs), FS)
        x = resample_poly(x, FS // g, int(fs) // g)
    n = len(x) // LENGTH
    if not n:
        return np.zeros((0, LENGTH), np.float32)
    out = robust(x[: n * LENGTH].reshape(n, LENGTH))
    return band(out) if filtered else out


def load_ptbxl(folder: Path, min_age: float = 18.0) -> dict[str, np.ndarray]:
    """Lead I of every PTB-XL record at 500 Hz as one 10 s strip; ages 18-89 only."""
    import polars as pl
    import wfdb

    folder = Path(folder)
    db = pl.read_csv(folder / "ptbxl_database.csv", infer_schema_length=None)
    db = db.filter(pl.col("age").is_not_null() & (pl.col("age") >= min_age) & (pl.col("age") < 120))
    xs = []
    for name in db["filename_hr"].to_list():
        rec = wfdb.rdrecord(str(folder / name), channel_names=["I"])
        xs.append(strips(rec.p_signal[:, 0], rec.fs)[0])
    return {"x": np.stack(xs), "age": db["age"].to_numpy().astype(np.float32), "sex": db["sex"].to_numpy(),
            "fold": db["strat_fold"].to_numpy(), "patient": db["patient_id"].to_numpy(), "ecg_id": db["ecg_id"].to_numpy()}


class Block(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(c_in, c_out, 7, stride, 3, bias=False)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, 7, 1, 3, bias=False)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.skip = (nn.Identity() if c_in == c_out and stride == 1
                     else nn.Sequential(nn.Conv1d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm1d(c_out)))

    def forward(self, x):
        y = torch.relu(self.bn1(self.conv1(x)))
        return torch.relu(self.bn2(self.conv2(y)) + self.skip(x))


class ECGAge(nn.Module):
    """A small 1D ResNet: one 2500-sample strip -> age in years (``mean + scale * output``)."""

    def __init__(self, width: int = 64, mean: float = 60.0, scale: float = 17.0, dropout: float = 0.2) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1, width, 15, 2, 7, bias=False), nn.BatchNorm1d(width), nn.ReLU())
        chans = [width, width, 2 * width, 2 * width, 4 * width, 4 * width]
        layers, c = [], width
        for i, co in enumerate(chans):
            layers.append(Block(c, co, 2 if i % 2 == 0 else 1))
            c = co
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * c, 1))
        self.register_buffer("mean", torch.tensor(float(mean)))
        self.register_buffer("scale", torch.tensor(float(scale)))

    def forward(self, x):
        h = self.body(self.stem(x.unsqueeze(1)))
        pooled = torch.cat([h.mean(-1), h.amax(-1)], dim=-1)
        return self.mean + self.scale * self.head(pooled).squeeze(-1)


def load_code15(folder: Path, out: Path, min_age: float = 18.0) -> dict[str, np.ndarray]:
    """Lead I of every CODE-15% exam as one 10 s strip at 250 Hz, cached in ``out`` (an ``.npz``).

    Each ``exams_partN.zip`` holds one HDF5 file with ``tracings`` ``(n, 4096, 12)`` at 400 Hz (10 s, zero-padded to
    4096; lead order DI, DII, ...) and ``exam_id``; ``exams.csv`` gives age, sex and the patient. Parts are unpacked
    one at a time into a scratch folder beside ``out`` and removed after reading.
    """
    import tempfile
    import zipfile

    import h5py
    import polars as pl
    from scipy.signal import resample_poly

    folder, out = Path(folder), Path(out)
    if out.is_file():
        return dict(np.load(out, allow_pickle=False))
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = pl.read_csv(folder / "exams.csv", infer_schema_length=None)
    info = {int(r["exam_id"]): r for r in meta.iter_rows(named=True)}
    xs, ages, males, patients, exams = [], [], [], [], []
    for part in sorted(folder.glob("exams_part*.zip")):
        with tempfile.TemporaryDirectory(dir=out.parent) as tmp, zipfile.ZipFile(part) as z:
            name = next(n for n in z.namelist() if n.endswith(".hdf5"))
            z.extract(name, tmp)
            with h5py.File(Path(tmp) / name, "r") as h:
                ids = h["exam_id"][:]
                tr = h["tracings"]
                for start in range(0, len(ids), 4096):
                    block = tr[start:start + 4096, :, 0].astype(np.float64)  # lead I
                    for j, eid in enumerate(ids[start:start + 4096]):
                        r = info.get(int(eid))
                        if r is None or r["age"] is None or r["age"] < min_age:
                            continue
                        x = block[j]
                        nz = np.flatnonzero(x != 0)
                        if len(nz) < 3800:  # padding or a flat lead
                            continue
                        x = x[nz[0]:nz[0] + 4000]
                        if len(x) < 4000:
                            continue
                        xs.append(robust(resample_poly(x, 5, 8)[None, :LENGTH])[0].astype(np.float16))
                        ages.append(float(r["age"]))
                        males.append(bool(r["is_male"]))
                        patients.append(int(r["patient_id"]))
                        exams.append(int(eid))
        print(f"{part.name}: {len(xs)} strips so far", flush=True)
    d = {"x": np.stack(xs), "age": np.asarray(ages, np.float32), "male": np.asarray(males),
         "patient": np.asarray(patients), "exam": np.asarray(exams)}
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **d)
    return d
