"""Portable HDF5 bridge for 1-D C-order flattened arrays.

Explicit flattened datasets avoid MATLAB/h5py dimension reversal ambiguity.
The schema is intentionally *not* IPASC. Lossless numeric round trip is tested
on Python; the MATLAB portion must be independently run on an installed MATLAB.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import h5py
import numpy as np
from .contracts import AcquisitionRecord

SCHEMA = "PA-HYBRID-EXCHANGE"
SCHEMA_VERSION = "0.1.0"


def _canonical(data: np.ndarray) -> np.ndarray:
    return np.asarray(data, dtype="<f8", order="C")


def _sha(data: np.ndarray) -> str:
    return hashlib.sha256(_canonical(data).tobytes(order="C")).hexdigest()


def write_acquisition(path: str | Path, record: AcquisitionRecord, *, overwrite: bool = False) -> Path:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    # One-dimensional datasets have identical element order in MATLAB and Python.
    signals = _canonical(record.signals)
    detectors = _canonical(record.detectors_m)
    with h5py.File(path, "w") as h:
        h.attrs["format"] = SCHEMA
        h.attrs["schema_version"] = SCHEMA_VERSION
        a = h.create_group("acquisition")
        a.attrs["n_detectors"] = signals.shape[0]
        a.attrs["n_samples"] = signals.shape[1]
        a.attrs["signal_axis_order"] = "detector,time"
        a.attrs["detector_axis_order"] = "detector,xyz"
        a.attrs["sample_interval_s"] = float(record.sample_interval_s)
        a.attrs["detector_coordinate_units"] = "m"
        a.attrs["signal_units"] = "unspecified"  # never invent Pa calibration
        a.attrs["signal_sha256_c_f64"] = _sha(signals)
        a.attrs["detector_sha256_c_f64"] = _sha(detectors)
        a.attrs["metadata_json"] = json.dumps(record.metadata(), ensure_ascii=False, sort_keys=True)
        a.create_dataset("signals_flat_c", data=signals.ravel(order="C"), compression="gzip", shuffle=True)
        a.create_dataset("detectors_flat_c", data=detectors.ravel(order="C"), compression="gzip", shuffle=True)
    return path


def read_acquisition(path: str | Path) -> AcquisitionRecord:
    with h5py.File(path, "r") as h:
        if h.attrs.get("format", "") != SCHEMA or h.attrs.get("schema_version", "") != SCHEMA_VERSION:
            raise ValueError("Unsupported HDF5 exchange schema/version")
        a = h["acquisition"]
        n, t = int(a.attrs["n_detectors"]), int(a.attrs["n_samples"])
        if n < 2 or t < 16 or a.attrs["signal_axis_order"] != "detector,time" or a.attrs["detector_axis_order"] != "detector,xyz":
            raise ValueError("Incorrect array shape or axes")
        if a.attrs["detector_coordinate_units"] != "m":
            raise ValueError("Sensor coordinates must be in metres")
        signal_flat = np.asarray(a["signals_flat_c"][:], dtype=np.float64)
        position_flat = np.asarray(a["detectors_flat_c"][:], dtype=np.float64)
        if signal_flat.size != n*t or position_flat.size != n*3:
            raise ValueError("Inconsistent data lengths")
        signals = signal_flat.reshape(n, t, order="C")
        positions = position_flat.reshape(n, 3, order="C")
        if _sha(signals) != a.attrs["signal_sha256_c_f64"] or _sha(positions) != a.attrs["detector_sha256_c_f64"]:
            raise ValueError("Data hash mismatch; file may be corrupted")
        meta = json.loads(a.attrs["metadata_json"])
        if not isinstance(meta, dict):
            raise ValueError("Invalid metadata JSON")
        return AcquisitionRecord(signals=signals, detectors_m=positions,
                                 sample_interval_s=float(a.attrs["sample_interval_s"]), **meta)


def inspect_exchange(path: str | Path) -> dict[str, object]:
    record = read_acquisition(path)
    return {"schema": SCHEMA_VERSION, "shape": list(record.shape), "domain": record.domain,
            "source_id": record.source_id, "acquisition_id": record.acquisition_id,
            "time_zero_known": record.time_zero_s is not None}
