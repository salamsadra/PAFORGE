"""Explicit adapter from PA-FORGE native HDF5 to the separate hybrid schema."""
from pathlib import Path
import h5py
import numpy as np
from .contracts import AcquisitionRecord


def from_native_hdf5(path: str | Path, *, acquisition_id: str, device_id: str,
                     session_id: str, source_id: str, coordinate_frame: str,
                     time_zero_s: float | None = None,
                     phantom_id: str | None = None,
                     subject_id: str | None = None) -> AcquisitionRecord:
    with h5py.File(path, "r") as h:
        if "synthetic" not in h.attrs:
            raise ValueError("Native source must state synthetic explicitly")
        if "raw_data" not in h or "detector_positions_m" not in h:
            raise ValueError("Native source lacks signals or explicit detector positions")
        if "sampling_rate_hz" not in h.attrs:
            raise ValueError("Native source lacks sampling rate")
        raw_domain = "synthetic" if bool(h.attrs["synthetic"]) else "experimental"
        wl = h.attrs.get("acquisition_wavelength_nm", None)
        wavelength = float(wl) if wl is not None else None
        return AcquisitionRecord(
            signals=np.asarray(h["raw_data"][:], dtype=np.float64),
            detectors_m=np.asarray(h["detector_positions_m"][:], dtype=np.float64),
            sample_interval_s=1/float(h.attrs["sampling_rate_hz"]),
            acquisition_id=acquisition_id, device_id=device_id, session_id=session_id,
            source_id=source_id, domain=raw_domain, coordinate_frame=coordinate_frame,
            time_zero_s=time_zero_s, wavelength_nm=wavelength,
            phantom_id=phantom_id, subject_id=subject_id,
            processing_history=(f"explicit native adapter: {Path(path).name}",))
