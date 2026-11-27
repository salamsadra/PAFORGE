"""Explicit adapter to legacy Python 2D reconstruction engine.

The historical code assumes a planar XY domain and a known time-zero of 0 s.
No attempt is made to infer it from signals or silently project geometry.
"""
import numpy as np
from .contracts import AcquisitionRecord


def to_legacy_reconstruction(record: AcquisitionRecord, *, fov_m: float,
                             grid_size: int, sound_speed_m_s: float):
    if record.time_zero_s is None or abs(record.time_zero_s) > 1e-15:
        raise ValueError("Legacy reconstruction requires explicitly calibrated time_zero_s=0")
    if not np.allclose(record.detectors_m[:, 2], 0.0, rtol=0, atol=1e-10):
        raise ValueError("Legacy XY reconstruction requires explicitly planar z=0 geometry")
    if record.coordinate_frame != "xy-plane-z0-metres":
        raise ValueError("Legacy XY reconstruction requires coordinate_frame='xy-plane-z0-metres'")
    from paforge.reconstruction_engine import Acquisition
    return Acquisition(record.signals, record.sample_interval_s,
                       record.detectors_m, fov_m=fov_m, grid_size=grid_size,
                       sound_speed_m_s=sound_speed_m_s,
                       wavelength_nm=record.wavelength_nm)
