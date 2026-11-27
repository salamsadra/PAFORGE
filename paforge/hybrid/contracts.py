"""Strict, explicit acquisition data contract for cross-language exchange.

No hidden 3-D to 2-D projection, inferred time zero, geometry, or experimental
calibration. Units and provenance are part of the record, not Python defaults.
"""
from dataclasses import dataclass, field
from typing import Any
import numpy as np


def _required_string(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty explicit string")


@dataclass(frozen=True)
class AcquisitionRecord:
    signals: np.ndarray
    detectors_m: np.ndarray
    sample_interval_s: float
    acquisition_id: str
    device_id: str
    session_id: str
    source_id: str
    domain: str  # synthetic / experimental / unknown
    coordinate_frame: str
    time_zero_s: float | None = None
    wavelength_nm: float | None = None
    phantom_id: str | None = None
    subject_id: str | None = None
    calibration_id: str | None = None
    calibration_kind: str = "none"  # none / estimated / measured
    processing_history: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        signals = np.asarray(self.signals)
        positions = np.asarray(self.detectors_m)
        if signals.ndim != 2 or signals.shape[0] < 2 or signals.shape[1] < 16:
            raise ValueError("signals must be [n_detectors>=2, n_samples>=16]")
        if signals.dtype.kind != "f" or not np.isfinite(signals).all():
            raise ValueError("signals must be finite real floating-point values")
        if positions.shape != (signals.shape[0], 3) or positions.dtype.kind not in "fi" or not np.isfinite(positions).all():
            raise ValueError("detectors_m must be finite [n_detectors, 3] coordinates")
        if not np.isfinite(self.sample_interval_s) or self.sample_interval_s <= 0:
            raise ValueError("sample_interval_s must be positive and finite")
        for name in ("acquisition_id", "device_id", "session_id", "source_id", "coordinate_frame"):
            _required_string(name, getattr(self, name))
        if self.domain not in {"synthetic", "experimental", "unknown"}:
            raise ValueError("domain must be synthetic, experimental, or unknown")
        if self.calibration_kind not in {"none", "estimated", "measured"}:
            raise ValueError("invalid calibration_kind")
        if self.calibration_kind != "none" and not self.calibration_id:
            raise ValueError("calibration_id required for an estimated/measured calibration")
        for name in ("time_zero_s", "wavelength_nm"):
            value = getattr(self, name)
            if value is not None and not np.isfinite(value):
                raise ValueError(f"{name} must be finite if supplied")
        if self.wavelength_nm is not None and self.wavelength_nm <= 0:
            raise ValueError("wavelength_nm must be positive")
        for name in ("phantom_id", "subject_id", "calibration_id"):
            val = getattr(self, name)
            if val is not None:
                _required_string(name, val)
        if not isinstance(self.processing_history, (tuple, list)) or not all(
            isinstance(x, str) and x.strip() for x in self.processing_history
        ):
            raise ValueError("processing_history must be a sequence of non-empty strings")
        # In-memory numeric type preserved, persisted representation is specified elsewhere.
        object.__setattr__(self, "signals", signals.copy())
        object.__setattr__(self, "detectors_m", positions.astype(np.float64, copy=True))
        object.__setattr__(self, "processing_history", tuple(self.processing_history))

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.signals.shape)

    def metadata(self) -> dict[str, Any]:
        names = ("acquisition_id", "device_id", "session_id", "source_id", "domain",
                 "coordinate_frame", "time_zero_s", "wavelength_nm", "phantom_id",
                 "subject_id", "calibration_id", "calibration_kind", "processing_history")
        return {name: getattr(self, name) for name in names}
