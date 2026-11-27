"""Small deterministic exchange fixture; synthetic test numbers, not measurements."""
from pathlib import Path
import numpy as np
from paforge.hybrid import AcquisitionRecord, write_acquisition, inspect_exchange

n_det, n_samples = 4, 64
signals = np.arange(n_det * n_samples, dtype=np.float64).reshape(n_det, n_samples) / 10
positions = np.stack([np.arange(n_det) * 0.0004, np.zeros(n_det), np.zeros(n_det)], axis=1)
record = AcquisitionRecord(signals=signals, detectors_m=positions, sample_interval_s=50e-9,
                           acquisition_id="fixture-0001", device_id="synthetic-test-device",
                           session_id="test-session", source_id="unit-test-source",
                           domain="synthetic", coordinate_frame="fixture-cartesian-xyz-m",
                           processing_history=("deterministic-fixture-no-real-physics",))
out = Path(__file__).with_name("test_exchange.h5")
write_acquisition(out, record, overwrite=True)
print(inspect_exchange(out))
