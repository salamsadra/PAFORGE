"""Import explicitly reviewed MATLAB PATBox raw exports.

Raw HDF5 has no hashes; importing to canonical PA-HYBRID-EXCHANGE attaches
SHA256 and enforces all PR-02 metadata checks. Does NOT assert physics fidelity.
"""
from __future__ import annotations
from pathlib import Path
import h5py
import numpy as np
from .contracts import AcquisitionRecord
from .exchange import write_acquisition
from .h5_attrs import scalar_int, scalar_float, scalar_text


def import_patbox_raw(path: str | Path, output: str | Path, *, source_id: str,
                      phantom_id: str | None = None, wavelength_nm: float | None = None) -> Path:
    with h5py.File(path,'r') as h:
        if (scalar_text(h.attrs,'format') != 'PA-PATBOX-RAW-EXPORT'
                or scalar_text(h.attrs,'schema_version') != '0.1.0'):
            raise ValueError('Wrong PATBox raw HDF5 format/version')
        g = h['raw']
        n, t = scalar_int(g.attrs,'n_detectors'), scalar_int(g.attrs,'n_samples')
        if n < 2 or t < 16 or n*t > 200_000_000:
            raise ValueError('Invalid or suspicious shape')
        note = scalar_text(g.attrs,'channel_mapping_note')
        if len(note.strip()) < 15:
            raise ValueError('Missing documented channel-order verification')
        # MATLAB may write 1-D logical streams as HDF5 [length,1] arrays.
        # Flat serialization is explicit; reshape only after size checks.
        flat = np.asarray(g['signals_flat_c'][:], dtype=np.float64).reshape(-1)
        posflat = np.asarray(g['detectors_flat_c'][:], dtype=np.float64).reshape(-1)
        if flat.size != n*t or posflat.size != n*3:
            raise ValueError('Broken PATBox raw lengths')
        record = AcquisitionRecord(signals=flat.reshape(n,t), detectors_m=posflat.reshape(n,3),
          sample_interval_s=scalar_float(g.attrs,'sample_interval_s'),
          acquisition_id=scalar_text(g.attrs,'acquisition_id'),
          device_id=scalar_text(g.attrs,'device_id'),
          session_id=scalar_text(g.attrs,'session_id'),
          source_id=source_id, domain='synthetic',
          coordinate_frame=scalar_text(g.attrs,'coordinate_frame'),
          time_zero_s=None, phantom_id=phantom_id, wavelength_nm=wavelength_nm,
          processing_history=('PATBox MATLAB raw export', 'channel mapping review: ' + note,
                              'SHA256 committed by Python after reading raw export'))
    return write_acquisition(output, record)
