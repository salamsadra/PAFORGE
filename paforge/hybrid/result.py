"""Read native MATLAB reconstruction HDF5 with strict scalar/singleton attrs.

MATLAB h5writeatt may encode a one-element attribute as rank-one HDF5;
Python h5py normally returns an ndarray. Never coerce multi-valued attrs.
"""
from pathlib import Path
import h5py
import numpy as np
from .h5_attrs import scalar_int, scalar_text


def read_matlab_result(path: str | Path):
    with h5py.File(path, 'r') as f:
        if scalar_text(f.attrs, 'format') != 'PA-HYBRID-RECONSTRUCTION':
            raise ValueError('Not a PA-HYBRID MATLAB reconstruction')
        if scalar_text(f.attrs, 'schema_version') != '0.1.0':
            raise ValueError('Unsupported reconstruction schema')
        g = f['reconstruction']
        h, w = scalar_int(g.attrs, 'height'), scalar_int(g.attrs, 'width')
        if h < 1 or w < 1 or h * w > 200_000_000:
            raise ValueError('Invalid image dimensions')
        flat = np.asarray(g['image_flat_c'][:], dtype=np.float64).reshape(-1)
        if flat.size != h*w or not np.isfinite(flat).all():
            raise ValueError('Invalid image size or nonfinite pixels')
        if scalar_int(g.attrs, 'quantitative') != 0:
            raise ValueError('Quantitative result not supported in this prototype')
        method_id = scalar_text(g.attrs, 'method_id')
        acquisition_id = scalar_text(g.attrs, 'acquisition_id')
        if not method_id or not acquisition_id:
            raise ValueError('Missing reconstruction method/acquisition identity')
        return flat.reshape(h, w), {
            'method_id': method_id,
            'acquisition_id': acquisition_id,
            'quantitative': False,
        }
