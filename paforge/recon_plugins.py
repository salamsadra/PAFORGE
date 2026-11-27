"""Explicit opt-in reconstruction plugin bridge for third-party algorithm code.

External algorithms are NOT bundled or vouched for. A user-installed Python
module must expose `reconstruct(traces, sampling_rate_hz, positions_m,
image_size, fov_m) -> ndarray`. This executes trusted local Python code.
The Zenodo record 23348355 has not been independently resolved or inspected.
"""
import importlib
from pathlib import Path
import numpy as np


def run_reconstruction_plugin(module_name, traces, sampling_rate_hz, positions_m, image_size, fov_m):
    if not module_name or any(ch not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.' for ch in module_name):
        raise ValueError('Expected a Python module dotted name, not a path or arbitrary expression')
    if sampling_rate_hz<=0 or image_size<8 or fov_m<=0:
        raise ValueError('Invalid reconstruction configuration')
    y=np.asarray(traces)
    xyz=np.asarray(positions_m)
    if y.ndim!=2 or xyz.shape!=(y.shape[0],3):
        raise ValueError('Expected signals [detectors,time] and positions [detectors,3]')
    mod=importlib.import_module(module_name)
    fn=getattr(mod,'reconstruct',None)
    if not callable(fn):
        raise TypeError(f'{module_name} does not define reconstruct(...)')
    image=np.asarray(fn(y,float(sampling_rate_hz),xyz,int(image_size),float(fov_m)))
    if image.shape!=(image_size,image_size) or not np.isfinite(image).all():
        raise ValueError('Plugin output must be finite 2D square image, same grid as requested')
    return image.astype('float32')
