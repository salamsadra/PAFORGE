"""Optional integration that calls the official *installed* PATATO Python package.

Writes a temporary documented PATATO HDF5 acquisition from the canonical 2D
interface. This is a bridge, NOT a copy of their algorithms. Backend and JAX/
OpenCL dependencies are installed separately by user. PATATO/PA-FORGE results
must be validated for orientation and physical/time-zero convention.
"""
import tempfile
from pathlib import Path
import h5py
import numpy as np


def reconstruct_patato(acq,method):
    try:import patato as pat
    except ImportError as e:
        raise RuntimeError('PATATO is an optional external package: pip install "pa-forge[patato]". No PATATO result is being simulated.') from e
    if acq.wavelength_nm is None:
        raise ValueError('Official PATATO bridge requires acquisition_wavelength_nm from the source metadata; it must not invent a wavelength')
    if method not in ('patato_bp','patato_model','patato_opencl_bp','patato_slow_bp'):
        raise ValueError('PATATO interface supports backprojection or model-based reconstruction')
    # Official PATATO format: [frame, wavelength, detector, time]; metres
    with tempfile.TemporaryDirectory(prefix='paforge_patato_') as td:
        source=Path(td)/'acquisition.hdf5'
        with h5py.File(source,'w') as f:
            f.attrs.update({'version':'PA-FORGE bridge using PATATO documented file format',
                            'fs':1/acq.dt_s,'speedofsound':float(acq.sound_speed_m_s),
                            'name':'PA-FORGE canonical 2D acquisition','date':'',
                            'comment':'Numerical native export; experimental provenance external'})
            f.create_dataset('raw_data',data=acq.traces[None,None].astype('float32'))
            f.create_dataset('GEOMETRY',data=acq.positions_m.astype('float64'))
            f.create_dataset('wavelengths',data=np.array([acq.wavelength_nm],dtype='float64'))
        pa=pat.PAData.from_hdf5(str(source))
        nx=int(acq.grid_size);fov=float(acq.fov_m)
        if method=='patato_bp':
            cls=getattr(pat,'ReferenceBackprojection',None) or getattr(pat,'Backprojection',None)
            if cls is None:raise RuntimeError('Installed PATATO has no documented ReferenceBackprojection/Backprojection; check package version')
            algorithm=cls(field_of_view=(fov,fov,0),n_pixels=(nx,nx,1))
        elif method in ('patato_opencl_bp','patato_slow_bp'):
            if method=='patato_opencl_bp':cls=getattr(pat,'OpenCLBackprojection',None)
            else:
                from patato.recon import SlowBackprojection as cls
            if cls is None:raise RuntimeError('Installed PATATO backend not found: '+method)
            algorithm=cls(field_of_view=(fov,fov,0),n_pixels=(nx,nx,1))
        else:
            cls=getattr(pat,'ModelBasedReconstruction',None)
            if cls is None:raise RuntimeError('Installed PATATO lacks ModelBasedReconstruction')
            algorithm=cls(field_of_view=(fov,fov,0),n_pixels=(nx,nx,1),pa_example=pa)
        rec,*_=algorithm.run(pa.get_time_series(),pa)
        image=np.asarray(rec.raw_data if hasattr(rec,'raw_data') else rec)
    image=np.squeeze(image)
    if image.shape!=(nx,nx) or not np.isfinite(image).all():
        raise ValueError(f'PATATO returned incompatible {image.shape}; expected 2D {(nx,nx)}. No silent resize/projection allowed.')
    return image.astype('float32')
