"""Experimental 2D reconstruction baselines and exploratory FDTD inverse.

FISTA here is the actual discretized 2D forward/adjoint solver, not a proxy for
vendor-provided device calibration. Sensor aperture, frequency response, delay,
physical p0 scale and 3D effects remain unmodeled. No accuracy claim.
"""
from pathlib import Path
import json
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import decimate,butter,sosfiltfilt
from .config import SimConfig
from .acoustic import reconstruct_ubp,reconstruct_das

from .reconstruction_engine import Acquisition, reconstruct, METHODS, SPECS


def reconstruct_experimental(input_h5,output,method='ubp',fov_m=0.04,grid_size=128,
                             sound_speed_m_s=1500.0,positions_file=None,
                             iterations=10,reg=0.001,time_decimate=4,
                             filter_low_hz=0.3e6,filter_high_hz=6e6,weights=None):
    if method not in METHODS:raise ValueError(f'Supported methods: {METHODS}')
    if not 0.005<=fov_m<=0.3:raise ValueError('FOV must be 5..300 mm')
    if not 1000<=sound_speed_m_s<=2200:raise ValueError('Check sound speed: 1000..2200 m/s')
    if not (1<=iterations<=100 and 0<=reg<=1 and time_decimate in (1,2,4,8,16)):
        raise ValueError('Iterations 1..100, reg 0..1, time_decimate 1/2/4/8/16 required')
    path=Path(input_h5)
    with h5py.File(path,'r') as f:
        if 'raw_data' not in f:raise ValueError('No /raw_data in source file')
        traces=np.asarray(f['raw_data'],dtype=np.float64)
        fs=float(f.attrs.get('sampling_rate_hz',0))
        if fs<=0:raise ValueError('Missing verified sampling rate')
        if positions_file is None:
            if 'detector_positions_m' not in f:
                raise ValueError('Measured detector geometry coordinates required. Do not invent a circle.')
            pos=np.asarray(f['detector_positions_m'])
        else:pos=np.load(positions_file,allow_pickle=False)
        wavelength_nm=f.attrs.get('acquisition_wavelength_nm',None)
        provenance=str(f.attrs.get('source_id','unspecified'))
        is_synthetic=bool(f.attrs.get('synthetic',False))
    if traces.ndim!=2 or traces.shape[0]<2 or traces.shape[1]<32 or not np.isfinite(traces).all():
        raise ValueError('Finite raw traces [detectors,time>=32] required')
    if pos.shape!=(traces.shape[0],3) or not np.isfinite(pos).all():
        raise ValueError('Finite detector positions [detectors,3] in metres required')
    if np.ptp(pos[:,2])>0.002:
        raise ValueError('Significant 3D sensor spread: a 2D projection is not valid for this geometry')
    acq=Acquisition(traces,1/fs,pos,fov_m,int(grid_size),float(sound_speed_m_s),wavelength_nm=float(wavelength_nm) if wavelength_nm is not None else None)
    rec=reconstruct(acq,method,iterations=iterations,reg=reg,time_decimate=time_decimate,
                    weights=weights,filter_low_hz=filter_low_hz,filter_high_hz=filter_high_hz)
    spec=SPECS[method]
    losses=None
    notes=[spec.description,'Numerical model: '+spec.model,
           'Common engine; applicable to both synthetic and measured 2D inputs under documented assumptions']
    if spec.requires_weights:
        notes.append('Supervised checkpoint required. Experimental domain shift has not been validated.')
    if spec.requires_fdtd:
        notes.append('FDTD is a simplified numerical model; device impulse response and time-zero not calibrated.')
    elif spec.category=='iterative':
        notes.append('Matrix-free straight-ray travel-time surrogate, not full-wave acoustics.')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with h5py.File(output/'experimental_reconstruction.h5','w') as f:
        f.attrs['method']=method;f.attrs['quantitative']=False
        f.attrs['source_id']=provenance;f.attrs['source_synthetic']=is_synthetic
        f.attrs['warnings']='Exploratory 2D approximation, unknown device response/time-zero/calibration and image registration.'
        f.create_dataset('image',data=rec,compression='gzip')
        if losses is not None:f.create_dataset('fista_objective',data=losses)
    plt.imsave(output/'experimental_reconstruction.png',rec,cmap='magma')
    manifest={'source':str(path),'source_id':provenance,'source_synthetic':is_synthetic,
              'method':method,'algorithm_family':spec.category,'algorithm_description':spec.description,'fov_m':fov_m,'grid_size':grid_size,
              'sound_speed_m_s':sound_speed_m_s,'signal_sampling_hz':fs,
              'iterations':iterations if spec.category in ('iterative','wave') else None,
              'regularization':reg if spec.category in ('iterative','wave') else None,
              'temporal_decimation':time_decimate if spec.requires_fdtd else None,
              'final_objective':None,
              'status':'qualitative experimental reconstruction ONLY' if not is_synthetic else 'synthetic reconstruction smoke test',
              'limitations':['Not validated or calibrated against publisher reconstruction or measured ground truth','No device impulse/angular response calibration','2D homogeneous medium',
                             'No experimental sound-speed/time-zero calibration',
                             'Reference image orientation, FOV and registration unestablished']+notes}
    (output/'reconstruction_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return rec
