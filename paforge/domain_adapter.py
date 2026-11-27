"""Apply a frozen, train-derived statistical spectral profile to simulated traces.

This operation is deterministic and DOES NOT generate stochastic noise. It is a
research domain adaptation baseline, not physics-calibrated sensor modeling.
"""
import json
from pathlib import Path
import h5py
import numpy as np
from .experimental import _adapt_record


def adapt_synthetic(input_h5,profile_json,output_h5):
    profile=json.loads(Path(profile_json).read_text(encoding='utf8'))
    with h5py.File(input_h5,'r') as f:
        if not bool(f.attrs.get('synthetic',False)):
            raise ValueError('Only known synthetic traces may be adapted')
        if 'raw_data' not in f:raise ValueError('No raw_data')
        x=f['raw_data'][:]
        fs=float(f.attrs.get('sampling_rate_hz',0))
        if x.shape != (profile['detectors'],profile['time_samples']) or not np.isclose(fs,profile['fs_hz'],rtol=1e-6):
            raise ValueError('Device dimensions and sampling frequency must match calibration training profile')
        attrs=dict(f.attrs)
    result=_adapt_record(x,profile).astype(np.float32)
    output_h5=Path(output_h5);output_h5.parent.mkdir(parents=True,exist_ok=True)
    with h5py.File(output_h5,'w') as f:
        for k,v in attrs.items():f.attrs[k]=v
        f.attrs['synthetic']=True
        f.attrs['domain_adapted']=True
        f.attrs['adaptation_kind']='train-only statistical spectral matching; NOT instrument calibration'
        f.attrs['calibration_profile_path']=str(Path(profile_json).resolve())
        f.attrs['source_simulation_path']=str(Path(input_h5).resolve())
        f.create_dataset('raw_data',data=result,compression='gzip')
    return output_h5
