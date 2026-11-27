"""Explicit metadata-preserving extraction of one acquisition from IPASC/vendor HDF5.

Do not infer axes, detector coordinates, wavelengths or sampling clocks. Source
files may contain [detectors,time,wavelength,frame] in a different order.
"""
from pathlib import Path
import hashlib
import h5py
import numpy as np


def extract_hdf5_acquisition(path, key, output, detector_axis, time_axis,
                             fixed_axes, sample_rate_hz, wavelength_nm,
                             source_id, group_id, experimental_attestation=False,
                             detector_positions_npy=None):
    if not experimental_attestation:
        raise ValueError("To extract experimental data, explicitly attest provenance after reviewing source documentation")
    path=Path(path);output=Path(output)
    if not path.is_file():raise FileNotFoundError(path)
    if not source_id or not group_id:raise ValueError("Source and patient/phantom group ID required")
    if sample_rate_hz<=0 or wavelength_nm<=0:raise ValueError("Positive sampling rate and wavelength required")
    with h5py.File(path,'r') as f:
        if 'synthetic' in f.attrs and bool(f.attrs['synthetic']):
            raise ValueError("Cannot extract a PA-FORGE synthetic file as experimental")
        key=key.lstrip('/')
        if key not in f:raise KeyError(f"Dataset {key} unavailable")
        d=f[key]
        if not isinstance(d,h5py.Dataset):raise ValueError("Dataset key must reference a numerical HDF5 array")
        nd=d.ndim
        if len({detector_axis,time_axis})!=2 or not all(0<=a<nd for a in [detector_axis,time_axis]):
            raise ValueError("Detector and time axes must be distinct and valid")
        fixed_axes={int(k):int(v) for k,v in fixed_axes.items()}
        required=set(range(nd))-{detector_axis,time_axis}
        if set(fixed_axes)!=required:
            raise ValueError(f"Specify an index for every non-signal axis: expected {sorted(required)}")
        selector=[]
        for i in range(nd):
            if i in required:
                if fixed_axes[i]<0 or fixed_axes[i]>=d.shape[i]:raise IndexError(f"Index {fixed_axes[i]} outside axis {i}")
                selector.append(fixed_axes[i])
            else:selector.append(slice(None))
        a=np.asarray(d[tuple(selector)],dtype=np.float64)
        kept=[i for i in range(nd) if i not in required]
        if kept.index(detector_axis)!=0:a=a.T
    if a.ndim!=2 or a.shape[0]<2 or a.shape[1]<32:raise ValueError("Invalid detector/time array")
    if not np.all(np.isfinite(a)):raise ValueError("Nonfinite signal values")
    positions=None
    if detector_positions_npy is not None:
        positions=np.load(detector_positions_npy,allow_pickle=False)
        if positions.shape!=(a.shape[0],3) or not np.isfinite(positions).all():
            raise ValueError("Coordinates must be [detectors,3] in metres")
    output.parent.mkdir(parents=True,exist_ok=True)
    with h5py.File(output,'w') as f:
        f.attrs.update({'format':'PA-FORGE-native-experimental-0.3 (not IPASC certified)',
                'synthetic':False, 'source_id':str(source_id),'group_id':str(group_id),
                'original_file_name':path.name,'original_dataset_key':key,
                'original_shape':str(nd), 'time_axis_original':time_axis,
                'detector_axis_original':detector_axis,
                'fixed_axes':str(fixed_axes),'sampling_rate_hz':float(sample_rate_hz),
                'acquisition_wavelength_nm':float(wavelength_nm),
                'signal_units':'unknown unless independently calibrated',
                'experimental_provenance_user_attested':True})
        f.create_dataset('raw_data',data=a.astype(np.float32),compression='gzip',shuffle=True)
        if positions is not None:f.create_dataset('detector_positions_m',data=positions)
    return output
