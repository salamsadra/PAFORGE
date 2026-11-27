"""IPASC-format HDF5 normalization, strictly metadata-driven and local-only."""
from pathlib import Path
import h5py
import numpy as np
from .experimental_import import extract_hdf5_acquisition


def normalize_ipasc(path,output,source_id,group_id,origin='experimental',wavelength_index=0,frame_index=0):
    """Transform a planar XY/XZ/YZ acquisition into 2D XY simulation axes.

    Caller must set origin truthfully. We do not silently call synthetic or
    unspecified files experimental. Device response metadata is preserved by
    keeping the original file; native output is not certified IPASC.
    """
    path=Path(path)
    with h5py.File(path,'r') as f:
        data=f['binary_time_series_data'];dims=data.shape
        if len(dims)!=4 or wavelength_index>=dims[2] or frame_index>=dims[3]:
            raise ValueError('Expected [detectors,time,wavelength,frame], valid indices required')
        rate=float(f['meta_data/ad_sampling_rate'][()]);wave=f['meta_data/acquisition_wavelengths'][()]
        wave=float(np.asarray(wave).flatten()[wavelength_index]) if np.asarray(wave).ndim else float(wave)
        det=f['meta_data_device/detectors'];names=sorted(det.keys())
        xyz=np.stack([det[k]['detector_position'][()] for k in names]).astype(float)
        if xyz.shape!=(dims[0],3):raise ValueError('Detector metadata count does not match signal axis')
        extent=np.ptp(xyz,axis=0)
        flat=np.argmin(extent)
        if extent[flat]>.002:raise ValueError('Non-planar detectors require a true 3D solver, cannot project')
        kept=[k for k in range(3) if k!=flat]
        p2d=np.column_stack([xyz[:,kept[0]],xyz[:,kept[1]],np.zeros(len(xyz))])
        if rate<=0:raise ValueError('Missing positive sample clock')
    if origin not in ('experimental','synthetic','unknown'):
        raise ValueError('origin must be experimental/synthetic/unknown')
    if origin=='experimental':
        # The original extraction method requires user attestation.
        output=Path(output)
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            posfile=Path(td)/'pos.npy';np.save(posfile,p2d)
            result=extract_hdf5_acquisition(path,'binary_time_series_data',output,0,1,{2:wavelength_index,3:frame_index},
                            rate,wave,source_id,group_id,True,posfile)
    else:
        output=Path(output);output.parent.mkdir(exist_ok=True,parents=True)
        with h5py.File(path,'r') as fin,h5py.File(output,'w') as fout:
            raw=np.asarray(fin['binary_time_series_data'][:,:,wavelength_index,frame_index],dtype='float32')
            fout.create_dataset('raw_data',data=raw,compression='gzip')
            fout.create_dataset('detector_positions_m',data=p2d)
            fout.attrs.update({'sampling_rate_hz':rate,'acquisition_wavelength_nm':wave,
                               'synthetic':origin=='synthetic','source_id':source_id,
                               'group_id':group_id,'origin_status':origin,'experimental_provenance_user_attested':False})
        result=output
    with h5py.File(result,'r+') as f:
        f.attrs['original_sensor_plane']=('XYZ'[flat]+' constant; projected axes '+''.join('XYZ'[i] for i in kept))
        f.attrs['original_sensor_spread_m']=np.array(extent)
        f.attrs['import_method']='IPASC metadata-driven 2D coordinate projection, no geometric resampling'
        f.attrs['source_original_hdf5']=str(path)
        f.attrs['origin_status']=origin
    return result
