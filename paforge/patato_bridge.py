"""Publisher-specification-grounded PATATO HDF5 → PA-FORGE 2D acquisition.

See https://patato.readthedocs.io/en/latest/dev/02_fileformatspecification.html
No invented detectors, sample rate or coordinates. 3D acquisitions are rejected
unless detector positions are (within tolerance) planar.
"""
from __future__ import annotations
from pathlib import Path
import json
import h5py
import numpy as np


def normalize_patato(path,out_h5,frame=0,wavelength_index=0,origin='unknown',
                     attest_experimental=False,group_id='unknown') -> dict:
    if origin not in ('experimental','unknown','synthetic'):
        raise ValueError('origin must be experimental/unknown/synthetic')
    if origin=='experimental' and not attest_experimental:
        raise ValueError('Confirm the publisher identifies this specific source as an experimental acquisition')
    src=Path(path);dst=Path(out_h5)
    with h5py.File(src,'r') as f:
        if any(k not in f for k in ('raw_data','GEOMETRY','wavelengths')):
            raise ValueError('Missing PATATO mandatory /raw_data, /GEOMETRY, or /wavelengths')
        raw=f['raw_data'];geometry=np.asarray(f['GEOMETRY'],dtype=np.float64)
        wavelengths=np.asarray(f['wavelengths'],dtype=float).ravel()
        fs=float(f.attrs.get('fs',0))
        if raw.ndim!=4:
            raise ValueError('Expected PATATO raw_data [frame, wavelength, detector, time], per publisher specification')
        nf,nw,nd,nt=map(int,raw.shape)
        if not (0<=int(frame)<nf and 0<=int(wavelength_index)<nw):
            raise ValueError('Frame or wavelength index out of bounds')
        if nt<32 or nd<2 or not np.isfinite(fs) or fs<=0:
            raise ValueError('Invalid raw_data detector/time shape or missing positive sample rate fs')
        if len(wavelengths)!=nw or not np.all(np.isfinite(wavelengths)):
            raise ValueError('Wavelength list does not match raw_data wavelength axis')
        if geometry.shape!=(nd,3) or not np.isfinite(geometry).all():
            raise ValueError('Geometry must be finite measured [detectors,3] metres')
        ext=np.ptp(geometry,axis=0)
        axis=int(np.argmin(ext))
        if ext[axis]>0.002:
            raise ValueError('Detector geometry is significantly 3D: 2D reconstruction unsupported')
        keep=[i for i in range(3) if i!=axis]
        positions=np.column_stack([geometry[:,keep[0]],geometry[:,keep[1]],np.zeros(nd)])
        signals=np.asarray(raw[int(frame),int(wavelength_index)],dtype=np.float32)
        if not np.isfinite(signals).all():raise ValueError('Nonfinite measured raw traces')
        available=[]
        if 'recons' in f:
            def visit(name,obj):
                if isinstance(obj,h5py.Dataset) and obj.ndim>=4:available.append('recons/'+name)
            f['recons'].visititems(visit)
        date=f.attrs.get('date','')
        label=f.attrs.get('name','')
    dst.parent.mkdir(parents=True,exist_ok=True)
    with h5py.File(dst,'w') as f:
        f.create_dataset('raw_data',data=signals,compression='gzip')
        f.create_dataset('detector_positions_m',data=positions)
        f.attrs.update({'sampling_rate_hz':fs,'acquisition_wavelength_nm':float(wavelengths[int(wavelength_index)]),
            'synthetic':origin=='synthetic','origin_status':origin,
            'source_id':'Cambridge_PATATO_'+src.stem,'group_id':str(group_id),
            'original_file':str(src),'original_frame_index':int(frame),'original_wavelength_index':int(wavelength_index),
            'original_detector_plane':'XYZ'[axis]+' approximately constant',
            'original_coordinate_axes':'XYZ'[keep[0]]+'XYZ'[keep[1]],
            'experimental_provenance_user_attested':bool(attest_experimental),
            'import_method':'PATATO documented array axes and geometric plane projection',
            'publisher_date':str(date),'scan_name':str(label)})
    report={'normalized_acquisition':str(dst),'source_file':str(src),'raw_shape':[nf,nw,nd,nt],
            'sampling_frequency_hz':fs,'wavelength_nm':float(wavelengths[int(wavelength_index)]),
            'selected_frame':int(frame),'selected_wavelength_index':int(wavelength_index),
            'original_geometry_axis_spread_m':ext.tolist(),'selected_plane':'XYZ'[axis],
            'reference_recon_keys':available,'origin':origin,
            'caveat':'Uncalibrated 2D instrument response/time-zero; reference registration not established.'}
    manifest=dst.with_suffix('.json');manifest.write_text(json.dumps(report,indent=2),encoding='utf-8')
    report['manifest_path']=str(manifest)
    return report


def extract_patato_reference(path,out_npy,frame=0,wavelength_index=0,recon_key='recons/Reference Backprojection/0') -> dict:
    """Extract an existing publisher reference, not a generated reconstruction."""
    with h5py.File(path,'r') as f:
        if recon_key not in f or not isinstance(f[recon_key],h5py.Dataset):
            raise ValueError('Reference reconstruction key absent; inspect available /recons methods first')
        ds=f[recon_key]
        if ds.ndim!=5:
            raise ValueError('Expected PATATO recon [frames,wavelengths,nx,ny,nz]')
        if not (0<=int(frame)<ds.shape[0] and 0<=int(wavelength_index)<ds.shape[1]):
            raise ValueError('Reconstruction frame/wavelength out of range')
        block=np.asarray(ds[int(frame),int(wavelength_index)],dtype=float)
        non_single=[n for n in block.shape if n!=1]
        if len(non_single)!=2:raise ValueError('Reference is volumetric (>2D); do not silently take a 3D slice')
        img=np.squeeze(block)
        if not np.isfinite(img).all():raise ValueError('Non-finite reconstruction')
    dest=Path(out_npy);dest.parent.mkdir(parents=True,exist_ok=True)
    np.save(dest,img.astype('float32'))
    return {'reference_npy':str(dest),'shape':list(img.shape),'source_key':recon_key,
            'provenance':'Publisher PATATO reconstruction; display/registration must be verified'}
