"""Inspect / import experimental acquisitions WITHOUT assuming vendor axis layout.

A mapping to a specific OADAT, PAISKINTONE, etc. archive MUST be validated
on a downloaded sample before it can be represented as a working adapter.
"""
from pathlib import Path
import numpy as np
import h5py
from scipy.io import loadmat


def inspect_file(filename):
    p=Path(filename)
    if not p.is_file():raise FileNotFoundError(p)
    if p.suffix.lower() in (".h5",".hdf5"):
        with h5py.File(p) as f:
            arrays={}
            def visit(name,obj):
                if isinstance(obj,h5py.Dataset):
                    arrays[name]={"shape":tuple(obj.shape),"dtype":str(obj.dtype)}
            f.visititems(visit)
            return arrays
    if p.suffix.lower() in (".mat",".npz"):
        if h5py.is_hdf5(p):return inspect_hdf5_mat(p)
        if p.suffix.lower()==".mat":
            data=loadmat(p)
            return {k:{"shape":v.shape,"dtype":str(v.dtype)} for k,v in data.items() if hasattr(v,"shape") and not k.startswith("__")}
        with np.load(p,allow_pickle=False) as data:
            return {k:{"shape":data[k].shape,"dtype":str(data[k].dtype)} for k in data.files}
    if p.suffix.lower()==".npy":
        a=np.load(p,mmap_mode="r",allow_pickle=False)
        return {"data":{"shape":a.shape,"dtype":str(a.dtype)}}
    raise ValueError("Supported extensions: .h5, .hdf5, .mat, .npz, .npy")


def inspect_hdf5_mat(p):
    with h5py.File(p) as f:
        output={}
        def visitor(name,obj):
            if isinstance(obj,h5py.Dataset):output[name]={"shape":tuple(obj.shape),"dtype":str(obj.dtype)}
        f.visititems(visitor)
        return output


def read_array(path,key):
    p=Path(path)
    if h5py.is_hdf5(p):
        with h5py.File(p) as f:
            if key not in f:raise KeyError(key)
            a=f[key][:]
    elif p.suffix==".npy":
        if key!="data":raise KeyError(".npy key must be data")
        a=np.load(p,allow_pickle=False)
    elif p.suffix==".npz":
        with np.load(p,allow_pickle=False) as f:a=f[key]
    elif p.suffix==".mat":
        a=loadmat(p)[key]
    else:raise ValueError("Unsupported input")
    if not np.issubdtype(a.dtype,np.number) or not np.isrealobj(a):raise ValueError("Expected real-valued numeric array")
    if not np.isfinite(a).all():raise ValueError("Non-finite array")
    return np.asarray(a)


def import_measurement(path,signal_key,output,detector_axis,sampling_rate_hz,wavelength_nm,
                       detector_positions_file=None, source_name="unverified-source"):
    """Import *only* raw 2D [detector, sample] matrix; explicit axes and metadata.

    Does NOT infer sampling rate or positions; positions missing -> flag unknown.
    """
    if detector_axis not in (0,1):raise ValueError("detector_axis must be 0 or 1")
    if sampling_rate_hz<=0:raise ValueError("Positive sampling rate required")
    a=read_array(path,signal_key)
    if a.ndim!=2:raise ValueError("Expected exactly 2D signal array; select frame explicitly")
    if detector_axis==1:a=a.T
    if a.shape[0]<2 or a.shape[1]<8:raise ValueError("Unreasonable detector/time dimensions")
    positions=None
    if detector_positions_file:
        positions=np.asarray(np.load(detector_positions_file,allow_pickle=False))
        if positions.shape!=(a.shape[0],3):raise ValueError("Positions must be [detector,xyz] in metres")
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with h5py.File(output,"w") as f:
        f.attrs["format"]="PA-FORGE experimental import (native, not certified IPASC)"
        f.attrs["source_file_name"]=Path(path).name
        f.attrs["source_dataset_key"]=signal_key
        f.attrs["source_name"]=source_name
        f.attrs["source_coordinates_verified"]=positions is not None
        f.attrs["sampling_rate_hz"]=float(sampling_rate_hz)
        f.attrs["acquisition_wavelength_nm"]=float(wavelength_nm)
        f.create_dataset("raw_data",data=a.astype(np.float32),compression="gzip")
        if positions is not None:f.create_dataset("detector_positions_m",data=positions)
    return output
