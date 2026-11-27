"""Strict, non-destructive inspection of experimental HDF5 before decoding.

No hidden assumptions on IPASC shape/metadata: the source file must be inspected
and the mapping from native axes to our standardized traces must be declared.
"""
from __future__ import annotations
from pathlib import Path
import h5py
import numpy as np
import json

SMALL_DATASETS = {
  'digital_twin_ipasc': {
    'record':'https://zenodo.org/records/15388429',
    'filename':'ipasc_file.hdf5','size_mb_approx':4.8,
    'md5':'f1d17e900f10341a05f6656fff194fdf',
    'intended_use':'IPASC metadata/shape and reconstruction interoperability; not full twin benchmark'},
  'openqpai_dis': {
    'record':'https://zenodo.org/records/14044853',
    'filename':'DIS.zip','size_mb_approx':132.1,
    'md5':'3ad5e85f91aed6fbaad978d8a8d94515',
    'intended_use':'independently measured optical absorption/scattering; NOT raw acoustic traces'},
  'openqpai_invision': {
    'record':'https://zenodo.org/records/14044853',
    'filename':'invision.zip','size_mb_approx':238.1,
    'md5':'a3ab81d0256feb2ffa8412b3ed40efa6',
    'intended_use':'experimental phantom raw acquisition; requires verified shape/geometry and annotation'},
}


def hdf5_inventory(path, max_nodes=200):
    path=Path(path)
    if not path.is_file(): raise FileNotFoundError(path)
    nodes=[]
    def val(v):
        if isinstance(v,(str,int,float,bool,np.integer,np.floating)):
            return v.item() if isinstance(v,(np.integer,np.floating)) else v
        if isinstance(v,bytes):return v.decode(errors='replace')[:200]
        if isinstance(v,np.ndarray):
            if v.size<=8 and v.dtype.kind in 'ifubS':return str(v.tolist())[:200]
            return f'array(shape={v.shape},dtype={v.dtype})'
        return str(v)[:200]
    with h5py.File(path) as f:
        def read_attrs(node):
            return {str(k):val(node.attrs[k]) for k in list(node.attrs)[:30]}
        nodes.append(dict(path='/', type='group',attrs=read_attrs(f)))
        def visit(key,item):
            if len(nodes)>=max_nodes:return
            props=dict(path='/'+key,type='dataset' if isinstance(item,h5py.Dataset) else 'group',
                attrs=read_attrs(item))
            if isinstance(item,h5py.Dataset):
                props.update(shape=list(item.shape),dtype=str(item.dtype), size_bytes=int(item.size*item.dtype.itemsize))
            nodes.append(props)
        f.visititems(visit)
    return {'file':str(path),'size_bytes':path.stat().st_size,'nodes':nodes,'truncated':len(nodes)>=max_nodes,
            'is_verified_experimental':False,'note':'Inventory only: provenance and physical metadata still require independent attestation'}
