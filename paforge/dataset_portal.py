"""Safe per-dataset inventory and visual inspection, without conflating data roles.

Only data arrays are read; never run publisher code. Compressed >256-MB members
require user extraction outside the preview process. This avoids in-memory 7-GB
unpacks and prevents arbitrary file execution from untrusted archives.
"""
from __future__ import annotations
from collections import Counter
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
import json
import tarfile
import tempfile
import zipfile
import numpy as np

from .dataset_registry import describe

SUPPORTED={'.h5','.hdf5','.npy','.npz','.mat','.zip','.xz','.csv','.png','.jpg','.jpeg','.tif','.tiff'}
MAX_PREVIEW_MEMBER=256*1024*1024
MAX_ARRAY_ELEMENTS=4_000_000


def _file_list(path: Path) -> dict:
    if path.is_dir():
        paths=[x for x in path.rglob('*') if x.is_file()]
        return {'kind':'directory','total_files':len(paths),
                'items':[{'name':str(p.relative_to(path)),'bytes':p.stat().st_size} for p in paths[:300]]}
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            items=[{'name':v.filename,'bytes':v.file_size,'compressed_bytes':v.compress_size}
                   for v in z.infolist() if not v.is_dir()]
        return {'kind':'zip','total_files':len(items),'items':items[:400],
                'note':'Inventory only. Select a known numeric member for preview; no bulk extraction.'}
    if path.name.lower().endswith('.tar.xz'):
        with tarfile.open(path,mode='r:xz') as t:
            items=[{'name':v.name,'bytes':v.size} for v in t if v.isfile()]
        return {'kind':'tar.xz','total_files':len(items),'items':items[:400],
                'note':'Compressed per-subject PAISKINTONE archive; no automatic raw-data axis assumptions.'}
    return {'kind':'file','total_files':1,'items':[{'name':path.name,'bytes':path.stat().st_size}]}


def _walk_hdf5(path: Path,limit=180):
    import h5py
    entries=[]
    with h5py.File(path,'r') as f:
        def visit(name,obj):
            if len(entries)>=limit:return
            if isinstance(obj,h5py.Dataset):
                entries.append({'key':name,'shape':list(obj.shape),'dtype':str(obj.dtype),
                    'attrs':{str(k):str(v)[:200] for k,v in list(obj.attrs.items())[:8]}})
        f.visititems(visit)
        attrs={str(k):str(v)[:250] for k,v in list(f.attrs.items())[:32]}
    return entries,attrs


def inspect_local_file(source_id: str,path: str|Path) -> dict:
    meta=describe(source_id)
    p=Path(path).expanduser()
    if not p.exists():raise FileNotFoundError(f'Cannot find local source {p}')
    listing=_file_list(p)
    result={'source_id':source_id,'source_title':meta['title'],'path':str(p),'listing':listing,
            'scientific_lanes':meta['lanes'],'prohibited_conversions':meta['excluded'],
            'integration_status':meta['status']}
    if p.is_file() and p.suffix.lower() in ('.h5','.hdf5'):
        result['datasets'],result['hdf5_attrs']=_walk_hdf5(p)
        if source_id=='cambridge-patato':
            keys={x['key'] for x in result['datasets']}
            result['patato_schema']={k:([x for x in result['datasets'] if x['key']==k] or [None])[0]
                                      for k in ('raw_data','recons','GEOMETRY','wavelengths')}
            result['patato_note']='PATATO raw_data may have multiple observation/wavelength dimensions; explicit axis inspection is required for reconstruction.'
    if p.is_file() and p.suffix.lower()=='.npz':
        with np.load(p,allow_pickle=False) as data:
            result['arrays']=[{'key':k,'shape':list(data[k].shape),'dtype':str(data[k].dtype)} for k in data.files[:120]]
    if p.is_file() and p.suffix.lower()=='.mat':
        try:
            import scipy.io
            result['arrays']=[{'key':k,'shape':list(shape),'dtype':str(dtype)} for k,shape,dtype in scipy.io.whosmat(p)]
        except (NotImplementedError,ValueError):
            result['note']='MAT v7.3 is HDF5; inspect it using the HDF5 editor/reader instead.'
    return result


@contextmanager
def _source_member(path:Path,member:str|None):
    if member is None or not str(member).strip():
        yield path
        return
    member=str(member)
    if '..' in Path(member).parts or Path(member).is_absolute():
        raise ValueError('Archive member must be a relative publisher path')
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            info=z.getinfo(member)
            if info.file_size>MAX_PREVIEW_MEMBER:
                raise ValueError('Archive member is too large for in-memory preview. Extract a single file to disk first.')
            with tempfile.TemporaryDirectory() as temp:
                name=Path(temp)/Path(member).name
                with z.open(info) as src,name.open('wb') as dst:
                    import shutil
                    shutil.copyfileobj(src,dst)
                yield name
        return
    if path.name.lower().endswith('.tar.xz'):
        with tarfile.open(path,'r:xz') as t:
            info=t.getmember(member)
            if not info.isfile() or info.size>MAX_PREVIEW_MEMBER:
                raise ValueError('Only small regular members can be previewed from TAR.XZ; extract larger raw files separately.')
            with tempfile.TemporaryDirectory() as temp:
                name=Path(temp)/Path(member).name
                with t.extractfile(info) as src,name.open('wb') as dst:
                    import shutil
                    shutil.copyfileobj(src,dst)
                yield name
        return
    raise ValueError('File is not a supported compressed archive')


def _read_2d(path:Path,key:str,index:int):
    import h5py
    if index<0:raise ValueError('Sample index must be >=0')
    suffix=path.suffix.lower()
    if suffix in ('.h5','.hdf5'):
        with h5py.File(path,'r') as f:
            if key not in f or not isinstance(f[key],h5py.Dataset):
                raise ValueError(f'HDF5 numeric key {key!r} not present')
            ds=f[key]
            if ds.ndim<2 or ds.ndim>6:raise ValueError('Expected 2D-6D array for preview')
            # Select extra dimensions explicitly by index; do not assume the
            # resulting matrix has detector or image semantics.
            selection=(min(index,ds.shape[0]-1),)*(ds.ndim-2)+(slice(None),slice(None))
            if ds.ndim==2:selection=(slice(None),slice(None))
            shape=ds.shape[-2:]
            if np.prod(shape)>MAX_ARRAY_ELEMENTS:raise ValueError('2D preview too large; select a cropped subset')
            arr=np.asarray(ds[selection])
            return arr,{'full_shape':list(ds.shape),'selected_slice':str(selection)}
    if suffix=='.npy':
        arr=np.load(path,allow_pickle=False,mmap_mode='r')
        if arr.ndim<2 or arr.ndim>6:raise ValueError('Not a 2D-6D numerical array')
        selection=(min(index,arr.shape[0]-1),)*(arr.ndim-2)+(slice(None),slice(None))
        if arr.ndim==2:selection=(slice(None),slice(None))
        if np.prod(arr.shape[-2:])>MAX_ARRAY_ELEMENTS:raise ValueError('Preview exceeds element budget')
        return np.asarray(arr[selection]),{'full_shape':list(arr.shape),'selected_slice':str(selection)}
    if suffix=='.npz':
        with np.load(path,allow_pickle=False) as file:
            if key not in file:raise ValueError('NPZ key not present; inspect keys first')
            arr=file[key]
        if arr.ndim<2 or arr.ndim>6 or np.prod(arr.shape[-2:])>MAX_ARRAY_ELEMENTS:
            raise ValueError('NPZ array is not a supported sized 2D-6D matrix')
        selection=(min(index,arr.shape[0]-1),)*(arr.ndim-2)+(slice(None),slice(None))
        return np.asarray(arr[selection]),{'full_shape':list(arr.shape),'selected_slice':str(selection)}
    if suffix=='.mat':
        from scipy.io import loadmat
        if not key or key.startswith('_'):raise ValueError('Select a named MAT variable')
        a=loadmat(path,variable_names=[key])[key]
        if a.ndim<2 or a.ndim>6 or np.prod(a.shape[-2:])>MAX_ARRAY_ELEMENTS:
            raise ValueError('MAT variable is not a supported sized matrix')
        selection=(min(index,a.shape[0]-1),)*(a.ndim-2)+(slice(None),slice(None))
        return np.asarray(a[selection]),{'full_shape':list(a.shape),'selected_slice':str(selection)}
    raise ValueError('Supported preview inputs: HDF5/NPY/NPZ/MAT')


def visualize_array(source_id:str,path:str|Path,key:str,output_dir:str|Path,
                    index:int=0,member:str|None=None,interpretation:str='image') -> tuple[list[str],dict]:
    """2D image or measured waveform plot, explicitly selected by the user."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    describe(source_id)
    if interpretation not in ('image','signal'):
        raise ValueError('Choose image or signal semantics according to publisher documentation')
    p=Path(path);out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    with _source_member(p,member) as selected:
        arr,selection=_read_2d(selected,key,int(index))
    if not np.issubdtype(arr.dtype,np.number):raise ValueError('Only numeric arrays are previewed')
    arr=np.asarray(arr)
    if np.iscomplexobj(arr):
        arr=np.abs(arr);complex_note='Displayed magnitude of complex data'
    else:complex_note=None
    if not np.isfinite(arr).all():raise ValueError('NaN or infinite values in selected array')
    figures=[]
    if interpretation=='signal':
        fig,ax=plt.subplots(figsize=(8,4))
        im=ax.imshow(arr,aspect='auto',origin='upper',cmap='RdBu_r')
        ax.set(xlabel='Time-sample index (axis assumed by user)',ylabel='Detector / trace index (assumed)',
               title='Selected measured or simulated signal matrix')
        fig.colorbar(im,ax=ax)
        outpath=out/'source_sinogram.png';fig.tight_layout();fig.savefig(outpath,dpi=130);plt.close(fig)
        figures.append(str(outpath))
        fig,ax=plt.subplots(figsize=(8,3))
        for i in np.unique(np.linspace(0,arr.shape[0]-1,min(6,arr.shape[0])).astype(int)):
            ax.plot(arr[i],lw=.8,label=f'row {i}')
        ax.legend(ncol=3,fontsize=7);ax.set(xlabel='Sample index',ylabel='Recorded units (uncalibrated)',title='Selected signal traces')
        outpath=out/'source_traces.png';fig.tight_layout();fig.savefig(outpath,dpi=130);plt.close(fig)
        figures.append(str(outpath))
    else:
        fig,ax=plt.subplots(figsize=(5,4))
        im=ax.imshow(arr,cmap='magma',origin='lower')
        fig.colorbar(im,ax=ax);ax.set(title='Publisher/user image or map • not reconstructed by PA-FORGE');ax.axis('off')
        outpath=out/'source_image.png';fig.tight_layout();fig.savefig(outpath,dpi=130);plt.close(fig)
        figures.append(str(outpath))
    summary={'source_id':source_id,'original_file':str(p),'archive_member':member,
             'array_key':key,'interpretation_user_selected':interpretation,'shape':list(arr.shape),
             'array_min':float(np.min(arr)),'array_max':float(np.max(arr)),
             'selection':selection,'complex_note':complex_note,
             'provenance':'Original source data, inspected only; no domain inference, reconstruction or learned synthesis.'}
    (out/'preview_manifest.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return figures,summary


def extract_skin_pairs(source_h5:str|Path,out_dir:str|Path,limit:int=16,
                       mua_key='mua',fluence_key='fluence',pressure_key='IP',signal_key='sinogram',
                       sample_axis=0)->dict:
    """Separate SYNTHETIC optical/truth/signal pairs; never rebrand as experimental.

    Schema-gated and lazy: does not load 100-GB shards into RAM. Export is
    deliberately a *paired reference corpus*, not phantom-material labels.
    """
    import h5py
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    source=Path(source_h5);out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    if limit<1 or limit>5000:raise ValueError('Select 1..5000 pairs per export')
    if sample_axis!=0:
        raise ValueError('This adapter has only verified sample-axis=0; use a validated schema converter for other layouts')
    with h5py.File(source,'r') as f:
        keys=[mua_key,fluence_key,pressure_key,signal_key]
        if any(k not in f for k in keys):raise ValueError('Required keys missing; inspect the actual HDF5 hierarchy first')
        arrays=[f[k] for k in keys]
        if any(ds.ndim!=3 for ds in arrays):
            raise ValueError('Verified paired layout requires [sample,spatial_1,spatial_2] for optical maps and [sample,detector,time] for sinogram')
        count=min(int(arrays[0].shape[0]),limit)
        if any(int(ds.shape[0])<count for ds in arrays):
            raise ValueError('Paired sample axis has inconsistent lengths')
        if not all(ds.shape[-2:]==arrays[0].shape[-2:] for ds in arrays[:3]):
            raise ValueError('mua, fluence and IP spatial dimensions must align')
        products=[]
        for i in range(count):
            values=[np.asarray(ds[i],dtype=np.float32) for ds in arrays]
            if any(a.size>MAX_ARRAY_ELEMENTS or not np.isfinite(a).all() for a in values):
                raise ValueError('Invalid/oversized sample or nonfinite optical/pressure/signal values')
            p=out/f'synthetic_skin_pair_{i:05d}.npz'
            np.savez_compressed(p,mua=values[0],fluence=values[1],IP=values[2],sinogram=values[3],
                                source_id='synthetic-skin',sample_index=np.int64(i))
            products.append(str(p))
            if i==0:
                fig,axes=plt.subplots(2,2,figsize=(10,8))
                for ax,a,k in zip(axes.flat,values,('mua','fluence','IP','simulated sinogram')):
                    im=ax.imshow(a,aspect='auto',origin='lower');ax.set_title(k);fig.colorbar(im,ax=ax)
                fig.tight_layout();fig.savefig(out/'paired_preview.png',dpi=135);plt.close(fig)
    report={'source':str(source),'source_id':'synthetic-skin','pairs_exported':count,
            'output_dir':str(out),'paired_keys':[mua_key,fluence_key,pressure_key,signal_key],
            'output_semantics':'SYNTHETIC 2D optical/initial pressure/sinogram pairs. Not measured data. Not categorical material segmentation.',
            'assumption':'Sample axis zero is source layout; inspect publisher before processing entire shard.'}
    (out/'paired_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def normalize_source_signals(source_id:str,path:str|Path,key:str,out_h5:str|Path,
                             detector_axis:int,time_axis:int,fixed_indices:dict[int,int],
                             sampling_rate_hz:float,wavelength_nm:float,
                             measured_positions_npy:str|Path,group_id:str,
                             attest_experimental:bool)->dict:
    """Only a measured-signal lane can enter the experimental reconstruction engine.

    Array axes and physical metadata are provided or verified by a researcher;
    no guessed positions, sampling clock, source units or experimental status.
    """
    desc=describe(source_id)
    if not any(x.startswith('measured_signal_') or x=='reconstruction_if_raw_metadata_verified'
               for x in desc['lanes']):
        raise ValueError(f'{source_id} is not an experimental raw-signal lane; optical/image/simulated sources are excluded')
    if not measured_positions_npy or not Path(measured_positions_npy).is_file():
        raise ValueError('An independently measured [detectors,3] coordinate NPY is required')
    positions=np.load(measured_positions_npy,allow_pickle=False)
    if positions.ndim!=2 or positions.shape[1]!=3 or not np.isfinite(positions).all():
        raise ValueError('Detector geometry NPY must be finite [detectors,3] in metres')
    if np.ptp(positions[:,2])>0.002:
        raise ValueError('The 2D experimental solver requires detector coordinates in the first two columns and a planar third dimension')
    from .experimental_import import extract_hdf5_acquisition
    result=extract_hdf5_acquisition(path,key,out_h5,int(detector_axis),int(time_axis),fixed_indices,
                     float(sampling_rate_hz),float(wavelength_nm),source_id,group_id,
                     bool(attest_experimental),measured_positions_npy)
    return {'native_h5':str(result),'source_id':source_id,'source_file':str(path),
            'detector_geometry':str(measured_positions_npy),'experimental_attestation':bool(attest_experimental),
            'label':'Measured raw acquisition only after explicit metadata verification; independent device calibration still required'}
