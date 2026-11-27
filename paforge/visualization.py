"""Dataset and generated-acquisition visual analytics for the local dashboard."""
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import json
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _read_npy_zip(z,name):
    with z.open(name) as f:return np.load(BytesIO(f.read()),allow_pickle=False)


def preview_invision_sample(archive_path,phantom_id,wavelength_nm,output_dir,device='invision'):
    """Render publisher reference, segmentation, optical maps and measured signals for InVision/TROPUS."""
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    stem=f'{phantom_id}_{int(wavelength_nm)}'
    if device not in ('invision','tropus'):raise ValueError('2D raw-data preview supports InVision and TROPUS; SVOT has no shared raw time series')
    paths={'reference':f'{device}/recon/{stem}.npy','labels':f'{device}/labels/{phantom_id}.npy',
           'mua':f'{device}/mua/{stem}.npy','musp':f'{device}/musp/{stem}.npy',
           'raw':f'{device}/raw/{stem}_ipasc.hdf5'}
    with ZipFile(archive_path) as z:
        names=set(z.namelist());missing=[v for v in paths.values() if v not in names]
        if missing:raise FileNotFoundError('Missing dataset assets: '+', '.join(missing))
        arrays={k:_read_npy_zip(z,v) for k,v in paths.items() if k!='raw'}
        with z.open(paths['raw']) as raw_stream:
            raw_bytes=raw_stream.read()
    with h5py.File(BytesIO(raw_bytes),'r') as f:
        if 'binary_time_series_data' not in f:raise ValueError('Raw IPASC HDF5 lacks binary_time_series_data')
        ds=f['binary_time_series_data'];data=np.asarray(ds[:,:,0,0] if ds.ndim==4 else ds[:],dtype=float)
        meta=f.get('meta_data')
        fs=float(meta['ad_sampling_rate'][()]) if meta and 'ad_sampling_rate' in meta else None
        signal_meta={'shape':list(data.shape),'sampling_rate_hz':fs,
                     'wavelength_nm':int(wavelength_nm),'source':f'OpenQPAIData {device} raw IPASC time series'}
    if data.ndim!=2:raise ValueError('Expected detector x time raw signal matrix')
    products=[]
    def save_image(name,a,title,cmap='magma',vmin=None,vmax=None):
        p=out/f'{name}.png';fig,ax=plt.subplots(figsize=(5,4));im=ax.imshow(a,cmap=cmap,origin='upper',vmin=vmin,vmax=vmax)
        ax.set_title(title);ax.axis('off');fig.colorbar(im,ax=ax,fraction=.046);fig.tight_layout();fig.savefig(p,dpi=145);plt.close(fig);products.append(str(p));return p
    save_image('dataset_reference',arrays['reference'],'Published InVision reconstruction')
    save_image('dataset_segmentation',arrays['labels'],'Manual material segmentation','tab10',0,max(3,int(np.max(arrays['labels']))))
    save_image('dataset_mua',arrays['mua'],'Mapped absorption μa')
    save_image('dataset_musp',arrays['musp'],'Mapped reduced scattering μs′')
    p=out/'dataset_raw_sinogram.png';fig,ax=plt.subplots(figsize=(9,4));im=ax.imshow(data,aspect='auto',origin='upper',cmap='RdBu_r')
    ax.set(title='Measured raw detector signals',xlabel='Time sample',ylabel='Detector');fig.colorbar(im,ax=ax,label='Recorded signal (native units)');fig.tight_layout();fig.savefig(p,dpi=145);plt.close(fig);products.append(str(p))
    p=out/'dataset_raw_traces.png';fig,ax=plt.subplots(figsize=(9,4));t=np.arange(data.shape[1])/(fs or 1.)*1e6
    for i in np.unique(np.linspace(0,data.shape[0]-1,min(8,data.shape[0])).astype(int)):
        ax.plot(t,data[i],lw=.8,label=f'Detector {i}')
    ax.set(title='Measured time-series examples',xlabel='Time (μs)' if fs else 'Sample index',ylabel='Signal (native units)');ax.legend(ncol=4,fontsize=7);fig.tight_layout();fig.savefig(p,dpi=145);plt.close(fig);products.append(str(p))
    report={'phantom_id':phantom_id,'device':device,'wavelength_nm':int(wavelength_nm),'reference_shape':list(arrays['reference'].shape),
        'segmentation_shape':list(arrays['labels'].shape),'material_labels':np.unique(arrays['labels']).astype(int).tolist(),
        'mua_shape':list(arrays['mua'].shape),'musp_shape':list(arrays['musp'].shape),'raw_signal':signal_meta,
        'assets':paths,'interpretation':'Reference image and labels are publisher-provided. Raw signals are measured acquisitions; plots do not calibrate their physical units.'}
    (out/'dataset_preview.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return products,report


def preview_generated_sample(sample_folder,output_dir=None):
    sample=Path(sample_folder);out=Path(output_dir or sample/'dashboard_visuals');out.mkdir(parents=True,exist_ok=True)
    with h5py.File(sample/'ground_truth.h5','r') as f:
        p0=f['initial_pressure_pa'][:];labels=f['labels'][:]
        mua=f['absorption_m_inv'][:] if 'absorption_m_inv' in f else None
        musp=f['scattering_reduced_m_inv'][:] if 'scattering_reduced_m_inv' in f else None
    with h5py.File(sample/'raw_signals.h5','r') as f:
        raw=f['raw_data'][:];fs=float(f.attrs['sampling_rate_hz'])
    with h5py.File(sample/'reconstruction.h5','r') as f:rec=f['image'][:]
    products=[]
    def image(name,a,title,cmap='magma'):
        p=out/f'{name}.png';fig,ax=plt.subplots(figsize=(5,4));im=ax.imshow(a,cmap=cmap,origin='lower');ax.set_title(title);ax.axis('off');fig.colorbar(im,ax=ax,fraction=.046);fig.tight_layout();fig.savefig(p,dpi=145);plt.close(fig);products.append(str(p))
    image('generated_initial_pressure',p0,'Generated initial pressure p₀')
    image('generated_material_labels',labels,'Generated material / anatomy labels','tab10')
    image('generated_reconstruction',rec,'PA-FORGE reconstructed image')
    p=out/'generated_raw_sinogram.png';fig,ax=plt.subplots(figsize=(9,4));im=ax.imshow(raw,aspect='auto',origin='upper',cmap='RdBu_r');ax.set(title='Generated raw detector signals',xlabel='Time sample',ylabel='Detector');fig.colorbar(im,ax=ax,label='Synthetic signal');fig.tight_layout();fig.savefig(p,dpi=145);plt.close(fig);products.append(str(p))
    p=out/'generated_raw_traces.png';fig,ax=plt.subplots(figsize=(9,4));t=np.arange(raw.shape[1])/fs*1e6
    for i in np.unique(np.linspace(0,raw.shape[0]-1,min(8,raw.shape[0])).astype(int)):ax.plot(t,raw[i],lw=.8,label=f'Detector {i}')
    ax.set(title='Generated time-series examples',xlabel='Time (μs)',ylabel='Synthetic signal');ax.legend(ncol=4,fontsize=7);fig.tight_layout();fig.savefig(p,dpi=145);plt.close(fig);products.append(str(p))
    p=out/'generated_distributions.png';fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    axes[0].hist(p0.ravel(),bins=50);axes[0].set_title('Initial pressure distribution')
    axes[1].hist(raw.ravel(),bins=50);axes[1].set_title('Raw signal distribution')
    axes[2].hist(rec.ravel(),bins=50);axes[2].set_title('Reconstruction intensity')
    for ax in axes:ax.set_ylabel('Count')
    fig.tight_layout();fig.savefig(p,dpi=145);plt.close(fig);products.append(str(p))
    if mua is not None:image('generated_mua',mua,'Generated absorption μa')
    if musp is not None:image('generated_musp',musp,'Generated reduced scattering μs′')
    return products


def preview_dis_archive(dis_zip,output_dir):
    """Plot DIS processed optical-property distributions without guessing units."""
    from zipfile import ZipFile
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    buckets={}
    inventory=[]
    with ZipFile(dis_zip) as z:
        files=[n for n in z.namelist() if n.lower().endswith('.npz') and '/processed/' in n.lower()]
        for name in files:
            with z.open(name) as stream, np.load(BytesIO(stream.read()),allow_pickle=False) as data:
                record={'file':name,'keys':list(data.files)}
                for key in data.files:
                    arr=np.asarray(data[key])
                    if not np.issubdtype(arr.dtype,np.number) or not arr.size or not np.isfinite(arr).all():continue
                    k=key.lower()
                    if any(t in k for t in ('mua','mu_a','absorption')):bucket='absorption'
                    elif any(t in k for t in ('musp','mu_s','mus','scattering')):bucket='scattering'
                    elif k in ('g','anisotropy','anisotropy_factor'):bucket='anisotropy'
                    elif any(t in k for t in ('wave','lambda','wl')):bucket='wavelength'
                    else:continue
                    buckets.setdefault(bucket,[]).append(arr.astype(float).ravel())
                inventory.append(record)
    products=[]
    for key,arrays in buckets.items():
        values=np.concatenate([a for a in arrays if a.size]) if arrays else np.array([])
        if not values.size:continue
        p=out/f'dis_{key}_distribution.png';fig,ax=plt.subplots(figsize=(7,4));ax.hist(values,bins=60)
        ax.set(title=f'DIS {key} distribution (native publisher units)',xlabel='Stored value (units require publisher confirmation)',ylabel='Count')
        fig.tight_layout();fig.savefig(p,dpi=145);plt.close(fig);products.append(str(p))
    report={'source':str(dis_zip),'processed_npz_count':len(files),'files':inventory,
            'recognized_numeric_properties':list(buckets),
            'warning':'Histograms show stored values only. They do not infer units, material mapping or calibrated optical coefficients.'}
    (out/'dis_visual_inventory.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return products,report


def preview_svot_slice(svot_zip,phantom_id,slice_index,output_dir):
    """Render a single labeled/reconstructed SVOT volume slice without unpacking the archive."""
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    paths={k:f'svot/{k}/{phantom_id}.npy' for k in ('labels','recon','mua','musp')}
    products=[];report={'phantom_id':phantom_id,'slice_index':int(slice_index),'source':'OpenQPAIData SVOT'}
    with ZipFile(svot_zip) as z:
        names=set(z.namelist())
        if paths['labels'] not in names:raise FileNotFoundError(paths['labels'])
        for kind,name in paths.items():
            if name not in names:continue
            with z.open(name) as stream:vol=np.load(stream,allow_pickle=False)
            if vol.ndim!=3:continue
            zi=max(0,min(int(slice_index),vol.shape[0]-1));sl=np.asarray(vol[zi])
            p=out/f'svot_{kind}_z{zi}.png';fig,ax=plt.subplots(figsize=(5,4));im=ax.imshow(sl,cmap='tab10' if kind=='labels' else 'magma',origin='lower')
            ax.set(title=f'SVOT {kind} · z={zi}');ax.axis('off');fig.colorbar(im,ax=ax,fraction=.046);fig.tight_layout();fig.savefig(p,dpi=145);plt.close(fig);products.append(str(p))
            report[kind]={'volume_shape':list(vol.shape),'slice_shape':list(sl.shape),'slice_min':float(np.min(sl)),'slice_max':float(np.max(sl))}
            del vol
    report['warning']='SVOT provides volumetric reconstructions/labels/optical maps; no raw SVOT time-series are shared.'
    (out/'svot_slice_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return products,report
