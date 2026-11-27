"""One-click apples-to-apples reconstruction comparisons from canonical HDF5.

Uses same measured/synthetic detector positions and traces, exports qualitative
images and timed manifests. Metrics against synthetic truth are supplied only
when the user supplies a same-grid truth image (never guessed).
"""
from pathlib import Path
import json,time
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .reconstruction_engine import Acquisition,reconstruct,METHODS,SPECS


def _acquisition(path,grid_size,fov_m,sound_speed_m_s):
    with h5py.File(path,'r') as f:
        if 'raw_data' not in f or 'detector_positions_m' not in f:
            raise ValueError('Canonical HDF5 needs raw_data and verified detector_positions_m; do not fabricate geometry')
        traces=f['raw_data'][:]
        pos=f['detector_positions_m'][:]
        fs=float(f.attrs.get('sampling_rate_hz',0))
        synthetic=bool(f.attrs.get('synthetic',False))
        source=str(f.attrs.get('source_id',f.attrs.get('source','unspecified')))
        wavelength_nm=f.attrs.get('acquisition_wavelength_nm',None)
    if not np.isfinite(fs) or fs<=0:raise ValueError('Positive sampling_rate_hz required')
    return Acquisition(traces,1/fs,pos,float(fov_m),int(grid_size),float(sound_speed_m_s),wavelength_nm=float(wavelength_nm) if wavelength_nm is not None else None),source,synthetic


def run_comparison(input_h5,output,methods,grid_size=64,fov_m=.1,sound_speed_m_s=1500,
                   iterations=8,reg=.001,time_decimate=4,weights=None,truth=None):
    if isinstance(methods,str):methods=[k.strip() for k in methods.split(',') if k.strip()]
    if not methods or len(methods)>12:raise ValueError('Choose 1..12 methods for one comparison run')
    if len(methods)!=len(set(methods)):raise ValueError('Duplicate algorithms selected')
    unknown=set(methods)-set(METHODS)
    if unknown:raise ValueError(f'Unknown methods: {sorted(unknown)}')
    acq,source,synthetic=_acquisition(input_h5,grid_size,fov_m,sound_speed_m_s)
    truth_img=None
    if truth:
        with h5py.File(truth,'r') as f:
            if 'initial_pressure_pa' not in f:raise ValueError('Truth file needs synthetic initial_pressure_pa')
            truth_img=np.asarray(f['initial_pressure_pa'],dtype=float)
        if truth_img.shape!=(int(grid_size),)*2:raise ValueError('Truth must match reconstruction grid (no silent resize)')
        if not synthetic:raise ValueError('Do not score experimental data against an unrelated synthetic truth file')
    folder=Path(output);folder.mkdir(parents=True,exist_ok=True)
    records=[];images={}
    for method in methods:
        start=time.perf_counter()
        try:
            image=reconstruct(acq,method,iterations=int(iterations),reg=float(reg),time_decimate=int(time_decimate),weights=weights)
        except Exception as e:
            records.append({'method':method,'status':'failed','exception_type':type(e).__name__,'detail':str(e),
                            'seconds':round(time.perf_counter()-start,4)})
            continue
        dt=time.perf_counter()-start
        np.save(folder/f'{method}.npy',image)
        plt.imsave(folder/f'{method}.png',image,cmap='magma')
        report={'method':method,'category':SPECS[method].category,'model':SPECS[method].model,
                'status':'ok','seconds':round(dt,5),'image_min':float(np.min(image)),'image_max':float(np.max(image))}
        if truth_img is not None:
            # Scale-independent metric only when explicitly given a known truth.
            img=np.asarray(image,dtype=float);p=np.asarray(truth_img,dtype=float)
            denom=float(np.vdot(img,img))
            alpha=float(np.vdot(img,p)/denom) if denom>0 else 0.
            err=np.sqrt(np.mean((alpha*img-p)**2))
            report.update({'best_linear_scale_to_truth':alpha,'scaled_rmse':float(err),
                           'scaled_nrmse':float(err/max(np.max(p)-np.min(p),1e-12)),
                           'note':'A posteriori scale-fitting uses ground truth: diagnostic only, not quantitative calibration'})
        records.append(report);images[method]=image
    # Diagnostic raw time trace and sinogram are always tied to the same input.
    fig,ax=plt.subplots(1,2,figsize=(10,3),layout='constrained')
    ax[0].imshow(acq.traces,aspect='auto',cmap='RdBu_r');ax[0].set_title('Input detector sinogram')
    ax[0].set_xlabel('Time sample');ax[0].set_ylabel('Detector')
    ax[1].plot(np.arange(acq.traces.shape[1])*acq.dt_s*1e6,acq.traces[0]);ax[1].set_title('Measured/simulated detector trace 0')
    ax[1].set_xlabel('Time (μs)');ax[1].set_ylabel('Signal (unvalidated units)')
    fig.savefig(folder/'input_signals.png',dpi=130);plt.close(fig)
    if images:
        cols=min(4,len(images));rows=int(np.ceil(len(images)/cols))
        fig,axs=plt.subplots(rows,cols,figsize=(3.2*cols,3.6*rows),squeeze=False,layout='constrained')
        for ax in axs.flat:ax.axis('off')
        for ax,(name,img) in zip(axs.flat,images.items()):
            ax.imshow(img,cmap='magma');ax.set_title(name);ax.axis('off')
        fig.savefig(folder/'comparison.png',dpi=130);plt.close(fig)
    report={'input':str(input_h5),'source_id':source,'synthetic':synthetic,
            'n_detectors':int(acq.traces.shape[0]),'n_times':int(acq.traces.shape[1]),
            'time_step_s':acq.dt_s,'grid_size':int(grid_size),'fov_m':float(fov_m),
            'sound_speed_m_s':float(sound_speed_m_s),'methods':records,
            'caveat':'Research-grade exploratory 2D; no calibrated experimental accuracy or guaranteed PATbox equivalence'}
    (folder/'comparison_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report
