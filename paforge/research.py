"""Reproducible synthetic-only inverse/reconstruction experiments.

These are numerical baselines, *not* real patient experiments or a valid
sim-to-real generalization claim. Matched FDTD inverse is an 'inverse crime'
and explicitly identified as such in the CSV/JSON outputs.
"""
import csv
import json
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .config import SimConfig, NoiseConfig, preset
from .pipeline import simulate
from .acoustic import reconstruct_das,reconstruct_ubp
from .inverse import from_simulation


def metrics(truth,image,mask=None):
    """Shape/correlation and best-scale (oracle) error used for diagnostics."""
    a=np.asarray(truth,dtype=np.float64)
    b=np.asarray(image,dtype=np.float64)
    if mask is None: mask=np.ones_like(a,dtype=bool)
    a=a[mask].ravel();b=b[mask].ravel()
    if a.size<3:raise ValueError('Evaluation mask too small')
    if a.std()<1e-15 or b.std()<1e-15:
        corr=0.
    else:corr=float(np.corrcoef(a,b)[0,1])
    # Evaluation ONLY: scalar fit uses the known synthetic truth.
    b=np.maximum(b,0)
    scale=float(np.dot(a,b)/max(np.dot(b,b),1.e-18))
    nrmse=float(np.linalg.norm(a-scale*b)/max(np.linalg.norm(a),1.e-18))
    return {'pearson_r':corr,'oracle_scale_nrmse':nrmse}


def run_benchmark(output, anatomy='vascular', seeds=(7,13), grid=64,
                  time_steps=240, transducers=24, iterations=10):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rows=[];examples=[]
    for seed in seeds:
        for case in ('noise_off','gaussian_colored'):
            noise=NoiseConfig(enabled=case!='noise_off',
                engines=['gaussian','colored'],relative_std=.055)
            base=preset(anatomy)
            cfg=SimConfig.model_validate({**base.model_dump(),
                'seed':int(seed),'grid_size':int(grid),'time_steps':int(time_steps),
                'transducers':int(transducers),'detector_filter':False,
                'noise':noise.model_dump(),'reconstruct':False})
            t0=time.perf_counter()
            sim=simulate(cfg)
            elapsed_forward=time.perf_counter()-t0
            op=from_simulation(cfg,sim.tissue.sound_speed_m_s,sim.dt_s,sim.detectors_m)
            t0=time.perf_counter()
            das=reconstruct_das(sim.raw_signals,sim.dt_s,sim.detectors_m,cfg)
            elapsed_das=time.perf_counter()-t0
            ubp=reconstruct_ubp(sim.raw_signals,sim.dt_s,sim.detectors_m,cfg)
            t0=time.perf_counter()
            inverse,losses=op.reconstruct(sim.raw_signals,iterations=iterations,reg=.001)
            elapsed_inverse=time.perf_counter()-t0
            for method,img,elapsed in [('das',das,elapsed_das),('ubp',ubp,elapsed_das),('iterative_matched_fdtd',inverse,elapsed_inverse)]:
                scored=metrics(sim.tissue.initial_pressure_pa,img,sim.tissue.mask)
                rows.append({'anatomy':anatomy,'seed':seed,'case':case,'method':method,
                    'added_noise':case!='noise_off','forward_seconds':round(elapsed_forward,4),
                    'reconstruction_seconds':round(elapsed,4),**scored,
                    'experiment_type':'synthetic internal consistency; NOT sim-to-real',
                    'matched_inverse_crime':method=='iterative_matched_fdtd',
                    'grid':grid,'transducers':transducers,'time_steps':time_steps})
            if len(examples)<2:
                examples.append((case,sim.tissue.initial_pressure_pa.copy(),
                                 sim.raw_signals.copy(),ubp,inverse,
                                 sim.tissue.vessel_type_mask.copy()))
    with (output/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={'status':'synthetic-only numerical experiment; no experimental data used',
             'scope':'2D FDTD, uncalibrated illustrative optical model',
             'inverse_crime_warning':'iterative inverse shares exact forward operator; this is an internal consistency test, not generalization',
             'metric_warning':'oracle_scale_nrmse uses true phantom to align overall image scale: diagnostic only',
             'settings':{'anatomy':anatomy,'seeds':list(seeds),'grid':grid,'time_steps':time_steps,
                         'transducers':transducers,'iterations':iterations},
             'results':rows}
    (output/'results.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    fig,axs=plt.subplots(len(examples),5,figsize=(14,5.8),layout='constrained')
    if len(examples)==1:axs=axs[None,:]
    for axrow,(case,truth,traces,das,inv,types) in zip(axs,examples):
        images=(types,truth,traces,das,inv)
        titles=('Vessel classes','True initial pressure','Sensor traces','Derivative backprojection','Matched FDTD inverse')
        for i,(ax,img,title) in enumerate(zip(axrow,images,titles)):
            ax.imshow(img,origin='lower',aspect='auto' if i==2 else 'equal',
                      cmap='tab10' if i==0 else 'RdBu_r' if i==2 else 'magma')
            ax.set_title(f'{title}\n{case}',fontsize=8)
            ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle('PA-FORGE: SYNTHETIC-ONLY internal numerical baseline (NOT experimental validation)',fontsize=11)
    fig.savefig(output/'comparison.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    keys=list(dict.fromkeys((r['case'],r['method']) for r in rows))
    labels=[f'{c}\n{m.replace("iterative_matched_fdtd","iterative")}' for c,m in keys]
    vals=[float(np.mean([r['pearson_r'] for r in rows if (r['case'],r['method'])==key])) for key in keys]
    ax.bar(np.arange(len(vals)),vals,color=['#64748b' if 'das' in k[1] else '#13a58e' if 'ubp' in k[1] else '#d99638' for k in keys])
    ax.set_xticks(np.arange(len(vals)),labels,fontsize=8);ax.set_ylabel('Mean Pearson correlation with synthetic ground truth')
    ax.set_title('Synthetic reconstruction benchmark (matched inverse is an inverse-crime diagnostic)')
    fig.savefig(output/'metric_summary.png',dpi=160);plt.close(fig)
    return output,summary
