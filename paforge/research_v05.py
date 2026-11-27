"""PA-FORGE v0.5: reproducible 2D acoustic model-mismatch benchmark.

This experiment has NO measured physical data. It compares inverse methods when
reconstruction assumptions differ from the numerical data-generating operator.
The same discretization remains in use: it does not prove sim-to-real transfer.
"""
from __future__ import annotations
from pathlib import Path
import csv
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .config import SimConfig, NoiseConfig, preset
from .pipeline import simulate
from .acoustic import reconstruct_ubp, reconstruct_das
from .inverse import from_simulation, ForwardOperator
from .research import metrics


def mismatch_benchmark(output, anatomy='vascular', seeds=(17, 29), grid=48,
                       time_steps=165, transducers=20, iterations=6,
                       assumed_sound_m_s=1630.0):
    """Compare DAS, derivative BP, matched/mismatched inversion on identical simulated data.

    A known synthetic phantom and forward speed map generate the traces; the
    mismatched solver assumes a fixed homogeneous speed. Report matched inverse
    separately because it shares the data-generating operator ('inverse crime').
    """
    if not (1200 <= assumed_sound_m_s <= 1900):
        raise ValueError('Assumed homogeneous speed must lie in [1200,1900] m/s')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    first = None
    for seed in seeds:
        for case in ('noise_off','gaussian_colored'):
            base = preset(anatomy)
            config = SimConfig.model_validate({**base.model_dump(), 'seed': int(seed),
                'grid_size': int(grid), 'time_steps': int(time_steps),
                'transducers': int(transducers), 'detector_filter': False,
                'reconstruct': False, 'noise': NoiseConfig(enabled=case!='noise_off',
                engines=['gaussian','colored'], relative_std=0.04).model_dump()})
            sim = simulate(config)
            sound_actual = np.asarray(sim.tissue.sound_speed_m_s, dtype=np.float64)
            assumed = np.full_like(sound_actual, assumed_sound_m_s)
            op_matched = from_simulation(config, sound_actual, sim.dt_s, sim.detectors_m)
            op_mismatched = from_simulation(config, assumed, sim.dt_s, sim.detectors_m)
            y = sim.raw_signals
            im = {}
            runtimes = {}
            for name, fn in (
                ('das', lambda: reconstruct_das(y, sim.dt_s, sim.detectors_m, config)),
                ('derivative_bp', lambda: reconstruct_ubp(y, sim.dt_s, sim.detectors_m, config)),
                ('iterative_matched_inverse_crime', lambda: op_matched.reconstruct(y, iterations=iterations)[0]),
                ('iterative_homogeneous_mismatch', lambda: op_mismatched.reconstruct(y, iterations=iterations)[0])):
                start = time.perf_counter()
                image = fn()
                runtimes[name] = time.perf_counter()-start
                im[name] = image
                m = metrics(sim.tissue.initial_pressure_pa,image,sim.tissue.mask)
                rows.append(dict(seed=seed, noise_condition=case, method=name,
                    **m, elapsed_seconds=round(runtimes[name],4),
                    true_sound_min_m_s=round(float(sound_actual.min()),4),
                    true_sound_max_m_s=round(float(sound_actual.max()),4),
                    assumed_sound_m_s=assumed_sound_m_s,
                    assumed_homogeneous=(name=='iterative_homogeneous_mismatch'),
                    inverse_crime=(name=='iterative_matched_inverse_crime'),
                    evidence='synthetic model mismatch only; not measured data'))
            if first is None:
                first = (sim.tissue.initial_pressure_pa, sim.tissue.vessel_type_mask,
                         y, im.copy(),sound_actual)
    with (output/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader();w.writerows(rows)
    by_method = {}
    for method in sorted(set(r['method'] for r in rows)):
        q=[r for r in rows if r['method']==method]
        by_method[method] = {
          'mean_pearson_r':float(np.mean([r['pearson_r'] for r in q])),
          'mean_oracle_scale_nrmse':float(np.mean([r['oracle_scale_nrmse'] for r in q])),
          'mean_seconds':float(np.mean([r['elapsed_seconds'] for r in q])),
          'n':len(q)}
    result=dict(status='synthetic numerical sensitivity only; NOT experimental or sim-to-real validation',
      scope='2D; fixed numerical wave solver; data/reconstructor differ in speed-of-sound map',
      interpretation='matched inversion shares the true forward model (inverse crime); mismatched inversion does not, but shares discretization',
      oracle_metric_warning='Oracle-scaled NRMSE aligns amplitude against known synthetic ground truth and cannot be used on unlabeled experimental data',
      settings=dict(anatomy=anatomy,seeds=list(seeds),grid=grid,time_steps=time_steps,
           transducers=transducers,iterations=iterations,assumed_sound_m_s=assumed_sound_m_s),
      summary=by_method, rows=rows)
    (output/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    if first:
        truth,types,raw,imgs,sound=first
        fig,axs=plt.subplots(2,4,figsize=(13.5,7.0),layout='constrained')
        panels=[('Vessel class masks',types,'tab10'),('Initial pressure truth',truth,'magma'),
                ('Numerical detector data',raw,'RdBu_r'),('True speed of sound',sound,'viridis'),
                ('DAS',imgs['das'],'magma'),('Derivative BP',imgs['derivative_bp'],'magma'),
                ('Matched iterative (inverse crime)',imgs['iterative_matched_inverse_crime'],'magma'),
                ('Mismatched iterative',imgs['iterative_homogeneous_mismatch'],'magma')]
        for ax,(title,img,cmap) in zip(axs.ravel(),panels):
            ax.imshow(img,origin='lower',aspect='auto' if title=='Numerical detector data' else 'equal',cmap=cmap)
            ax.set_title(title,fontsize=9)
            ax.axis('off')
        fig.suptitle('PA-FORGE v0.5 — SYNTHETIC 2D numerical benchmark, NOT experimental validation',fontsize=12)
        fig.savefig(output/'figure_model_mismatch.png',dpi=150)
        plt.close(fig)
        methods=list(by_method)
        fig,ax=plt.subplots(figsize=(8,3.6),layout='constrained')
        ax.barh(range(len(methods)),[by_method[m]['mean_pearson_r'] for m in methods])
        ax.set_yticks(range(len(methods)),[m.replace('_',' ') for m in methods],fontsize=8)
        ax.set_xlabel('Mean Pearson correlation with known synthetic truth')
        ax.set_title('Model-mismatch benchmark: not a real-data result')
        fig.savefig(output/'figure_metrics.png',dpi=150)
        plt.close(fig)
    return result
