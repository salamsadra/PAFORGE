import csv
import json
import numpy as np
from paforge.config import preset,SimConfig
from paforge.pipeline import simulate,run_batch
from paforge.research import run_benchmark,metrics
from paforge.inverse import from_simulation
from paforge.acoustic import reconstruct_ubp,reconstruct_das


def test_vascular_sample_exports_blood_type_and_reconstruction(tmp_path):
    c=preset('vascular').model_copy(update={'grid_size':64,'time_steps':230,'transducers':16,
                                              'detector_filter':False})
    p=run_batch(c,tmp_path)[0]
    import h5py
    assert (p/'vessel_types.png').exists()
    with h5py.File(p/'ground_truth.h5') as h:
        types=h['vessel_type_mask'][:]
        assert (types==1).any() and (types==2).any()
    with h5py.File(p/'reconstruction.h5') as h:
        assert h['image'].shape==(64,64)


def test_heterogeneous_speed_adjoint():
    cfg=SimConfig(grid_size=48,transducers=12,time_steps=110,detector_filter=False,reconstruct=False)
    r=simulate(cfg)
    op=from_simulation(cfg,r.tissue.sound_speed_m_s,r.dt_s,r.detectors_m)
    rng=np.random.default_rng(2)
    x=rng.normal(size=(48,48));y=rng.normal(size=(12,110))
    assert np.isclose(np.vdot(op.forward(x),y),np.vdot(x,op.adjoint(y)),rtol=1e-9)


def test_ubp_improves_noiseless_vascular_pearson_vs_plain_das():
    c=SimConfig.model_validate({**preset('vascular').model_dump(),
                                'grid_size':64,'time_steps':240,'transducers':24,
                                'detector_filter':False,'reconstruct':False,'seed':7})
    r=simulate(c)
    das=reconstruct_das(r.raw_signals,r.dt_s,r.detectors_m,c)
    ubp=reconstruct_ubp(r.raw_signals,r.dt_s,r.detectors_m,c)
    assert metrics(r.tissue.initial_pressure_pa,ubp,r.tissue.mask)['pearson_r'] > metrics(r.tissue.initial_pressure_pa,das,r.tissue.mask)['pearson_r']


def test_benchmark_writes_reproducible_metrics(tmp_path):
    path,result=run_benchmark(tmp_path,seeds=(7,),grid=48,time_steps=150,transducers=12,iterations=3)
    assert len(result['results'])==6
    assert (path/'metrics.csv').exists()
    assert (path/'comparison.png').exists()
    with (path/'metrics.csv').open() as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==6
    assert all(row['experiment_type'].startswith('synthetic') for row in rows)
    assert 'inverse_crime_warning' in json.loads((path/'results.json').read_text())


def test_group_disjoint_splits(tmp_path):
    import csv
    from paforge.splits import group_split
    file=tmp_path/'metadata.csv'
    with file.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['sample_id','subject_id'])
        w.writeheader()
        for sid in range(10):
            for j in range(3):w.writerow({'sample_id':f'{sid}_{j}','subject_id':f'subj{sid}'})
    m=group_split(file,tmp_path/'split_output')
    g=[set(v) for v in m['splits'].values()]
    assert all(len(gg)>0 for gg in g)
    assert not(g[0]&g[1] or g[0]&g[2] or g[1]&g[2])


def test_uncalibrated_learned_mode_rejects_fabricated_class_identity(tmp_path):
    from paforge.vascular_cvae import create_demo_masks,train,sample
    # Sanity check trainer and explicit no-fabricated-vessel fallback contract.
    path=create_demo_masks(tmp_path/'tiny.npz',n=9)
    weights=tmp_path/'demo.pt'
    history=train(path,weights,epochs=1,batch_size=9)
    assert len(history)==1
    import pytest
    try:
        types=sample(weights,np.random.default_rng(9),64)
        assert (types==1).any() and (types==2).any()
    except ValueError as e:
        assert 'collapsed' in str(e)
