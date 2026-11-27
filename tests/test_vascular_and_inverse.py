import numpy as np
import pytest
from paforge.config import preset,SimConfig
from paforge.anatomy import generate_tissue
from paforge.pipeline import simulate
from paforge.inverse import from_simulation,ForwardOperator

@pytest.mark.parametrize('pattern',['branching','paired','plexus'])
def test_vascular_types_and_noise_off(pattern):
    cfg=preset('vascular').model_copy(update={'vascular_pattern':pattern,'grid_size':64,
        'transducers':16,'time_steps':250,'detector_filter':False})
    r=simulate(cfg)
    types=r.tissue.vessel_type_mask
    assert (types==1).any() and (types==2).any(),(pattern,np.unique(types,return_counts=True))
    assert np.all((types>0)==r.tissue.vessel_mask)
    assert np.array_equal(r.raw_signals,r.clean_signals)
    assert np.isfinite(r.reconstruction).all()


def test_adjoint_dot_product():
    cfg=SimConfig(grid_size=48,transducers=11,time_steps=95,detector_filter=False)
    sound=np.full((48,48),1500.,dtype=float)
    from paforge.acoustic import sensor_positions
    pos=sensor_positions(cfg)
    dt=cfg.cfl*(cfg.fov_m/cfg.grid_size)/(np.sqrt(2)*1500.)
    op=ForwardOperator(sound,pos,cfg.fov_m,dt,95)
    rng=np.random.default_rng(9)
    p=rng.normal(size=(48,48));v=rng.normal(size=(11,95))
    dot1=np.vdot(op.forward(p),v)
    dot2=np.vdot(p,op.adjoint(v))
    assert np.isclose(dot1,dot2,rtol=1e-9,atol=1e-9),(dot1,dot2)


def test_iterative_inverse_fits_measurements_better_than_zero():
    cfg=SimConfig(grid_size=48,transducers=20,time_steps=175,detector_filter=False,
                  reconstruct=False)
    sim=simulate(cfg)
    op=from_simulation(cfg,sim.tissue.sound_speed_m_s,sim.dt_s,sim.detectors_m)
    p,losses=op.reconstruct(sim.raw_signals,iterations=6,reg=.001)
    assert np.all(p>=0) and np.isfinite(p).all()
    assert np.linalg.norm(op.forward(p)-sim.raw_signals)<np.linalg.norm(sim.raw_signals)
    assert len(losses)==6


def test_no_filtered_iterative_mode():
    with pytest.raises(ValueError,match='detector_filter'):
        SimConfig(reconstruction_method='iterative',detector_filter=True)
