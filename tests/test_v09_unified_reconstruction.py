"""v0.9 controlled numerical and dashboard integration tests; NOT physical QA."""
import sys,types
import h5py
import numpy as np
import pytest
from paforge.reconstruction_engine import Acquisition,RayOperator,reconstruct,METHODS,method_catalog


@pytest.fixture
def fixture_acq():
    n=48;nt=240;det=12;fov=.06;dt=1e-7
    ang=np.linspace(0,2*np.pi,det,endpoint=False)
    positions=np.column_stack([.019*np.cos(ang),.019*np.sin(ang),np.zeros(det)])
    acq=Acquisition(np.random.default_rng(4).standard_normal((det,nt))*0.1,dt,positions,fov,n,1500.,wavelength_nm=800.)
    return acq


def test_ray_dot_product(fixture_acq):
    op=RayOperator(fixture_acq)
    x=np.random.default_rng(7).normal(size=(48,48))
    y=np.random.default_rng(8).normal(size=fixture_acq.traces.shape)
    assert np.allclose(np.vdot(op.forward(x),y),np.vdot(x,op.adjoint(y)),rtol=1e-12,atol=1e-12)


def test_native_method_registry(fixture_acq):
    acq=fixture_acq
    assert len(METHODS)>=25
    assert len(METHODS)==len(method_catalog())
    methods=[m for m in METHODS if m not in ('patato_bp','patato_model','patato_opencl_bp','patato_slow_bp','learned_unet','learned_rescnn','pnp_fista')]
    for m in methods:
        out=reconstruct(acq,m,iterations=2,time_decimate=1,filter_low_hz=.4e6,filter_high_hz=4e6)
        assert out.shape==(48,48) and np.isfinite(out).all(),m


def test_corrupt_geometry_and_untrained_models(fixture_acq):
    a=fixture_acq
    with pytest.raises(ValueError,match='checkpoint'):
        reconstruct(a,'learned_unet')
    badpos=a.positions_m.copy();badpos[0,2]=.01
    with pytest.raises(ValueError,match='3D'):
        Acquisition(a.traces,a.dt_s,badpos,a.fov_m,a.grid_size)


def test_ray_model_inverse_on_matched_data(fixture_acq):
    a=fixture_acq
    op=RayOperator(a)
    p0=np.zeros((48,48));p0[19:25,20:24]=1
    new=Acquisition(op.forward(p0),a.dt_s,a.positions_m,a.fov_m,a.grid_size,a.sound_speed_m_s)
    initial=np.linalg.norm(new.traces)
    rec=reconstruct(new,'cg_tikhonov',iterations=30,reg=.00001)
    assert np.isfinite(rec).all()
    assert np.linalg.norm(op.forward(rec)-new.traces)<initial


def test_comparison_outputs_and_failure_isolation(tmp_path,fixture_acq):
    a=fixture_acq
    h5=tmp_path/'synthetic.h5'
    with h5py.File(h5,'w') as f:
        f.create_dataset('raw_data',data=a.traces)
        f.create_dataset('detector_positions_m',data=a.positions_m)
        f.attrs['sampling_rate_hz']=1/a.dt_s;f.attrs['synthetic']=True
    from paforge.reconstruction_lab import run_comparison
    report=run_comparison(h5,tmp_path/'result',['das','ubp','cg_tikhonov','learned_unet'],
                          grid_size=48,fov_m=a.fov_m,iterations=3)
    assert [r['status'] for r in report['methods']]==['ok','ok','ok','failed']
    assert (tmp_path/'result'/'input_signals.png').exists()
    assert (tmp_path/'result'/'comparison.png').exists()
    assert (tmp_path/'result'/'cg_tikhonov.npy').exists()


def test_patato_official_api_bridge_mock(monkeypatch,fixture_acq):
    a=fixture_acq
    class FakePA:
        @staticmethod
        def from_hdf5(path):
            with h5py.File(path,'r') as f:
                assert f['raw_data'].shape==(1,1,len(a.traces),a.traces.shape[1])
                assert f['GEOMETRY'].shape==a.positions_m.shape
                assert f['wavelengths'][0]==800
            return FakePA()
        def get_time_series(self):return object()
    class FakeBackprojection:
        def __init__(self,field_of_view,n_pixels,**kwargs):self.n=n_pixels[0]
        def run(self,series,padata,**kwargs):
            return types.SimpleNamespace(raw_data=np.full((1,1,1,self.n,self.n),3.)),{},None
    fake=types.SimpleNamespace(PAData=FakePA,ReferenceBackprojection=FakeBackprojection,
                               ModelBasedReconstruction=FakeBackprojection)
    monkeypatch.setitem(sys.modules,'patato',fake)
    for method in ('patato_bp','patato_model'):
        rec=reconstruct(a,method)
        assert rec.shape==(48,48) and np.allclose(rec,3.)


def _write_training_pair(folder,index,architecture='das'):
    folder.mkdir(parents=True)
    h,w=32,32
    x=np.linspace(-1,1,w)[None,:]+np.zeros((h,1))
    y=np.linspace(-1,1,h)[:,None]+np.zeros((1,w))
    source=np.exp(-((x-index*.09)**2+(y-index*.02)**2)/.1)
    rec=source+.05*np.random.default_rng(index).normal(size=(h,w))
    with h5py.File(folder/'reconstruction.h5','w') as f:
        f.attrs['method']='das (synthetic)';f.create_dataset('image',data=rec)
    with h5py.File(folder/'ground_truth.h5','w') as f:
        f.create_dataset('initial_pressure_pa',data=source)


@pytest.mark.parametrize('family',['unet','rescnn'])
def test_train_learned_and_infer(tmp_path,family):
    pytest.importorskip('torch')
    from paforge.learned_reconstruction import train_paired,apply_checkpoint
    for i in range(4):_write_training_pair(tmp_path/f'sample_{i}',i)
    weights=tmp_path/(family+'.pt')
    info=train_paired(tmp_path,weights,family,epochs=1,batch_size=2,seed=3,device='cpu')
    assert weights.is_file() and info['training_samples']+info['validation_samples']==4
    assert not (set(info['train_groups']) & set(info['validation_groups']))
    out=apply_checkpoint(np.ones((32,32)),weights,expected_family=family)
    assert out.shape==(32,32) and np.isfinite(out).all()
    with pytest.raises(ValueError,match='architecture'):
        apply_checkpoint(np.ones((32,32)),weights,expected_family='rescnn' if family=='unet' else 'unet')


def test_gradio_unified_dashboard_smoke():
    pytest.importorskip('gradio')
    from paforge.ui import build_ui
    ui=build_ui()
    assert len(ui.blocks)>450


def test_same_engine_synthetic_and_experimental_paths(tmp_path):
    from paforge.config import SimConfig
    from paforge.pipeline import simulate
    from paforge.experimental_reconstruction import reconstruct_experimental
    cfg=SimConfig(anatomy='vascular',grid_size=48,transducers=12,time_steps=120,
                  detector_filter=False,reconstruction_method='cf_das')
    sim=simulate(cfg)
    p=tmp_path/'canonical.h5'
    with h5py.File(p,'w') as f:
        f.create_dataset('raw_data',data=sim.raw_signals)
        f.create_dataset('detector_positions_m',data=sim.detectors_m)
        f.attrs['sampling_rate_hz']=1/sim.dt_s;f.attrs['synthetic']=True
    rec=reconstruct_experimental(p,tmp_path/'output','cf_das',cfg.fov_m,cfg.grid_size,1480.)
    assert np.allclose(rec,sim.reconstruction,rtol=2e-6,atol=1e-6)


def test_self_supervised_dip_fixture(fixture_acq):
    pytest.importorskip('torch')
    rec=reconstruct(fixture_acq,'deep_image_prior',iterations=2,reg=.001)
    assert rec.shape==(48,48) and np.min(rec)>=0 and np.isfinite(rec).all()
