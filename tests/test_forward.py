import numpy as np
import pytest
from paforge.config import SimConfig,NoiseConfig
from paforge.anatomy import generate_tissue
from paforge.acoustic import simulate_pressure
from paforge.pipeline import simulate


@pytest.fixture(scope="module")
def no_noise():
    return SimConfig(grid_size=64,transducers=16,time_steps=250,detector_filter=False)


def test_no_noise_bit_identical(no_noise):
    r=simulate(no_noise)
    assert np.array_equal(r.clean_signals,r.raw_signals)
    assert r.noise_components=={}
    assert r.info["computed_noise_snr_db"] is None


def test_no_noise_even_with_engines_configured(no_noise):
    cfg=no_noise.model_copy(update={"noise":NoiseConfig(enabled=False,engines=["gaussian","impulsive","colored","drift","empirical"])})
    a=simulate(cfg);b=simulate(no_noise)
    assert np.array_equal(a.clean_signals,b.clean_signals)
    assert np.array_equal(a.raw_signals,b.raw_signals)


def test_deterministic_across_repeated_runs(no_noise):
    a=simulate(no_noise);b=simulate(no_noise)
    assert np.array_equal(a.raw_signals,b.raw_signals)
    assert np.array_equal(a.tissue.initial_pressure_pa,b.tissue.initial_pressure_pa)


def test_noise_does_not_change_anatomy(no_noise):
    with_noise=no_noise.model_copy(update={"noise":NoiseConfig(enabled=True,engines=["gaussian","colored"],relative_std=.1)})
    a=simulate(no_noise);b=simulate(with_noise)
    assert np.array_equal(a.tissue.initial_pressure_pa,b.tissue.initial_pressure_pa)
    assert not np.array_equal(a.raw_signals,b.raw_signals)
    assert all(k in b.noise_components for k in ["gaussian","colored"])
    assert np.isfinite(b.info["computed_noise_snr_db"])


@pytest.mark.parametrize("name",["forearm","skin","breast","carotid"])
def test_anatomy_and_geometry_generate(name):
    from paforge.config import preset
    cfg=preset(name).model_copy(update={"grid_size":64,"time_steps":200,"transducers":16,"detector_filter":False})
    result=simulate(cfg)
    assert result.raw_signals.shape==(16,200)
    assert result.reconstruction.shape==(64,64)
    assert np.max(result.tissue.initial_pressure_pa)>0
    assert np.all(np.isfinite(result.raw_signals))
    assert result.detectors_m.shape==(16,3)


def test_zero_pressure_yields_zero_wave(no_noise):
    n=no_noise.grid_size
    wave,dt,pos=simulate_pressure(np.zeros((n,n)),np.full((n,n),1500.),no_noise)
    assert np.count_nonzero(wave)==0
    assert dt>0


def test_wave_arrival_travel_time(no_noise):
    n=no_noise.grid_size
    yy,xx=np.mgrid[:n,:n]
    p0=100*np.exp(-((xx-(n-1)/2)**2+(yy-(n-1)/2)**2)/(2*2**2))
    c=1500.
    wave,dt,pos=simulate_pressure(p0,np.full((n,n),c),no_noise)
    travel=np.linalg.norm(pos[0,:2])/c
    observed=np.argmax(np.abs(wave[0]))*dt
    # A finite Gaussian wavepacket has broad support; arrival peak tolerance.
    assert abs(observed-travel)<2.5e-6,(observed,travel)
