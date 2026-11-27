from pathlib import Path
import numpy as np
import pytest
from paforge.config import SimConfig,NoiseConfig
from paforge.noise import add_noise, measured_added_snr_db


@pytest.mark.parametrize("engine",["gaussian","colored","impulsive","drift"])
def test_single_noise_engines(engine):
    clean=np.ones((16,500),dtype=np.float32)
    cfg=SimConfig(noise=NoiseConfig(enabled=True,engines=[engine],impulse_probability=.2))
    out,components=add_noise(clean,cfg,np.random.default_rng(7))
    assert out.shape==clean.shape
    assert engine in components
    assert np.any(out!=clean)


def test_noise_off_with_undeclared_empirical_path():
    clean=np.ones((3,32),dtype=np.float32)
    cfg=SimConfig(noise=NoiseConfig(enabled=False,engines=["empirical"],empirical_path=None))
    out,components=add_noise(clean,cfg,np.random.default_rng(2))
    assert np.array_equal(out,clean)
    assert components=={}


def test_empirical_engine(tmp_path):
    p=tmp_path/"noise.npy"
    np.save(p,np.random.default_rng(2).normal(size=(3,100)))
    cfg=SimConfig(noise=NoiseConfig(enabled=True,engines=["empirical"],empirical_path=str(p)))
    x=np.ones((4,70),dtype=np.float32)
    y,components=add_noise(x,cfg,np.random.default_rng(1))
    assert not np.array_equal(y,x)
    assert np.isfinite(measured_added_snr_db(x,y))


def test_empirical_file_not_faked(tmp_path):
    cfg=SimConfig(noise=NoiseConfig(enabled=True,engines=["empirical"],empirical_path=str(tmp_path/"missing.npy")))
    with pytest.raises(FileNotFoundError):
        add_noise(np.ones((2,40)),cfg,np.random.default_rng(1))
