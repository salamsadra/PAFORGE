import numpy as np
from paforge.config import SimConfig,NoiseConfig
from paforge.pipeline import simulate
from paforge.ui import build_ui


def test_all_five_engines_together(tmp_path):
    recording=tmp_path/"recorded_noise.npy"
    np.save(recording,np.random.default_rng(9).normal(size=(3,1000)))
    cfg=SimConfig(grid_size=64,time_steps=220,transducers=16,detector_filter=False,
          noise=NoiseConfig(enabled=True,
            engines=["gaussian","colored","impulsive","drift","empirical"],
            empirical_path=str(recording), impulse_probability=.1))
    result=simulate(cfg)
    assert len(result.noise_components)==5
    assert np.any(result.raw_signals!=result.clean_signals)
    assert len(result.info["noise_components"])==5


def test_gradio_panel_constructs():
    panel=build_ui()
    try:
        assert type(panel).__name__=="Blocks"
    finally:
        panel.close()


def test_dropout_does_not_trigger_noise_engine():
    cfg=SimConfig(grid_size=64,time_steps=220,transducers=16,detector_filter=False,
                  channel_dropout_probability=1,noise=NoiseConfig(enabled=False))
    result=simulate(cfg)
    assert np.count_nonzero(result.raw_signals)==0
    assert result.noise_components=={}
    assert len(result.info["dropped_channels"])==16
