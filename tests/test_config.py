import pytest
from paforge.config import NoiseConfig, SimConfig,preset


def test_noise_is_off_by_default():
    assert not SimConfig().noise.enabled
    assert not preset("skin").noise.enabled


def test_all_anatomy_presets():
    for name in ("forearm","skin","breast","carotid"):
        assert preset(name).anatomy==name


def test_empirical_requires_file_when_enabled():
    with pytest.raises(ValueError,match="empirical_path"):
        NoiseConfig(enabled=True,engines=["empirical"])
    NoiseConfig(enabled=False,engines=["empirical"])


def test_cvae_requires_checkpoint():
    with pytest.raises(ValueError,match="generator_weights"):
        SimConfig(anatomy_engine="cvae")
