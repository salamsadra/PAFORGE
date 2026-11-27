"""Isolated candidate feature tests; never assert instrument calibration."""
import numpy as np
from paforge.config import SimConfig
from paforge.pipeline import _material_geometry_diagnostic
from paforge.acoustic import sensor_positions


def test_geometry_preflight_original_sensor_in_sponge():
    cfg=SimConfig(anatomy='material_phantom',anatomy_engine='empirical_material',
                  geometry='circle',optical_profile='dummy',generator_weights='dummy',
                  grid_size=64,transducers=48,detector_filter=False)
    p0=np.zeros((64,64),dtype=float)
    p0[8:56,8:56]=1
    result=_material_geometry_diagnostic(p0,cfg,cfg)
    assert not result['geometry_gate_pass']
    assert result['receiver_nearest_boundary_px']<9


def test_padding_preflight_separates_support_from_sponge():
    cfg=SimConfig(anatomy='material_phantom',anatomy_engine='empirical_material',
                  geometry='circle',optical_profile='dummy',generator_weights='dummy',
                  grid_size=64,transducers=48,detector_filter=False,
                  material_solver_padding_factor=2)
    p0=np.zeros((64,64),dtype=float)
    p0[8:56,8:56]=1
    large=np.pad(p0,32)
    big=cfg.model_copy(update={'grid_size':128,'fov_m':2*cfg.fov_m})
    result=_material_geometry_diagnostic(large,cfg,big,pad_px=32)
    assert result['geometry_gate_pass']
    assert result['receiver_nearest_boundary_px']>=9
    assert result['source_receiver_clearance_px']>2


def test_padding_does_not_apply_to_unrelated_anatomy():
    import pytest
    with pytest.raises(ValueError,match='material_phantom'):
        SimConfig(anatomy='forearm',material_solver_padding_factor=2)
