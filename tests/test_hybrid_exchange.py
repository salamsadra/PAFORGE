import h5py
import numpy as np
import pytest
from paforge.hybrid import AcquisitionRecord, write_acquisition, read_acquisition, from_native_hdf5
from paforge.hybrid.result import read_matlab_result


def _record(nd=4, nt=64, **kwargs):
    meta = dict(acquisition_id="A001", device_id="D1", session_id="S1",
                source_id="original-source", domain="synthetic",
                coordinate_frame="xyz-centred-metres", time_zero_s=None,
                wavelength_nm=None, processing_history=("test fixture",))
    meta.update(kwargs)
    return AcquisitionRecord(np.arange(nd*nt, dtype=np.float64).reshape(nd,nt)/10.,
                             np.stack((np.arange(nd)*.0004, np.zeros(nd), np.zeros(nd)), axis=1),
                             5e-8, **meta)


def test_roundtrip_non_square_and_exact_channel_values(tmp_path):
    x = _record(); path = tmp_path/'data.h5'
    write_acquisition(path,x)
    y = read_acquisition(path)
    np.testing.assert_array_equal(y.signals,x.signals)
    np.testing.assert_array_equal(y.detectors_m,x.detectors_m)
    assert y.time_zero_s is None and y.domain == 'synthetic'
    assert y.signals[0,1]==.1 and y.signals[1,0]==6.4
    with pytest.raises(FileExistsError):write_acquisition(path,x)


def test_square_shape_and_distinct_order(tmp_path):
    x=_record(16,16)
    path=tmp_path/'square.h5';write_acquisition(path,x)
    y=read_acquisition(path)
    np.testing.assert_array_equal(x.signals,y.signals)
    assert y.signals[0,1]==.1 and y.signals[1,0]==1.6


def test_reject_unknown_implicit_metadata():
    with pytest.raises(ValueError, match='device_id'):
        _record(device_id='')
    with pytest.raises(ValueError, match='calibration_id'):
        _record(calibration_kind='measured')


def test_reject_shape_and_nonfinite():
    with pytest.raises(ValueError):
        AcquisitionRecord(np.zeros((3,16),dtype=float), np.zeros((2,3)), 1e-7,
                          'a','d','s','src','unknown','frame')
    x=_record()
    x.signals[0,0]=float('nan')
    with pytest.raises(ValueError):
        AcquisitionRecord(x.signals,x.detectors_m,5e-8,'a','d','s','src','unknown','frame')


def test_corrupt_signal_is_detected(tmp_path):
    p=tmp_path/'corrupt.h5';write_acquisition(p,_record())
    with h5py.File(p,'r+') as f:
        f['acquisition/signals_flat_c'][1]=999
    with pytest.raises(ValueError,match='hash'):
        read_acquisition(p)


def test_legacy_adapter_requires_geometry_and_provenance(tmp_path):
    p=tmp_path/'legacy.h5'
    with h5py.File(p,'w') as f:
        f.attrs['synthetic']=True
        f.attrs['sampling_rate_hz']=20e6
        f.create_dataset('raw_data', data=np.ones((4,64),dtype=float))
    with pytest.raises(ValueError,match='detector positions'):
        from_native_hdf5(p, acquisition_id='a',device_id='d',session_id='s',
                         source_id='src',coordinate_frame='xyz')
    with h5py.File(p,'r+') as f:
        f.create_dataset('detector_positions_m', data=np.zeros((4,3)))
    x=from_native_hdf5(p,acquisition_id='a',device_id='d',session_id='s',
                       source_id='src',coordinate_frame='xyz')
    assert x.domain=='synthetic' and x.time_zero_s is None


def test_matlab_result_schema_python_reader(tmp_path):
    p=tmp_path/'matlab.h5'
    with h5py.File(p,'w') as f:
        f.attrs['format']='PA-HYBRID-RECONSTRUCTION'; f.attrs['schema_version']='0.1.0'
        g=f.create_group('reconstruction')
        g.attrs.update({'height':2,'width':3,'method_id':'PATBox:DAS','acquisition_id':'A001','quantitative':0})
        g.create_dataset('image_flat_c',data=np.array([1.,2.,3.,4.,5.,6.]))
    image, meta = read_matlab_result(p)
    np.testing.assert_array_equal(image, [[1,2,3],[4,5,6]])
    assert meta['method_id']=='PATBox:DAS'


def test_legacy_reconstruction_requires_explicit_planar_and_time_zero():
    from paforge.hybrid.adapters import to_legacy_reconstruction
    x = _record(time_zero_s=None)
    with pytest.raises(ValueError,match='time_zero'):
        to_legacy_reconstruction(x,fov_m=.025,grid_size=32,sound_speed_m_s=1500)
    x2 = _record(time_zero_s=0.,coordinate_frame='xy-plane-z0-metres')
    acq = to_legacy_reconstruction(x2, fov_m=.025, grid_size=32, sound_speed_m_s=1500)
    assert acq.traces.shape == (4,64)
    with pytest.raises(ValueError,match='coordinate_frame'):
        to_legacy_reconstruction(_record(time_zero_s=0.), fov_m=.025,
                                 grid_size=32, sound_speed_m_s=1500)
