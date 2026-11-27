"""Regression tests for MATLAB-style singleton numeric/string HDF5 attributes.

No MATLAB executable is required: these are explicitly synthetic HDF5 fixtures.
"""
from pathlib import Path
import h5py
import numpy as np
import pytest
from paforge.hybrid.h5_attrs import scalar_int, scalar_float, scalar_text
from paforge.hybrid.patbox_import import import_patbox_raw
from paforge.hybrid.exchange import read_acquisition
from paforge.hybrid.pr04 import _pr04_raw_matches_acquisition


def _raw(path, *, attrs_singleton=True, columns=True):
    n, t = 32, 3649
    s = np.arange(n*t, dtype=np.float64).reshape(n,t)/1000
    p = np.column_stack([np.full(n,-.01905), np.linspace(-.00465,.00465,n), np.zeros(n)])
    def maybe(x):
        if attrs_singleton and isinstance(x, str):
            return np.asarray([x.encode("utf-8")])
        return np.asarray([x]) if attrs_singleton else x
    with h5py.File(path,'w') as h:
        h.attrs['format'] = maybe('PA-PATBOX-RAW-EXPORT')
        h.attrs['schema_version'] = maybe('0.1.0')
        g = h.create_group('raw')
        for key,val in {'n_detectors':n,'n_samples':t,'sample_interval_s':1e-8,
                        'acquisition_id':'PR04-SYNTHETIC-TEST',
                        'device_id':'PATBox-sim','session_id':'PR04',
                        'coordinate_frame':'PATBox-audited-xyz-m',
                        'channel_mapping_note':'Synthetic contract-only row-order test'}.items():
            g.attrs[key] = maybe(val)
        signals = s.ravel(order='C')
        detectors = p.ravel(order='C')
        if columns:
            signals=signals[:,None];detectors=detectors[:,None]
        g.create_dataset('signals_flat_c',data=signals)
        g.create_dataset('detectors_flat_c',data=detectors)
    return s,p


@pytest.mark.parametrize('singleton,columns', [(True,True),(True,False),(False,True),(False,False)])
def test_matlab_singleton_attrs_and_flat_arrays_roundtrip(tmp_path, singleton, columns):
    raw=tmp_path/'patbox_raw.h5'; out=tmp_path/'acquisition.h5'
    signals,detectors=_raw(raw,attrs_singleton=singleton,columns=columns)
    imported=import_patbox_raw(raw,out,source_id='PATBox-reviewed',phantom_id='Example1')
    acq=read_acquisition(imported)
    assert acq.shape == (32,3649)
    np.testing.assert_array_equal(acq.signals,signals)
    np.testing.assert_array_equal(acq.detectors_m,detectors)
    assert _pr04_raw_matches_acquisition(raw,acq) is True


def test_reject_multivalued_detector_count(tmp_path):
    raw=tmp_path/'raw.h5'; _raw(raw)
    with h5py.File(raw,'r+') as h:
        del h['raw'].attrs['n_detectors']
        h['raw'].attrs['n_detectors'] = np.asarray([32,33],dtype=np.int64)
    with pytest.raises(ValueError,match='must be scalar/singleton'):
        import_patbox_raw(raw,tmp_path/'out.h5',source_id='PATBox-reviewed')


def test_reject_fractional_detector_count(tmp_path):
    raw=tmp_path/'raw.h5'; _raw(raw)
    with h5py.File(raw,'r+') as h:
        del h['raw'].attrs['n_detectors']
        h['raw'].attrs['n_detectors'] = np.asarray([32.5])
    with pytest.raises(ValueError,match='finite integer'):
        import_patbox_raw(raw,tmp_path/'out.h5',source_id='PATBox-reviewed')


def test_reject_nonfinite_sample_interval(tmp_path):
    raw=tmp_path/'raw.h5'; _raw(raw)
    with h5py.File(raw,'r+') as h:
        del h['raw'].attrs['sample_interval_s']
        h['raw'].attrs['sample_interval_s'] = np.asarray([np.nan])
    with pytest.raises(ValueError,match='must be finite'):
        import_patbox_raw(raw,tmp_path/'out.h5',source_id='PATBox-reviewed')
