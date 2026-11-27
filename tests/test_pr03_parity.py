from __future__ import annotations
import h5py
import numpy as np
import pytest
from paforge.hybrid.pr03 import make_fixture, verify_roundtrip, main
from paforge.hybrid.exchange import read_acquisition, write_acquisition
from paforge.hybrid.parity import compare_results, write_test_image
from paforge.hybrid.patbox_import import import_patbox_raw


def _img(tmp_path, name, data=None, **kwargs):
    if data is None: data=np.arange(15, dtype=float).reshape(3,5)
    opts=dict(method_id='PATBox:DAS',acquisition_id='A',dx_m=1e-4,dy_m=1e-4,
              origin_x_m=-.01, origin_y_m=-.01, orientation='rows=x+,cols=y+')
    opts.update(kwargs)
    return write_test_image(tmp_path/name,data,**opts)


def test_fixture_roundtrip_and_intentional_diff(tmp_path):
    a=make_fixture(tmp_path/'fixture.h5')
    x=read_acquisition(a)
    assert x.shape==(5,73) and x.domain=='synthetic'
    assert x.time_zero_s is None
    with pytest.raises(ValueError, match='distinct output'):
        verify_roundtrip(a,a)
    y=x.signals.copy();y[1,0]+=0.5
    from paforge.hybrid.contracts import AcquisitionRecord
    import dataclasses
    b=write_acquisition(tmp_path/'altered.h5',dataclasses.replace(x,signals=y))
    with h5py.File(b,'r+') as f:
        f.attrs['producer']='MATLAB:pa_bridge_roundtrip'  # simulated marker for Python unit test only
    assert verify_roundtrip(a,b)['status']=='FAIL'


def test_matlab_schema_transport_without_matlab_cannot_be_claimed(tmp_path):
    a=make_fixture(tmp_path/'a.h5')
    with pytest.raises(ValueError,match='distinct output'):
        main(['roundtrip',str(a),str(a)])
    b=make_fixture(tmp_path/'b.h5')
    with pytest.raises(ValueError,match='producer marker'):
        main(['roundtrip',str(a),str(b)])


def test_parity_same_values(tmp_path):
    a=_img(tmp_path,'a.h5'); b=_img(tmp_path,'b.h5')
    rep=compare_results(a,b)
    assert rep.status=='PASS' and rep.max_abs==0
    assert rep.shape==(3,5)
    assert main(['compare',str(a),str(b),'--json',str(tmp_path/'rep.json')])==0


def test_parity_detects_tolerance_and_reports_failure(tmp_path):
    a=_img(tmp_path,'a.h5')
    x=np.arange(15,dtype=float).reshape(3,5); x[0,0]+=1
    b=_img(tmp_path,'b.h5',data=x)
    rep=compare_results(a,b)
    assert rep.status=='FAIL' and rep.rmse>0
    assert main(['compare',str(a),str(b)])==1


def test_parity_rejects_mismatched_identity(tmp_path):
    a=_img(tmp_path,'a.h5')
    b=_img(tmp_path,'b.h5',method_id='PATBox:DMAS')
    with pytest.raises(ValueError, match='method_id'):
        compare_results(a,b)
    c=_img(tmp_path,'c.h5',acquisition_id='different')
    with pytest.raises(ValueError, match='acquisition_id'):
        compare_results(a,c)


def test_parity_rejects_unaligned_grids(tmp_path):
    a=_img(tmp_path,'a.h5')
    b=_img(tmp_path,'b.h5',origin_x_m=-.009)
    with pytest.raises(ValueError,match='grid_origin_x_m'):
        compare_results(a,b)
    with h5py.File(b, 'r+') as f:
        del f['reconstruction'].attrs['grid_orientation']
    with pytest.raises(ValueError,match='Missing grid metadata'):
        compare_results(a,b)


def test_parity_cannot_hide_signal_scaling(tmp_path):
    a=_img(tmp_path,'a.h5');b=_img(tmp_path,'b.h5',data=np.arange(15,dtype=float).reshape(3,5)*2)
    assert compare_results(a,b).status=='FAIL'


def test_import_patbox_raw(tmp_path):
    raw=tmp_path/'raw.h5'
    with h5py.File(raw,'w') as h:
        h.attrs['format']='PA-PATBOX-RAW-EXPORT'; h.attrs['schema_version']='0.1.0'
        g=h.create_group('raw')
        g.attrs.update({'n_detectors':4,'n_samples':32,'sample_interval_s':1e-7,
            'acquisition_id':'PAT-A','device_id':'SIM','session_id':'S',
            'coordinate_frame':'xyz-centred-metres',
            'channel_mapping_note':'Verified sensor mask indexing manually'})
        g.create_dataset('signals_flat_c',data=np.arange(128,dtype=float))
        g.create_dataset('detectors_flat_c',data=np.arange(12,dtype=float)/1000)
    out=import_patbox_raw(raw,tmp_path/'canonical.h5',source_id='PATBox-local')
    acq=read_acquisition(out)
    assert acq.shape==(4,32) and acq.signals[1,0]==32
    assert acq.time_zero_s is None


def test_import_requires_honest_verification_note(tmp_path):
    p=tmp_path/'raw.h5'
    with h5py.File(p,'w') as h:
        h.attrs['format']='PA-PATBOX-RAW-EXPORT';h.attrs['schema_version']='0.1.0'
        g=h.create_group('raw');g.attrs.update({'n_detectors':4,'n_samples':32,'sample_interval_s':1e-7,
            'acquisition_id':'x','device_id':'x','session_id':'x','coordinate_frame':'x',
            'channel_mapping_note':'unverified'})
        g.create_dataset('signals_flat_c',data=np.ones(128))
        g.create_dataset('detectors_flat_c',data=np.ones(12))
    with pytest.raises(ValueError,match='verification'):
        import_patbox_raw(p,tmp_path/'canonical.h5',source_id='x')


def test_roundtrip_acceptance_unit_path_with_simulated_marker(tmp_path):
    # This is a Python unit test of the acceptance logic, NOT a MATLAB run.
    a=make_fixture(tmp_path/'original.h5')
    b=make_fixture(tmp_path/'candidate.h5')
    with h5py.File(b,'r+') as f:
        f.attrs['producer']='MATLAB:pa_bridge_roundtrip'  # simulated for test only
    assert verify_roundtrip(a,b)['status']=='PASS'
