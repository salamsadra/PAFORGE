"""Contract tests: MATLAB HDF5 singleton attrs, synthetic fixtures only.

No physical MATLAB execution is claimed by these tests.
"""
import h5py
import numpy as np
import pytest
from paforge.hybrid.result import read_matlab_result
from paforge.hybrid.parity import write_test_image, compare_results
from paforge.hybrid.pr04 import _result_metadata


def _wrap(path, *, multivalue=None):
    """Mimic MATLAB h5writeatt by replacing scalars with rank-one attrs."""
    with h5py.File(path,'a') as h:
        for group in (h, h['reconstruction']):
            for key in list(group.attrs.keys()):
                value=group.attrs[key]
                del group.attrs[key]
                if isinstance(value, (bytes,str)):
                    asbytes=value.encode() if isinstance(value,str) else value
                    group.attrs[key]=np.asarray([asbytes],dtype=f'S{max(1,len(asbytes))}')
                else:
                    group.attrs[key]=np.asarray([value])
        if multivalue is not None:
            key=multivalue
            g=h['reconstruction']
            del g.attrs[key]
            g.attrs[key]=[8,9]


def _file(path, value=0):
    a=np.arange(20,dtype=float).reshape(4,5)/10+value
    write_test_image(path,a,method_id='PATBox:DAS',acquisition_id='PR04-TEST',
        dx_m=7.5e-5,dy_m=7.5e-5,origin_x_m=-.02,origin_y_m=-.02,
        orientation='rows=x,cols=y (native k-Wave)')
    with h5py.File(path,'a') as h:
        h.attrs['producer']='MATLAB:PATBox:direct'
        h.attrs['matlab_release']='R2025a'
        h.attrs['patbox_entrypoint']='/PATBox/patReconstruct.m'
        h.attrs['acquisition_signal_sha256']='a'*64
    return a


def test_read_matlab_singleton_reconstruction(tmp_path):
    p=tmp_path/'matlab.h5'
    expected=_file(p)
    _wrap(p)
    im,meta=read_matlab_result(p)
    np.testing.assert_array_equal(im,expected)
    assert meta['method_id']=='PATBox:DAS'
    assert meta['acquisition_id']=='PR04-TEST'
    meta2=_result_metadata(p,'MATLAB:PATBox:direct',acq_hash='a'*64)
    assert meta2['matlab_release']=='R2025a'


def test_read_matlab_singleton_numerical_grid_parity(tmp_path):
    a,b=tmp_path/'ref.h5',tmp_path/'bridge.h5'
    _file(a);_file(b)
    _wrap(a)
    _wrap(b)
    report=compare_results(a,b,require_grid_metadata=True)
    assert report.status=='PASS'
    assert report.rmse==0
    assert report.method_id=='PATBox:DAS'


def test_mismatched_matlab_reconstruction_detected(tmp_path):
    a,b=tmp_path/'ref.h5',tmp_path/'bridge.h5'
    _file(a);_file(b,.01)
    _wrap(a);_wrap(b)
    report=compare_results(a,b,atol=1e-10,rtol=1e-8)
    assert report.status=='FAIL'
    assert report.rmse>0


def test_reject_multivalued_height(tmp_path):
    p=tmp_path/'bad.h5';_file(p);_wrap(p,multivalue='height')
    with pytest.raises(ValueError,match='scalar/singleton'):
        read_matlab_result(p)


def test_reject_multivalued_method(tmp_path):
    p=tmp_path/'bad.h5';_file(p);_wrap(p)
    with h5py.File(p,'a') as h:
        g=h['reconstruction']
        del g.attrs['method_id']
        g.attrs['method_id']=np.array([b'PATBox:DAS',b'PATBox:UBP'])
    with pytest.raises(ValueError,match='scalar/singleton'):
        read_matlab_result(p)


def test_reject_multivalued_grid(tmp_path):
    a,b=tmp_path/'ref.h5',tmp_path/'bridge.h5'
    _file(a);_file(b)
    _wrap(a,multivalue='grid_dx_m');_wrap(b)
    with pytest.raises(ValueError,match='scalar/singleton'):
        compare_results(a,b)


def test_full_pr04_gate_with_matlab_style_recon_attrs(tmp_path):
    """Synthetic end-to-end gate; cannot substitute for actual MATLAB run."""
    from paforge.hybrid.contracts import AcquisitionRecord
    from paforge.hybrid.exchange import write_acquisition
    from paforge.hybrid.pr04 import verify_bundle
    sig=np.linspace(-1,1,72,dtype=float).reshape(3,24)
    xyz=np.array([[0,0,0],[.001,0,0],[.002,0,0]])
    a=AcquisitionRecord(signals=sig,detectors_m=xyz,sample_interval_s=1e-8,
        acquisition_id='PR04-UNIT',device_id='sim-test',session_id='unit',
        source_id='unit',domain='synthetic',coordinate_frame='audited test frame')
    write_acquisition(tmp_path/'acquisition.h5',a)
    with h5py.File(tmp_path/'patbox_raw.h5','w') as h:
        h.attrs['format']='PA-PATBOX-RAW-EXPORT'
        h.attrs['schema_version']='0.1.0'
        g=h.create_group('raw')
        g.attrs.update({'n_detectors':3,'n_samples':24,'sample_interval_s':1e-8})
        g.create_dataset('signals_flat_c',data=sig.ravel(order='C')[:,None])
        g.create_dataset('detectors_flat_c',data=xyz.ravel(order='C')[:,None])
    with h5py.File(tmp_path/'acquisition.h5','r') as h:
        hsh=h['acquisition'].attrs['signal_sha256_c_f64']
    for name,producer in (('direct_patbox.h5','MATLAB:PATBox:direct'),
                          ('bridge_patbox.h5','MATLAB:PAForge:bridge')):
        p=tmp_path/name
        _file(p)
        with h5py.File(p,'a') as h:
            h['reconstruction'].attrs['acquisition_id']='PR04-UNIT'
            h.attrs['producer']=producer
            h.attrs['acquisition_signal_sha256']=hsh
        _wrap(p)
    (tmp_path/'patbox_sim.mat').write_bytes(b'UNIT-ONLY-NOT-MATLAB')
    (tmp_path/'matlab_environment.txt').write_text('UNIT TEST ONLY\n')
    report=verify_bundle(tmp_path)
    assert report['status']=='PASS_ARTIFACT_PARITY'
    assert report['matlab_execution_verified'] is False
