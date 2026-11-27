"""Unit tests with fabricated fixtures; never a real MATLAB validation."""
import json
from pathlib import Path
import h5py
import numpy as np
import pytest

from paforge.hybrid.contracts import AcquisitionRecord
from paforge.hybrid.exchange import write_acquisition
from paforge.hybrid.parity import write_test_image
from paforge.hybrid.pr04 import verify_bundle, write_report, main


def make_bundle(tmp_path: Path, *, divergence: float = 0.) -> Path:
    # Metadata claim fixtures in a TEST ONLY harness. Not real MATLAB evidence.
    (tmp_path/'patbox_sim.mat').write_bytes(b'UNIT TEST PLACEHOLDER - NOT PATBOX')
    n, t = 3, 24
    signals=np.arange(n*t,dtype=float).reshape(n,t)/100
    coords=np.array([[.002, .001, 0.], [0,.001,0.],[-.002,.001,0.]])
    acq=AcquisitionRecord(signals=signals,detectors_m=coords,sample_interval_s=1e-7,
        acquisition_id='unit-only',device_id='PATBox-unit',session_id='unit',
        source_id='unit-test',domain='synthetic',coordinate_frame='reviewed',
        processing_history=('unit test only',))
    write_acquisition(tmp_path/'acquisition.h5',acq)
    with h5py.File(tmp_path/'patbox_raw.h5','w') as h:
        h.attrs['format']='PA-PATBOX-RAW-EXPORT';h.attrs['schema_version']='0.1.0'
        r=h.create_group('raw')
        r.attrs.update({'n_detectors':n,'n_samples':t,'sample_interval_s':1e-7})
        r.create_dataset('signals_flat_c',data=signals.ravel(order='C'))
        r.create_dataset('detectors_flat_c',data=coords.ravel(order='C'))
    with h5py.File(tmp_path/'acquisition.h5','r') as h:
        sig_hash=h['acquisition'].attrs['signal_sha256_c_f64']
    image=np.arange(12).reshape(3,4).astype(float)
    for name, prod, offset in [('direct_patbox.h5','MATLAB:PATBox:direct',0),
                               ('bridge_patbox.h5','MATLAB:PAForge:bridge',divergence)]:
        path=tmp_path/name
        write_test_image(path,image+offset,method_id='PATBox:DAS',acquisition_id='unit-only',
            dx_m=.001,dy_m=.002,origin_x_m=-.003,origin_y_m=-.004,
            orientation='rows=x,cols=y (native k-Wave)')
        with h5py.File(path,'a') as h:
            h.attrs['producer']=prod
            h.attrs['matlab_release']='TEST-ONLY-NOT-RUNTIME'
            h.attrs['patbox_entrypoint']='/tmp/PATBox/patReconstruct.m'
            h.attrs['acquisition_signal_sha256']=sig_hash
    (tmp_path/'matlab_environment.txt').write_text('UNIT TEST - NOT MATLAB\n')
    return tmp_path


def test_pr04_pending_is_not_pass(tmp_path):
    report=verify_bundle(tmp_path)
    assert report['status']=='PENDING_MATLAB'
    assert not report['matlab_execution_verified']
    assert main(['verify',str(tmp_path)])==3


def test_pr04_parity_fixture_only(tmp_path):
    folder=make_bundle(tmp_path)
    r=verify_bundle(folder)
    assert r['status']=='PASS_ARTIFACT_PARITY'
    assert r['parity']['max_abs']==0
    assert r['physical_validation'] is False
    assert r['matlab_execution_verified'] is False
    assert 'claimed' not in r.get('scope','').lower()
    out=write_report(folder)
    assert json.loads(out.read_text())['status']=='PASS_ARTIFACT_PARITY'
    with pytest.raises(FileExistsError):write_report(folder)


def test_pr04_parity_detects_numeric_drift(tmp_path):
    r=verify_bundle(make_bundle(tmp_path,divergence=.02))
    assert r['status']=='FAIL'
    assert r['parity']['rmse']>0


def test_pr04_rejects_forged_producer(tmp_path):
    folder=make_bundle(tmp_path)
    with h5py.File(folder/'bridge_patbox.h5','a') as h:
        h.attrs['producer']='Python:fake_m'
    with pytest.raises(ValueError,match='producer'):
        verify_bundle(folder)


def test_pr04_rejects_untied_acquisition(tmp_path):
    folder=make_bundle(tmp_path)
    with h5py.File(folder/'direct_patbox.h5','a') as h:
        h.attrs['acquisition_signal_sha256']='0'*64
    with pytest.raises(ValueError,match='hash'):
        verify_bundle(folder)


def test_pr04_rejects_raw_data_tampering(tmp_path):
    folder=make_bundle(tmp_path)
    with h5py.File(folder/'patbox_raw.h5','a') as h:
        h['raw/signals_flat_c'][0]+=1
    with pytest.raises(ValueError,match='changed'):
        verify_bundle(folder)


def test_pr04_rejects_missing_grid(tmp_path):
    folder=make_bundle(tmp_path)
    with h5py.File(folder/'direct_patbox.h5','a') as h:
        del h['reconstruction'].attrs['grid_origin_y_m']
    with pytest.raises(ValueError,match='Missing grid metadata'):
        verify_bundle(folder)


def test_pr04_rejects_method_mismatch(tmp_path):
    folder=make_bundle(tmp_path)
    with h5py.File(folder/'direct_patbox.h5','a') as h:
        h['reconstruction'].attrs['method_id']='PATBox:UBP'
    with pytest.raises(ValueError,match='method_id mismatch'):
        verify_bundle(folder)


def test_pr04_rejects_nan_tolerances(tmp_path):
    with pytest.raises(ValueError,match='Tolerances'):
        verify_bundle(tmp_path,atol=float('nan'))


def test_pr04_rejects_cross_release(tmp_path):
    folder=make_bundle(tmp_path)
    with h5py.File(folder/'bridge_patbox.h5','a') as h:
        h.attrs['matlab_release']='different release'
    with pytest.raises(ValueError,match='MATLAB release'):
        verify_bundle(folder)
