import csv
import json
import numpy as np
import h5py
import pytest
from paforge.experimental import read_manifest, run_experiment, _JS, _adapt_record
from paforge.experimental_import import extract_hdf5_acquisition
from paforge.experimental_reconstruction import reconstruct_experimental


def write_h5(path,signal,synthetic):
    with h5py.File(path,'w') as f:
        f.create_dataset('raw_data',data=signal)
        f.attrs['synthetic']=synthetic
        f.attrs['sampling_rate_hz']=10e6


def make_fixture(tmp_path, same_groups=False):
    # All waves are FAKE. They are ONLY exercising software contracts.
    rows=[];rng=np.random.default_rng(12)
    for domain in ('experimental','simulation'):
        for split in ('train','test'):
            for k in range(2):
                group=(f'{domain}_group_0' if same_groups else f'{domain}_{split}_{k}')
                t=np.arange(256)/1e7
                carrier=1e6 if domain=='experimental' else .7e6
                a=np.sin(2*np.pi*carrier*t)[None,:]*np.linspace(.4,1,8)[:,None]
                a+=rng.normal(scale=.08,size=a.shape)
                path=tmp_path/f'{domain}_{split}_{k}.h5';write_h5(path,a,domain=='simulation')
                rows.append(dict(path=str(path),signal_key='raw_data',detector_axis='0',
                    domain=domain,split=split,group_id=group,source_id='MOCK_SOURCE_NOT_REAL',
                    sample_rate_hz='10000000',experimental_attestation='yes',
                    quiet_start='0',quiet_stop='32'))
    path=tmp_path/'manifest.csv'
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return path


def test_no_leakage_and_frozen_test(tmp_path):
    manifest=make_fixture(tmp_path)
    output=tmp_path/'eval'
    report=run_experiment(manifest,output,quiet_mode='extract')
    assert report['heldout_metrics']['before']['n_real_independent_groups']==2
    assert report['heldout_metrics']['after']['spectral_js_nats']>=0
    assert (output/'calibration_profile.json').exists()
    with np.load(output/'empirical_quiet_noise.npz') as f:
        assert f['noise'].shape==(16,32)
    fit=report['calibration']['fit_groups']
    assert set(fit['experimental'])=={'experimental_train_0','experimental_train_1'}
    assert len(report['provenance']['records'])==8
    assert all(len(r['sha256'])==64 for r in report['provenance']['records'])


def test_reject_leakage(tmp_path):
    manifest=make_fixture(tmp_path,same_groups=True)
    with pytest.raises(ValueError,match='LEAKAGE'):read_manifest(manifest)


def test_reject_synthetic_mislabeled_real(tmp_path):
    p=tmp_path/'fake.h5';write_h5(p,np.zeros((8,64)),True)
    manifest=make_fixture(tmp_path)
    text=manifest.read_text().replace('experimental_train_0.h5','fake.h5',1)
    manifest.write_text(text)
    with pytest.raises(ValueError,match='SYNTHETIC'):read_manifest(manifest)


def test_extract_four_dimensional_and_attest(tmp_path):
    src=tmp_path/'h4.h5'
    with h5py.File(src,'w') as f:
        f.create_dataset('binary_time_series_data',data=np.arange(8*64*2*3).reshape(8,64,2,3))
    out=tmp_path/'converted.h5'
    with pytest.raises(ValueError,match='attest'):
        extract_hdf5_acquisition(src,'binary_time_series_data',out,0,1,{2:0,3:1},10e6,800,'DOI','phantom-1')
    extract_hdf5_acquisition(src,'binary_time_series_data',out,0,1,{2:0,3:1},10e6,800,'DOI','phantom-1',True)
    with h5py.File(out) as f:
        assert f['raw_data'].shape==(8,64)
        assert f.attrs['synthetic']==False


def test_reconstruction_requires_real_detector_geometry(tmp_path):
    p=tmp_path/'real.h5';write_h5(p,np.zeros((8,120)),False)
    with pytest.raises(ValueError,match='geometry'):
        reconstruct_experimental(p,tmp_path/'recon',grid_size=48)
    theta=np.linspace(0,2*np.pi,8,endpoint=False)
    coords=np.column_stack([.016*np.cos(theta),.016*np.sin(theta),np.zeros(8)])
    coord=tmp_path/'coords.npy';np.save(coord,coords)
    rec=reconstruct_experimental(p,tmp_path/'recon',grid_size=48,positions_file=coord)
    assert rec.shape==(48,48)
    assert (tmp_path/'recon'/'experimental_reconstruction.png').exists()

def test_apply_profile_only_sim_and_matched_dimensions(tmp_path):
    from paforge.domain_adapter import adapt_synthetic
    manifest=make_fixture(tmp_path)
    run_experiment(manifest,tmp_path/'report')
    with pytest.raises(ValueError,match='Only known synthetic'):
        adapt_synthetic(tmp_path/'experimental_train_0.h5',tmp_path/'report/calibration_profile.json',tmp_path/'no.h5')
    p=adapt_synthetic(tmp_path/'simulation_test_1.h5',tmp_path/'report/calibration_profile.json',tmp_path/'adapted.h5')
    with h5py.File(p) as f:
        assert f['raw_data'].shape==(8,256)
        assert bool(f.attrs['domain_adapted'])

def test_download_refuses_invalid_archive(tmp_path):
    from paforge.download_data import fetch_openqpai
    with pytest.raises(ValueError,match='Choose one of'):
        fetch_openqpai(tmp_path,kind='invalid')
