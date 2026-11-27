import json
import h5py
import numpy as np
import pytest
from paforge.dataset_hub import SOURCE_BY_ID,pick_assets
from paforge.research_inputs import SMALL_DATASETS, hdf5_inventory
from paforge.recon_plugins import run_reconstruction_plugin
from paforge.research_v05 import mismatch_benchmark


def test_digital_twin_catalog_small_first():
    assert 'digital-twin-iqa' in SOURCE_BY_ID
    a=[{'filename':'data.zip','bytes':2_900_000_000},
       {'filename':'ipasc_file.hdf5','bytes':4_800_000},
       {'filename':'README.md','bytes':7100}]
    names={r['filename'] for r in pick_assets('digital-twin-iqa',a,'starter')}
    assert 'ipasc_file.hdf5' in names
    assert 'data.zip' not in names
    assert SMALL_DATASETS['digital_twin_ipasc']['md5']=='f1d17e900f10341a05f6656fff194fdf'


def test_hdf5_inventory_no_raw_loading(tmp_path):
    p=tmp_path/'sample.h5'
    with h5py.File(p,'w') as f:
        f.attrs['manufacturer']='test only'
        f.create_dataset('raw_time_series_data',shape=(3,11,2),dtype='float32')
    r=hdf5_inventory(p)
    assert r['is_verified_experimental'] is False
    assert any(x.get('shape')==[3,11,2] for x in r['nodes'])
    assert r['nodes'][0]['attrs']['manufacturer']=='test only'


def test_plugin_contract_validation():
    with pytest.raises(ValueError):run_reconstruction_plugin('bad;module',np.zeros((3,20)),1.,np.zeros((3,3)),32,.04)
    with pytest.raises(ValueError):run_reconstruction_plugin('math',np.zeros((3,20)),1.,np.zeros((3,2)),32,.04)
    with pytest.raises(TypeError):run_reconstruction_plugin('math',np.zeros((3,20)),1.,np.zeros((3,3)),32,.04)


def test_model_mismatch_numeric(tmp_path):
    result=mismatch_benchmark(tmp_path,anatomy='vascular',seeds=(11,),grid=48,
           time_steps=115,transducers=8,iterations=2,assumed_sound_m_s=1660)
    assert len(result['rows'])==8
    assert (tmp_path/'metrics.csv').exists()
    assert (tmp_path/'figure_model_mismatch.png').exists()
    assert result['status'].startswith('synthetic')
    d=json.loads((tmp_path/'results.json').read_text())
    assert sum(bool(x['inverse_crime']) for x in d['rows'])==2
    assert d['summary']['iterative_homogeneous_mismatch']['n']==2
