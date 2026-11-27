import io
import json
import zipfile
from pathlib import Path
import numpy as np
import h5py
import pytest
from typer.testing import CliRunner
from paforge.dataset_registry import describe,all_descriptions
from paforge import dataset_hub as hub
from paforge.dataset_wizard import plan_datasets
from paforge.dataset_portal import inspect_local_file,visualize_array,extract_skin_pairs
from paforge.patato_bridge import normalize_patato,extract_patato_reference
from paforge.cli import app


def test_catalog_accurate_scientific_separation():
    rows=all_descriptions()
    assert len(rows)>=12
    assert 'measured_signal_reconstruction' in describe('cambridge-patato')['lanes'] or 'measured_signal_reconstruction_if_calibrated' in describe('cambridge-patato')['lanes']
    assert 'optical_initial_pressure_training' in describe('synthetic-skin')['lanes']
    assert 'material CVAE' in ' '.join(describe('paiskintone')['excluded'])
    assert 'unverified' in describe('zenodo-23348355')['status']
    assert 'cambridge-patato' in hub.SOURCE_BY_ID
    with pytest.raises(KeyError):describe('unknown')


def test_cambridge_api_real_publisher_metadata_only(monkeypatch):
    responses=[
        {'_embedded':{'bundles':[{'name':'ORIGINAL','_links':{'bitstreams':{'href':'https://www.repository.cam.ac.uk/server/api/core/bundles/ID/bitstreams'}}}]}},
        {'_embedded':{'bitstreams':[{'name':'preclinical_phantom.hdf5','sizeBytes':1234,'uuid':'a',
           '_links':{'content':{'href':'https://www.repository.cam.ac.uk/server/api/core/bitstreams/a/content'}}}]}},
    ]
    monkeypatch.setattr(hub,'_json_url',lambda url:responses.pop(0))
    assets=hub.discover('cambridge-patato')['assets']
    assert len(assets)==1 and assets[0]['bytes']==1234
    assert assets[0]['url'].startswith('https://www.repository.cam.ac.uk/')
    assert not responses


def test_cambridge_api_refuses_external_asset(monkeypatch):
    responses=[
       {'_embedded':{'bundles':[{'name':'ORIGINAL','_links':{'bitstreams':{'href':'https://www.repository.cam.ac.uk/list'}}}]}},
       {'_embedded':{'bitstreams':[{'name':'x.hdf5','sizeBytes':123,
          '_links':{'content':{'href':'https://untrusted.invalid/file.hdf5'}}}]}},
    ]
    monkeypatch.setattr(hub,'_json_url',lambda url:responses.pop(0))
    with pytest.raises(hub.SourceError,match='external asset'):
        hub.discover('cambridge-patato')


def test_per_source_budget_whole_file_and_mandatory_optics(monkeypatch,tmp_path):
    def fake_discover(source):
        files={'openqpai':[('invision.zip',800_000_000),('DIS.zip',100_000_000),
                             ('material_mapping.json',1000),('svot.zip',10_000_000_000)],
               'synthetic-skin':[('small.h5',100_000_000),('large.h5',5_000_000_000)]}
        return {'source':{'title':source,'provider':'zenodo','role':'fixture'},'assets':[
             {'filename':n,'bytes':size,'url':'https://zenodo.org/a',
              'checksum':None,'checksum_algorithm':None} for n,size in files[source]]}
    monkeypatch.setattr('paforge.dataset_wizard.discover',fake_discover)
    with pytest.raises(ValueError,match='mandatory files'):
        plan_datasets('experiment_informed_generation',['synthetic-skin'],'complete',tmp_path,{'openqpai':0.5})
    plan=plan_datasets('experiment_informed_generation',['synthetic-skin'],'complete',tmp_path,{'openqpai':1.0,'synthetic-skin':0.2})
    assert 'svot.zip' not in [a['filename'] for a in plan['assets']]
    assert 'small.h5' in [a['filename'] for a in plan['assets']]
    assert 'large.h5' not in [a['filename'] for a in plan['assets']]


def _make_patato(path):
    with h5py.File(path,'w') as f:
        f.attrs['fs']=40e6
        f.attrs['name']='test experimental phantom'
        # detector ring in XZ plane, Y constant
        theta=np.linspace(0,2*np.pi,6,endpoint=False)
        f.create_dataset('GEOMETRY',data=np.column_stack([.03*np.sin(theta),np.zeros(6),.03*np.cos(theta)]))
        raw=np.arange(2*3*6*64,dtype=np.float32).reshape(2,3,6,64)
        f.create_dataset('raw_data',data=raw)
        f.create_dataset('wavelengths',data=[700,750,800])
        f.create_dataset('recons/Reference Backprojection/0',data=np.random.default_rng(1).random((2,3,48,48,1)))
    return raw


def test_patato_normalization_and_reference(tmp_path):
    source=tmp_path/'patato.hdf5'
    raw=_make_patato(source)
    dest=tmp_path/'native.h5'
    rep=normalize_patato(source,dest,1,2,'experimental',True,'phantom_07')
    assert rep['sampling_frequency_hz']==40e6 and rep['wavelength_nm']==800
    assert rep['selected_plane']=='Y'
    with h5py.File(dest,'r') as f:
        np.testing.assert_equal(f['raw_data'][:],raw[1,2])
        assert f['detector_positions_m'].shape==(6,3)
        assert (f['detector_positions_m'][:,2]==0).all()
        assert f.attrs['origin_status']=='experimental'
    ref=extract_patato_reference(source,tmp_path/'reference.npy',1,2)
    assert np.load(ref['reference_npy']).shape==(48,48)
    with pytest.raises(ValueError,match='Confirm'):
        normalize_patato(source,tmp_path/'bad.h5',origin='experimental')


def test_patato_blocks_nonplanar_geometry(tmp_path):
    src=tmp_path/'3d.hdf5';_make_patato(src)
    with h5py.File(src,'r+') as f:
        f['GEOMETRY'][:,1]=np.linspace(-.01,.01,6)
    with pytest.raises(ValueError,match='3D'):
        normalize_patato(src,tmp_path/'bad.h5')


def test_portal_visualization_zip_and_mat(tmp_path):
    src=tmp_path/'mini.npz'
    np.savez(src,raw_data=np.arange(6*64).reshape(6,64),image=np.ones((16,16)))
    zpath=tmp_path/'mixed.zip'
    with zipfile.ZipFile(zpath,'w') as z:z.write(src,'mini.npz')
    report=inspect_local_file('cambridge-patato',zpath)
    assert report['listing']['total_files']==1
    figures,meta=visualize_array('cambridge-patato',zpath,'raw_data',tmp_path/'plots',member='mini.npz',interpretation='signal')
    assert len(figures)==2 and all(Path(p).is_file() for p in figures)
    assert meta['provenance'].startswith('Original source data')
    with pytest.raises(ValueError,match='relative'):
        visualize_array('cambridge-patato',zpath,'image',tmp_path/'plots',member='../bad.npz')


def test_synthetic_skin_pairs_guard_and_export(tmp_path):
    src=tmp_path/'synthetic_skin.h5'
    with h5py.File(src,'w') as f:
        for key in ('mua','fluence','IP'):
            f.create_dataset(key,data=np.ones((3,16,16),dtype='float32'))
        f.create_dataset('sinogram',data=np.ones((3,8,64),dtype='float32'))
    rep=extract_skin_pairs(src,tmp_path/'paired',limit=2)
    assert rep['pairs_exported']==2
    assert (tmp_path/'paired'/'paired_preview.png').exists()
    with np.load(tmp_path/'paired'/'synthetic_skin_pair_00000.npz',allow_pickle=False) as f:
        assert f['sinogram'].shape==(8,64) and f['IP'].shape==(16,16)
    with pytest.raises(ValueError,match='sample-axis=0'):
        extract_skin_pairs(src,tmp_path/'bad',sample_axis=1)


def test_gpu_selector_enforces_available_device(tmp_path,monkeypatch):
    from paforge.material_cvae import train
    import torch
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError,match='CUDA explicitly requested'):
            train(tmp_path/'absent.npz',tmp_path/'absent.json',tmp_path/'weights.pt',device='cuda')


def test_cli_lists_new_tools():
    result=CliRunner().invoke(app,['catalog','--source','cambridge-patato'])
    assert result.exit_code==0,result.output
    assert 'PATATO' in result.output
    result=CliRunner().invoke(app,['import-patato','--help'])
    assert result.exit_code==0,result.output
