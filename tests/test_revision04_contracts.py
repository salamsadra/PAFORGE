"""Artificial fixtures only: no publisher data or historical GEN evidence."""
import io
import json
import zipfile
import numpy as np
import pytest
from paforge.experimental_learning import prepare_openqpai, sample_optics, _partition
from paforge.material_cvae import sample_empirical, train
from paforge.material_contracts import file_sha256


@pytest.fixture
def material_inputs(tmp_path):
    a=np.zeros((64,64),np.uint8);a[8:56,8:56]=1
    b=np.roll(a,2,axis=0); c=np.roll(a,2,axis=1)
    corpus=tmp_path/'corpus.npz'; profile=tmp_path/'profile.json'
    np.savez_compressed(corpus,masks=np.stack([a,b,c]),groups=['A','B','C'],classes=2)
    p={'format':'PA-FORGE-optical-profile-0.6','classes':2,
       'groups':{'train':['A'],'validation':['B'],'test':['C']},
       'measurements':[{'group':'A','label':k,'wavelength_nm':700,'absorption_m_inv':1.+k,'scattering_reduced_m_inv':10.+k} for k in (0,1)]}
    profile.write_text(json.dumps(p));return corpus,profile,p,a


@pytest.mark.parametrize('other', ['validation','test'])
def test_empirical_rejects_overlapping_partitions(material_inputs,other):
    corpus,profile,p,_=material_inputs;p['groups'][other].append('A');profile.write_text(json.dumps(p))
    with pytest.raises(ValueError,match='Data leakage'):sample_empirical(corpus,profile,np.random.default_rng(1))


@pytest.mark.parametrize('group', ['B','C','UNKNOWN'])
def test_optics_rejects_nontrain_rows(material_inputs,group):
    _,profile,p,a=material_inputs;p['measurements'][0]['group']=group;profile.write_text(json.dumps(p))
    with pytest.raises(ValueError,match='not in declared TRAIN'):sample_optics(profile,a,700,np.random.default_rng(1))


def test_empirical_source_provenance(material_inputs):
    corpus,profile,_,a=material_inputs
    got,prov=sample_empirical(corpus,profile,np.random.default_rng(1),distort_pixels=0,return_provenance=True)
    assert np.array_equal(got,a) and prov['source_group']=='A' and prov['source_mask_index']==0
    import hashlib
    assert prov['source_mask_sha256']==hashlib.sha256(a.tobytes()).hexdigest()


def test_declared_taxonomy_and_train_only_optical_read(tmp_path):
    archive=tmp_path/'source.zip';groups=['A','B','C'];split=_partition(groups,42)
    def put(z,name,array):
        buf=io.BytesIO();np.save(buf,array);z.writestr(name,buf.getvalue())
    with zipfile.ZipFile(archive,'w') as z:
        for g in groups:
            a=np.zeros((64,64),np.uint8);a[8:56,8:56]=1
            if g in split['test']:a[24:32,24:32]=2
            put(z,f'invision/labels/{g}.npy',a)
            if g in split['train']:
                for key in ('mua','musp'):put(z,f'invision/{key}/{g}_700.npy',np.ones_like(a,float))
            else:
                for key in ('mua','musp'):z.writestr(f'invision/{key}/{g}_700.npy',b'not a numpy file')
    with pytest.raises(ValueError,match='outside frozen material taxonomy'):
        prepare_openqpai(archive,tmp_path/'rejected','1/m','1/m')
    out=prepare_openqpai(archive,tmp_path/'declared','1/m','1/m',declared_classes=3)
    p=json.loads(open(out['optical_profile']).read())
    assert p['classes']==3 and p['taxonomy_source']=='declared'
    assert {r['group'] for r in p['measurements']}==set(split['train'])
    assert out['source_zip_sha256']==file_sha256(archive)


@pytest.mark.parametrize('beta',[.001,.01])
def test_kl_weight_checkpoint(material_inputs,tmp_path,beta):
    corpus,profile,_,_=material_inputs;weights=tmp_path/'weights.pt'
    r=train(corpus,profile,weights,epochs=1,batch_size=1,device='cpu',kl_weight=beta)
    import torch
    w=torch.load(weights,weights_only=True)
    assert r['kl_weight']==w['kl_weight']==beta
    assert r['profile_sha256']==w['profile_sha256']==file_sha256(profile)
    assert r['corpus_sha256']==w['corpus_sha256']==file_sha256(corpus)
    assert r['heldout_test'] is None


@pytest.mark.parametrize('beta',[-1,float('nan'),float('inf')])
def test_kl_weight_rejects_invalid(material_inputs,tmp_path,beta):
    corpus,profile,_,_=material_inputs
    with pytest.raises(ValueError,match='KL weight'):
        train(corpus,profile,tmp_path/'never.pt',kl_weight=beta)
    assert not (tmp_path/'never.pt').exists()


def test_physical_partition_export(material_inputs,tmp_path):
    from paforge.material_partitions import separate
    corpus,profile,_,_=material_inputs
    out=tmp_path/'parts'; r=separate(corpus,profile,out)
    for key, groups in {'train':{'A'},'validation':{'B'},'test':{'C'},'train_validation':{'A','B'}}.items():
        path=out/r['outputs'][key]['file']
        with np.load(path,allow_pickle=False) as f:assert set(f['groups'])==groups
        assert r['outputs'][key]['sha256']==file_sha256(path)
    with pytest.raises(FileExistsError):separate(corpus,profile,out)
