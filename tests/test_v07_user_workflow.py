from pathlib import Path
import io,json,zipfile
import numpy as np
import h5py
import pytest
from paforge import dataset_hub
from paforge.dataset_wizard import plan_datasets,execute_dataset_plan
from paforge.multidevice_learning import prepare_openqpai_complete
from paforge.experimental_learning import sample_optics
from paforge.visualization import preview_invision_sample,preview_generated_sample


def _npy(a):
    b=io.BytesIO();np.save(b,a);return b.getvalue()


def _make_device_zip(path,device,n=6):
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for i in range(n):
            group=f'P.5.{i+1}';yy,xx=np.mgrid[:32,:32]
            lab=np.zeros((32,32),np.uint8)
            lab[(xx-16)**2+(yy-16)**2<14**2]=1
            lab[((xx-12-i%3)**2/20+(yy-15)**2/45)<1]=2
            lab[((xx-22)**2+(yy-16)**2)<8]=3
            z.writestr(f'{device}/labels/{group}.npy',_npy(lab))
            for wave in (700,750):
                mua=np.choose(lab,[.01,.03,.08,.12]).astype(float)*(1+.1*(wave-700)/50)
                musp=np.choose(lab,[.1,.2,.3,.4]).astype(float)
                z.writestr(f'{device}/mua/{group}_{wave}.npy',_npy(mua))
                z.writestr(f'{device}/musp/{group}_{wave}.npy',_npy(musp))


def _make_svot_zip(path,n=6):
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for i in range(n):
            group=f'P.5.{i+1}';vol=np.zeros((12,32,32),np.uint8)
            for k in range(12):
                yy,xx=np.mgrid[:32,:32]
                vol[k][(xx-16)**2+(yy-16)**2<14**2]=1
                vol[k][((xx-12-i%3)**2/20+(yy-15)**2/45)<1]=2
                vol[k][((xx-22)**2+(yy-16)**2)<8]=3
            z.writestr(f'svot/labels/{group}.npy',_npy(vol))


def _make_dis_zip(path):
    b=io.BytesIO();np.savez(b,mua=np.array([.01,.02]),musp=np.array([.1,.2]),g=np.array([.8,.9]),wavelengths=np.array([700,800]))
    with zipfile.ZipFile(path,'w') as z:z.writestr('DIS/processed/T.5.1.a/T.5.1.npz',b.getvalue())


def test_multidevice_complete_corpus_train_only_optics(tmp_path):
    inv=tmp_path/'invision.zip';tro=tmp_path/'tropus.zip';svot=tmp_path/'svot.zip';dis=tmp_path/'DIS.zip';mapping=tmp_path/'material_mapping.json'
    _make_device_zip(inv,'invision');_make_device_zip(tro,'tropus');_make_svot_zip(svot);_make_dis_zip(dis)
    mapping.write_text(json.dumps({'P.5.1':{'0':'T.5.1.a'},'P.5.2':{'1':'T.5.1.a'}}))
    report=prepare_openqpai_complete(inv,tro,svot,dis,mapping,tmp_path/'prepared','1/cm','1/cm',seed=7,svot_slices_per_volume=4)
    assert report['counts']['unique_phantom_groups']==6
    assert report['counts']['invision_masks']==6
    assert report['counts']['tropus_masks']==6
    assert report['counts']['svot_2d_slices']>=18
    assert Path(report['dis_inventory']).exists()
    profile=json.loads(Path(report['optical_profile']).read_text())
    assert profile['format']=='PA-FORGE-optical-profile-0.7'
    assert not(set(profile['groups']['train']) & set(profile['groups']['test']))
    labels=np.ones((16,16),dtype=np.uint8)
    mua,musp,selected=sample_optics(report['optical_profile'],labels,700,np.random.default_rng(4))
    assert mua.shape==labels.shape and musp.shape==labels.shape
    assert all(r['group'] in profile['groups']['train'] for r in profile['measurements'])


def test_complete_preparation_requires_dis_and_mapping(tmp_path):
    inv=tmp_path/'invision.zip';_make_device_zip(inv,'invision')
    with pytest.raises(FileNotFoundError,match='DIS.zip'):
        prepare_openqpai_complete(inv,output_dir=tmp_path/'out')


def test_dataset_visual_explorer_renders_original_signal_and_maps(tmp_path):
    archive=tmp_path/'invision.zip'
    lab=np.zeros((32,32),np.uint8);lab[4:28,4:28]=1;lab[12:20,12:20]=2
    b=io.BytesIO()
    with h5py.File(b,'w') as f:
        f.create_dataset('binary_time_series_data',data=np.random.default_rng(3).normal(size=(8,100,1,1)))
        m=f.create_group('meta_data');m.create_dataset('ad_sampling_rate',data=40e6)
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('invision/recon/P.5.1_700.npy',_npy(np.ones((32,32))))
        z.writestr('invision/labels/P.5.1.npy',_npy(lab))
        z.writestr('invision/mua/P.5.1_700.npy',_npy(np.ones((32,32))*.02))
        z.writestr('invision/musp/P.5.1_700.npy',_npy(np.ones((32,32))*.2))
        z.writestr('invision/raw/P.5.1_700_ipasc.hdf5',b.getvalue())
    files,report=preview_invision_sample(archive,'P.5.1',700,tmp_path/'preview')
    assert len(files)==6 and all(Path(p).exists() for p in files)
    assert report['raw_signal']['shape']==[8,100]


def test_dataset_wizard_requires_optical_source_and_plans_complete(monkeypatch,tmp_path):
    def fake_discover(sid):
        assets=[]
        if sid=='openqpai':
            for name,size in [('DIS.zip',132_000_000),('invision.zip',238_000_000),('tropus.zip',286_000_000),('svot.zip',19_800_000_000),('material_mapping.json',2000),('README_data.md',6000)]:
                assets.append({'filename':name,'bytes':size,'url':'https://zenodo.org/a','checksum_algorithm':'md5','checksum':'a'*32})
        else:
            assets=[{'filename':'data.zip','bytes':2_900_000_000,'url':'https://zenodo.org/b','checksum_algorithm':'md5','checksum':'b'*32},
                    {'filename':'ipasc_file.hdf5','bytes':4_800_000,'url':'https://zenodo.org/c','checksum_algorithm':'md5','checksum':'c'*32}]
        return {'source':{'title':sid,'provider':'zenodo','role':'test'},'assets':assets}
    monkeypatch.setattr('paforge.dataset_wizard.discover',fake_discover)
    plan=plan_datasets('full_research_corpus',[],'complete',tmp_path/'data')
    assert plan['mandatory_sources']==['openqpai','digital-twin-iqa']
    assert {x['filename'] for x in plan['assets']} >= {'DIS.zip','invision.zip','tropus.zip','svot.zip','data.zip'}
    assert plan['totals']['remaining_download_bytes']>23_000_000_000
    assert Path(plan['plan_path']).exists()


def test_execute_dataset_plan_respects_license_and_writes_report(monkeypatch,tmp_path):
    plan={'task':'custom','profile_resolved':'complete','output_dir':str(tmp_path/'data'),
          'totals':{'preflight_ok':True,'remaining_download_gb':1,'free_disk_gb':10,'already_present_bytes':0},
          'assets':[{'source_id':'openqpai','filename':'test.bin','bytes':3,'estimated_bytes':0,
                     'url':'https://zenodo.org/test','checksum_algorithm':None,'checksum':None}]}
    monkeypatch.setattr('paforge.dataset_wizard.fetch_asset',lambda asset,folder,**kw:{'filename':asset['filename'],'status':'downloaded','bytes':3})
    with pytest.raises(ValueError,match='license'):
        execute_dataset_plan(plan,False)
    out=execute_dataset_plan(plan,True)
    assert out['sources']['openqpai']['status']=='complete_selected'
    assert Path(out['report_path']).exists()


def test_dis_and_svot_visual_diagnostics(tmp_path):
    from paforge.visualization import preview_dis_archive,preview_svot_slice
    dis=tmp_path/'DIS.zip';_make_dis_zip(dis)
    files,report=preview_dis_archive(dis,tmp_path/'displots')
    assert files and all(Path(p).exists() for p in files)
    svot=tmp_path/'svot.zip';_make_svot_zip(svot)
    files,report=preview_svot_slice(svot,'P.5.1',5,tmp_path/'svotplots')
    assert len(files)==1 and Path(files[0]).exists()
    assert report['labels']['volume_shape']==[12,32,32]


def test_archive_inventory_without_extracting(tmp_path):
    from paforge.dataset_wizard import inspect_downloaded_archives
    folder=tmp_path/'external'/'openqpai';folder.mkdir(parents=True)
    archive=folder/'tiny.zip'
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('invision/raw/a.hdf5',b'abc')
        z.writestr('invision/labels/a.npy',b'def')
    report=inspect_downloaded_archives(tmp_path/'external',tmp_path/'inventory.json')
    assert report['archive_count']==1
    assert report['archives'][0]['role_counts']['raw']==1
    assert report['archives'][0]['role_counts']['labels']==1


def test_cli_dataset_wizard_commands_exist():
    from typer.testing import CliRunner
    from paforge.cli import app
    r=CliRunner().invoke(app,['datasets','--help'])
    assert r.exit_code==0, r.output
    assert 'plan' in r.output and 'download-plan' in r.output


def test_one_click_full_workflow_orchestrator(monkeypatch,tmp_path):
    import paforge.ui as ui
    root=tmp_path/'external';oq=root/'openqpai';oq.mkdir(parents=True)
    for name in ('invision.zip','DIS.zip','material_mapping.json','tropus.zip','svot.zip'):(oq/name).write_bytes(b'x')
    plan={'output_dir':str(root),'task':'full_research_corpus','profile_resolved':'complete','totals':{'preflight_ok':True}}
    monkeypatch.setattr('paforge.dataset_wizard.execute_dataset_plan',lambda *a,**k:{'sources':{'openqpai':{'status':'complete_selected'}}})
    monkeypatch.setattr('paforge.dataset_wizard.inspect_downloaded_archives',lambda *a,**k:{'archive_count':5})
    prepared={'training_corpus':'corpus.npz','optical_profile':'profile.json','counts':{'unique_phantom_groups':6}}
    monkeypatch.setattr('paforge.multidevice_learning.prepare_openqpai_complete',lambda *a,**k:prepared)
    def fake_train(corpus,profile,weights,**kwargs):
        Path(weights).parent.mkdir(parents=True,exist_ok=True);Path(weights).write_bytes(b'weights')
        return {'weights':str(weights),'best_validation_loss':.5,'history':[]}
    monkeypatch.setattr('paforge.material_cvae.train',fake_train)
    def fake_run(cfg,folder,callback=None):
        sample=Path(folder)/'material_phantom_00000';sample.mkdir(parents=True,exist_ok=True)
        for name in ('raw_signals.h5','ground_truth.h5','reconstruction.h5'):(sample/name).write_bytes(b'fixture')
        if callback:callback(1,1,sample)
        return [sample]
    monkeypatch.setattr('paforge.pipeline.run_batch',fake_run)
    monkeypatch.setattr('paforge.io.plot_preview',lambda sample:Path(sample)/'overview.png')
    monkeypatch.setattr('paforge.visualization.preview_generated_sample',lambda *a,**k:['generated.png'])
    monkeypatch.setattr(ui,'_bundle',lambda folder:str(Path(folder)/'bundle.zip'))
    overview,gallery,bundle,report=ui._ui_run_complete_workflow(json.dumps(plan),True,str(tmp_path/'run'),
        '1/cm','1/cm',6,2,2,42,1,64,100,8,700,70,'das',2,False,.025)
    assert overview.endswith('overview.png') and gallery==['generated.png']
    assert bundle.endswith('bundle.zip')
    assert json.loads(report)['generated_samples']==1
