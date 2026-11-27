"""No network or experimental-data claims: all publisher payloads here are mocked."""
import hashlib
import io
import json
from pathlib import Path
import zipfile

import pytest
from typer.testing import CliRunner
from paforge import dataset_hub as hub
from paforge.cli import app


class Response(io.BytesIO):
    def __init__(self, data: bytes, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {}


def test_offline_registry_covers_all_requested_sources():
    assert {'openqpai','oadat','paiskintone','synthetic-skin','breast-pact',
            'pam3','photiqa','shanghaitech-vessel','cambridge-vascular'} <= set(hub.SOURCE_BY_ID)
    assert hub.SOURCE_BY_ID['oadat'].provider == 'guided'
    assert hub.SOURCE_BY_ID['breast-pact'].role.startswith('experimental reconstructed')


def test_zenodo_publisher_metadata_and_checksums(monkeypatch):
    metadata={'files':[{'key':'invision.zip','size':42, 'checksum':'md5:'+'a'*32,
                         'links':{'content':'https://zenodo.org/api/records/14044853/files/invision.zip/content'}},
                        {'key':'README_data.md','size':5,'checksum':'md5:'+'b'*32,
                         'links':{'content':'https://zenodo.org/README_data.md'}}]}
    monkeypatch.setattr(hub, '_json_url', lambda u:metadata)
    result=hub.discover('openqpai')
    assert result['assets'][0]['checksum_algorithm']=='md5'
    assert len(hub.pick_assets('openqpai',result['assets'],'starter'))==2
    assert len(hub.pick_assets('openqpai',result['assets'],'full'))==2


def test_dryad_discovery_and_filename(monkeypatch):
    def fake(url):
        if url.endswith('/versions'):
            return {'_embedded':{'stash:versions':[{'id':2,'versionNumber':1},{'id':6,'versionNumber':2}]}}
        return {'_embedded':{'stash:files':[{'id':12,'path':'README.md','size':18,
                    'digestType':'md5','digest':'a'*32}]}}
    monkeypatch.setattr(hub,'_json_url',fake)
    assets=hub.discover('breast-pact')['assets']
    assert assets[0]['url']=='https://datadryad.org/api/v2/files/12/download'
    assert assets[0]['checksum']=='a'*32
    assert len(hub.pick_assets('breast-pact',assets,'starter'))==1


def test_guard_unsafe_file_names_and_urls():
    for n in ('../test','/abs','dir/file','', '.env','..\\bad'):
        with pytest.raises(hub.SourceError): hub._basename(n)
    with pytest.raises(hub.SourceError):hub._https('http://zenodo.org/unsafe')
    with pytest.raises(hub.SourceError):hub._https('file:///etc/passwd')


def test_stream_resume_correct_range(monkeypatch,tmp_path):
    content=b'0123456789abcdefghij'; digest=hashlib.md5(content).hexdigest()
    a={'filename':'test.bin','url':'https://zenodo.org/test','bytes':len(content),
       'checksum_algorithm':'md5','checksum':digest}
    folder=tmp_path/'samples';folder.mkdir()
    (folder/'test.bin.part').write_bytes(content[:8])
    def fake(url,headers=None,timeout=40):
        assert headers['Range']=='bytes=8-'
        return Response(content[8:],status=206,headers={'Content-Range':'bytes 8-19/20'})
    monkeypatch.setattr(hub,'_open',fake)
    res=hub.fetch_asset(a,folder)
    assert res['status']=='downloaded'
    assert (folder/'test.bin').read_bytes()==content
    assert not (folder/'test.bin.part').exists()
    assert hub.fetch_asset(a,folder)['status']=='already_verified'


def test_range_ignored_restarts_instead_of_appending(monkeypatch,tmp_path):
    full=b'goodfile';a={'filename':'test.bin','url':'https://zenodo.org/test','bytes':len(full),
         'checksum_algorithm':'md5','checksum':hashlib.md5(full).hexdigest()}
    folder=tmp_path/'data';folder.mkdir()
    (folder/'test.bin.part').write_bytes(b'junk')
    monkeypatch.setattr(hub,'_open',lambda *args,**kwargs:Response(full,status=200))
    hub.fetch_asset(a,folder)
    assert (folder/'test.bin').read_bytes()==full


def test_bad_checksum_and_partial_transfer_remain_untrusted(monkeypatch,tmp_path):
    a={'filename':'test.bin','url':'https://zenodo.org/test','bytes':5,
       'checksum_algorithm':'md5','checksum':hashlib.md5(b'hello').hexdigest()}
    monkeypatch.setattr(hub,'_open',lambda *args,**kwargs:Response(b'jello'))
    with pytest.raises(hub.DownloadError,match='checksum mismatch'):
        hub.fetch_asset(a,tmp_path)
    assert not (tmp_path/'test.bin').exists()
    assert not (tmp_path/'test.bin.part').exists()
    monkeypatch.setattr(hub,'_open',lambda *args,**kwargs:Response(b'he'))
    with pytest.raises(hub.DownloadError,match='incomplete transfer'):
        hub.fetch_asset(a,tmp_path)
    assert (tmp_path/'test.bin.part').exists()


def test_download_budget_guided_and_manifest(monkeypatch,tmp_path):
    fake=[{'filename':'README_data.md','bytes':5,'url':'https://zenodo.org/README_data.md',
            'checksum_algorithm':None,'checksum':None}]
    monkeypatch.setattr(hub,'discover',lambda sid:{'assets':fake,'state':'discoverable'})
    monkeypatch.setattr(hub,'_open',lambda *args,**kwargs:Response(b'abcde'))
    out=hub.run_download(['openqpai','oadat'],tmp_path/'data',accept_licenses=True)
    assert out['sources']['openqpai']['status']=='complete_selected'
    assert out['sources']['oadat']['status']=='guided_external_download'
    report=json.loads((tmp_path/'data'/'download_report.json').read_text())
    assert report['sources']['oadat']['status']=='guided_external_download'
    assert (tmp_path/'data'/'openqpai'/'README_data.md').read_bytes()==b'abcde'
    with pytest.raises(ValueError,match='accept-licenses'):
        hub.run_download(['openqpai'],tmp_path)


def test_mock_catalog_dry_run_still_reports_guided(monkeypatch,tmp_path):
    monkeypatch.setattr(hub,'discover',lambda sid:{'assets':[{'filename':'DATA.zip','bytes':6_000_000_000,'checksum':None,'checksum_algorithm':None,'url':'https://datadryad.org/a'}]})
    report=hub.run_download(['breast-pact','cambridge-vascular'],tmp_path,dry_run=True,profile='full')
    assert report['sources']['breast-pact']['assets'][0]['status']=='skipped_file_budget'
    assert report['sources']['cambridge-vascular']['status']=='no_files_downloaded'
    assert not (tmp_path/'data').exists()


def test_safe_unpack_zip_traversal_and_symlink(tmp_path):
    good=tmp_path/'test.zip'
    with zipfile.ZipFile(good,'w') as z:z.writestr('safe/file.txt','abc')
    hub.safe_unpack(good,tmp_path/'extracted')
    assert (tmp_path/'extracted/safe/file.txt').read_text()=='abc'
    bad=tmp_path/'bad.zip'
    with zipfile.ZipFile(bad,'w') as z:z.writestr('../escape.txt','evil')
    with pytest.raises(hub.DownloadError,match='Unsafe path'):
        hub.safe_unpack(bad,tmp_path/'extracted')


def test_cli_help_and_list_command():
    r=CliRunner().invoke(app,['datasets','list'])
    assert r.exit_code==0, r.output
    assert len(json.loads(r.output))>=12
    r=CliRunner().invoke(app,['datasets','download','--help'])
    assert r.exit_code==0
    assert '--accept-licenses' in r.output


def test_unknown_checksum_html_redirect_rejected(monkeypatch,tmp_path):
    fake = {'filename':'public.zip','bytes':0,'estimated_bytes':10000,
            'url':'https://www.repository.cam.ac.uk/bitstreams/example/download',
            'checksum':None,'checksum_algorithm':None}
    monkeypatch.setattr(hub,'_open',lambda *args,**kwargs:Response(b'<html>Login required</html>'))
    with pytest.raises(hub.DownloadError,match='valid ZIP'):
        hub.fetch_asset(fake,tmp_path)
    assert not (tmp_path/'public.zip').exists()


def test_cambridge_publisher_bitstreams_and_starter():
    source=hub.discover('cambridge-vascular')
    assert len(source['assets'])==4
    chosen=hub.pick_assets('cambridge-vascular',source['assets'],'starter')
    assert [a['filename'] for a in chosen]==['InSilico.zip']
    assert chosen[0]['estimated_bytes']>400_000_000
    assert chosen[0]['bytes']==0  # publisher site rounded; not an exact checksum/length
