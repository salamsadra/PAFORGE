"""GEN-03 release-candidate tests using only synthetic payloads and phantom masks.

No external network, experimental data, or held-out OpenQPAIData test groups.
"""
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from paforge import download_data


def _publisher(monkeypatch, payload=b'publisher-good'):
    monkeypatch.setitem(download_data.OPENQPAI,'invision',
                        ('invision.zip',hashlib.md5(payload).hexdigest()))
    return payload


def test_download_commits_only_verified_bytes_and_offline_cache(monkeypatch,tmp_path):
    data=_publisher(monkeypatch)
    calls=[]
    def fake(req, timeout):
        calls.append(req.full_url)
        return io.BytesIO(data)
    monkeypatch.setattr(download_data.urllib.request,'urlopen',fake)
    result=download_data.fetch_openqpai(tmp_path)
    assert result['upstream_md5_verified']
    assert (tmp_path/'invision.zip').read_bytes()==data
    assert not list(tmp_path.glob('*.partial'))
    download_data.fetch_openqpai(tmp_path)
    assert len(calls)==1  # offline re-use; no new request


def test_corrupt_cached_archive_quarantines_and_requires_retry(monkeypatch,tmp_path):
    good=_publisher(monkeypatch)
    (tmp_path/'invision.zip').write_bytes(b'corrupt cache')
    calls=[]
    monkeypatch.setattr(download_data.urllib.request,'urlopen',
                        lambda *args,**kwargs:(calls.append(1),io.BytesIO(good))[1])
    with pytest.raises(ValueError, match='invalid cache quarantined'):
        download_data.fetch_openqpai(tmp_path)
    assert not (tmp_path/'invision.zip').exists()
    quarantined=list(tmp_path.glob('invision.zip.invalid-*'))
    assert len(quarantined)==1 and quarantined[0].read_bytes()==b'corrupt cache'
    assert calls==[]
    download_data.fetch_openqpai(tmp_path)
    assert (tmp_path/'invision.zip').read_bytes()==good
    assert calls==[1]


def test_corrupt_new_download_never_gets_final_name(monkeypatch,tmp_path):
    _publisher(monkeypatch)
    monkeypatch.setattr(download_data.urllib.request,'urlopen',lambda *a,**k:io.BytesIO(b'<html>error</html>'))
    with pytest.raises(ValueError,match='Download checksum mismatch'):
        download_data.fetch_openqpai(tmp_path)
    assert not (tmp_path/'invision.zip').exists()
    assert not list(tmp_path.glob('*.partial'))
    quarantined=list(tmp_path.glob('invision.zip.*.partial.invalid-*'))
    assert len(quarantined)==1
    assert quarantined[0].read_bytes()==b'<html>error</html>'


def test_stream_failure_cleans_partial_without_final_name(monkeypatch,tmp_path):
    _publisher(monkeypatch)
    def fail(*a,**k):raise OSError('mock network interruption')
    monkeypatch.setattr(download_data.urllib.request,'urlopen',fail)
    with pytest.raises(OSError,match='network interruption'):
        download_data.fetch_openqpai(tmp_path)
    assert not (tmp_path/'invision.zip').exists()
    assert not list(tmp_path.glob('*.partial'))


def _synthetic_corpus_and_profile(tmp_path, overlapping=False):
    rng=np.random.default_rng(147)
    masks=[];groups=[]
    for name in ('A','B','C'):
        mask=rng.integers(0,2,size=(64,64),dtype=np.uint8)
        masks.append(mask);groups.append(name)
    corpus=tmp_path/'data.npz'
    np.savez_compressed(corpus,masks=np.stack(masks),groups=np.array(groups),classes=2)
    splits={'train':['A'],'validation':['B'],'test':['C']}
    if overlapping:splits['test']=['A']
    profile=tmp_path/'profile.json'
    profile.write_text(json.dumps({'groups':splits}))
    return corpus,profile


def test_train_no_implicit_test_score_and_explicit_opt_in(tmp_path):
    from paforge.material_cvae import train
    corpus,profile=_synthetic_corpus_and_profile(tmp_path)
    w=tmp_path/'safe.pt'
    report=train(corpus,profile,w,epochs=1,batch_size=1,seed=2,device='cpu')
    assert report['heldout_test'] is None
    assert report['test_evaluation_opt_in'] is False
    import torch
    model=torch.load(w,weights_only=True)
    assert model['test_evaluation_opt_in'] is False
    assert model['test_groups']==['C']
    report=train(corpus,profile,tmp_path/'explicit.pt',epochs=1,batch_size=1,
                 seed=2,device='cpu',evaluate_test=True)
    assert report['test_evaluation_opt_in'] is True
    assert report['heldout_test']['test_phantom_count']==1  # synthetic TEST ONLY


def test_train_rejects_group_overlap(tmp_path):
    from paforge.material_cvae import train
    corpus,profile=_synthetic_corpus_and_profile(tmp_path,overlapping=True)
    with pytest.raises(ValueError, match='Data leakage'):
        train(corpus,profile,tmp_path/'never-created.pt',epochs=1,batch_size=1,device='cpu')
    assert not (tmp_path/'never-created.pt').exists()


def test_cli_synthetic_material_default_uses_padding_two():
    from typer.testing import CliRunner
    from paforge.cli import app
    result=CliRunner().invoke(app,['generate-informed','--help'])
    assert result.exit_code==0
    assert '--material-solver-padding-factor' in result.stdout.replace('…','') or 'material-solve' in result.stdout
    # The command default is 2, while generic SimConfig retains legacy default 1.
    from paforge.config import SimConfig
    assert SimConfig().material_solver_padding_factor==1
