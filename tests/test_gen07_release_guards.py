"""Non-destructive release-output guards; no download, MPS, or experiment data."""
from pathlib import Path
from zipfile import ZipFile
import pytest
from paforge.config import SimConfig
from paforge.pipeline import run_batch
from paforge.ui import _bundle


def test_run_batch_refuses_existing_sample_before_simulation(tmp_path):
    target=tmp_path/'vascular_00000'
    target.mkdir()
    sentinel=target/'raw_signals.h5'
    sentinel.write_bytes(b'original scientific data')
    cfg=SimConfig(anatomy='vascular',samples=2)
    with pytest.raises(FileExistsError,match='Refusing to overwrite'):
        run_batch(cfg,tmp_path)
    assert sentinel.read_bytes()==b'original scientific data'
    assert not (tmp_path/'vascular_00001').exists()


def test_bundle_includes_current_samples_only(tmp_path):
    sample=tmp_path/'material_phantom_00000'
    sample.mkdir()
    (sample/'manifest.json').write_text('{}')
    (tmp_path/'do_not_include_test_data.txt').write_text('private')
    (tmp_path/'other_sample_00000').mkdir()
    (tmp_path/'other_sample_00000'/'old.h5').write_bytes(b'old')
    archive=Path(_bundle(tmp_path,only_samples=[sample]))
    with ZipFile(archive) as z:
        assert z.namelist()==['material_phantom_00000/manifest.json']
    with pytest.raises(FileExistsError,match='Refusing to overwrite'):
        _bundle(tmp_path,only_samples=[sample])


def test_bundle_rejects_outside_folder(tmp_path):
    inside=tmp_path/'output';inside.mkdir()
    outside=tmp_path/'test';outside.mkdir()
    with pytest.raises(ValueError,match='direct sample folders'):
        _bundle(inside,only_samples=[outside])
