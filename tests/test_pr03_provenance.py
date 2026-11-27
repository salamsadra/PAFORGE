import pytest
from paforge.hybrid.provenance import BenchmarkRun, validate_group_splits, file_sha256


def run(**kwargs):
    fields=dict(run_id='r1',project='PATBox',source_commit='a'*40,
        acquisition_id='A',phantom_id='P',device_id='D',split='train',algorithm='DAS',
        input_sha256='b'*64,software_versions={'MATLAB':'R2025b','k-Wave':'1.4'},
        configuration={'sound_speed_m_s':1500},output_filename='recon.h5')
    fields.update(kwargs)
    return BenchmarkRun(**fields)


def test_manifest_roundtrip(tmp_path):
    f=run().save(tmp_path/'manifest.json')
    assert BenchmarkRun.load(f)==run()
    with pytest.raises(FileExistsError):run().save(f)


def test_manifest_missing_commit_or_software():
    with pytest.raises(ValueError,match='commit'):
        run(source_commit='main')
    with pytest.raises(ValueError,match='Software'):
        run(software_versions={})


def test_group_disjointness():
    a=run(split='train')
    b=run(split='test',run_id='r2',acquisition_id='B',phantom_id='P')
    with pytest.raises(ValueError,match='leakage'):
        validate_group_splits([a,b])
    c=run(split='test',run_id='r3',acquisition_id='C',phantom_id='Q',device_id='E')
    assert len(validate_group_splits([a,c]))==2
    assert len(validate_group_splits([a,c],group_by='device_id'))==2


def test_file_sha256(tmp_path):
    path=tmp_path/'x';path.write_bytes(b'abc')
    assert file_sha256(path).startswith('ba7816')
