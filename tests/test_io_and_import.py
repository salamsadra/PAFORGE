import h5py
import numpy as np
import pytest
from paforge.config import SimConfig
from paforge.pipeline import run_batch
from paforge.datasets import inspect_file,import_measurement


def test_separate_exports_and_native_metadata(tmp_path):
    cfg=SimConfig(grid_size=64,transducers=16,time_steps=200,detector_filter=False)
    paths=run_batch(cfg,tmp_path)
    p=paths[0]
    assert (p/"raw_signals.h5").exists()
    assert (p/"reconstruction.h5").exists()
    assert (p/"ground_truth.h5").exists()
    assert (p/"manifest.json").exists()
    with h5py.File(p/"raw_signals.h5") as f:
        assert f["raw_data"].shape==(16,200)
        assert f["detector_positions_m"].shape==(16,3)
        assert np.array_equal(f["raw_data"][:],f["clean_detector_signals"][:])
        assert not f.attrs["format"].startswith("IPASC")


def test_inspect_and_import_explicit_axes(tmp_path):
    path=tmp_path/"source.h5"
    with h5py.File(path,"w") as f:f.create_dataset("measurements/traces",data=np.ones((100,8)))
    inspected=inspect_file(path)
    assert inspected["measurements/traces"]["shape"]==(100,8)
    dest=import_measurement(path,"measurements/traces",tmp_path/"imported.h5",1,20e6,800)
    with h5py.File(dest) as f:
        assert f["raw_data"].shape==(8,100)
        assert not bool(f.attrs["source_coordinates_verified"])


def test_incorrect_import_without_metadata_rejected(tmp_path):
    with pytest.raises(ValueError):
        import_measurement(tmp_path/"missing.npy","data",tmp_path/"out.h5",0,-1,800)
