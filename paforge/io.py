"""PA-FORGE native HDF5; intentionally NOT claimed to pass IPASC/PACFISH QC."""
import json
from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _dataset(group,name,data):
    return group.create_dataset(name,data=data,compression="gzip",compression_opts=2,shuffle=True)


def export_result(root,idx,config,tissue,clean,raw,traces_noise,dt,positions,rec,info):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    stem=f"{config.anatomy}_{idx:05d}"
    sample=root/stem;sample.mkdir(exist_ok=True)
    with h5py.File(sample/"raw_signals.h5","w") as f:
        f.attrs["format"]="PA-FORGE-native-0.2 (NOT certified IPASC)"
        f.attrs["synthetic"] = True
        f.attrs["time_unit"]="s"; f.attrs["pressure_unit"]="Pa (unvalidated simulator units)"
        f.attrs["source"]="PA-FORGE approximate 2D wave solver"
        f.attrs["config_json"]=config.model_dump_json()
        _dataset(f,"raw_data",raw)
        _dataset(f,"time_s",np.arange(raw.shape[1])*dt)
        _dataset(f,"detector_positions_m",positions)
        _dataset(f,"clean_detector_signals",clean)
        if traces_noise:
            noise=f.create_group("noise_components")
            for key,value in traces_noise.items():_dataset(noise,key,value)
        f.attrs["sampling_rate_hz"]=1/dt
        f.attrs["acquisition_wavelength_nm"]=config.wavelength_nm
        f.attrs["anatomy"] = config.anatomy
    if config.export_images and rec is not None:
        with h5py.File(sample/"reconstruction.h5","w") as f:
            f.attrs["method"]=config.reconstruction_method+" (unvalidated synthetic physics)"
            f.attrs["quantitative"] = False
            f.attrs["fov_m"] = config.fov_m
            _dataset(f,"image",rec)
        plt.imsave(sample/"reconstruction.png",rec,cmap="magma")
    if config.anatomy=="vascular" and config.export_truth:
        from matplotlib.colors import ListedColormap, BoundaryNorm
        cm=ListedColormap(["#111927","#e45454","#508cfc","#efba57","#9b9b9b"])
        plt.imsave(sample/"vessel_types.png",tissue.vessel_type_mask,cmap=cm,vmin=0,vmax=4)
    if config.export_truth:
        with h5py.File(sample/"ground_truth.h5","w") as f:
            f.attrs["note"]="Synthetic tissue ground truth; source is procedural or trained-phantom-informed, not human measurements"
            f.attrs["fov_m"]=config.fov_m
            for name in ("labels","absorption_m_inv","scattering_reduced_m_inv","sound_speed_m_s","gruneisen","vessel_mask","vessel_type_mask","fluence_j_m2","initial_pressure_pa"):
                _dataset(f,name,getattr(tissue,name))
    with open(sample/"manifest.json","w",encoding="utf-8") as f:
        json.dump({"sample_id":stem,"synthetic":True,"information":info,"config":config.model_dump(mode="json"),
                   "raw_format":"PA-FORGE native HDF5, NOT IPASC conformance-certified",
                   "unit_warning":"approximate optical and 2D acoustic solver; not experimentally calibrated"},f,indent=2,allow_nan=False)
    return sample


def plot_preview(sample_folder):
    sample_folder=Path(sample_folder)
    with h5py.File(sample_folder/"raw_signals.h5") as f:
        raw=f["raw_data"][:]; dt=1/float(f.attrs["sampling_rate_hz"])
    with h5py.File(sample_folder/"ground_truth.h5") as f:
        p0=f["initial_pressure_pa"][:]
    with h5py.File(sample_folder/"reconstruction.h5") as f:
        rec=f["image"][:]
    anatomical=sample_folder.name.startswith("vascular_")
    fig,axes=plt.subplots(1,4 if anatomical else 3,figsize=(15 if anatomical else 12,3.5),layout="constrained")
    if anatomical:
        with h5py.File(sample_folder/"ground_truth.h5") as f:
            vessel_types=f["vessel_type_mask"][:]
        from matplotlib.colors import ListedColormap
        axes[0].imshow(vessel_types,cmap=ListedColormap(["#101827","#e45454","#508cfc","#efba57","#888888"]),vmin=0,vmax=4,origin="lower")
        axes[0].set_title("Vessels: red artery / blue vein / gold capillary")
        axes=axes[1:]
    axes[0].imshow(p0,cmap="magma",origin="lower");axes[0].set_title("Initial pressure (synthetic)")
    axes[1].imshow(raw,aspect="auto",cmap="RdBu_r",extent=[0,raw.shape[1]*dt*1e6,raw.shape[0],0]);axes[1].set_title("Raw sensor traces");axes[1].set_xlabel("Time (μs)")
    axes[2].imshow(rec,cmap="magma",origin="lower");axes[2].set_title("Reconstructed image (qualitative)")
    for a in (axes[0],axes[2]):a.set_xticks([]);a.set_yticks([])
    out=sample_folder/"overview.png";fig.savefig(out,dpi=150);plt.close(fig)
    return out
