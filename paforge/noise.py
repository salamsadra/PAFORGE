"""Independent noise components. Noise OFF bypasses every engine exactly."""
from pathlib import Path
import h5py
import numpy as np
from scipy.signal import lfilter, resample


def _empirical_array(path):
    p=Path(path)
    if not p.is_file(): raise FileNotFoundError(f"Empirical noise file does not exist: {p}")
    if p.suffix==".npy": arr=np.load(p,allow_pickle=False)
    elif p.suffix==".npz":
        with np.load(p,allow_pickle=False) as f:
            if "noise" not in f:raise ValueError(".npz requires key 'noise'")
            arr=f["noise"]
    elif p.suffix in (".h5",".hdf5"):
        with h5py.File(p) as f:
            if "noise" not in f:raise ValueError(".h5 requires dataset /noise")
            arr=f["noise"][:]
    else:raise ValueError("Empirical noise supports .npy/.npz/.h5")
    if arr.ndim not in (1,2) or arr.size<8:raise ValueError("Empirical noise must be a nonempty 1D or 2D array")
    if not np.isfinite(arr).all():raise ValueError("Empirical noise contains non-finite values")
    return np.asarray(arr,dtype=np.float64)


def add_noise(clean,config,rng):
    """Return noisy traces and named components, array shape (sensors,time)."""
    cfg=config.noise
    if not cfg.enabled or not cfg.engines:
        return clean.copy(), {}
    if len(set(cfg.engines))!=len(cfg.engines):raise ValueError("Duplicate noise engines")
    shape=clean.shape
    amp=float(np.sqrt(np.mean(np.asarray(clean,dtype=np.float64)**2)))
    # A nonzero reference for silence prevents zero-variance normalization failures.
    scale=max(amp,1e-12)
    contributions={}
    for kind in cfg.engines:
        if kind=="gaussian":
            v=rng.standard_normal(shape)*cfg.relative_std*scale
        elif kind=="colored":
            v=lfilter([1.0],[1.0,-cfg.colored_ar],rng.standard_normal(shape),axis=1)
            v*=cfg.relative_std*scale/max(float(np.std(v)),1e-12)
        elif kind=="impulsive":
            hits=rng.random(shape)<cfg.impulse_probability
            v=hits*rng.normal(size=shape)*cfg.impulse_amplitude*scale
        elif kind=="drift":
            # Smooth per-channel low-frequency drift via stochastic low-pass.
            v=lfilter([0.005],[1.,-.995],rng.normal(size=shape),axis=1)
            v*=cfg.drift_amplitude*scale/max(float(np.std(v)),1e-12)
        elif kind=="empirical":
            source=_empirical_array(cfg.empirical_path)
            if source.ndim==1:source=source[None,:]
            choices=rng.integers(0,source.shape[0],size=shape[0])
            v=source[choices]
            if v.shape[1]!=shape[1]:v=resample(v,shape[1],axis=1)
            v=v-v.mean(axis=1,keepdims=True)
            v*=cfg.relative_std*scale/max(float(np.std(v)),1e-12)
        else:raise ValueError(kind)
        contributions[kind]=np.asarray(v,dtype=np.float32)
    combined=clean.astype(np.float64).copy()
    for v in contributions.values():combined+=v
    return combined.astype(np.float32), contributions


def measured_added_snr_db(clean,changed):
    resid=changed.astype(float)-clean.astype(float)
    signal=float(np.sqrt(np.mean(clean.astype(float)**2)))
    noise=float(np.sqrt(np.mean(resid**2)))
    if noise==0:return None
    if signal==0:return float("-inf")
    return 20*np.log10(signal/noise)
