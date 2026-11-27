"""Provenance-gated waveform calibration and held-out sim-to-real *signal-domain* evaluation.

This module never claims sensor pressure calibration, optical-parameter validity,
image reconstruction accuracy, or human-data validation on synthetic input.
All fitting uses TRAIN only; evaluation reads frozen TEST only. Group identities
must be disjoint. Profiles describe domain-adaptation filters, not transducer IRs.
"""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
from dataclasses import dataclass

import h5py
import numpy as np
from scipy.signal import welch
from scipy.stats import wasserstein_distance

REQUIRED = {"path", "group_id", "domain", "split", "signal_key", "detector_axis", "sample_rate_hz", "source_id"}
DOMAIN = {"experimental", "simulation"}
SPLITS = {"train", "validation", "test"}


def _finite_float(s, name):
    try: v=float(s)
    except (ValueError, TypeError) as e: raise ValueError(f"Invalid {name}: {s!r}") from e
    if not np.isfinite(v) or v<=0: raise ValueError(f"{name} must be finite and positive")
    return v


def _truthy(v):
    return str(v).strip().lower() in {"true", "1", "yes", "y"}


def _sha256(path):
    sha=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda: f.read(1024*1024),b""):sha.update(block)
    return sha.hexdigest()


@dataclass
class Observation:
    signal: np.ndarray
    sample_rate_hz: float
    group_id: str
    domain: str
    split: str
    source_id: str
    filename: str
    sha256: str
    quiet_interval: tuple[int,int] | None


def _attr_to_text(a):
    return a.decode(errors="replace") if isinstance(a,bytes) else str(a)


def load_observation(row, manifest_dir):
    file_path=Path(row['path'])
    if not file_path.is_absolute():file_path=(manifest_dir/file_path).resolve()
    if not file_path.is_file():raise FileNotFoundError(file_path)
    if file_path.suffix.lower() not in {".h5",".hdf5"}:
        raise ValueError("v0.3 accepts only uncompressed HDF5; convert other formats using import-data")
    dom=row['domain'].strip().lower()
    if dom not in DOMAIN:raise ValueError(f"Unknown domain {dom}")
    split=row['split'].strip().lower()
    if split not in SPLITS:raise ValueError(f"Unknown split {split}")
    group=row['group_id'].strip()
    if not group:raise ValueError("group_id cannot be blank")
    key=row['signal_key'].strip().lstrip('/')
    axis=int(row['detector_axis'])
    if axis not in (0,1):raise ValueError("detector_axis must be 0 or 1")
    fs=_finite_float(row['sample_rate_hz'],'sample_rate_hz')
    source=row['source_id'].strip()
    if not source:raise ValueError("source_id cannot be blank")
    with h5py.File(file_path,'r') as f:
        if key not in f:raise KeyError(f"{key} missing in {file_path}")
        if f[key].ndim!=2:raise ValueError(f"Expected 2D slice [detector,time] in {file_path}")
        if dom=="experimental" and ('synthetic' in f.attrs and _truthy(f.attrs['synthetic'])):
            raise ValueError(f"SYNTHETIC dataset cannot be declared experimental: {file_path}")
        # Explicit provenance is required. This is an author-attested field, not independently verified.
        if dom=="experimental" and not _truthy(row.get('experimental_attestation','')):
            raise ValueError("Experimental row requires experimental_attestation=yes and traceable source_id")
        a=np.asarray(f[key][:], dtype=np.float64)
        if axis==1:a=a.T
        if dom=="simulation" and 'synthetic' in f.attrs and not _truthy(f.attrs['synthetic']):
            raise ValueError("A file marked nonsynthetic cannot be declared simulation")
    if min(a.shape)<2 or a.shape[1]<32:raise ValueError("Requires >=2 detectors, >=32 time samples")
    if not np.isfinite(a).all():raise ValueError("Nonfinite traces")
    qa=row.get('quiet_start','').strip(); qb=row.get('quiet_stop','').strip()
    if bool(qa)!=bool(qb):raise ValueError("quiet_start and quiet_stop must be provided together")
    quiet=None
    if qa:
        quiet=(int(qa),int(qb))
        if not (0<=quiet[0]<quiet[1]<=a.shape[1]) or quiet[1]-quiet[0]<16:
            raise ValueError("Invalid quiet window: need >=16 time samples")
    return Observation(a,fs,group,dom,split,source,str(file_path),_sha256(file_path),quiet)


def read_manifest(csv_file):
    csv_file=Path(csv_file)
    with csv_file.open(newline='',encoding='utf8') as f:
        reader=csv.DictReader(f)
        missing=REQUIRED-set(reader.fieldnames or ())
        if missing:raise ValueError(f"Missing manifest columns: {sorted(missing)}")
        rows=list(reader)
    if not rows:raise ValueError("Manifest is empty")
    seen={}
    for r in rows:
        key=(r['domain'],r['group_id'])
        if key in seen and r['split']!=seen[key]:
            raise ValueError(f"LEAKAGE: group {key} assigned to both {seen[key]} and {r['split']}")
        seen[key]=r['split']
    items=[load_observation(r,csv_file.resolve().parent) for r in rows]
    for d in DOMAIN:
        for s in ['train','test']:
            if not any(o.domain==d and o.split==s for o in items):
                raise ValueError(f"Need {d} {s} measurements; test must be unseen groups")
    train=items[0]
    dims={(o.signal.shape[0],o.signal.shape[1],round(o.sample_rate_hz,3)) for o in items}
    if len(dims)!=1:
        raise ValueError("Acquisition mismatch: all arrays must share detector count, time samples and sampling rate. Harmonize/resample with a documented adapter first; never silently warp measurements.")
    return items


def _psd(ob, nperseg=128):
    _,p=welch(ob.signal,fs=ob.sample_rate_hz,axis=1,nperseg=min(nperseg,ob.signal.shape[1]),scaling='spectrum',detrend='constant')
    p=np.asarray(p.mean(axis=0),dtype=np.float64)
    p=np.maximum(p,1e-30)
    return p/np.sum(p)


def _rms(o):return float(np.sqrt(np.mean(o.signal**2)))


def _JS(p,q):
    p=np.asarray(p,dtype=float);q=np.asarray(q,dtype=float)
    p=p/max(p.sum(),1e-30);q=q/max(q.sum(),1e-30)
    m=(p+q)*.5
    return float(.5*(np.sum(p*np.log(np.maximum(p,1e-30)/np.maximum(m,1e-30)))+
                      np.sum(q*np.log(np.maximum(q,1e-30)/np.maximum(m,1e-30)))))


def _avg_spectrum(items):return np.mean([_psd(o) for o in items],axis=0)


def fit_profile(observations, quiet_mode='none'):
    """Statistical train-only match: log-PSD transfer and digital scale. NOT transducer calibration.

    A nuisance spectral filter fit on organ and device recordings conflates the tissue
    source spectrum with detector response; interpretation must reflect that.
    """
    tr=[o for o in observations if o.split=='train']
    r=[o for o in tr if o.domain=='experimental']
    s=[o for o in tr if o.domain=='simulation']
    if not r or not s:raise ValueError("Need training signals from both domains")
    realpsd=_avg_spectrum(r);simpsd=_avg_spectrum(s)
    # bounded gain so profile cannot create spurious out-of-band peaks
    response=np.clip(np.sqrt((realpsd+1e-10)/(simpsd+1e-10)),.25,4.)
    scale=float(np.median([_rms(o) for o in r])/max(np.median([_rms(o) for o in s]),1e-12))
    profile={'spectral_gain':response.tolist(), 'digital_gain':scale,
        'n_frequency_bins':len(response),'fs_hz':r[0].sample_rate_hz,'time_samples':r[0].signal.shape[1],
        'detectors':r[0].signal.shape[0],
        'fit_groups':{'experimental':sorted(set(o.group_id for o in r)),
                      'simulation':sorted(set(o.group_id for o in s))},
        'caveat': 'Data-driven spectral domain matching (training groups only), NOT measured instrument impulse response or physical pressure calibration.'}
    if quiet_mode=='extract':
        if not all(o.quiet_interval is not None for o in r):raise ValueError("Every real train record must specify quiet_start, quiet_stop")
        blocks=[]
        for o in r:
            a,b=o.quiet_interval
            blocks.append(o.signal[:,a:b])
        length=min(x.shape[1] for x in blocks)
        noise=np.concatenate([x[:,:length] for x in blocks],axis=0)
        profile['quiet_window_length']=length
        return profile,noise
    return profile,None


def _adapt_record(signal,profile):
    """Interpolate TRAIN-ONLY estimated transfer to fft length and rescale.

    The zero-frequency bin can alter baselines; this is not a reconstruction operator.
    """
    x=np.asarray(signal,dtype=np.float64)
    n=x.shape[1]
    src=np.linspace(0,1,len(profile['spectral_gain']))
    tgt=np.linspace(0,1,n//2+1)
    gain=np.interp(tgt,src,np.asarray(profile['spectral_gain']))
    # Spectral shaping with output per-record energy reset to training gain; keeps
    # global amplitude statistics comparable while preventing all-zero input blowups.
    transformed=np.fft.irfft(np.fft.rfft(x,axis=1)*gain[None,:],n=n,axis=1)
    return transformed*profile['digital_gain']


def _metric(real,sim):
    p=_avg_spectrum(real);q=_avg_spectrum(sim)
    rr=np.log10(np.maximum([_rms(o) for o in real],1e-30))
    ss=np.log10(np.maximum([_rms(o) for o in sim],1e-30))
    return {'spectral_js_nats':_JS(p,q),
            'log10_rms_wasserstein':float(wasserstein_distance(rr,ss)),
            'n_real_records':len(real),'n_sim_records':len(sim),
            'n_real_independent_groups':len(set(o.group_id for o in real)),
            'n_sim_independent_groups':len(set(o.group_id for o in sim))}


def heldout_evaluate(observations, profile):
    """Frozen test evaluation; absolutely no fitting or threshold selection on test."""
    test=[o for o in observations if o.split=='test']
    real=[o for o in test if o.domain=='experimental']
    sim=[o for o in test if o.domain=='simulation']
    train_real=set(profile['fit_groups']['experimental'])
    train_sim=set(profile['fit_groups']['simulation'])
    if train_real.intersection(o.group_id for o in real):raise ValueError('real train/test group leakage')
    if train_sim.intersection(o.group_id for o in sim):raise ValueError('sim train/test group leakage')
    after=[Observation(_adapt_record(o.signal,profile),o.sample_rate_hz,o.group_id,o.domain,o.split,o.source_id,o.filename,o.sha256,o.quiet_interval) for o in sim]
    return {'before':_metric(real,sim),'after':_metric(real,after),
        'interpretation':'Signal-distribution matching only. Improvement in spectral JS is not reconstruction accuracy or clinical/generalization proof.'}


def run_experiment(manifest, out, quiet_mode='none'):
    items=read_manifest(manifest)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    profile,noise=fit_profile(items,quiet_mode)
    verdict=heldout_evaluate(items,profile)
    provenance={'records':[{'domain':o.domain,'split':o.split,'group_id':o.group_id,
        'source_id':o.source_id,'path':o.filename,'sha256':o.sha256,'sample_rate_hz':o.sample_rate_hz} for o in items]}
    sources=sorted(set(o.source_id for o in items if o.domain=='experimental'))
    result={'status':'held-out real-vs-synthetic signal-distribution comparison' ,
        'real_sources_user_attested_not_independently_authenticated':sources,
        'calibration':profile,'heldout_metrics':verdict,'provenance':provenance,
        'limitations':['Statistical spectral matching is not measurement of physical detector impulse response.',
          'Not a reference-image reconstruction benchmark.',
          'No evidence of inverse model generalization without downstream experiment.',
          'An author-provided provenance assertion is not independent verification.',
          'Differences in anatomy/illumination can dominate spectral differences.']}
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8')
    (out/'calibration_profile.json').write_text(json.dumps(profile,indent=2,allow_nan=False),encoding='utf8')
    if noise is not None:
        np.savez_compressed(out/'empirical_quiet_noise.npz',noise=noise.astype(np.float32))
    return result
