"""Optical/material-data preparation from publisher OpenQPAIData InVision ZIP.

Training and optical statistics use TRAIN phantom groups only; wavelengths of a
phantom never cross split boundaries. No vessel/clinical anatomy labels assumed.
Units require affirmative user declaration from publisher documentation.
"""
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
from collections import defaultdict
import hashlib
import json
import re
import numpy as np
from scipy.ndimage import zoom
from .material_contracts import validate_optical_rows, class_count, file_sha256

PATTERN = re.compile(r'^invision/(labels|mua|musp)/([^/]+)\.npy$', re.IGNORECASE)
SPECTRAL = re.compile(r'^(.*?)_(\d{3,4})$')


def _resize(a, size=64, order=0):
    if a.shape == (size,size): return np.asarray(a)
    out=zoom(a, (size/a.shape[0],size/a.shape[1]),order=order,mode='nearest')
    return out[:size,:size]


def _read_np(archive, filename):
    with archive.open(filename) as f:
        data=np.load(BytesIO(f.read()),allow_pickle=False)
    if data.ndim!=2 or not np.isfinite(data).all():
        raise ValueError(f'Expected finite 2D map: {filename}')
    return data


def _partition(groups,seed=42,holdout_fraction=.2):
    if len(groups)<3:raise ValueError('At least three distinct phantom IDs are required for train/validation/test splitting')
    names=np.array(sorted(groups),dtype=object)
    rng=np.random.default_rng(seed);rng.shuffle(names)
    if len(names)<5:
        # One train, one validation, one test, when 3 groups are available.
        n_test=n_val=1
    else:
        n_test=max(1,int(round(len(names)*holdout_fraction)))
        n_val=max(1,int(round(len(names)*holdout_fraction)))
    return {'test':sorted(names[:n_test].tolist()),'validation':sorted(names[n_test:n_test+n_val].tolist()),
            'train':sorted(names[n_test+n_val:].tolist())}


def prepare_openqpai(archive_path, output_dir, absorption_unit, scattering_unit,
                     seed=42, max_groups=None, holdout_fraction=.2, declared_classes=None):
    """Create 64x64 grouped training masks and empirical optical profiles.

    absorption_unit/scattering_unit: '1/m' or '1/cm' as confirmed by user.
    A unit cannot be inferred merely from observed numerical magnitudes.
    """
    valid=('1/m','1/cm')
    if absorption_unit not in valid or scattering_unit not in valid:
        raise ValueError('Optical units must be explicitly supplied as 1/m or 1/cm; refer to the data publisher')
    factor_a=100. if absorption_unit=='1/cm' else 1.
    factor_s=100. if scattering_unit=='1/cm' else 1.
    archive_path=Path(archive_path)
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    if not archive_path.is_file():raise FileNotFoundError(archive_path)
    with ZipFile(archive_path) as z:
        index={'labels':{},'mua':{},'musp':{}}
        for entry in z.namelist():
            m=PATTERN.match(entry)
            if m:index[m.group(1).lower()][m.group(2)]=entry
        groups=sorted(index['labels'])
        if max_groups is not None:
            if max_groups<3:raise ValueError('max_groups must be >=3')
            groups=groups[:max_groups]
        if len(groups)<3:raise ValueError('The archive requires >=3 labeled phantoms, no duplicate wavelength pseudo-groups')
        split=_partition(groups,seed,holdout_fraction)
        labels_cache={}
        masks=[]; ids=[]
        for g in groups:
            a=_read_np(z,index['labels'][g]);
            if np.any(a<0) or np.max(a)>15 or not np.all(a==np.round(a)):
                raise ValueError(f'Unsupported material class codes in {g}; 0..15 integer codes expected')
            labels_cache[g]=a.astype(np.uint8)
            masks.append(_resize(a,64,0).astype(np.uint8));ids.append(g)
        n_classes=class_count(max(np.max(labels_cache[g]) for g in split['train']), declared_classes)
        if any(np.max(v)>=n_classes for v in labels_cache.values()):
            raise ValueError('Label outside frozen material taxonomy; declare classes independently of held-out masks')
        mask_file=output_dir/'material_corpus.npz'
        np.savez_compressed(mask_file,masks=np.stack(masks),groups=np.asarray(ids,dtype='U128'),classes=n_classes)
        rows=[]; unmatched=[]
        for k,entry in index['mua'].items():
            match=SPECTRAL.match(k)
            if not match:continue
            group,wavelength=match.group(1),int(match.group(2))
            if group not in split['train']:continue
            skey=k
            if skey not in index['musp']:
                unmatched.append(k);continue
            ab=_read_np(z,entry)*factor_a
            sc=_read_np(z,index['musp'][skey])*factor_s
            lab=labels_cache[group]
            if ab.shape!=lab.shape or sc.shape!=lab.shape:
                raise ValueError(f'Map shape mismatch for {k}; registration cannot be presumed')
            if np.any(ab<0) or np.any(sc<0):raise ValueError(f'Negative optical parameter values in {k}')
            if group not in split['train']:continue
            for cls in np.unique(lab):
                selection=lab==cls
                if selection.sum()<8:continue
                # One statistical observation per phantom/class/wavelength,
                # not tens of thousands of dependent pixels.
                av=ab[selection];sv=sc[selection]
                rows.append({'group':group,'label':int(cls),'wavelength_nm':wavelength,
                             'absorption_m_inv':float(np.median(av)),
                             'scattering_reduced_m_inv':float(np.median(sv)),
                             'absorption_iqr':float(np.subtract(*np.percentile(av,[75,25]))),
                             'scattering_iqr':float(np.subtract(*np.percentile(sv,[75,25]))),
                             'pixel_count':int(selection.sum())})
    if not rows:
        raise ValueError('No wavelength-matched mua/musp optical maps found; cannot fit an optical profile')
    profile={'format':'PA-FORGE-optical-profile-0.6','source':'OpenQPAIData InVision',
             'training_scope':'TRAIN phantom IDs only; no test/validation optical leakage',
             'units':'SI: 1/m','declared_source_units':{'mua':absorption_unit,'musp':scattering_unit},
             'classes':n_classes,'taxonomy_source':'declared' if declared_classes is not None else 'train_max',
             'groups':split,'measurements':rows,
             'unmatched_spectral_files':unmatched,
             'limitations':['Measured/derived phantom material maps, NOT human anatomical labels',
                'Group samples are correlated across wavelengths; no patient population model',
                'Acoustic parameters and Gruneisen are uncalibrated and are never inferred from these optical maps',
                'Optical distributions are train-only pooled group-level summaries, not calibrated posterior uncertainty']}
    (output_dir/'optical_profile.json').write_text(json.dumps(profile,indent=2),encoding='utf-8')
    manifest={'training_corpus':str(mask_file),'optical_profile':str(output_dir/'optical_profile.json'),
              'source_zip':str(archive_path),'source_zip_sha256':file_sha256(archive_path),'counts':{'groups':len(groups),'train':len(split['train']),
                 'validation':len(split['validation']),'test':len(split['test']),
                 'material_classes':n_classes,'optical_group_class_wavelength_rows':len(rows)},
              'split_by':'distinct phantom group ID, never individual wavelength/image'}
    (output_dir/'ingestion_report.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest


def sample_optics(profile_path,labels,wavelength_nm,rng,scattering_scale=1.0):
    """Sample joint (mua,musp) from one measured group-level optical record per material.

    Do not fabricate an interpolated spectrum. Request an observed wavelength.
    """
    profile=json.loads(Path(profile_path).read_text(encoding='utf-8'))
    if profile.get('format') not in ('PA-FORGE-optical-profile-0.6','PA-FORGE-optical-profile-0.7'):raise ValueError('Unsupported optical profile format')
    validate_optical_rows(profile)
    if not np.isfinite(scattering_scale) or scattering_scale<0:raise ValueError('Invalid scattering scale')
    available={int(r['wavelength_nm']) for r in profile['measurements']}
    if int(wavelength_nm) not in available:raise ValueError(f'Wavelength {wavelength_nm} nm unavailable; observed: {sorted(available)}')
    mua=np.zeros(labels.shape,float);musp=np.zeros(labels.shape,float);chosen={}
    for c in np.unique(labels):
        rows=[r for r in profile['measurements'] if r['label']==int(c) and r['wavelength_nm']==int(wavelength_nm)]
        if not rows:raise ValueError(f'No TRAIN optical observations for class {c} @ {wavelength_nm} nm')
        # Balance by independent phantom first; multiple devices/wavelength files
        # must not silently give one phantom a larger sampling probability.
        by_group=defaultdict(list)
        for candidate in rows:by_group[candidate['group']].append(candidate)
        chosen_group=sorted(by_group)[int(rng.integers(len(by_group)))]
        candidates=by_group[chosen_group]
        row=candidates[int(rng.integers(len(candidates)))]
        # Joint row sampling avoids assuming independence of mua and musp.
        aa=max(0,float(row['absorption_m_inv']))
        ss=max(0,float(row['scattering_reduced_m_inv']))*scattering_scale
        mua[labels==c]=aa;musp[labels==c]=ss
        chosen[str(int(c))]={'group':row['group'],'absorption_m_inv':aa,'musp_m_inv':ss}
    return mua,musp,chosen
