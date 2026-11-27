"""Multi-device OpenQPAIData preparation for material-prior learning.

Uses 2-D InVision/TROPUS label maps and selected 2-D slices from 3-D SVOT
volumes. Group splits are by phantom ID across every device. InVision/TROPUS
mua/musp maps provide wavelength-specific train-only optical profiles. DIS is
independently inventoried and retained as a reference measurement source; it is
not silently merged unless material IDs and array semantics can be verified.
"""
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import json, re
import numpy as np
from scipy.ndimage import zoom
from .experimental_learning import _partition, _read_np, SPECTRAL


def _load_npy_zip(z, name):
    with z.open(name) as stream:
        a=np.load(stream,allow_pickle=False)
    if not isinstance(a,np.ndarray) or not np.issubdtype(a.dtype,np.number):
        raise ValueError(f'Expected numeric .npy array: {name}')
    if not np.isfinite(a).all():raise ValueError(f'Non-finite array: {name}')
    return a


def _resize64(a, order=0):
    if a.shape==(64,64):return np.asarray(a)
    out=zoom(a,(64/a.shape[0],64/a.shape[1]),order=order,mode='nearest')
    return out[:64,:64]


def _phantom_id(path):
    stem=Path(path).stem
    m=re.match(r'^(P\.[A-Za-z0-9.]+?)(?:_\d{3,4})?$',stem)
    return m.group(1) if m else stem


def _flatten_mapping(obj,prefix=''):
    out=[]
    if isinstance(obj,dict):
        for k,v in obj.items():out.extend(_flatten_mapping(v,f'{prefix}/{k}'))
    elif isinstance(obj,list):
        for i,v in enumerate(obj):out.extend(_flatten_mapping(v,f'{prefix}/{i}'))
    else:out.append({'path':prefix,'value':str(obj)})
    return out


def inspect_dis_archive(dis_zip, mapping_json=None, output_json=None):
    """Create a safe inventory of DIS processed NPZ arrays without guessing units."""
    report={'source':str(dis_zip),'processed_npz':[],'raw_files':0,'mapping':None,
            'warning':'DIS arrays are inventoried, not silently unit-converted or joined to phantom classes.'}
    with ZipFile(dis_zip) as z:
        names=z.namelist();npz=[n for n in names if n.lower().endswith('.npz') and '/processed/' in n.lower()]
        report['raw_files']=sum('/raw/' in n.lower() for n in names)
        for name in npz:
            try:
                with z.open(name) as f, np.load(BytesIO(f.read()),allow_pickle=False) as data:
                    arrays=[]
                    for key in data.files:
                        a=np.asarray(data[key])
                        item={'key':key,'shape':list(a.shape),'dtype':str(a.dtype)}
                        if np.issubdtype(a.dtype,np.number) and a.size and np.isfinite(a).all():
                            item.update({'min':float(np.min(a)),'median':float(np.median(a)),'max':float(np.max(a))})
                        arrays.append(item)
                    report['processed_npz'].append({'file':name,'arrays':arrays})
            except Exception as exc:
                report['processed_npz'].append({'file':name,'error':f'{type(exc).__name__}: {exc}'})
    if mapping_json and Path(mapping_json).is_file():
        mapping=json.loads(Path(mapping_json).read_text(encoding='utf-8-sig'))
        report['mapping']={'path':str(mapping_json),'entries':_flatten_mapping(mapping)}
    report['processed_npz_count']=len(npz)
    if output_json:
        p=Path(output_json);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def prepare_openqpai_complete(invision_zip, tropus_zip=None, svot_zip=None, dis_zip=None,
                              mapping_json=None, output_dir='models/openqpai_complete',
                              absorption_unit='1/cm', scattering_unit='1/cm', seed=42,
                              svot_slices_per_volume=6):
    """Build a multi-device grouped material corpus and train-only optical profile.

    The 19.8 GB SVOT archive is read lazily volume-by-volume from the ZIP. Only
    evenly spaced 2-D label slices are retained for the 2-D material CVAE; the
    original 3-D files are never expanded wholesale to disk.
    """
    if absorption_unit not in ('1/m','1/cm') or scattering_unit not in ('1/m','1/cm'):
        raise ValueError('Declare source optical units as 1/m or 1/cm')
    if not invision_zip or not Path(invision_zip).is_file():
        raise FileNotFoundError('InVision ZIP is mandatory for this OpenQPAIData material-learning workflow')
    if not dis_zip or not Path(dis_zip).is_file():
        raise FileNotFoundError('DIS.zip is mandatory for the complete optical-reference workflow')
    if not mapping_json or not Path(mapping_json).is_file():
        raise FileNotFoundError('material_mapping.json is mandatory to audit material-label to DIS sample identities')
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    mask_rows=[];group_rows=[];device_rows=[];sample_rows=[];labels_by_device={}
    optical_rows=[];unmatched=[];group_names=set();max_class=0
    factor_a=100. if absorption_unit=='1/cm' else 1.
    factor_s=100. if scattering_unit=='1/cm' else 1.

    for device,archive_path in [('invision',invision_zip),('tropus',tropus_zip)]:
        if not archive_path or not Path(archive_path).is_file():continue
        with ZipFile(archive_path) as z:
            entries=z.namelist();base=f'{device}/'
            label_files={Path(n).stem:n for n in entries if n.startswith(base+'labels/') and n.lower().endswith('.npy')}
            mua_files={Path(n).stem:n for n in entries if n.startswith(base+'mua/') and n.lower().endswith('.npy')}
            musp_files={Path(n).stem:n for n in entries if n.startswith(base+'musp/') and n.lower().endswith('.npy')}
            for stem,name in sorted(label_files.items()):
                lab=_load_npy_zip(z,name)
                if lab.ndim!=2 or np.any(lab<0) or not np.all(lab==np.round(lab)):
                    raise ValueError(f'Unsupported 2D categorical label map: {name}')
                lab=lab.astype(np.uint8);group=_phantom_id(name);group_names.add(group)
                max_class=max(max_class,int(lab.max()))
                mask_rows.append(_resize64(lab,0).astype(np.uint8));group_rows.append(group)
                device_rows.append(device);sample_rows.append(stem)
                labels_by_device[(device,group)]=lab
            for key,name in sorted(mua_files.items()):
                match=SPECTRAL.match(key)
                if not match:continue
                group,wave=match.group(1),int(match.group(2))
                if group not in label_files:continue
                if key not in musp_files:
                    unmatched.append(f'{device}:{key}:missing_musp');continue
                lab=labels_by_device[(device,group)]
                mua=_load_npy_zip(z,name).astype(float)*factor_a
                musp=_load_npy_zip(z,musp_files[key]).astype(float)*factor_s
                if mua.shape!=lab.shape or musp.shape!=lab.shape:
                    unmatched.append(f'{device}:{key}:shape_mismatch');continue
                if (mua<0).any() or (musp<0).any():raise ValueError(f'Negative optical map values in {device}:{key}')
                # Cache for train-only filtering after all devices are split.
                for cls in np.unique(lab):
                    sel=lab==cls
                    if sel.sum()<8:continue
                    optical_rows.append({'group':group,'device':device,'label':int(cls),'wavelength_nm':wave,
                        'absorption_m_inv':float(np.median(mua[sel])),
                        'scattering_reduced_m_inv':float(np.median(musp[sel])),
                        'absorption_iqr':float(np.subtract(*np.percentile(mua[sel],[75,25]))),
                        'scattering_iqr':float(np.subtract(*np.percentile(musp[sel],[75,25]))),
                        'pixel_count':int(sel.sum())})

    svot_volume_count=0
    if svot_zip and Path(svot_zip).is_file():
        with ZipFile(svot_zip) as z:
            names=[n for n in z.namelist() if n.startswith('svot/labels/') and n.lower().endswith('.npy')]
            for name in sorted(names):
                volume=_load_npy_zip(z,name)
                if volume.ndim!=3 or not np.all(volume==np.round(volume)):
                    continue
                volume=volume.astype(np.uint8);group=_phantom_id(name);group_names.add(group)
                max_class=max(max_class,int(volume.max()));svot_volume_count+=1
                depth=volume.shape[0]
                count=max(1,int(svot_slices_per_volume))
                indices=np.unique(np.linspace(0,depth-1,count).round().astype(int))
                for zi in indices:
                    sl=volume[int(zi)]
                    if len(np.unique(sl))<2:continue
                    mask_rows.append(_resize64(sl,0).astype(np.uint8));group_rows.append(group)
                    device_rows.append('svot_slice');sample_rows.append(f'{Path(name).stem}:z{zi}')
                del volume

    if len(group_names)<3:raise ValueError('Need at least three distinct phantom IDs across selected devices')
    if not mask_rows:raise ValueError('No usable 2D or sampled SVOT segmentation maps found')
    split=_partition(sorted(group_names),seed)
    train_groups=set(split['train'])
    train_optics=[r for r in optical_rows if r['group'] in train_groups]
    if not train_optics:raise ValueError('No matched train-only mua/musp maps; include InVision or TROPUS optical maps')
    masks=np.stack(mask_rows).astype(np.uint8);groups=np.asarray(group_rows,dtype='U128')
    corpus=out/'material_corpus_multidevice.npz'
    np.savez_compressed(corpus,masks=masks,groups=groups,devices=np.asarray(device_rows,dtype='U32'),
                        sample_ids=np.asarray(sample_rows,dtype='U160'),classes=int(max_class+1))
    dis_report=None
    if dis_zip and Path(dis_zip).is_file():
        dis_report=inspect_dis_archive(dis_zip,mapping_json,out/'dis_inventory.json')
    profile={'format':'PA-FORGE-optical-profile-0.7','source':'OpenQPAIData InVision/TROPUS mapped optical maps; DIS independently inventoried',
        'training_scope':'TRAIN phantom IDs only across all devices; no test/validation optical leakage',
        'units':'SI: 1/m','declared_source_units':{'mua':absorption_unit,'musp':scattering_unit},
        'classes':int(max_class+1),'groups':split,'measurements':train_optics,
        'dis_inventory':str(out/'dis_inventory.json') if dis_report else None,
        'limitations':['InVision/TROPUS mapped mua/musp arrays provide wavelength-specific optical profiles.',
          'DIS processed/raw files are inventoried independently; material_mapping join and NPZ semantics must be reviewed before merging them into training.',
          'SVOT contributes 2D slices from 3D label volumes to the 2D CVAE; SVOT raw time series are not available from publisher.',
          'Acoustic properties and Gruneisen parameter are not inferred from optical measurements.',
          'All generated signals remain synthetic.']}
    profile_path=out/'optical_profile_multidevice.json'
    profile_path.write_text(json.dumps(profile,indent=2),encoding='utf-8')
    report={'training_corpus':str(corpus),'optical_profile':str(profile_path),
        'counts':{'unique_phantom_groups':len(group_names),'mask_samples':len(mask_rows),
          'invision_masks':int(sum(d=='invision' for d in device_rows)),
          'tropus_masks':int(sum(d=='tropus' for d in device_rows)),
          'svot_2d_slices':int(sum(d=='svot_slice' for d in device_rows)),
          'svot_volumes_scanned':svot_volume_count,'material_classes':int(max_class+1),
          'train_optical_rows':len(train_optics),'all_optical_rows':len(optical_rows)},
        'split_by':'distinct phantom ID across devices; repeated wavelength/device samples remain in same split',
        'dis_inventory':str(out/'dis_inventory.json') if dis_report else None,
        'warning':'DIS is independently inventoried, not silently fused into mapped optical labels.'}
    (out/'multidevice_ingestion_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report
