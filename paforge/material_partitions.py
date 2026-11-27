"""Curator-side export of physically separated material corpora.
This decodes the legacy combined corpus; it must run outside training/evaluation.
Only declared per-mask arrays are copied. Unknown fields are not propagated.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .material_contracts import validate_partition, validate_optical_rows, file_sha256


def separate(corpus, profile, output):
    profile = Path(profile); corpus = Path(corpus); output = Path(output)
    p = json.loads(profile.read_text()); parts = validate_partition(p['groups'])
    validate_optical_rows(p)
    with np.load(corpus, allow_pickle=False) as f:
        masks = f['masks']; groups = f['groups'].astype(str); classes = int(f['classes'])
        if masks.ndim != 3 or masks.shape[1:] != (64,64) or groups.shape != (len(masks),):
            raise ValueError('Expected aligned categorical masks and groups')
        if not np.isfinite(masks).all() or np.any(masks != np.round(masks)) or np.any(masks < 0) or np.any(masks >= classes):
            raise ValueError('Invalid mask labels')
        if classes != int(p['classes']): raise ValueError('Profile/corpus taxonomy mismatch')
        if set(groups) != set.union(*parts.values()) or any(not v for v in parts.values()):
            raise ValueError('Complete nonempty partition membership required')
        arrays = {'masks': masks, 'groups': groups}
        for key in ('sample_ids','devices'):
            if key in f:
                if f[key].shape != groups.shape: raise ValueError(f'Misaligned {key}')
                arrays[key] = f[key]
    output.mkdir(parents=True, exist_ok=False)
    report = {'source_corpus_sha256':file_sha256(corpus),'profile_sha256':file_sha256(profile),
              'classes':classes,'outputs':{}, 'scope':'Curator decoded combined input. No claim of historical byte isolation.'}
    for key, members in {**parts,'train_validation':parts['train']|parts['validation']}.items():
        keep=np.isin(groups,list(members)); target=output/f'material_{key}.npz'
        np.savez_compressed(target,classes=classes,**{k:v[keep] for k,v in arrays.items()})
        report['outputs'][key]={'file':target.name,'sha256':file_sha256(target),'groups':sorted(members),'masks':int(keep.sum())}
    (output/'partition_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('corpus');parser.add_argument('profile');parser.add_argument('output')
    a=parser.parse_args();print(json.dumps(separate(a.corpus,a.profile,a.output),indent=2))
