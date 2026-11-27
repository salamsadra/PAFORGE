"""Group-disjoint train/validation/test splits to prevent subject leakage."""
import csv
import json
from pathlib import Path
import numpy as np


def group_split(metadata_csv,output,group_column='subject_id',train_fraction=.7,val_fraction=.15,seed=42):
    if not (0<train_fraction<1 and 0<val_fraction<1 and train_fraction+val_fraction<1):
        raise ValueError('Fractions must leave a strictly positive held-out test portion')
    with open(metadata_csv,newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f)
        if group_column not in (reader.fieldnames or []):
            raise ValueError(f'Metadata CSV must contain group column {group_column!r}')
        data=list(reader)
    if not data or any(not row.get(group_column) for row in data):
        raise ValueError('Each record must have a nonempty group ID')
    groups=np.array(sorted(set(row[group_column] for row in data)))
    if len(groups)<3:raise ValueError('At least 3 independent groups needed for train/val/test')
    rng=np.random.default_rng(seed)
    rng.shuffle(groups)
    n=len(groups)
    ntrain=max(1,int(np.floor(train_fraction*n)))
    nval=max(1,int(np.floor(val_fraction*n)))
    if ntrain+nval>=n: ntrain=n-2;nval=1
    train=set(groups[:ntrain]);val=set(groups[ntrain:ntrain+nval]);test=set(groups[ntrain+nval:])
    if train & val or train & test or val & test:raise AssertionError('Group leakage')
    assignments=[]
    for row in data:
        g=row[group_column]
        assignments.append({**row,'split':'train' if g in train else 'validation' if g in val else 'test'})
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with (output/'assignments.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(assignments[0]));w.writeheader();w.writerows(assignments)
    manifest={'group_column':group_column,'seed':seed,'dataset_metadata_path':str(metadata_csv),
              'splits':{'train':sorted(train),'validation':sorted(val),'test':sorted(test)},
              'num_records':len(data),'num_independent_groups':n,
              'warning':'No subject-level independence if group column is not an actual subject/patient identifier.'}
    (output/'split_manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest
