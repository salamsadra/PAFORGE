"""Candidate input guards, not historical validation evidence."""
import hashlib
import numpy as np


def file_sha256(path):
    with open(path, 'rb') as stream:
        h = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def validate_partition(partition):
    sets = {}
    for key in ('train', 'validation', 'test'):
        values = partition.get(key)
        if not isinstance(values, list) or any(not isinstance(g, str) or not g.strip() for g in values):
            raise ValueError(f'Invalid or missing {key} group list')
        if len(values) != len(set(values)):
            raise ValueError(f'Duplicate IDs in {key} group list')
        sets[key] = set(values)
    if not sets['train']:
        raise ValueError('No training groups')
    if sets['train'] & sets['validation'] or sets['train'] & sets['test'] or sets['validation'] & sets['test']:
        raise ValueError('Data leakage: train, validation and test group IDs must be disjoint')
    return sets


def validate_optical_rows(profile):
    splits = validate_partition(profile['groups'])
    for row in profile['measurements']:
        if row['group'] not in splits['train']:
            raise ValueError('Optical record is not in declared TRAIN groups')
        values = np.asarray([row['absorption_m_inv'], row['scattering_reduced_m_inv']], dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError('Optical values must be finite and nonnegative')
    return splits


def class_count(train_max, declared_classes=None):
    if declared_classes is None:
        count = int(train_max) + 1
    elif isinstance(declared_classes, (int, np.integer)) and not isinstance(declared_classes, bool):
        count = int(declared_classes)
    else:
        raise ValueError('Declared classes must be an integer')
    if not 2 <= count <= 16 or train_max >= count:
        raise ValueError('Invalid material taxonomy (2..16 contiguous codes, covering TRAIN)')
    return count
