"""Pre-registered provenance gates for IMPACT/PATBox/PR-03 benchmark runs."""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

ALLOWED_SOURCE = {'PATBox', 'IMPACT', 'Python'}
ALLOWED_SPLIT = {'train', 'validation', 'test'}


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class BenchmarkRun:
    run_id: str
    project: str
    source_commit: str
    acquisition_id: str
    phantom_id: str
    device_id: str
    split: str
    algorithm: str
    input_sha256: str
    software_versions: dict[str, str]
    configuration: dict
    output_filename: str
    source_type: str = 'synthetic'

    def __post_init__(self):
        if self.project not in ALLOWED_SOURCE:
            raise ValueError('Unsupported source project')
        if self.split not in ALLOWED_SPLIT:
            raise ValueError('Invalid split')
        if self.source_type not in {'synthetic','experimental'}:
            raise ValueError('Source type must be explicit')
        for key in ('run_id','acquisition_id','phantom_id','device_id','algorithm','output_filename'):
            if not isinstance(getattr(self,key),str) or not getattr(self,key).strip():
                raise ValueError(f'{key} required')
        if len(self.source_commit)!=40 or any(c not in '0123456789abcdefABCDEF' for c in self.source_commit):
            raise ValueError('40-character source Git commit required; cannot substitute branch name')
        if len(self.input_sha256)!=64 or any(c not in '0123456789abcdefABCDEF' for c in self.input_sha256):
            raise ValueError('Input file SHA256 required')
        if not isinstance(self.configuration,dict):
            raise ValueError('Configuration must be a JSON object')
        if not isinstance(self.software_versions,dict) or not self.software_versions:
            raise ValueError('Software version details required')
        if not all(isinstance(v,str) and v.strip() for v in self.software_versions.values()):
            raise ValueError('Software versions cannot be blank')

    def save(self, path: str | Path) -> Path:
        path=Path(path)
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(asdict(self),indent=2,sort_keys=True)+'\n',encoding='utf-8')
        return path

    @classmethod
    def load(cls, path: str | Path) -> 'BenchmarkRun':
        return cls(**json.loads(Path(path).read_text(encoding='utf-8')))


def validate_group_splits(runs: Iterable[BenchmarkRun], *, group_by: str='phantom_id') -> dict[str,list[str]]:
    """Forbid the same phantom or device appearing across train/validation/test.

    Call separately with group_by='device_id' for leave-one-device-out tests.
    """
    if group_by not in {'phantom_id','device_id','acquisition_id'}:
        raise ValueError('Only approved data grouping keys supported')
    groups: dict[str,set[str]] = {}
    for run in runs:
        groups.setdefault(getattr(run,group_by),set()).add(run.split)
    leaked={name: sorted(splits) for name,splits in groups.items() if len(splits)>1}
    if leaked:
        raise ValueError(f'Data leakage across {group_by} groups: {leaked}')
    return {name:sorted(splits) for name,splits in groups.items()}
