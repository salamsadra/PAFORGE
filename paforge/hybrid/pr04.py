"""PR-04 executable evidence gate for independently run PATBox/MATLAB.

NOT physics-validation, experimental-data validation, or an independent
Python algorithm implementation. This is a reproducible integration-parity
pipeline. Python-only CI MUST NOT report MATLAB execution as successful.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np

from .exchange import read_acquisition
from .parity import compare_results
from .patbox_import import import_patbox_raw
from .h5_attrs import scalar_int, scalar_float, scalar_text
from .provenance import file_sha256

REQUIRED_PRODUCERS = ('MATLAB:PATBox:direct', 'MATLAB:PAForge:bridge')
REQUIRED_FILES = ('patbox_sim.mat', 'patbox_raw.h5', 'acquisition.h5', 'direct_patbox.h5',
                  'bridge_patbox.h5', 'matlab_environment.txt')


def _attr(value):
    """Normalize missing-proof-optional identity attrs without array stringification.

    Required attributes are read with scalar_text at the call site.
    """
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError('HDF5 identity attribute is not scalar/singleton')
        value = value.reshape(-1)[0]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode('utf-8').rstrip('\x00')
    if not isinstance(value, str):
        raise ValueError('HDF5 identity attribute must be text')
    return value.rstrip('\x00')


def _pr04_raw_matches_acquisition(raw_path: Path, acq) -> bool:
    """Check the imported dataset still matches original raw PATBox array values.

    Hashing of the canonical acquisition is performed by read_acquisition().
    The raw file is not trusted solely based on the importer output filename.
    """
    with h5py.File(raw_path, 'r') as h:
        if scalar_text(h.attrs,'format') != 'PA-PATBOX-RAW-EXPORT':
            raise ValueError('Not original raw PATBox export')
        group = h['raw']
        if (scalar_int(group.attrs,'n_detectors'), scalar_int(group.attrs,'n_samples')) != acq.shape:
            raise ValueError('Original raw shape mismatch')
        flat = np.asarray(group['signals_flat_c'][:],dtype=np.float64).reshape(-1)
        pos = np.asarray(group['detectors_flat_c'][:],dtype=np.float64).reshape(-1)
        if not np.array_equal(flat, acq.signals.ravel(order='C')):
            raise ValueError('Original PATBox signals changed during import')
        if not np.array_equal(pos, acq.detectors_m.ravel(order='C')):
            raise ValueError('Original PATBox detector coordinates changed during import')
        if scalar_float(group.attrs,'sample_interval_s') != acq.sample_interval_s:
            raise ValueError('PATBox sample interval changed during import')
    return True


def _result_metadata(path: Path, producer: str, *, acq_hash: str) -> dict:
    with h5py.File(path,'r') as h:
        found = scalar_text(h.attrs, 'producer')
        if found != producer:
            raise ValueError(f'Invalid producer on {path.name}: expected {producer}, found {found!r}')
        if scalar_text(h.attrs, 'acquisition_signal_sha256') != acq_hash:
            raise ValueError(f'{path.name}: signal hash not tied to acquisition')
        release = scalar_text(h.attrs, 'matlab_release')
        entrypoint = scalar_text(h.attrs, 'patbox_entrypoint')
        if not release or not entrypoint.lower().endswith('patreconstruct.m'):
            raise ValueError(f'{path.name}: MATLAB/PATBox provenance is incomplete')
        return {'producer': producer, 'matlab_release': release,
                'patbox_entrypoint': entrypoint,
                'file_sha256': file_sha256(path)}


def verify_bundle(folder: str | Path, *, atol: float = 1e-10,
                  rtol: float = 1e-8) -> dict:
    """Verify a real MATLAB bundle. Fail *closed* for missing or forged data.

    Returns PENDING_MATLAB, FAIL or PASS_ARTIFACT_PARITY. The last status does not
    indicate accuracy against a physical reference or cross-method parity.
    """
    p = Path(folder).resolve()
    if not np.isfinite([atol,rtol]).all() or min(atol,rtol) < 0:
        raise ValueError('Tolerances must be finite and nonnegative')
    missing = [name for name in REQUIRED_FILES if not (p/name).is_file()]
    if missing:
        return {'status':'PENDING_MATLAB', 'scope':'integration only',
                'missing_files':missing, 'matlab_execution_verified':False,
                'physical_validation':False, 'experimental_validation':False}
    ref_path, cand_path = p/'direct_patbox.h5', p/'bridge_patbox.h5'
    if ref_path.resolve() == cand_path.resolve():
        raise ValueError('Reference and bridge output paths must be distinct')
    record = read_acquisition(p/'acquisition.h5')
    if record.domain != 'synthetic':
        raise ValueError('PR04 PATBox reference must be labeled synthetic')
    _pr04_raw_matches_acquisition(p/'patbox_raw.h5',record)
    with h5py.File(p/'acquisition.h5','r') as h:
        hash_value = _attr(h['acquisition'].attrs['signal_sha256_c_f64'])
    ref_meta = _result_metadata(ref_path,REQUIRED_PRODUCERS[0],acq_hash=hash_value)
    cand_meta = _result_metadata(cand_path,REQUIRED_PRODUCERS[1],acq_hash=hash_value)
    if ref_meta['patbox_entrypoint'] != cand_meta['patbox_entrypoint']:
        raise ValueError('PATBox entrypoint differs between direct and bridge routes')
    if ref_meta['matlab_release'] != cand_meta['matlab_release']:
        raise ValueError('MATLAB release differs between direct and bridge routes')
    parity=compare_results(ref_path,cand_path,atol=atol,rtol=rtol,
                          require_grid_metadata=True)
    status='PASS_ARTIFACT_PARITY' if parity.status=='PASS' else 'FAIL'
    return {'status':status,'scope':'PATBox direct-vs-bridge, same method',
            'matlab_execution_verified':False,
            'execution_provenance':'reported by MATLAB output metadata; requires independent run-log review',
            'physical_validation':False,
            'experimental_validation':False,'acquisition_id':record.acquisition_id,
            'acquisition_shape':list(record.shape),'algorithm':parity.method_id,
            'sample_interval_s':record.sample_interval_s,
            'input_files_sha256':{k:file_sha256(p/k) for k in ('patbox_sim.mat','patbox_raw.h5','acquisition.h5')},
            'result_provenance':{'direct':ref_meta,'bridge':cand_meta},
            'parity':parity.to_dict(),'atol':atol,'rtol':rtol}


def write_report(folder: str | Path, *, atol: float = 1e-10,
                 rtol: float = 1e-8, overwrite: bool = False) -> Path:
    path=Path(folder)/'PR04_validation.json'
    if path.exists() and not overwrite:
        raise FileExistsError(f'Refusing to overwrite {path}')
    report=verify_bundle(folder,atol=atol,rtol=rtol)
    path.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
    return path


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(prog='python -m paforge.hybrid.pr04',
        description='PAForge PR04 PATBox MATLAB integration benchmark')
    sub=parser.add_subparsers(dest='action',required=True)
    imp=sub.add_parser('import-raw',help='Convert original MATLAB raw export to canonical acquisition')
    imp.add_argument('folder'); imp.add_argument('--source-id',required=True)
    imp.add_argument('--phantom-id',default=None)
    v=sub.add_parser('verify',help='Gate PATBox direct/bridge parity with source provenance')
    v.add_argument('folder'); v.add_argument('--atol',type=float,default=1e-10)
    v.add_argument('--rtol',type=float,default=1e-8);v.add_argument('--save-json',action='store_true')
    args=parser.parse_args(argv)
    if args.action=='import-raw':
        p=Path(args.folder);p.mkdir(parents=True,exist_ok=True)
        dest=import_patbox_raw(p/'patbox_raw.h5',p/'acquisition.h5',
                              source_id=args.source_id,phantom_id=args.phantom_id)
        print(dest)
        return 0
    report=verify_bundle(args.folder,atol=args.atol,rtol=args.rtol)
    print(json.dumps(report,indent=2,allow_nan=False))
    if args.save_json:
        write_report(args.folder,atol=args.atol,rtol=args.rtol)
    return {'PASS_ARTIFACT_PARITY':0,'FAIL':1,'PENDING_MATLAB':3}[report['status']]


if __name__=='__main__':
    try:
        sys.exit(main())
    except (ValueError,KeyError,FileNotFoundError,FileExistsError,OSError) as exc:
        print('PR04 validation error: '+str(exc),file=sys.stderr)
        sys.exit(2)
