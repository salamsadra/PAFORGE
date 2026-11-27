"""PR-03 command-line verification harness: real MATLAB runs are opt-in."""
from __future__ import annotations
import argparse
import json
import h5py
from pathlib import Path
import sys
import numpy as np

from .contracts import AcquisitionRecord
from .exchange import read_acquisition, write_acquisition
from .parity import compare_results, write_comparison_json


def make_fixture(path: str | Path) -> Path:
    """Deterministic non-square channel/time test fixture, never called experimental."""
    n, t = 5, 73
    idx = np.arange(n)[:, None]
    tt = np.arange(t)[None, :]
    traces = (np.sin((idx + 1) * (tt + 2) / 17.) + 0.01*idx + .0001*tt).astype('float64')
    coords = np.column_stack((np.linspace(-.004,.004,n),
                              np.linspace(.001,.003,n), np.zeros(n)))
    rec = AcquisitionRecord(signals=traces, detectors_m=coords,
        sample_interval_s=4.e-8, acquisition_id='PR03-ARRAY-FIXTURE',
        device_id='simulated-fixture', session_id='PR03-QA',
        source_id='PR03-deterministic-array-QA', domain='synthetic',
        coordinate_frame='xyz-centred-metres', time_zero_s=None,
        processing_history=('deterministic array transport QA only; NOT a physical wave simulation',))
    return write_acquisition(path, rec)


def verify_roundtrip(reference_h5: str | Path, matlab_h5: str | Path) -> dict:
    """Check MATLAB roundtrip traces, metadata and geometry without tolerance."""
    if Path(reference_h5).resolve() == Path(matlab_h5).resolve():
        raise ValueError('A MATLAB roundtrip must use a distinct output file, not the input itself')
    with h5py.File(matlab_h5, 'r') as h:
        if h.attrs.get('producer', '') != 'MATLAB:pa_bridge_roundtrip':
            raise ValueError('Candidate lacks explicit MATLAB roundtrip producer marker')
    ref, output = read_acquisition(reference_h5), read_acquisition(matlab_h5)
    equal_signals = np.array_equal(ref.signals, output.signals)
    equal_geometry = np.array_equal(ref.detectors_m, output.detectors_m)
    equal_metadata = ref.metadata() == output.metadata()
    equal_dt = ref.sample_interval_s == output.sample_interval_s
    return {'status': 'PASS' if all((equal_signals, equal_geometry, equal_metadata, equal_dt)) else 'FAIL',
            'identical_signals': bool(equal_signals), 'identical_detectors': bool(equal_geometry),
            'identical_metadata': bool(equal_metadata), 'identical_sample_interval': bool(equal_dt),
            'n_channels': ref.shape[0], 'n_samples': ref.shape[1],
            'validation_level': 'cross-language array transport only'}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description='PR-03 fail-closed verification tools')
    sub = p.add_subparsers(dest='command', required=True)
    f = sub.add_parser('fixture', help='Create deterministic Python array fixture')
    f.add_argument('output')
    rr = sub.add_parser('roundtrip', help='Compare MATLAB-returned acquisition against input')
    rr.add_argument('reference'); rr.add_argument('candidate')
    imp = sub.add_parser('import-patbox', help='Convert reviewed MATLAB PATBox raw export to canonical exchange')
    imp.add_argument('input'); imp.add_argument('output'); imp.add_argument('--source-id', required=True)
    imp.add_argument('--phantom-id'); imp.add_argument('--wavelength-nm', type=float)
    c = sub.add_parser('compare', help='Same-acquisition, same-method, same-grid image parity')
    c.add_argument('reference'); c.add_argument('candidate')
    c.add_argument('--atol', type=float, default=1e-8)
    c.add_argument('--rtol', type=float, default=1e-5)
    c.add_argument('--json', dest='json_path', default=None)
    c.add_argument('--allow-unregistered-grid', action='store_true',
                   help='Unsafe for physical parity: only for legacy debugging')
    args = p.parse_args(argv)
    if args.command == 'fixture':
        print(make_fixture(args.output))
        return 0
    if args.command == 'import-patbox':
        from .patbox_import import import_patbox_raw
        print(import_patbox_raw(args.input,args.output,source_id=args.source_id,
            phantom_id=args.phantom_id,wavelength_nm=args.wavelength_nm))
        return 0
    if args.command == 'roundtrip':
        report = verify_roundtrip(args.reference, args.candidate)
    else:
        report_obj = compare_results(args.reference,args.candidate, atol=args.atol,rtol=args.rtol,
                                     require_grid_metadata=not args.allow_unregistered_grid)
        report = report_obj.to_dict()
        if args.json_path:
            write_comparison_json(args.json_path, report_obj)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError, OSError, KeyError) as exc:
        print(f'PR03 validation error: {exc}', file=sys.stderr)
        sys.exit(2)
