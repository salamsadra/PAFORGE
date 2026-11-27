"""Fail-closed numerical comparison of PR-02/PR-03 reconstruction results.

This module deliberately does NOT imply that a comparison between different
algorithms, grid extents or phantoms constitutes implementation parity.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any
import h5py
import numpy as np

from .result import read_matlab_result
from .h5_attrs import scalar_float, scalar_text


@dataclass(frozen=True)
class ParityReport:
    status: str
    method_id: str
    acquisition_id: str
    shape: tuple[int, int]
    rmse: float
    mae: float
    max_abs: float
    relative_l2: float | None
    pearson_r: float | None
    reference_sha256: str
    candidate_sha256: str
    atol: float
    rtol: float
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _image_hash(x: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(x, dtype='<f8', order='C').tobytes(order='C')).hexdigest()


def _scalar(x: np.ndarray) -> float:
    return float(np.asarray(x))


def _load_result(path: str | Path) -> tuple[np.ndarray, dict[str, str]]:
    im, meta = read_matlab_result(path)
    with h5py.File(path, 'r') as h:
        g = h['reconstruction']
        # The PR-02 schema lacks explicit image coordinate/orientation metadata.
        # Absence of these attributes MUST NOT be assumed to represent a common grid.
        extra = {}
        for k in ('grid_dx_m', 'grid_dy_m', 'grid_origin_x_m',
                  'grid_origin_y_m', 'grid_orientation'):
            if k in g.attrs:
                extra[k] = (scalar_text(g.attrs, k) if k == 'grid_orientation'
                            else scalar_float(g.attrs, k))
        meta.update(extra)
    return im, meta


def compare_results(reference_path: str | Path, candidate_path: str | Path, *,
                    atol: float = 1e-8, rtol: float = 1e-5,
                    require_grid_metadata: bool = True) -> ParityReport:
    """Compare two instances of the SAME method on the SAME acquisition.

    Default requires explicit equal grid metadata; legacy results without it
    are rejected rather than assumed geometrically aligned. Tolerances apply
    elementwise: abs(a-b) <= atol + rtol*abs(reference).
    """
    if not all(np.isfinite([atol, rtol])) or atol < 0 or rtol < 0:
        raise ValueError('atol and rtol must be nonnegative finite numbers')
    ref, rm = _load_result(reference_path)
    cand, cm = _load_result(candidate_path)
    for k in ('acquisition_id', 'method_id'):
        if rm.get(k) != cm.get(k):
            raise ValueError(f'{k} mismatch; cannot claim same-method parity')
    if ref.shape != cand.shape:
        raise ValueError('Image shape mismatch; resampling is forbidden in parity mode')
    keys = ('grid_dx_m', 'grid_dy_m', 'grid_origin_x_m', 'grid_origin_y_m', 'grid_orientation')
    if require_grid_metadata:
        missing = [k for k in keys if k not in rm or k not in cm]
        if missing:
            raise ValueError(f'Missing grid metadata for parity: {missing}; obtain a calibrated grid')
    for k in keys:
        present = (k in rm, k in cm)
        if present[0] != present[1]:
            raise ValueError(f'{k} known only on one image')
        if present[0]:
            if k == 'grid_orientation':
                if str(rm[k]) != str(cm[k]):
                    raise ValueError(f'{k} mismatch')
            elif not np.isclose(float(rm[k]), float(cm[k]), rtol=1e-10, atol=1e-14):
                raise ValueError(f'{k} mismatch')
    diff = cand - ref
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))
    max_abs = float(np.max(np.abs(diff)))
    reference_norm = float(np.linalg.norm(ref.ravel()))
    relative_l2 = float(np.linalg.norm(diff.ravel()) / reference_norm) if reference_norm else (0.0 if not np.any(diff) else None)
    rs = ref.ravel(); cs = cand.ravel()
    pearson = float(np.corrcoef(rs, cs)[0, 1]) if np.std(rs) > 0 and np.std(cs) > 0 else None
    passed = bool(np.all(np.abs(diff) <= atol + rtol*np.abs(ref)))
    return ParityReport(
        status='PASS' if passed else 'FAIL', method_id=rm['method_id'], acquisition_id=rm['acquisition_id'],
        shape=tuple(map(int, ref.shape)), rmse=rmse, mae=mae, max_abs=max_abs,
        relative_l2=relative_l2, pearson_r=pearson,
        reference_sha256=_image_hash(ref), candidate_sha256=_image_hash(cand),
        atol=float(atol), rtol=float(rtol),
        failure_reason=None if passed else 'Elementwise absolute/relative tolerance exceeded',
    )


def write_comparison_json(path: str | Path, report: ParityReport, *, overwrite: bool = False) -> Path:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return path


def write_test_image(path: str | Path, image: np.ndarray, *, method_id: str, acquisition_id: str,
                     dx_m: float, dy_m: float, origin_x_m: float, origin_y_m: float,
                     orientation: str, overwrite: bool = False) -> Path:
    """Testing utility; not a physical reconstruction or quantitative output."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    im = np.asarray(image, dtype=np.float64)
    if im.ndim != 2 or not im.size or not np.isfinite(im).all():
        raise ValueError('Expected nonempty finite 2-D image')
    if not method_id or not acquisition_id or not orientation:
        raise ValueError('method_id, acquisition_id and orientation required')
    if dx_m <= 0 or dy_m <= 0 or not np.isfinite([dx_m,dy_m,origin_x_m,origin_y_m]).all():
        raise ValueError('Invalid pixel spacing or origin')
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, 'w') as h:
        h.attrs['format'] = 'PA-HYBRID-RECONSTRUCTION'
        h.attrs['schema_version'] = '0.1.0'
        g = h.create_group('reconstruction')
        g.attrs.update({'height': im.shape[0], 'width': im.shape[1],
            'method_id': method_id, 'acquisition_id': acquisition_id,
            'quantitative': 0, 'grid_dx_m': dx_m, 'grid_dy_m': dy_m,
            'grid_origin_x_m': origin_x_m, 'grid_origin_y_m': origin_y_m,
            'grid_orientation': orientation})
        g.create_dataset('image_flat_c', data=im.ravel(order='C'))
    return path
