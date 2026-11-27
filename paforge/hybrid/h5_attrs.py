"""Strict reading of MATLAB/Python HDF5 scalar attributes.

MATLAB h5writeatt may store scalar values using singleton dataspaces. h5py
can therefore return a 1-element ndarray rather than a Python scalar.
This module never takes the first element of a multi-valued attribute.
"""
from __future__ import annotations

from numbers import Integral, Real
import numpy as np


def scalar_attr(attrs, key: str):
    """Read an HDF5 scalar or singleton attribute, rejecting multi-values."""
    if key not in attrs:
        raise ValueError(f"Missing HDF5 attribute {key!r}")
    value = attrs[key]
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(
                f"HDF5 attribute {key!r} must be scalar/singleton, "
                f"got shape {value.shape}, dtype {value.dtype}"
            )
        value = value.reshape(-1)[0]
    if isinstance(value, np.generic):
        value = value.item()
    return value


def scalar_int(attrs, key: str) -> int:
    value = scalar_attr(attrs, key)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"HDF5 attribute {key!r} must be a numeric integer")
    number = float(value)
    if not np.isfinite(number) or number != np.floor(number):
        raise ValueError(f"HDF5 attribute {key!r} must be a finite integer")
    # Numeric HDF5 sizes in this application are far below 2^53.
    return int(number)


def scalar_float(attrs, key: str) -> float:
    value = scalar_attr(attrs, key)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"HDF5 attribute {key!r} must be numeric")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"HDF5 attribute {key!r} must be finite")
    return number


def scalar_text(attrs, key: str) -> str:
    value = scalar_attr(attrs, key)
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"HDF5 attribute {key!r} is not UTF-8") from exc
    if not isinstance(value, str):
        raise ValueError(f"HDF5 attribute {key!r} must be a text scalar")
    return value.rstrip("\x00")
