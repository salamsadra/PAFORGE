"""Opt-in MATLAB/Python exchange bridge (PR-02 prototype, not validated reconstruction)."""
from .contracts import AcquisitionRecord
from .exchange import read_acquisition, write_acquisition, inspect_exchange
from .legacy import from_native_hdf5
__all__ = ["AcquisitionRecord", "read_acquisition", "write_acquisition", "inspect_exchange", "from_native_hdf5"]
