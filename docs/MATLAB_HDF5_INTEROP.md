# MATLAB interoperability

Optional bridge scripts are included in `paforge/hybrid/matlab/`. MATLAB, PATBox and k-Wave must be installed separately when required.

After generating a synthetic sample, read its native HDF5 files in MATLAB:

```matlab
addpath('/path/to/PAForge/tools/matlab');
PAForge_Read_Native('/path/to/sample_folder');
```

The reader loads `/raw_data` from `raw_signals.h5`, `/initial_pressure_pa` from `ground_truth.h5`, and `/image` from `reconstruction.h5`. HDF5 exchange does not establish numerical equivalence between MATLAB and Python algorithms or scanner calibration.
