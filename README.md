# PAForge

**Experiment-informed photoacoustic data generation**

PAForge is a modular research framework for configurable synthetic photoacoustic acquisitions and reconstruction of compatible measured data. It integrates phantom generation, optical source modelling, two-dimensional acoustic propagation, detector sampling, signal perturbations, image reconstruction and HDF5 export. MATLAB interoperability is optional.

## Installation

Python 3.10+ is required. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
paforge --help
```

On Windows, activate with `.venv\Scripts\activate` in Command Prompt. For the graphical interface, install `python -m pip install -e ".[ui]"` and run `paforge ui --port 7860`. Open http://127.0.0.1:7860. Alternatively, use `PAForge_Launch.sh` (macOS/Linux) or `PAForge_Launch_Windows.bat` (Windows).

## Numerical simulation

```bash
paforge demo --output outputs/demo --samples 1 --seed 42 \
  --grid-size 64 --time-steps 380 --transducers 48 --no-filter
```

For custom parameters, edit `examples/config.yaml` and use `paforge generate examples/config.yaml --output outputs/custom`.

## Experiment-informed generation

Download the original InVision archive separately from [OpenQPAIData](https://doi.org/10.5281/zenodo.14044853). The following settings apply to that archive; confirm optical units and material taxonomy for other inputs.

```bash
paforge prepare-openqpai /path/to/invision.zip \
  --output models/invision --absorption-unit 1/cm \
  --scattering-unit 1/cm --declared-classes 4 --seed 42

python -m paforge.material_partitions models/invision/material_corpus.npz \
  models/invision/optical_profile.json models/invision_partitions

paforge generate-informed --engine empirical_material \
  --model-source models/invision_partitions/material_train.npz \
  --optical-profile models/invision/optical_profile.json \
  --output outputs/informed --samples 2 --seed 42 \
  --grid-size 64 --wavelength-nm 700 --fov-m 0.07 \
  --time-steps 380 --transducers 48 --method ubp \
  --material-solver-padding-factor 2 --no-noise --no-filter
```

The empirical route requires no trained neural network. Preserve group partitions and use a fresh output directory for each generation. Optional learned methods require additional dependencies (`.[train]`).

## Outputs and measured data

A typical synthetic sample contains `ground_truth.h5`, `raw_signals.h5`, `reconstruction.h5` and acquisition metadata. Generated RF is synthetic, not additional experimental measurements. The reconstruction interface also accepts compatible measured signals with acquisition metadata.

## Documentation

- [Dataset discovery and download](docs/DATASETS.md)
- [Measured data](docs/MEASURED_DATA.md)
- [Reconstruction methods](docs/RECONSTRUCTION_METHODS.md)
- [MATLAB interoperability](docs/MATLAB_HDF5_INTEROP.md)
- [Persian quickstart](docs/QUICKSTART_FA.md)

## Repository structure

- `paforge/` — computational engines, data interfaces, CLI, UI and MATLAB bridge
- `examples/` — example configurations and scripts
- `tools/` — reusable utilities
- `tests/` — regression tests
- `docs/` — user documentation

To run tests, install `python -m pip install -e ".[dev,ui,train]"` and run `python -m pytest -q`.

## Scope and license

PAForge uses approximate 2D numerical models; it is not a calibrated scanner digital twin or a clinical tool. Optional learned methods are experimental. External MATLAB toolboxes and experimental datasets are not bundled and retain their respective licenses. See [LICENSE](LICENSE) and [CITATION.cff](CITATION.cff).
