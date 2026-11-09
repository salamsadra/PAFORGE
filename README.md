# PA-FORGE v0.9.0 — Unified Reconstruction Research Workbench

**Research software — 2D numerical prototype, not a clinically validated PAT device reconstruction package.**

v0.9 preserves v0.8's domain-aware dataset downloading, generation from experimental material/optical information, CVAE, GPU selection, signal+image export, archives/visualization, and CLI. The major addition is a **single algorithm registry and acquisition interface** shared between simulated phantoms and experimental signals. It currently registers **33 method labels** (29 built-in variants, including four neural pathways; four methods delegate to the optional original PATATO package). Algorithm availability depends on the input geometry, forward model, optional trained weights and installed packages.

### Quick-start on Windows

1. Extract this ZIP to `C:\Research\PA-FORGE-v0.9` (a separate directory from v0.8; **do not delete the existing datasets**).
2. Double-click `START_PA_FORGE.bat`. The first launch checks/installs the local package and basic Gradio dependencies using Python `py`.
3. Open `http://127.0.0.1:7860` in your browser.
4. **Tab 00**: choose CPU PyTorch if needed, or official PATATO if you want its backprojection/model-based methods. Installing the main package never auto-executes untrusted external source code.
5. **Tab 10 — Unified Reconstruction Lab**: enter the path of the native synthetic `raw_signals.h5` OR a previously normalized experimental HDF5; select 1–12 methods; pick FOV, sound speed, grid, iterations and optional learned model checkpoint. Click **Compare selected reconstruction methods**.
6. **Tab 11 — Train Learned Reconstructors**: if desired, train a U-Net or residual CNN on your own paired **synthetic DAS reconstruction and synthetic initial-pressure p0 truth**. Feed its `.pt` checkpoint back into Tab 10. Data from the same source phantom must stay in one split.
7. Tabs 01, 02, 04, 05 and 06 still provide generation, experimental reconstruction, full-dataset learning and benchmarks. Reconstruction dropdowns expose the shared registry.

No external dataset, trained reconstruction checkpoint or 20+ GB raw archive is embedded in the software ZIP. The 30-phantom OpenQPAIData corpus, PATATO Cambridge example datasets and Digital Twin must be obtained separately in accordance with publisher licenses. Your existing v0.8 downloads can be reused.

### One-click vs optional external dependencies

**Original PATATO**: Uses the actual installed `patato` Python package, not a rewritten imitation. In Dashboard Tab 00 select **Official PATATO Python backend**, or install through Windows PowerShell:

```powershell
py -m pip install -e ".[ui,dev,train,patato]"
```

PATATO may also require a compatible JAX/OpenCL/GPU runtime according to the selected official backend. If a dependency, backend or required wavelength is missing, PA-FORGE reports the failure; it does not silently substitute an internal algorithm. PATATO backend methods are *unverified in this release with the actual PATATO install* because the build container could not download it. The local bridge and documented API shape were tested using a test double.

**PATbox MATLAB**: This release implements corresponding **independent Python algorithmic variants** for DAS, CF-DAS, DMAS, DS-DMAS, MV-DAS, VDAS, SCF-DAS, CFMV-DAS, FBP, UBP, DMAS-UBP, VDAS-UBP, TR, iterative DAS and iterative TR. These are **NOT a line-by-line port of PATbox**, and some use simplified/global estimates rather than PATbox's original adaptive formulation. The Zenodo record `23348355` remains unverified as to exact downloadable MATLAB source contents. See `MATLAB_PATBOX_PORT_AUDIT.md`. Faithful source-to-source equivalence, licensing and output parity cannot be claimed until the MATLAB source is available and compared case-by-case.

---

## Shared acquisition contract

| Input | Required meaning |
|---|---|
| `/raw_data` | 2D numeric array `[detectors, samples]`, finite; measured or generated acoustic traces |
| `/detector_positions_m` | Actual/acoustically simulated sensor positions `[detectors, 3]` in metres |
| `sampling_rate_hz` | Positive sampling rate in Hz in HDF5 attributes |
| `synthetic` | Boolean source classification; experiment is never inferred from file naming |
| `acquisition_wavelength_nm` | Real known wavelength, required for PATATO backend. Never fabricated |
| FOV / grid / effective sound speed | Selected numerical model; must include all detectors for FDTD |
| optional synthetic `/ground_truth.h5` | Only for matched-grid *synthetic* diagnostic comparisons; not used as real experimental ground truth |

Only substantially planar geometry is supported. FDTD inversion requires the sensors to be inside its computational grid and a stable time step. For 3D measurements, data without known geometry, or time-zero/aperture-calibrated inverse problems the engine must stop rather than manufacture coordinates.

## Built-in reconstruction families

**Beamforming and analytic:** DAS, derivative UBP-like, filtered UBP-like, ramp-filtered DAS, CF-DAS, sign-CF-DAS, DMAS, DS-DMAS, variance-DAS, globally covariance-weighted MV-DAS, CFMV-DAS, DMAS-UBP and variance-UBP. Most are independent *qualitative approximate* methods; they are not exact universal formulae across all detector layouts.

**Full-wave numerical (prototype):** FDTD adjoint, nonnegative FISTA-L2 FDTD, reverse-time source injection, iterative reverse-time residual injection. Current numerical solver uses simplified 2D acoustics, limited grid size and no calibrated device response. Note that forward and inverse solve the *same approximate model* in matched synthetic tests (inverse crime).

**Approximate ray-model inverse:** matrix-free forward/adjoint with an adjoint dot-product test, CG-Tikhonov, projected nonnegative FISTA-L2, Landweber, SIRT, residual iterative DAS, LSQR, TV-FISTA and ADMM-TV. They are not the wave-equation model; interpretation of derived initial pressure is qualitative for real measurements.

**Learning:** supervised DAS-to-p0 U-Net, supervised DAS-to-p0 residual CNN, plug-and-play model-based FISTA with a **trained** residual CNN, and per-acquisition *self-supervised* Deep Image Prior using the ray-based data consistency operator. Deep Image Prior learns from the input acquisition itself (no pretrained network needed). Other learned methods require a valid `.pt` checkpoint; no off-the-shelf human or clinical weights are shipped.

**Upstream PATATO original package:** official ReferenceBackprojection (JAX), ModelBasedReconstruction, OpenCLBackprojection and SlowBackprojection (NumPy) through the installed Python API. These are external optional backends, not duplicated source. Their upstream version and hardware dependencies can change.

See `RECONSTRUCTION_METHOD_MATRIX.md` for the complete 33-method reference.

## CLI workflows (PowerShell, run in the extracted project folder)

```powershell
py -m pip install -e ".[ui,train,dev]"
py -m paforge.cli reconstruction-methods
py -m paforge.cli ui
```

Create a small paired synthetic dataset (DAS is required for the current learning-target pipeline):

```powershell
py -m paforge.cli demo --anatomy vascular --samples 8 --seed 42 --grid-size 64 --time-steps 380 --transducers 48 --no-filter --reconstruction-method das --output outputs/paired_das_training
```

Train a supervised reconstruction model on synthetic DAS/p0 pairs with **disjoint source groups** (our procedural sample folders are independent; for experiment-informed generated samples grouping follows the source-phantom provenance when present):

```powershell
py -m paforge.cli train-reconstruction outputs/paired_das_training --weights models/unet_das.pt --architecture unet --epochs 10 --batch-size 4 --device auto
```

Compare multiple methods on **exactly one shared raw measurement**, with measured geometry:

```powershell
py -m paforge.cli reconstruct-compare outputs/paired_das_training/vascular_00000/raw_signals.h5 --methods das,ubp,cf_das,dmas,cg_tikhonov,admm_tv,deep_image_prior --grid-size 64 --fov-m 0.025 --iterations 8 --time-decimate 1 --truth outputs/paired_das_training/vascular_00000/ground_truth.h5 --output experiments/comparison_v09
```

Run the learned model after training:

```powershell
py -m paforge.cli reconstruct-compare outputs/paired_das_training/vascular_00000/raw_signals.h5 --methods das,learned_unet --weights models/unet_das.pt --grid-size 64 --fov-m 0.025 --output experiments/learned_comparison
```

Reconstruct a normalized real InVision acquisition (choose measured detector array encompassing FOV; algorithm may enforce a larger FOV):

```powershell
py -m paforge.cli reconstruct-acquisition data/real/invision_P.5.10.2_700.h5 --method cf_das --grid-size 128 --fov-m 0.1 --sound-speed-m-s 1500 --output experiments/invision_cf
```

Direct synthetic generator from the same registry:

```powershell
py -m paforge.cli demo --anatomy vascular --samples 1 --grid-size 64 --time-steps 380 --transducers 48 --no-filter --reconstruction-method admm_tv --output outputs/v09_example
```

The `reconstruct-compare` output folder contains: `input_signals.png`, `comparison.png`, one numerical `.npy` and plotted `.png` per successful method, plus `comparison_report.json` with runtime, individual errors and limited diagnostic metrics (only when synthetic truth provided). Failure of one optional backend does not prevent other methods in a comparison from completing.

## Training and reproducibility boundaries

- The U-Net/ResCNN trainers **require** an existing sample directory hierarchy with `reconstruction.h5` (`attrs['method']='das...'`) and `ground_truth.h5` (`initial_pressure_pa`). They do not claim to train on the full real measured dataset, which lacks true calibrated p0 ground truth.
- Group-disjoint validation is required. The source group is recorded in the checkpoint report; metadata for source group should be retained when deriving multiple augmented frames from one physical phantom.
- U-Net/ResCNN currently train on a consistent DAS image normalization; they are **refinement networks**, not raw-waveform-to-image direct inverse networks. Comparing with actual measured data needs independent QA.
- `deep_image_prior` optimizes network parameters per acquisition, using *approximate* travel-time waveform consistency; it is not a trained diffusion model.
- GPU is supported for PyTorch learning via CUDA/MPS if the correct backend is already installed. FDTD, NumPy beamforming and ray iterative methods are primarily CPU NumPy/SciPy; PATATO JAX/OpenCL may use accelerators if installed correctly.
- Experimental FOV, sensor aperture/angular response, frequency response, time-zero, sound speed and 3D effects require external calibration. Reconstructed image pixels must not be interpreted as quantitative absorption/pressure without validation.
- No MATLAB package code is bundled; avoid claiming equivalence to PATbox until the original MATLAB source is examined, licensed and numerically parity-tested.

## Scientific baseline tests

```powershell
py -m pytest -q
```

In the release environment **99 tests passed** (original 88 tests plus 11 new integration tests). This includes: ray forward–adjoint dot product, synthetic/experimental identical-image path for CF-DAS, method dispatch, checkpoint training on toy synthetic pairs, optional upstream API mocked integration, Deep Image Prior smoke, simulated waveform reconstruction and dashboard build. **The full 23.4 GB datasets, trained-model realism and actual upstream PATATO runtime were not tested in this build environment.** This is a software integration milestone, not independent real-data validation.

## Continuing the previous v0.8 workflows

The download wizard, catalog entries, domain-aware optical/material learning, trained material CVAE, GPU training, experimental import (IPASC and PATATO source format), visual explorer and image/signal exports remain in the package. See `docs/README_v0.8_ARCHIVE.md` for all previous dataset-specific step-by-step operations. Existing downloaded files under `data/external/` can be reused; copying or specifying their existing directory is sufficient. No automatic 23 GB re-download is required for v0.9.

## Core references and implementation provenance

- Upstream PATATO original Python implementation, MIT license: https://github.com/BohndiekLab/patato and https://patato.readthedocs.io/en/latest/examples/02_pafilter.html
- Official PATATO HDF5 format (detector geometry, sampling frequency, axes): https://patato.readthedocs.io/en/latest/dev/02_fileformatspecification.html
- PATbox user-provided upstream record, **unverified in build environment**: https://zenodo.org/records/23348355
- OpenQPAIData: https://zenodo.org/records/14044853
- Cambridge PATATO example dataset: https://www.repository.cam.ac.uk/items/bf455f97-1ff0-4694-87ca-b72c85366dc6

**A public redistribution license for PA-FORGE itself has not been selected.** An explicit LICENSE file and provenance audit are needed before open-source redistribution. Third-party PATATO remains installed separately under its own MIT license.
