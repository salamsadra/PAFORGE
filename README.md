# PA-FORGE

### Photoacoustic data generation, physics-based simulation, and image reconstruction

**Release guide:** v0.5.1 Dashboard Edition (v0.5 simulation/reconstruction core)  
**Primary platform for this tutorial:** Windows 10/11 + PowerShell  
**Python:** 3.10 or newer (3.11/3.12 recommended; the documented Windows workflow was also executed with Python 3.13)  
**What this is:** a local, configurable **2D photoacoustic research prototype**.  
**What this is not:** a clinically validated scanner, a full 3D solver, or a certified producer of realistic human measurements.

> **Start here.** This guide reproduces the actual end-to-end workflow used to test PA-FORGE: install → synthetic phantom → raw acoustic traces → iterative reconstruction → numerical benchmark → download small published datasets → inspect measured HDF5 signals → reconstruct an InVision phantom → compare to a published image → use the local dashboard. You do **not** need to download a large dataset to try the simulator.

---

## Quick start — synthetic data in four commands

Open PowerShell **inside the extracted project folder** and run:

```powershell
py -m pip install -e ".[dev,ui,train]"
py -m pytest -q
py -m paforge.cli demo --anatomy vascular --vascular-pattern branching --grid-size 96 --time-steps 420 --transducers 48 --reconstruction-method iterative --no-filter --no-noise --samples 1 --output output_vascular
py -m paforge.cli ui
```

The demo produces generated phantom maps, **raw simulated detector signals**, and a reconstructed image. Open the local dashboard at <http://127.0.0.1:7860> (or the address printed in your terminal). **No external dataset is needed** for this quick start. For the real InVision workflow, follow §§6–13 **in order**; do not skip the geometry and acquisition-normalization steps.

---

## Contents

1. [What PA-FORGE does](#1-what-pa-forge-does)
2. [Get the project and install Python dependencies](#2-get-the-project-and-install-python-dependencies)
3. [Check installation and run the automated tests](#3-check-installation-and-run-the-automated-tests)
4. [Run your first complete synthetic simulation](#4-run-your-first-complete-synthetic-simulation)
5. [Run the reconstruction benchmarks](#5-run-the-reconstruction-benchmarks)
6. [Download a small, verified experimental dataset bundle](#6-download-a-small-verified-experimental-dataset-bundle)
7. [Inspect the IPASC HDF5 sample](#7-inspect-the-ipasc-hdf5-sample)
8. [Extract one real InVision acquisition](#8-extract-one-real-invision-acquisition)
9. [Normalize the detector coordinate plane](#9-normalize-the-detector-coordinate-plane)
10. [Convert the real acquisition to PA-FORGE format](#10-convert-the-real-acquisition-to-pa-forge-format)
11. [Reconstruct an experimental photoacoustic image](#11-reconstruct-an-experimental-photoacoustic-image)
12. [View the published reference and segmentation](#12-view-the-published-reference-and-segmentation)
13. [Launch and use the v0.5.1 dashboard](#13-launch-and-use-the-v051-dashboard)
14. [Further configuration and generative-model options](#14-further-configuration-and-generative-model-options)
15. [Troubleshooting](#15-troubleshooting)
16. [Scientific limits, attribution, and release checklist](#16-scientific-limits-attribution-and-release-checklist)

---

## 1. What PA-FORGE does

The **PA** prefix stands for **Photoacoustic**. A proposed expansion of **FORGE** is *Framework for Optical Reconstruction and Generative Experimentation*; this is a project naming convention, not a standardized scientific acronym.

Two independent workflows are supported:

**A. Create synthetic data from scratch** (requires **no downloaded datasets**):

```text
Choose anatomy / vessel network
       ↓
Generate a 2D tissue phantom and material maps
       ↓
Compute approximate optical absorption / illumination / initial pressure
       ↓
Propagate pressure with a 2D heterogeneous acoustic FDTD solver
       ↓
Sample detector waveforms (raw time-series)
       ↓
Optionally add independently controlled noise and device effects
       ↓
Reconstruct a 2D image (DAS, derivative-weighted UBP, adjoint, or iterative FISTA)
       ↓
Export raw HDF5 signals + reconstructed images + synthetic ground truth + manifest
```

**B. Reconstruct a published experimental acquisition**:

```text
Publisher's real detector time-series + measured detector positions
       ↓
Explicitly inspect axes, timing, units, detector geometry, and provenance
       ↓
Extract one acquisition into a normalized PA-FORGE HDF5 file
       ↓
Qualitative 2D UBP / DAS reconstruction
       ↓
Preview / rotate / flip in dashboard (display only)
       ↓
Compare visually with the publisher's reconstruction and segmentation
```

**Anatomy presets:** `forearm`, `skin`, `breast`, `carotid`, and standalone `vascular`. The vascular generator supports `branching`, `paired`, and `plexus` networks; its procedural masks distinguish artery, vein, and capillary structures.

**Phantom generation options:** `procedural` (working default), `cvae`, and `vascular_cvae` (the latter two **require compatible trained local weights**). No model pretrained on real human vascular labels is included.

**Noise:** OFF by default; optional `gaussian`, `colored`, `impulsive`, `drift`, and user-supplied `empirical` noise. Acquisition effects such as filtering, laser jitter, and dropped channels are separately controlled.

**Spatial scope:** acoustic propagation, phantoms, and reconstructions are **2D**, even though detector coordinates are stored as three-component XYZ vectors.

---

## 2. Get the project and install Python dependencies

### 2.1 Download and extract

Download the **PA-FORGE v0.5.1 Dashboard Edition** project archive and extract it. The main project folder inside the archive is still named **`PA-FORGE-v0.5`**; this is expected. The v0.5.1 update changes the dashboard, **not** the physics core. `pyproject.toml` may therefore still report package version `0.5.0`.

For example, your folder may be:

```text
C:\Research\PA-FORGE-v0.5\
```

The project directory must contain `pyproject.toml`, `paforge/`, `tests/`, `examples/`, and this `README.md`.

**Open PowerShell in the project directory:** in Windows File Explorer, open the `PA-FORGE-v0.5` folder, type `powershell` into the address bar, and press Enter. This avoids quoting a long path with spaces and parentheses. Alternatively, use a quoted path:

```powershell
Set-Location 'C:\Research\PA-FORGE-v0.5'
```

### 2.2 Check Python and pip

```powershell
py --version
py -m pip --version
```

If `py` is not installed, install Python from <https://www.python.org/downloads/> and reopen PowerShell.

### 2.3 Install **directly**, without `venv` or `.env`

These are the exact command forms used in the Windows walkthrough:

```powershell
py -m pip install --upgrade pip
py -m pip install -e ".[dev,ui,train]"
```

This installs the PA-FORGE project in editable mode plus the test runner, Gradio dashboard, and PyTorch generative-model dependencies. **No virtual environment is required.** Note that a global/user installation can affect other Python projects; a dedicated environment is recommended for shared laboratory work but not required for this walkthrough. PyTorch may be a relatively large download.

To verify that the project is installed:

```powershell
py -m pip show pa-forge
```

> **Windows PATH tip:** `paforge --help` can fail with “The term 'paforge' is not recognized” when Python's Scripts directory is not on PATH. This does **not** necessarily mean the package is missing. All commands in this README use the reliable equivalent `py -m paforge.cli ...`.

---

## 3. Check installation and run the automated tests

```powershell
py -m paforge.cli --help
py -m pytest -q
```

**Expected for the complete optional-dependency installation:** the supplied v0.5/v0.5.1 test suite previously passed **63 tests**. The total can differ if tests are added later. The important criterion is **zero failures**.

If only the minimal Python dependencies are installed, two tests may fail because `gradio` or `torch` is missing. Install the optional extras as in §2.3. For a synthetic-core-only check, you can temporarily exclude those two optional-dependency tests:

```powershell
py -m pytest -q -k "not test_gradio_panel_constructs and not test_uncalibrated_learned_mode_rejects_fabricated_class_identity"
```

**Do not mistake skipped tests for passing tests.** For the full dashboard/generative release, use the complete test command.

**Note about the bundled helper script:** `scripts/first_real_data.ps1` was originally written for an activated virtual environment and calls `paforge` directly. If you are following this **no-venv** Windows guide, use the `py -m paforge.cli ...` commands below instead; they are the verified command form for this installation path.

---

## 4. Run your first complete synthetic simulation

**No external dataset is needed.** The `demo` command creates its own procedural phantom; the output is *synthetic numerical data*, not a published experimental acquisition.

Run the actual vascular example used during development:

```powershell
py -m paforge.cli demo --anatomy vascular --vascular-pattern branching --grid-size 96 --time-steps 420 --transducers 48 --reconstruction-method iterative --no-filter --no-noise --samples 1 --output output_vascular
```

This selects a `96 × 96` grid, 48 transducers, 420 time steps, iterative FDTD reconstruction using FISTA, and zero **added** noise. `--no-filter` is required for the current matched iterative FDTD implementation.

Look in:

```text
output_vascular/
└── vascular_00000/
    ├── ground_truth.h5          # synthetic truth: initial pressure, material/vessel maps
    ├── raw_signals.h5           # simulated multichannel detector time-series
    ├── reconstruction.h5       # numeric reconstructed image
    ├── reconstruction.png      # reconstructed image preview
    ├── vessel_types.png        # vessel-class display (vascular mode)
    ├── manifest.json           # configuration and provenance
    └── overview.png            # multi-panel experiment preview
```

**What a successful run proves:** the selected numerical forward model, detector sampling, reconstruction, and file export execute. It does *not* establish realism or quantitative accuracy on real equipment.

### Other preset anatomies

```powershell
py -m paforge.cli demo --anatomy forearm --no-noise --output output_forearm
py -m paforge.cli demo --anatomy skin --no-noise --output output_skin
py -m paforge.cli demo --anatomy breast --no-noise --output output_breast
py -m paforge.cli demo --anatomy carotid --no-noise --output output_carotid
```

### Optional noise example

```powershell
py -m paforge.cli demo --anatomy vascular --noise --engine gaussian --engine colored --output output_noisy
```

Noise remains OFF unless explicitly enabled. `--no-noise` disables stochastic noise generators; it does not automatically disable all deterministic device effects.

---

## 5. Run the reconstruction benchmarks

### 5.1 Standard synthetic benchmark

```powershell
py -m paforge.cli benchmark --anatomy vascular --seeds 3 --grid 64 --transducers 48 --iterations 20 --output experiments/high_quality_matched
```

This benchmark compares synthetic reconstructions; the iterative method uses FISTA. It needs no downloaded dataset.

### 5.2 Model-mismatch benchmark (the one used in our actual test)

```powershell
py -m paforge.cli benchmark-mismatch --output experiments/mismatch --seeds 3 --grid 48 --transducers 20 --time-steps 175 --iterations 8 --assumed-sound-m-s 1630
```

Key outputs:

```text
experiments/mismatch/
  metrics.csv
  results.json
  figure_model_mismatch.png
  figure_metrics.png
```

The figure displays **DAS**, **derivative-weighted backprojection**, **matched FDTD + FISTA**, and **mismatched homogeneous-speed FDTD + FISTA**. **FISTA has not been removed.** The matched case shares the generating forward operator (an *inverse crime*); the deliberately mismatched case assumes incorrect sound speed. Poor DAS images and degraded mismatched reconstructions are not, by themselves, defects in the data generator.

![Synthetic 2D model-mismatch benchmark: NOT experimental validation](experiments/mismatch/figure_model_mismatch.png)

For a more demanding test (more compute and memory; improvement is not guaranteed):

```powershell
py -m paforge.cli benchmark-mismatch --output experiments/mismatch_highres --seeds 3 --grid 64 --transducers 48 --time-steps 260 --iterations 20 --assumed-sound-m-s 1630
```

**Important:** The project is a research prototype, and the simple DAS is not an optimized scanner-specific implementation. Compare reconstruction methods under controlled geometry, sound-speed, preprocessing, and sampling assumptions before claiming superiority.

---

## 6. Download a small, verified experimental dataset bundle

For the **synthetic** pipeline, skip this section. For the real-data path, use these small Zenodo assets; we specifically chose them to avoid the huge published archives.

| Source | Selected asset | Download size (approx.) | Purpose |
|---|---|---:|---|
| [Digital Twins for PA QA](https://zenodo.org/records/15388429) | `ipasc_file.hdf5` | 4.8 MB | Standard-format HDF5 interoperability/geometry inspection; **not independently proven to be a physical measurement** |
| [OpenQPAIData](https://zenodo.org/records/14044853) | `invision.zip` | 238.1 MB | Real photoacoustic acquisitions, publisher reconstructions, segmentation and property maps |
| [OpenQPAIData](https://zenodo.org/records/14044853) | `DIS.zip` | 132.1 MB | Optical-material measurements for future calibration; **not acoustic detector signals** |
| Both | README / material mapping metadata | <0.1 MB | Provenance and file interpretation |

**Expected total selected download:** **375,015,995 bytes** (~375 MB decimal, excluding local Python dependencies and ZIP extraction). The published data carry their own terms; review the current publisher record before using `--accept-licenses`. Installation and dashboard startup **do not automatically download third-party data**.

### 6.1 List and inspect sources

```powershell
py -m paforge.cli datasets list
py -m paforge.cli datasets inspect digital-twin-iqa
py -m paforge.cli datasets inspect openqpai
```

### 6.2 Do a dry run (downloads nothing)

```powershell
py -m paforge.cli datasets download --dataset digital-twin-iqa --dataset openqpai --profile starter --dry-run --max-file-gb 0.5 --max-total-gb 0.6 --output data/external
```

In our verified walkthrough, the dry-run reported:

```text
status: digital-twin-iqa=planned; openqpai=planned
scheduled_gb: 0.375
```

Review the **actual `assets` list**; a `starter` profile is a selection rule, not a permanent guarantee of identical filenames if publishers update records.

### 6.3 Download the same selected bundle

```powershell
py -m paforge.cli datasets download --dataset digital-twin-iqa --dataset openqpai --profile starter --accept-licenses --max-file-gb 0.5 --max-total-gb 0.6 --output data/external
```

The tool downloads to source-specific folders, resumes where supported, and verifies the publisher checksums if present. In our real Windows run, all **six selected files** were marked `downloaded`, `checksum_verified: true`, and both sources reported `complete_selected`.

### 6.4 Confirm files and inspect report

```powershell
Get-ChildItem data\external -Recurse -File | Select-Object FullName, Length
Get-Content data\external\download_report.json
```

Expected key files:

```text
data/external/
├── digital-twin-iqa/
│   ├── README.md
│   └── ipasc_file.hdf5
└── openqpai/
    ├── README_data.md
    ├── material_mapping.json
    ├── DIS.zip
    └── invision.zip
```

`complete_selected` means *selected files downloaded correctly*. It does **not** mean they have already been ingested, calibrated, reconstructed, or scientifically validated. An internal downloader report may say version `0.4.0`; this is a module/report version and not evidence that the v0.5.1 UI is absent.

---

## 7. Inspect the IPASC HDF5 sample

### 7.1 Generate a non-destructive inventory

```powershell
py -m paforge.cli inventory-hdf5 "data/external/digital-twin-iqa/ipasc_file.hdf5" --output "experiments/ipasc_inventory.json"
Get-Content experiments\ipasc_inventory.json
```

An inventory may end in `"truncated": true` because there are **many detector metadata nodes**; that does not imply the source file is truncated.

### 7.2 Inspect the actual scalar metadata and a few detector positions

Copy this entire **PowerShell here-string** (including `@'` and `'@ | py -`):

```powershell
@'
import h5py
import numpy as np

path = "data/external/digital-twin-iqa/ipasc_file.hdf5"

with h5py.File(path, "r") as f:
    print("SIGNAL SHAPE:", f["binary_time_series_data"].shape)

    print("\nACQUISITION METADATA:")
    for key, value in f["meta_data"].items():
        if isinstance(value, h5py.Dataset):
            print(key, "=", value[()])

    detectors = f["meta_data_device/detectors"]
    names = sorted(detectors.keys())
    print("\nDETECTOR COUNT:", len(names))
    print("\nFIRST 3 DETECTORS:")

    for name in names[:3]:
        d = detectors[name]
        print("Detector:", name)
        for key in ["detector_position", "detector_orientation",
                    "detector_geometry_type", "detector_geometry"]:
            if key in d:
                print(" ", key, "=", d[key][()])

    positions = np.array([
        detectors[name]["detector_position"][()]
        for name in names
    ])
    print("\nPOSITION MIN:", positions.min(axis=0))
    print("POSITION MAX:", positions.max(axis=0))
    print("POSITION SPREAD:", np.ptp(positions, axis=0))
'@ | py -
```

**Actual inspected sample:** `(256, 2030, 1, 1)` signals, **256 detectors**, **40 MHz** sampling (`25 ns` between samples), **700 nm** wavelength, and detectors lying in the **XZ plane** (`Y=0`). The first coordinates and complete ranges are reported by the script; do not assume all IPASC datasets share this geometry. The small Digital Twins sample's experimental-versus-synthetic provenance still needs publisher-level verification; do **not** automatically mark it experimental.

---

## 8. Extract one real InVision acquisition

### 8.1 Look inside the ZIP without extracting it

```powershell
tar -tf "data/external/openqpai/invision.zip" | Select-Object -First 50

tar -tf "data/external/openqpai/invision.zip" | Select-String '^invision/raw/' | Select-Object -First 20

tar -tf "data/external/openqpai/invision.zip" | Select-String '^invision/recon/' | Select-Object -First 15
```

Read the publisher's file guide:

```powershell
Get-Content "data/external/openqpai/README_data.md" -Encoding UTF8 | Select-Object -First 100
```

The `-Encoding UTF8` flag avoids garbled accented author names and file-tree symbols in some PowerShell installations.

From the source guide, InVision contains measured `raw/` IPASC HDF5 arrays, published `recon/` NumPy images, `labels/` segmentation maps, plus `mua/` and `musp/` optical maps. InVision is a **2D** target; the much larger SVOT branch includes **3D** volumes, which this PA-FORGE release cannot reconstruct volumetrically.

### 8.2 Extract only the selected acquisition and reference arrays

```powershell
New-Item -ItemType Directory -Force "data\selected_invision"

tar -xf "data/external/openqpai/invision.zip" -C "data/selected_invision" "invision/raw/P.5.10.2_700_ipasc.hdf5" "invision/recon/P.5.10.2_700.npy" "invision/labels/P.5.10.2.npy"
```

This avoids unpacking every acquisition. It does not modify the original publisher ZIP.

### 8.3 Verify this exact acquisition and its reference files

```powershell
@'
import h5py
import numpy as np

path = "data/selected_invision/invision/raw/P.5.10.2_700_ipasc.hdf5"

with h5py.File(path, "r") as f:
    print("SIGNALS:", f["binary_time_series_data"].shape)
    print("\nMETADATA:")
    for name in ["ad_sampling_rate", "acquisition_wavelengths",
                 "dimensionality", "scanning_method"]:
        print(name, "=", f["meta_data"][name][()])

    detectors = f["meta_data_device/detectors"]
    names = sorted(detectors.keys())
    positions = np.array([
        detectors[n]["detector_position"][()] for n in names
    ])
    print("\nDETECTORS:", len(names))
    print("MIN XYZ:", positions.min(axis=0))
    print("MAX XYZ:", positions.max(axis=0))
    print("SPREAD XYZ:", np.ptp(positions, axis=0))
    print("FINITE:", np.isfinite(positions).all())

print("\nREFERENCE IMAGE:")
ref = np.load("data/selected_invision/invision/recon/P.5.10.2_700.npy")
print("Shape:", ref.shape)
print("Range:", ref.min(), ref.max())

print("\nSEGMENTATION:")
seg = np.load("data/selected_invision/invision/labels/P.5.10.2.npy")
print("Shape:", seg.shape)
print("Labels:", np.unique(seg))
'@ | py -
```

**Observed for this sample:** `256 × 2030 × 1 × 1` raw signals, 40 MHz, 700 nm, 256 sensors; a published **288 × 288** reference image; and a **288 × 288** segmentation with labels `0, 1, 2, 3`. The sensor `Y` coordinate is zero, while `X` and `Z` vary. Note that these integer segmentation labels are **material classes** from the source, **not** inherently artery/vein/capillary labels.

---

## 9. Normalize the detector coordinate plane

**This step matters.** PA-FORGE's qualitative **2D** reconstructor uses the **first two columns** of the provided XYZ detector-coordinate array as the reconstruction plane, and rejects substantial spread in the third column. InVision's reported sensors lie in **XZ**, so we explicitly map:

```text
Original physical coordinate:   (X,  Y=0,  Z)
PA-FORGE 2D coordinate array:   (X,  Z,    0)
```

This is **not** the same as rotating the image preview. It is a necessary, documented coordinate convention for this particular acquisition. **Never apply it blindly to a different instrument or 3D acquisition.**

```powershell
@'
import h5py
import numpy as np
from pathlib import Path

source = Path("data/selected_invision/invision/raw/P.5.10.2_700_ipasc.hdf5")
output = Path("data/selected_invision/detector_positions_2d.npy")

with h5py.File(source, "r") as f:
    detectors = f["meta_data_device/detectors"]
    names = sorted(detectors.keys())
    original = np.array([
        detectors[name]["detector_position"][()]
        for name in names
    ], dtype=np.float64)

assert original.shape == (256, 3)
assert np.all(np.isfinite(original))
assert np.allclose(original[:, 1], 0)

converted = np.column_stack([
    original[:, 0],
    original[:, 2],
    np.zeros(len(original))
])

np.save(output, converted)
print("Saved:", output)
print("Shape:", converted.shape)
print("X range:", converted[:, 0].min(), converted[:, 0].max())
print("Y range:", converted[:, 1].min(), converted[:, 1].max())
print("Z range:", converted[:, 2].min(), converted[:, 2].max())
'@ | py -
```

**Expected:** `(256, 3)` coordinates, with the new `Z` column exactly zero. The original HDF5 and original measured coordinates remain unchanged.

---

## 10. Convert the real acquisition to PA-FORGE format

Make a folder for normalized experimental measurements:

```powershell
New-Item -ItemType Directory -Force "data\real"
```

The original signal shape has detector axis **0**, time axis **1**, and singleton axes **2** and **3**. We explicitly select index 0 for both extra axes and attach the transformed measured positions:

```powershell
py -m paforge.cli extract-acquisition "data/selected_invision/invision/raw/P.5.10.2_700_ipasc.hdf5" --key binary_time_series_data --detector-axis 0 --time-axis 1 --fixed-axis 2=0 --fixed-axis 3=0 --sampling-rate-hz 40000000 --wavelength-nm 700 --source-id "OpenQPAIData_InVision_P.5.10.2_700" --group-id "P.5.10.2" --positions "data/selected_invision/detector_positions_2d.npy" --attest-experimental --output "data/real/invision_P.5.10.2_700.h5"
```

**Expected terminal message:**

```text
Exported original experimental acquisition to data\real\invision_P.5.10.2_700.h5
```

The output is **PA-FORGE-native HDF5**, not an IPASC-certified derivative. It records the original source ID, group ID, wavelength, sample clock, detector geometry, and normalized `raw_data` shape `(256, 2030)`.

**Provenance warning:** `--attest-experimental` is a user declaration that this particular acquisition originates from real published measurements. It is appropriate here because OpenQPAIData documents InVision phantom measurements. **Do not use this flag to relabel a synthetic Digital Twins file as experimental.**

---

## 11. Reconstruct an experimental photoacoustic image

### 11.1 The exact successful command

```powershell
py -m paforge.cli reconstruct-acquisition "data/real/invision_P.5.10.2_700.h5" --method ubp --fov-m 0.06 --grid-size 256 --sound-speed-m-s 1500 --output "experiments/real_invision_ubp"
```

**Do not use `--grid-size 288`** in this version. Although the publisher reference is `288 × 288`, `SimConfig` currently limits `grid_size` to **256**. The original attempted 288 grid failed validation; 256 ran successfully. A matching pixel count would not, by itself, establish matching FOV or orientation.

The example assumes a **60 mm FOV** and a **homogeneous 1500 m/s sound speed** as *initial exploratory settings*, **not published scanner calibration values**. The grid is `256 × 256`, and the full 256 channels / 2030 samples per channel are retained.

**Actual observed terminal result:**

```json
{"output": "experiments\\real_invision_ubp", "image_shape": [256, 256], "quantitative": false}
```

**Files created:**

```text
experiments/real_invision_ubp/
├── experimental_reconstruction.h5
├── experimental_reconstruction.png
└── reconstruction_manifest.json
```

The first reconstruction visibly recovered the large circular phantom outline and some internal features. It also showed artifacts and an orientation difference from the publisher reconstruction. This establishes **functional experimental-data ingestion and qualitative reconstruction**, not quantitative accuracy, validated physical calibration, or superiority to reference methods.

### 11.2 What about FISTA on experimental data?

**FISTA remains available in the *synthetic* iterative reconstruction pipeline**, but the current `reconstruct-acquisition` command supports only `ubp` and `das`. Do not claim that this InVision reconstruction ran FISTA. Experimental iterative reconstruction requires an independently verified scanner-specific forward operator and calibration.

---

## 12. View the published reference and segmentation

### 12.1 Make a side-by-side qualitative comparison

```powershell
@'
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

ref = np.load(
    "data/selected_invision/invision/recon/P.5.10.2_700.npy"
)
ours = Image.open(
    "experiments/real_invision_ubp/experimental_reconstruction.png"
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(ref, cmap="magma")
axes[0].set_title("OpenQPAIData Reference")
axes[1].imshow(ours)
axes[1].set_title("PA-FORGE UBP")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig("experiments/real_invision_ubp/comparison.png", dpi=200)
plt.show()
'@ | py -
```

### 12.2 Plot the publisher's four-class segmentation

```powershell
@'
import numpy as np
import matplotlib.pyplot as plt

path = "data/selected_invision/invision/labels/P.5.10.2.npy"
labels = np.load(path)

plt.figure(figsize=(7, 7))
plt.imshow(labels, cmap="tab10", vmin=0, vmax=3)
plt.title("OpenQPAIData - Phantom P.5.10.2 Segmentation")
plt.colorbar(ticks=[0, 1, 2, 3])
plt.axis("off")
plt.tight_layout()
plt.savefig("experiments/real_invision_ubp/segmentation_reference.png", dpi=200)
plt.show()
'@ | py -
```

### 12.3 Optional orientation-diagnostic grid

If the reconstructed phantom appears rotated relative to the reference, examine display-only alternatives:

```powershell
@'
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

ref = np.load("data/selected_invision/invision/recon/P.5.10.2_700.npy")
ours = np.array(Image.open(
    "experiments/real_invision_ubp/experimental_reconstruction.png"
).convert("L"))

variants = {
    "Original": ours,
    "Rotate 90 CCW": np.rot90(ours, 1),
    "Rotate 90 CW": np.rot90(ours, -1),
    "Rotate 180": np.rot90(ours, 2),
    "Flip LR": np.fliplr(ours),
    "Flip UD": np.flipud(ours),
}

fig, axes = plt.subplots(2, 4, figsize=(15, 8))
axes[0, 0].imshow(ref, cmap="magma")
axes[0, 0].set_title("Reference")
for ax, (name, img) in zip(axes.flat[1:], variants.items()):
    ax.imshow(img, cmap="magma")
    ax.set_title(name)
for ax in axes.flat:
    ax.axis("off")
plt.tight_layout()
plt.savefig("experiments/real_invision_ubp/orientation_test.png", dpi=180)
plt.show()
'@ | py -
```

These plots are for **visual inspection**. The published `288 × 288` image and our `256 × 256` image are **not** physically registered or photometrically calibrated. **Do not compute or publish raw pixelwise PSNR, SSIM, Pearson correlation, or material segmentation accuracy from these side-by-side PNGs without first verifying image coordinates, FOV, registration, and intensity convention.**

---

## 13. Launch and use the v0.5.1 dashboard

### 13.1 Start the local app

The v0.5.1 full archive **already includes the updated `paforge/ui.py`**. If you are still using an older v0.5 folder and have only the `PA-FORGE_v0.5.1_ui_only.py` patch, back up `paforge/ui.py` and replace **only that file**. No optical, acoustic, reconstruction, or dataset modules need editing.

```powershell
py -m paforge.cli ui
```

Open the local address printed by PowerShell (normally <http://127.0.0.1:7860>). The app binds to `127.0.0.1`: it is intended to run locally, not be exposed on the public internet.

### 13.2 Tab 01 — Synthetic Generator

Choose anatomy (`vascular` for the first test), network pattern/branch levels, sample count/seed, wavelengths, detector geometry/count, grid size, time steps, noise and reconstruction method. For iterative reconstruction, disable the detector bandpass filter. The procedural generator works without neural-network weights. Click **Generate simulated signals and images**; inspect the preview and export ZIP. For a physically controlled baseline, leave added noise OFF.

### 13.3 Tab 02 — Experimental Reconstruction

Use the normalized InVision file from §10:

```text
Acquisition HDF5 path:
data/real/invision_P.5.10.2_700.h5

Optional reference NumPy image:
data/selected_invision/invision/recon/P.5.10.2_700.npy

Method: UBP
Grid: 256
FOV: 60 mm
Assumed homogeneous sound speed: 1500 m/s
```

Click **Inspect acquisition metadata**, then **Reconstruct acquisition**. The dashboard displays the PA-FORGE image and, if provided, the unmodified published reference. A downloadable archive includes the numeric reconstructed HDF5, original reconstruction PNG, manifest, and dashboard settings.

**User-controlled display orientation:** select counterclockwise rotation **0° / 90° / 180° / 270°**, optionally **Flip left/right** or **Flip up/down**. These transforms update the **preview only**, without recomputing reconstruction or changing the saved numeric HDF5 or detector time-series. This gives users flexibility when different vendors publish different image orientations. It is **not** a substitute for physically verified coordinate conversion or scientific image registration.

**Important:** the dashboard expects **normalized PA-FORGE HDF5** with `/raw_data` and `/detector_positions_m`, not a raw, vendor-specific IPASC HDF5 directly. Perform §9–§10 first.

### 13.4 Tab 03 — Research Information

Explains the numerical/experimental boundary, modeling assumptions, non-3D scope, and pretrained-weight requirements.

---

## 14. Further configuration and generative-model options

### 14.1 Run a reproducible YAML configuration

```powershell
py -m paforge.cli generate examples/vascular_config.yaml --output output_yaml
```

The included example uses procedural vascular generation and matched iterative reconstruction with `detector_filter: false`. For more precise configuration than the dashboard exposes (e.g., regularization), edit a **copy** of the example YAML. Do not silently change the published experiment's original configuration.

### 14.2 Synthetic demonstration of the CVAE training path

With PyTorch installed:

```powershell
py -m paforge.cli train-demo --output models/demo_cvae.pt --epochs 1 --samples 32
py -m paforge.cli train-vascular-demo --weights models/synthetic_vascular.pt --samples 64 --epochs 8
```

These commands train on **procedurally generated masks only**. Passing them does not establish learned realism or clinical validity. For actual vascular-CVAE training, provide an independently curated `types` segmentation dataset with shape `[N,64,64]` and integer labels `0..3`:

```powershell
py -m paforge.cli train-vascular your_training_types.npz models/vascular.pt --epochs 50
```

Training on real data requires proper licensing, annotation provenance and group-disjoint validation; the publisher's `labels` array of **material regions** is **not** interchangeable with arterial/venous classes.

### 14.3 Optional external reconstruction plugin

The `reconstruct-plugin` interface can call a **separately reviewed and installed** trusted module that implements a compatible `reconstruct(traces, sampling_rate_hz, positions_m, image_size, fov_m)` API. It does **not** mean the proposed Zenodo reconstruction package at <https://zenodo.org/records/23348355> was automatically integrated or proven inferior. Review external code, licensing and algorithm assumptions before integrating it. See `RESEARCH_SPRINT_README.md` for the plugin example.

---

## 15. Troubleshooting

| Symptom | Explanation and exact remedy |
|---|---|
| `paforge : The term 'paforge' is not recognized` | Windows cannot find the installed console script. Use `py -m paforge.cli --help` and consistently replace `paforge` with `py -m paforge.cli`. |
| `ModuleNotFoundError: No module named 'gradio'` | Install the dashboard extra: `py -m pip install -e ".[ui]"`, or the full command `py -m pip install -e ".[dev,ui,train]"`. |
| `ModuleNotFoundError: No module named 'torch'` | Install the training extra: `py -m pip install -e ".[train]"`. It may be a large dependency. |
| Two failing tests and 61 passing | The two originally observed failures were due to missing `gradio`/`torch`, not demonstrated failures of numerical physics. Install extras and rerun **all** tests. |
| `grid_size: Input should be less than or equal to 256` | The reconstructor does not accept `--grid-size 288`. Use `--grid-size 256` as in §11. |
| `Significant 3D sensor spread` | Current experimental reconstruction supports a justified 2D geometry only. Confirm the **physical** detector coordinate plane; §9 applies only to the inspected planar XZ InVision sample. Do not project truly 3D data by guessing. |
| Iterative reconstruction complains about detector filter | Add `--no-filter` to the `demo` command, or disable filter in dashboard. |
| `README_data.md` shows `GrÃ¶hl` or unreadable line symbols | Use `Get-Content ... -Encoding UTF8`. Encoding corruption in PowerShell display does not imply corrupt data. |
| `"truncated": true` in inventory | The metadata scanner has a node-count reporting limit; the HDF5 is not necessarily truncated. |
| Reference image seems rotated by 90° | Use the dashboard's **display-only rotation**, not a destructive edit to raw data; scientific comparison still requires real coordinate-frame registration. |
| Reference is 288 × 288, ours is 256 × 256 | They can differ in pixels and FOV. This is not, by itself, a processing failure. Do not score unmatched pixels directly. |
| Download fails or file size differs | Rerun `datasets inspect`, review `download_report.json`, check network/proxy and publisher metadata. Verify checksums before analysis. |
| Dashboard cannot find experimental file | Use `data/real/invision_P.5.10.2_700.h5` from §10, verify the current project directory, or use an absolute path. Original IPASC files are not directly interchangeable with normalized input. |
| FISTA does not appear under real-acquisition methods | It is intentionally available for the synthetic matched-operator experiment; the real-acquisition CLI/UI currently offers only UBP/DAS. |

For command-specific usage:

```powershell
py -m paforge.cli --help
py -m paforge.cli demo --help
py -m paforge.cli datasets download --help
py -m paforge.cli extract-acquisition --help
py -m paforge.cli reconstruct-acquisition --help
```

---

## 16. Scientific limits, attribution, and release checklist

### What has actually been demonstrated

- **Synthetic functionality:** procedural phantoms for multiple anatomies, approximate optical source modeling, 2D FDTD detector time-series, iterative matched reconstruction and export.
- **Synthetic numerical benchmark:** a reproducible experiment showing that matched-model FISTA can recover a synthetic phantom better than the deliberately mismatched or simple backprojection baselines under the tested conditions. The matched result uses the generating forward operator (*inverse crime*).
- **Dataset acquisition:** in the Windows walkthrough, **six files / ~375 MB** were downloaded from the two Zenodo records, and their reported publisher MD5 checksums verified.
- **Real HDF5 processing:** **one published InVision tissue-mimicking-phantom acquisition** (`P.5.10.2`, 700 nm) was inspected, normalized and qualitatively reconstructed; its outer phantom structure was visible and a difference in display orientation was observed.
- **v0.5.1 interface:** synthetic generation and qualitative experimental reconstruction are accessible from the local Gradio application; rotation and flip affect previews only. Automated dashboard tests were conducted with controlled fixtures; the specific Windows interactive behavior should be smoke-tested before distribution.

### What has **not** been established

- Patient-level or human-anatomical realism of the synthetic phantoms; training or validation of the CVAEs on representative real anatomy.
- Fully calibrated light transport, scanner impulse/point-spread response, phase, timing offsets, acoustic attenuation/dispersion, or heterogeneous experimental sound-speed maps.
- Accurate 3D volumetric acoustics/reconstruction or IPASC **conformance certification**.
- Quantitative registration, calibrated absorption estimates, blinded accuracy scoring against reference masks, independent scanner comparison, or statistically validated superiority over other packages.
- Native implementation/verification of the separate Zenodo record `23348355` reconstruction methods.

### Published data and credit

- **OpenQPAIData:** Gröhl and collaborators, *OpenQPAIData: Measurements of test objects with reference labels for quantitative photoacoustic imaging research*. Publisher record: <https://zenodo.org/records/14044853>; code: <https://github.com/BohndiekLab/OpenQPAIData>. Dataset README reports CC BY 4.0 for data and MIT for associated code; **verify current publisher terms** and cite the exact version used.
- **Digital Twins for reconstruction QA:** <https://zenodo.org/records/15388429>. Publisher metadata in the downloader reported CC BY 4.0; verify the record. The small sample is not the complete benchmark.
- **Larger optional Cambridge dataset (not downloaded in this tutorial):** <https://www.repository.cam.ac.uk/items/bf455f97-1ff0-4694-87ca-b72c85366dc6>.

**No third-party raw archives are included** in the PA-FORGE project ZIP. The updated release ZIP also contains a fresh `checksums_release_v051.sha256` manifest; older `checksums.sha256` files are historical and may refer to pre-dashboard package contents. Make sure the PA-FORGE *software itself* has an explicit distribution license before publishing it as an open-source package; the upstream datasets' CC BY license does not automatically license PA-FORGE source code.

### Before sharing PA-FORGE publicly

- [ ] Run `py -m pytest -q` on a clean Windows machine with full optional dependencies installed.
- [ ] Run one synthetic `demo` and inspect `raw_signals.h5`, reconstruction, and `manifest.json`.
- [ ] Launch all three dashboard tabs and confirm preview, ZIP downloads, rotation/flip and error messages.
- [ ] Include an explicit source-code license, a versioned changelog, and citation/attribution guidance.
- [ ] Keep device-specific example parameters labeled **illustrative** until independently calibrated.
- [ ] Retain experiment configurations and provenance with any images or published metrics.
- [ ] Do not call a visually plausible, unregistered reconstruction *quantitatively validated*.

---

## One-page command recap (Windows PowerShell)

Run these **from the project root**. Detailed explanations and safeguards appear in the sections above.

```powershell
# Install directly, no venv
py -m pip install -e ".[dev,ui,train]"

# Check package and tests
py -m paforge.cli --help
py -m pytest -q

# Synthetic full pipeline
py -m paforge.cli demo --anatomy vascular --vascular-pattern branching --grid-size 96 --time-steps 420 --transducers 48 --reconstruction-method iterative --no-filter --no-noise --samples 1 --output output_vascular

# Synthetic benchmark
py -m paforge.cli benchmark-mismatch --output experiments/mismatch --seeds 3 --grid 48 --transducers 20 --time-steps 175 --iterations 8 --assumed-sound-m-s 1630

# Small dataset plan, then actual download after license review
py -m paforge.cli datasets download --dataset digital-twin-iqa --dataset openqpai --profile starter --dry-run --max-file-gb 0.5 --max-total-gb 0.6 --output data/external
py -m paforge.cli datasets download --dataset digital-twin-iqa --dataset openqpai --profile starter --accept-licenses --max-file-gb 0.5 --max-total-gb 0.6 --output data/external
Get-Content data\external\download_report.json

# Inspect IPASC, then follow §§8–10 for verified geometry/extraction
py -m paforge.cli inventory-hdf5 "data/external/digital-twin-iqa/ipasc_file.hdf5" --output "experiments/ipasc_inventory.json"

# After extracting and normalizing the selected InVision acquisition
py -m paforge.cli reconstruct-acquisition "data/real/invision_P.5.10.2_700.h5" --method ubp --fov-m 0.06 --grid-size 256 --sound-speed-m-s 1500 --output "experiments/real_invision_ubp"

# Dashboard
py -m paforge.cli ui
```

**Final principle:** maintain the difference between **synthetic ground truth**, **measured detector signals**, **qualitative reconstructed images**, and **quantitatively validated experimental results**. PA-FORGE is useful precisely when those outputs and their provenance remain separate and reproducible.
