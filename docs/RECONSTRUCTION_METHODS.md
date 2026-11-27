# Reconstruction methods

The canonical `Acquisition` interface is the same whether the input came from a simulator or an experimental scanner. A ✓ below means **the method is callable for that type if its required geometry/checkpoint/backend exists**, not that it was validated on real acquisitions. An internal method marked approximate remains exploratory.

| Method key | Family | Simulated | Experimental | Model/notes |
|---|---|---|---|---|
| `das` | beamforming | ✓ conditional | ✓ conditional | straight-ray TOF |
| `ubp` | analytic | ✓ conditional | ✓ conditional | homogeneous TOF |
| `filtered_ubp` | analytic | ✓ conditional | ✓ conditional | homogeneous TOF |
| `fbp` | analytic | ✓ conditional | ✓ conditional | homogeneous TOF |
| `cf_das` | beamforming | ✓ conditional | ✓ conditional | straight-ray TOF |
| `scf_das` | beamforming | ✓ conditional | ✓ conditional | straight-ray TOF |
| `dmas` | beamforming | ✓ conditional | ✓ conditional | straight-ray TOF |
| `ds_dmas` | beamforming | ✓ conditional | ✓ conditional | straight-ray TOF |
| `vdas` | beamforming | ✓ conditional | ✓ conditional | straight-ray TOF |
| `mv_das` | beamforming | ✓ conditional | ✓ conditional | straight-ray TOF |
| `cfmv_das` | beamforming | ✓ conditional | ✓ conditional | straight-ray TOF |
| `dmas_ubp` | beamforming | ✓ conditional | ✓ conditional | homogeneous TOF |
| `vdas_ubp` | beamforming | ✓ conditional | ✓ conditional | homogeneous TOF |
| `adjoint_fdtd` | wave | ✓ conditional | ✓ conditional | homogeneous/heterogeneous 2D FDTD; sensors inside stable 2D wave grid |
| `fista` | wave | ✓ conditional | ✓ conditional | homogeneous/heterogeneous 2D FDTD; sensors inside stable 2D wave grid |
| `time_reversal` | wave | ✓ conditional | ✓ conditional | 2D FDTD; sensors inside stable 2D wave grid |
| `iterative_time_reversal` | wave | ✓ conditional | ✓ conditional | 2D FDTD; sensors inside stable 2D wave grid |
| `cg_tikhonov` | iterative | ✓ conditional | ✓ conditional | approximate travel-time operator |
| `fista_ray` | iterative | ✓ conditional | ✓ conditional | approximate travel-time operator |
| `landweber` | iterative | ✓ conditional | ✓ conditional | approximate travel-time operator |
| `sirt` | iterative | ✓ conditional | ✓ conditional | approximate travel-time operator |
| `iterative_das` | iterative | ✓ conditional | ✓ conditional | approximate travel-time operator |
| `lsqr` | iterative | ✓ conditional | ✓ conditional | approximate travel-time operator |
| `tv_fista` | iterative | ✓ conditional | ✓ conditional | approximate travel-time operator |
| `admm_tv` | iterative | ✓ conditional | ✓ conditional | approximate travel-time operator |
| `learned_unet` | learned | ✓ conditional | ✓ conditional | supervised synthetic paired data; **trained .pt needed** |
| `learned_rescnn` | learned | ✓ conditional | ✓ conditional | supervised synthetic paired data; **trained .pt needed** |
| `pnp_fista` | learned | ✓ conditional | ✓ conditional | ray model + trained denoiser; **trained .pt needed** |
| `deep_image_prior` | learned | ✓ conditional | ✓ conditional | ray-model signal consistency; no pretraining |
| `patato_bp` | external | ✓ conditional | ✓ conditional | PATATO external backend; **official PATATO package + runtime needed** |
| `patato_model` | external | ✓ conditional | ✓ conditional | PATATO external backend; **official PATATO package + runtime needed** |
| `patato_opencl_bp` | external | ✓ conditional | ✓ conditional | PATATO external backend; **official PATATO package + runtime needed** |
| `patato_slow_bp` | external | ✓ conditional | ✓ conditional | PATATO external backend; **official PATATO package + runtime needed** |

### Crucial distinctions

- The 13 beamforming/analytic variations are not interchangeable with full-wave inversion, and the 'FBP'/'UBP' methods are explicitly heuristic baselines. Variance/minimum-variance choices use approximate estimators, e.g. global rather than per-pixel covariance in `mv_das`.
- Four wave-based methods are experimental **2D FDTD**; `time_reversal` is reverse-time source injection, not an exact k-Wave time reversal implementation.
- Eight iterative ray-based methods use a consistent matrix-free interpolating forward/adjoint operator, not an acoustic wave-equation propagator. The ray operator has an adjoint dot-product test, but not device-specific forward-model calibration.
- Four learned modes: `learned_unet`, `learned_rescnn`, `pnp_fista` require pretrained local checkpoints; `deep_image_prior` trains a CNN for each input. PnP model uses a DAS-trained residual network and is explicitly exploratory/out-of-distribution when applied to ray-inversion iterates.
- Four PATATO methods use *original upstream code* via optional installation; they were **not executed with the actual PATATO binary** in this environment. Missing runtime is a reported error, not a fallback.
- Several method names correspond to familiar PATbox MATLAB algorithm families but have *not* been validated against the original MATLAB source code.
