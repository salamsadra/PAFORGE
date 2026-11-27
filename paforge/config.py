"""All lengths, times and frequencies are SI units internally."""
from typing import Literal
from pydantic import BaseModel, Field, model_validator


class NoiseConfig(BaseModel):
    enabled: bool = False  # IMPORTANT: noise OFF is a first-class default
    engines: list[Literal["gaussian", "colored", "impulsive", "drift", "empirical"]] = Field(default_factory=lambda: ["gaussian"])
    relative_std: float = Field(default=0.025, ge=0, le=2.0)
    colored_ar: float = Field(default=0.92, ge=0, lt=1)
    impulse_probability: float = Field(default=0.001, ge=0, le=1)
    impulse_amplitude: float = Field(default=0.30, ge=0, le=20)
    drift_amplitude: float = Field(default=0.04, ge=0, le=10)
    empirical_path: str | None = None

    @model_validator(mode="after")
    def check_empirical(self):
        if self.enabled and "empirical" in self.engines and not self.empirical_path:
            raise ValueError("Empirical noise requires an existing empirical_path; it is NEVER fabricated.")
        return self


class SimConfig(BaseModel):
    anatomy: Literal["forearm", "skin", "breast", "carotid", "vascular", "material_phantom"] = "forearm"
    # Dedicated vessel-network generator; applies when anatomy="vascular".
    vascular_pattern: Literal["branching", "paired", "plexus"] = "branching"
    vascular_branch_levels: int = Field(default=3, ge=1, le=5)
    vascular_density: float = Field(default=1.0, ge=0.2, le=3.0)
    # Reconstruction methods: DAS is descriptive, iterative inversion uses an
    # exactly paired FDTD forward/adjoint without detector filtering.
    reconstruction_method: Literal["das", "ubp", "adjoint", "iterative", "adjoint_fdtd", "fista", "filtered_ubp", "fbp", "cf_das", "scf_das", "dmas", "ds_dmas", "vdas", "mv_das", "cfmv_das", "dmas_ubp", "vdas_ubp", "time_reversal", "iterative_time_reversal", "cg_tikhonov", "fista_ray", "landweber", "sirt", "iterative_das", "lsqr", "tv_fista", "admm_tv", "learned_unet", "learned_rescnn", "pnp_fista", "deep_image_prior", "patato_bp", "patato_model", "patato_opencl_bp", "patato_slow_bp"] = "ubp"
    reconstruction_iterations: int = Field(default=8, ge=1, le=100)
    reconstruction_weights: str | None = None
    reconstruction_l2: float = Field(default=0.001, ge=0, le=1.)
    seed: int = Field(default=42, ge=0)
    samples: int = Field(default=1, ge=1, le=10_000)
    grid_size: int = Field(default=96, ge=48, le=256)
    fov_m: float = Field(default=0.04, gt=0.001, le=0.3)
    time_steps: int = Field(default=380, ge=50, le=2000)
    cfl: float = Field(default=0.35, gt=0, le=0.50)
    wavelength_nm: int = Field(default=800, ge=660, le=1300)
    illumination_model: Literal["attenuation_proxy", "uniform_idealized"] = "attenuation_proxy"
    incident_fluence_j_m2: float = Field(default=30.0, gt=0, le=1000)
    transducers: int = Field(default=64, ge=8, le=512)
    geometry: Literal["circle", "arc", "linear"] = "circle"
    center_frequency_hz: float = Field(default=1.8e6, gt=0)
    bandwidth_hz: float = Field(default=1.8e6, gt=0)
    detector_filter: bool = True
    sound_speed_offset_m_s: float = Field(default=0, ge=-250, le=250)
    sound_speed_variation_fraction: float = Field(default=1, ge=0, le=4)
    scattering_scale: float = Field(default=1, ge=0.05, le=20)
    pulse_energy_jitter_fraction: float = Field(default=0, ge=0, le=0.5)
    channel_dropout_probability: float = Field(default=0, ge=0, le=1)
    reconstruct: bool = True
    export_images: bool = True
    export_truth: bool = True
    noise: NoiseConfig = Field(default_factory=NoiseConfig)
    # Conditional VAE is an optional *learned vessel prior*, not an end-to-end waveform generator.
    anatomy_engine: Literal["procedural", "cvae", "vascular_cvae", "material_cvae", "empirical_material"] = "procedural"
    generator_weights: str | None = None
    optical_profile: str | None = None
    material_temperature: float = Field(default=0.9, ge=0.1, le=2.)
    material_composition: list[float] | None = None
    material_deform_px: float = Field(default=1.0, ge=0, le=8)
    # Optional larger numerical propagation grid for synthetic material phantoms.
    # The 64x64 truth grid and its dx remain unchanged; only the acoustic domain grows.
    material_solver_padding_factor: Literal[1,2] = 1

    @model_validator(mode="after")
    def validate_engine(self):
        if self.reconstruct and self.reconstruction_method in ("iterative", "adjoint", "adjoint_fdtd", "fista", "time_reversal", "iterative_time_reversal") and self.detector_filter:
            raise ValueError("Exact FDTD iterative reconstruction requires detector_filter=False")
        if self.anatomy_engine == "vascular_cvae" and self.anatomy != "vascular":
            raise ValueError("vascular_cvae is only defined for anatomy=vascular")
        if self.material_solver_padding_factor == 2 and (self.anatomy != "material_phantom" or self.geometry != "circle"):
            raise ValueError("Material solver padding factor 2 currently requires material_phantom and circular receivers")
        if self.anatomy_engine in ("material_cvae","empirical_material"):
            if self.anatomy!="material_phantom":
                raise ValueError("material generators require anatomy=material_phantom")
            if not self.optical_profile:
                raise ValueError("Material phantom generation requires optical_profile")
        if self.anatomy=="material_phantom" and self.anatomy_engine not in ("material_cvae","empirical_material"):
            raise ValueError("material_phantom requires learned or empirical material generator")
        if self.anatomy_engine in ("cvae","vascular_cvae","material_cvae","empirical_material") and not self.generator_weights:
            raise ValueError("cvae anatomy engine requires generator_weights")
        return self


def preset(anatomy: str = "forearm") -> SimConfig:
    """Illustrative research presets, NOT configurations of a calibrated clinical instrument."""
    overrides = {
        "forearm": dict(geometry="circle", wavelength_nm=800),
        "skin": dict(geometry="linear", wavelength_nm=800, fov_m=0.025),
        "breast": dict(geometry="circle", wavelength_nm=800, fov_m=0.065),
        "carotid": dict(geometry="arc", wavelength_nm=800, fov_m=0.05),
        "vascular": dict(geometry="circle", wavelength_nm=800, fov_m=0.025, illumination_model="uniform_idealized"),
    }
    if anatomy not in overrides:
        raise ValueError(f"Unknown anatomy {anatomy}")
    return SimConfig(anatomy=anatomy, **overrides[anatomy])
