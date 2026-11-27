"""Reproducible end-to-end photoacoustic forward simulation orchestration."""
from dataclasses import dataclass
import numpy as np
from .config import SimConfig
from .anatomy import generate_tissue
from .acoustic import simulate_pressure,apply_detector_filter,reconstruct_das,reconstruct_ubp
from .noise import add_noise, measured_added_snr_db
from .io import export_result


def _material_geometry_diagnostic(p0, config, receiver_config, pad_px=0, sponge_pixels=9):
    """Track support vs circular sensor ring and sponge; not real hardware calibration."""
    from .acoustic import sensor_positions
    n=receiver_config.grid_size
    mid=(n-1)/2
    active=np.asarray(p0)>0
    if not active.any(): raise ValueError('Material phantom has no positive initial-pressure support')
    yy,xx=np.indices((n,n), dtype=float)
    source_max=float(np.hypot(yy-mid,xx-mid)[active].max())
    pos=sensor_positions(receiver_config)
    receiver_r=float(np.hypot(pos[:,0]-mid,pos[:,1]-mid).min())
    boundary_margin=float(np.min(np.stack((pos[:,0],pos[:,1],n-1-pos[:,0],n-1-pos[:,1]))))
    return {'source_max_radius_px':source_max,'receiver_radius_px':receiver_r,
            'source_receiver_clearance_px':receiver_r-source_max,
            'receiver_nearest_boundary_px':boundary_margin,
            'sponge_width_px':sponge_pixels,
            'geometry_gate_pass':bool(source_max+2<=receiver_r and boundary_margin>=sponge_pixels),
            'simulation_only':True,'crop_padding_each_side_px':int(pad_px),
            'measurement_device_calibrated':False}


@dataclass
class SampleResult:
    tissue: object
    clean_signals: np.ndarray
    raw_signals: np.ndarray
    noise_components: dict
    dt_s: float
    detectors_m: np.ndarray
    reconstruction: np.ndarray | None
    info: dict


def simulate(config: SimConfig, index: int=0) -> SampleResult:
    if index<0:raise ValueError("index must be nonnegative")
    # Deterministically isolate random streams: toggling noise cannot modify anatomy.
    seed=np.random.SeedSequence([config.seed,index])
    a,n,l=seed.spawn(3)
    rng_anatomy=np.random.default_rng(a);rng_noise=np.random.default_rng(n);rng_laser=np.random.default_rng(l)
    learned_prior=None
    learned_types=None
    if config.anatomy_engine=="vascular_cvae":
        from .vascular_cvae import sample
        learned_types=sample(config.generator_weights,rng_anatomy,config.grid_size)
    if config.anatomy_engine=="cvae":
        from .learned import sample_vessel_prior
        learned_prior=sample_vessel_prior(config, rng_anatomy)
    if config.anatomy=="material_phantom":
        from .informed_phantom import generate_material_phantom
        tissue=generate_material_phantom(config,rng_anatomy)
    else:
        tissue=generate_tissue(config,rng_anatomy,learned_prior,learned_types)
    jitter=0.
    if config.pulse_energy_jitter_fraction>0:
        jitter=float(rng_laser.normal(scale=config.pulse_energy_jitter_fraction))
        jitter=max(-.9,jitter)
        tissue.initial_pressure_pa*=1+jitter
        tissue.fluence_j_m2*=1+jitter
    # Experimental material-phantom solver-domain enlargement. Keep the material
    # label/p0 pixel scale and truth FOV fixed; only propagate on a padded domain.
    # Not a calibrated acquisition geometry and not a quantitative reconstruction.
    propagation_cfg=config
    padded_source=tissue.initial_pressure_pa
    padded_sound=tissue.sound_speed_m_s
    pad=0
    material_geometry_audit=None
    if config.anatomy=='material_phantom' and config.geometry=='circle':
        if config.material_solver_padding_factor == 2:
            pad=config.grid_size//2
            propagation_cfg=config.model_copy(update={
                'grid_size':2*config.grid_size,'fov_m':2*config.fov_m})
            padded_source=np.pad(tissue.initial_pressure_pa,pad,mode='constant',constant_values=0)
            padded_sound=np.pad(tissue.sound_speed_m_s,pad,mode='edge')
        material_geometry_audit=_material_geometry_diagnostic(
            padded_source,config,propagation_cfg,pad)
        if config.material_solver_padding_factor==2 and not material_geometry_audit['geometry_gate_pass']:
            raise ValueError('Padded domain does not meet material/receiver/sponge preflight geometry contract')
    waveform,dt,positions=simulate_pressure(padded_source,padded_sound,propagation_cfg)
    clean=apply_detector_filter(waveform,dt,propagation_cfg)
    # Channel dropout is an ACQUISITION DEFECT, not an injected noise process.
    dropped=np.zeros(config.transducers,dtype=bool)
    if config.channel_dropout_probability>0:
        dropped = rng_laser.random(config.transducers)<config.channel_dropout_probability
        clean[dropped]=0.
    raw,components=add_noise(clean,config,rng_noise)
    inversion_losses=None
    from .reconstruction_engine import Acquisition, reconstruct as unified_reconstruct
    if not config.reconstruct:
        rec=None
    else:
        acq=Acquisition(raw,dt,positions,propagation_cfg.fov_m,propagation_cfg.grid_size,
                        1480.+config.sound_speed_offset_m_s,padded_sound,config.wavelength_nm)
        rec=unified_reconstruct(acq,config.reconstruction_method,
                                iterations=config.reconstruction_iterations,
                                reg=config.reconstruction_l2,time_decimate=1,
                                weights=config.reconstruction_weights)
        if pad:
            rec=rec[pad:pad+config.grid_size,pad:pad+config.grid_size].copy()
    inversion_losses=None
    snr=measured_added_snr_db(clean,raw)
    info={"material_geometry_preflight":material_geometry_audit,
          "material_solver_padding_factor":config.material_solver_padding_factor,
          "propagation_grid_size":propagation_cfg.grid_size,
          "propagation_fov_m":propagation_cfg.fov_m,
          "reconstruction_cropped_to_truth_grid":bool(pad),
          "material_provenance":getattr(tissue,"provenance",None),
          "reconstruction_method":config.reconstruction_method,
          "inversion_losses":inversion_losses,
          "vascular_pattern":config.vascular_pattern if config.anatomy=="vascular" else None,
          "vascular_type_pixel_counts":{name:int((tissue.vessel_type_mask==typ).sum()) for typ,name in [(1,"artery"),(2,"vein"),(3,"capillary"),(4,"untyped")]},
          "sample_index":index,"sampling_rate_hz":1/dt,"dt_s":dt,"recording_duration_s":dt*config.time_steps,
          "computed_noise_snr_db":None if snr is None or not np.isfinite(snr) else float(snr),
          "noise_added":bool(components),"noise_components":list(components),
          "laser_energy_jitter_draw":jitter,"dropped_channels":np.flatnonzero(dropped).tolist(),
          "experimental_calibration":False,"solver":"approximate heterogeneous 2D wave equation, sponge boundaries",
          "optical_model":config.illumination_model+"; NOT optical Monte Carlo",
          "reconstruction":"PA-FORGE unified uncalibrated 2D reconstruction: " + config.reconstruction_method if rec is not None else None}
    return SampleResult(tissue,clean,raw,components,dt,positions,rec,info)


def run_batch(config: SimConfig,folder,callback=None):
    # Refuse collisions before any simulation or HDF5 writes, even for a partial
    # previous run. A new output directory is required for a new scientific run.
    from pathlib import Path
    folder=Path(folder)
    target_names=[folder/f"{config.anatomy}_{i:05d}" for i in range(config.samples)]
    existing=[str(p) for p in target_names if p.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing sample folders; choose a new output directory: "
                              +", ".join(existing[:3]))
    paths=[]
    for index in range(config.samples):
        result=simulate(config,index)
        p=export_result(folder,index,config,result.tissue,result.clean_signals,result.raw_signals,
                        result.noise_components,result.dt_s,result.detectors_m,result.reconstruction,result.info)
        paths.append(p)
        if callback:callback(index+1,config.samples,p)
    return paths
