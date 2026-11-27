"""Experimental-phantom-informed 2D physics inputs (not real new measurements)."""
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt
from .anatomy import Tissue
from .experimental_learning import sample_optics
from .material_contracts import file_sha256


def generate_material_phantom(config,rng):
    mask_provenance=None
    if config.anatomy_engine=='material_cvae':
        from .material_cvae import sample
        labels=sample(config.generator_weights,rng,config.grid_size,config.material_temperature,config.material_composition)
        source='learned categorical CVAE of training-only phantom material maps'
    elif config.anatomy_engine=='empirical_material':
        from .material_cvae import sample_empirical
        labels,mask_provenance=sample_empirical(config.generator_weights,config.optical_profile,rng,
                                config.grid_size,config.material_deform_px,return_provenance=True)
        source='train-only empirical material mask with smooth stochastic deformation'
    else:raise ValueError('Unknown material generator')
    mua,musp,selected=sample_optics(config.optical_profile,labels,config.wavelength_nm,
                                   rng,config.scattering_scale)
    body=labels!=0
    gamma=np.zeros_like(mua);gamma[body]=.12   # explicit uncalibrated assumption
    sound=np.full_like(mua,1480.+config.sound_speed_offset_m_s)
    # Optical maps do not identify acoustic parameters. Constant sound speed is
    # an EXPLICIT user-assumed acoustic medium; heterogeneity is not fitted.
    depth=distance_transform_edt(body)*config.fov_m/config.grid_size
    if config.illumination_model=='uniform_idealized':
        fluence=np.full_like(mua,config.incident_fluence_j_m2)
    else:
        mu_eff=np.sqrt(3*mua*np.maximum(mua+musp,1e-9))
        fluence=config.incident_fluence_j_m2*np.exp(-mu_eff*depth)
    p0=gamma*mua*fluence;p0[~body]=0
    zeros=np.zeros_like(labels,dtype=np.uint8)
    tissue=Tissue(labels,mua,musp,sound,gamma,body,zeros.astype(bool),zeros,fluence,p0)
    tissue.provenance={'material_source':source,'sampled_optical_training_groups':selected,
        'optical_profile':str(config.optical_profile),'generator_weights':str(config.generator_weights),
        'optical_profile_sha256':file_sha256(config.optical_profile),
        'generator_input_sha256':file_sha256(config.generator_weights),
        'empirical_source':mask_provenance,
        'sound_speed_assumed_constant_m_s':float(sound[0,0]),
        'gruneisen_assumed':.12,'label_identity':'phantom material classes, NOT blood vessel classes',
        'data_claim':'synthetic, conditioned on group-held-out-train OpenQPAIData material/optical maps'}
    return tissue
