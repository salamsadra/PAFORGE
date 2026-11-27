"""Stochastic 2D anatomy and optical/acoustic property map construction.

IMPORTANT: these anatomies and spectral mappings are ILLUSTRATIVE phantoms;
not learned from patients, not a biological ground-truth atlas.
"""
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from .vascular import make_vascular_network

# 0 coupling, 1 soft, 2 skin, 3 muscle/gland, 4 bone,
# 5 generic vessel/artery, 6 vein, 7 capillary
# Optical values below are demonstration-scale tissue model parameters, NOT
# wavelength-resolved measurements from a particular device or donor.
PROPERTIES = {
    0: (0.3, 0.01, 1480.0, 0.02, 0.0),
    1: (3.0, 600., 1510.0, 0.11, 0.0),
    2: (25., 1100., 1610.0, 0.13, 0.0),
    3: (8.0, 900., 1570.0, 0.12, 0.0),
    4: (40., 1300., 2300.0, 0.14, 0.0),
    5: (120., 700., 1580.0, 0.13, 0.0),
    6: (100., 720., 1580.0, 0.13, 0.0),
    7: (78., 800., 1540.0, 0.13, 0.0),
}

@dataclass
class Tissue:
    labels: np.ndarray
    absorption_m_inv: np.ndarray
    scattering_reduced_m_inv: np.ndarray
    sound_speed_m_s: np.ndarray
    gruneisen: np.ndarray
    mask: np.ndarray
    vessel_mask: np.ndarray
    vessel_type_mask: np.ndarray  # 0 none, 1 artery, 2 vein, 3 capillary, 4 unspecified
    fluence_j_m2: np.ndarray
    initial_pressure_pa: np.ndarray


def _ellipse(x, y, cx, cy, rx, ry, angle=0.):
    c, s = np.cos(angle), np.sin(angle)
    u, v = c*(x-cx)+s*(y-cy), -s*(x-cx)+c*(y-cy)
    return (u/rx)**2+(v/ry)**2 <= 1


def _random_vessel_path(y, x, rng, width=0.017):
    """Curving tubes for illustrative vascular geometry."""
    rows = y.shape[0]
    pos = rng.uniform(-.26, .26)
    slope = rng.uniform(-.24,.24)
    wav = rng.uniform(0.018,0.07)
    center = pos + slope*y + wav*np.sin(12*y+rng.uniform(0,2*np.pi))
    return np.abs(x-center) < width


def generate_tissue(config, rng, vessel_prior=None, vascular_type_prior=None):
    n=config.grid_size
    yy, xx = np.mgrid[0:n,0:n].astype(np.float64)
    x=(xx-(n-1)/2)/(n/2)
    y=(yy-(n-1)/2)/(n/2)
    labels=np.zeros((n,n), dtype=np.uint8)
    vascular_types=np.zeros((n,n),dtype=np.uint8)
    if config.anatomy == "forearm":
        outer = _ellipse(x,y,0,0,.80,.72, angle=-.08)
        inner = _ellipse(x,y,0,0,.76,.68, angle=-.08)
        labels[outer]=2;labels[inner]=1
        muscle=_ellipse(x,y,.06,-.02,.57,.49,angle=.12)
        labels[muscle]=3
        for cx,cy,r in [(-.26,.06,.13),(.29,.17,.095)]:
            labels[_ellipse(x,y,cx,cy,r,r) & inner]=4
        vessels = (_ellipse(x,y,-.49,-.29,.045,.045)| _ellipse(x,y,.48,-.22,.055,.04))
        vessels |= _ellipse(x,y,.07,-.52,.035,.03)
    elif config.anatomy == "skin":
        outer = (np.abs(x)<.86)&(y>-.54)&(y<.48)
        epidermis=outer&(y<-.47)
        dermis=outer&(y>=-.47)&(y<-.16)
        labels[outer]=1; labels[epidermis]=2; labels[dermis]=3
        vessels=np.zeros((n,n), bool)
        for _ in range(8):
            xc=rng.uniform(-.8,.8)
            yc=rng.uniform(-.13,.30)
            vessels|=_ellipse(x,y,xc,yc,rng.uniform(.014,.035),rng.uniform(.014,.03))
        for _ in range(2):
            vessels|=_random_vessel_path(y,x,rng,.013)&(y>-.12)&(y<.45)
    elif config.anatomy == "breast":
        outer=_ellipse(x,y,0,0,.80,.78)
        inner=_ellipse(x,y,0,0,.77,.75)
        labels[outer]=2; labels[inner]=1
        sm=gaussian_filter(rng.normal(size=(n,n)),sigma=max(2,n/14))
        if np.any(inner):
            labels[inner&(sm>np.quantile(sm[inner],.63))]=3
        vessels=np.zeros_like(inner)
        for _ in range(5):
            angle=rng.uniform(0,2*np.pi)
            r=rng.uniform(.20,.60)
            vessels|=_ellipse(x,y,r*np.cos(angle),r*np.sin(angle),.021,.055,angle)
        # optional lesion-shaped region with increased absorption (NOT real tumour model)
        labels[_ellipse(x,y,.18,.12,.12,.10)&inner]=3
    elif config.anatomy == "carotid":
        outer=_ellipse(x,y,0,0,.82,.78)
        inner=_ellipse(x,y,0,0,.78,.74)
        labels[outer]=2; labels[inner]=1
        labels[_ellipse(x,y,.23,-.05,.35,.25)&inner]=3
        labels[_ellipse(x,y,-.26,.28,.12,.10)&inner]=4
        vessels=_ellipse(x,y,.28,-.20,.12,.13)|_ellipse(x,y,-.14,-.14,.10,.14)
    elif config.anatomy == "vascular":
        outer=_ellipse(x,y,0,0,.81,.81)
        inner=_ellipse(x,y,0,0,.77,.77)
        labels[outer]=2; labels[inner]=1
        # Optically/acoustically heterogeneous soft background, not a human atlas.
        texture=gaussian_filter(rng.normal(size=(n,n)),max(2,n/22))
        labels[inner & (texture>np.quantile(texture[inner],.88))]=3
        if vascular_type_prior is None:
            vascular_types=make_vascular_network(n,config.fov_m,rng,
                config.vascular_branch_levels,config.vascular_pattern,config.vascular_density)
        else:
            if vascular_type_prior.shape!=(n,n) or np.any(vascular_type_prior>3):
                raise ValueError('Learned vascular segmentation must contain class codes 0..3')
            vascular_types=vascular_type_prior.copy()
        vascular_types[~inner]=0
        vessels=vascular_types>0
    else:
        raise ValueError(config.anatomy)
    vessels &= (labels!=0)&(labels!=4)
    if vessel_prior is not None:
        if vessel_prior.shape!=(n,n): raise ValueError("Learned vessel prior shape mismatch")
        # Restrict model predictions to existing soft tissue.
        valid=(labels!=0)&(labels!=4)&(labels!=2)
        # Calibrate sparse-model output using only anatomically valid tissue.
        # This is a demonstration sampling policy, not a validated vessel prior.
        cutoff=np.quantile(vessel_prior[valid],.985) if np.any(valid) else np.inf
        vessels=(vessel_prior>=cutoff)&valid
        vascular_types[:]=0  # learned mask does not carry arterial/venous identity
    if config.anatomy == "vascular" and vessel_prior is None:
        labels[vessels & (vascular_types==1)]=5
        labels[vessels & (vascular_types==2)]=6
        labels[vessels & (vascular_types==3)]=7
    else:
        labels[vessels]=5
        vascular_types[vessels]=4  # untyped vessels: don't claim artery/vein
    mua=np.zeros((n,n)); mus=np.zeros_like(mua); sound=np.zeros_like(mua); gamma=np.zeros_like(mua)
    for lab,(a,s,c,g,_) in PROPERTIES.items():
        m=labels==lab;mua[m]=a;mus[m]=s;sound[m]=c;gamma[m]=g
    # Demonstration-only spectral scaling, not calibrated Hb/HbO2 curves.
    spectrum=1 + 0.25*np.sin((config.wavelength_nm-660)/640*np.pi)
    mua[np.isin(labels,(5,6,7))]*=spectrum
    mua[labels==2]*=1 + .12*(config.wavelength_nm-800)/500
    mus*=config.scattering_scale
    background=np.zeros_like(sound)
    background[labels!=0] = (sound[labels!=0]-1480.)*config.sound_speed_variation_fraction
    sound=1480.+background+config.sound_speed_offset_m_s
    # Analytical attenuation-length proxy: NOT diffusion equation / MC transport.
    # Distance of each tissue pixel to illuminated outer boundary.
    body = labels!=0
    depth=distance_transform_edt(body)*config.fov_m/n
    mu_eff=np.sqrt(3*mua*np.maximum(mua+mus,1e-6))
    if config.illumination_model == "uniform_idealized":
        # Only a mathematical idealization of homogenous optical excitation,
        # not a measurement-equivalent fluence distribution in human tissue.
        fluence=np.full_like(mua,config.incident_fluence_j_m2)
    else:
        fluence=config.incident_fluence_j_m2*np.exp(-mu_eff*depth)
    p0=gamma*mua*fluence
    p0[~body]=0
    return Tissue(labels,mua,mus,sound,gamma,body,vessels,vascular_types,fluence,p0)
