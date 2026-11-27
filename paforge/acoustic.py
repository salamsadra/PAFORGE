"""2D acoustic wave equation with zero initial velocity and absorbing sponge.

Research PROTOTYPE solver; not k-Wave, no acoustic attenuation, no PML,
no transducer angular response or pressure calibration against physical data.
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt


def sensor_positions(config):
    n=config.grid_size; middle=(n-1)/2
    if config.geometry=="circle":
        ang=np.linspace(0,2*np.pi,config.transducers,endpoint=False)
        x=middle+.418*n*np.cos(ang); y=middle+.418*n*np.sin(ang)
    elif config.geometry=="arc":
        ang=np.linspace(np.pi*.10,np.pi*.90,config.transducers)
        x=middle+.418*n*np.cos(ang); y=middle-.418*n*np.sin(ang)
    else:
        x=np.linspace(n*.10,n*.90,config.transducers)
        y=np.full_like(x,n*.105)
    return np.column_stack([y,x])


def _acceleration(p, c, rho, dx):
    """rho*c^2*div((1/rho)*grad(p)); no-flux edges."""
    invrho=1./rho
    fx=.5*(invrho[:,1:]+invrho[:,:-1])*(p[:,1:]-p[:,:-1])
    fy=.5*(invrho[1:,:]+invrho[:-1,:])*(p[1:,:]-p[:-1,:])
    div=np.zeros_like(p)
    div[:,1:-1]+=(fx[:,1:]-fx[:,:-1])
    div[1:-1,:]+=(fy[1:,:]-fy[:-1,:])
    div[:,0]+=fx[:,0]; div[:,-1]-=fx[:,-1]
    div[0,:]+=fy[0,:]; div[-1,:]-=fy[-1,:]
    return rho*c*c*div/(dx*dx)


def _sponge(n, pixels=9, strength=.055):
    ii=np.arange(n)
    distance=np.minimum(ii,n-1-ii)
    edge=np.maximum(0,(pixels-distance)/pixels)
    damping=strength*(edge[:,None]**2+edge[None,:]**2)
    return np.exp(-damping)


def _sample_bilinear(p, positions):
    y,x=positions[:,0],positions[:,1]
    y0=np.floor(y).astype(int);x0=np.floor(x).astype(int)
    wy=y-y0;wx=x-x0
    return ((1-wx)*(1-wy)*p[y0,x0]+wx*(1-wy)*p[y0,x0+1]
            +(1-wx)*wy*p[y0+1,x0]+wx*wy*p[y0+1,x0+1])


def simulate_pressure(p0, sound, config, rho=None):
    if p0.shape != (config.grid_size,config.grid_size): raise ValueError("p0 shape")
    if not np.isfinite(p0).all() or not np.isfinite(sound).all(): raise ValueError("Nonfinite physical map")
    if (sound<=0).any():raise ValueError("Positive sound speed required")
    n=config.grid_size;dx=config.fov_m/n
    dt=config.cfl*dx/(np.sqrt(2)*sound.max())
    if rho is None:rho=np.full_like(sound,1000.)
    damp=_sponge(n)
    positions=sensor_positions(config)
    prev=p0+0.5*dt**2*_acceleration(p0,sound,rho,dx)
    cur=p0.copy()
    traces=np.empty((config.transducers,config.time_steps), dtype=np.float32)
    for t in range(config.time_steps):
        traces[:,t]=_sample_bilinear(cur,positions)
        nxt=(2*cur-prev+dt**2*_acceleration(cur,sound,rho,dx))*damp
        prev,cur=cur,nxt
        if not np.isfinite(cur).all():raise FloatingPointError("Acoustic solver diverged")
    positions_m=np.column_stack([(positions[:,1]-(n-1)/2)*dx,(positions[:,0]-(n-1)/2)*dx, np.zeros(len(positions))])
    return traces,dt,positions_m


def apply_detector_filter(traces,dt,config):
    if not config.detector_filter:return traces.copy()
    fs=1/dt;nyq=fs/2
    low=(config.center_frequency_hz-config.bandwidth_hz/2)/nyq
    high=(config.center_frequency_hz+config.bandwidth_hz/2)/nyq
    if high>=.999 or low<=0:
        raise ValueError(f"Detector passband invalid for simulated sampling rate {fs:.3g} Hz")
    sos=butter(3,[low,high],btype="bandpass",output="sos")
    return sosfiltfilt(sos,traces,axis=1).astype(np.float32)


def reconstruct_das(traces,dt,positions_m,config):
    """Unweighted travel-time delay-and-sum (qualitative reconstruction ONLY).

    Uses one homogeneous effective c and interpolation; is not exact 2D PACT
    inversion, NOT a quantitative initial pressure estimate.
    """
    n=config.grid_size;dx=config.fov_m/n
    yy,xx=np.mgrid[:n,:n]
    x=(xx-(n-1)/2)*dx;y=(yy-(n-1)/2)*dx
    c=1480+config.sound_speed_offset_m_s
    t_axis=np.arange(traces.shape[1],dtype=float)
    out=np.zeros((n,n),dtype=np.float64)
    for i,pos in enumerate(positions_m):
        r=np.sqrt((x-pos[0])**2+(y-pos[1])**2)
        sample=r/(c*dt)
        out+=np.interp(sample.ravel(),t_axis,traces[i],left=0,right=0).reshape(n,n)
    out/=len(positions_m)
    return out.astype(np.float32)


def reconstruct_ubp(traces,dt,positions_m,config,smoothing_samples=1.5):
    """Heuristic derivative-weighted universal-backprojection-like baseline.

    p - t*dp/dt at travel-time (acoustic PACT convention), averaged over
    detectors. This is NOT a validated universal backprojection formula for
    all acquisition geometries; finite aperture and angular weighting omitted.
    """
    from scipy.ndimage import gaussian_filter1d
    n=config.grid_size;dx=config.fov_m/n
    yy,xx=np.mgrid[:n,:n]
    x=(xx-(n-1)/2)*dx;y=(yy-(n-1)/2)*dx
    c=1480.+config.sound_speed_offset_m_s
    t_idx=np.arange(traces.shape[1],dtype=float)
    smooth=gaussian_filter1d(np.asarray(traces,dtype=float),smoothing_samples,axis=1)
    derivative=np.gradient(smooth,dt,axis=1)
    out=np.zeros((n,n),dtype=np.float64)
    for i,pos in enumerate(positions_m):
        t=np.hypot(x-pos[0],y-pos[1])/c
        sample=t/dt
        p=np.interp(sample.ravel(),t_idx,smooth[i],left=0,right=0).reshape(n,n)
        dp=np.interp(sample.ravel(),t_idx,derivative[i],left=0,right=0).reshape(n,n)
        out += p-t*dp
    return (out/len(positions_m)).astype(np.float32)
