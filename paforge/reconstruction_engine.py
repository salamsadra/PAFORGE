"""PA-FORGE v0.9 shared 2D reconstruction registry.

Inputs for synthetic and measured acquisitions are *identical*. Method labels
carry a physics/capability disclaimer: some are exploratory beamforming variants
or inverse solutions of a simplified ray/time-of-flight surrogate, not
calibrated reconstructions or certified ports of PATBox/PATATO.
"""
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, sosfiltfilt


@dataclass(frozen=True)
class MethodSpec:
    name: str
    category: str
    description: str
    model: str
    requires_weights: bool = False
    requires_fdtd: bool = False


_SPECS = [
    MethodSpec('das','beamforming','Unweighted delay and sum','straight-ray TOF'),
    MethodSpec('ubp','analytic','Derivative-weighted backprojection-like baseline','homogeneous TOF'),
    MethodSpec('filtered_ubp','analytic','Bandpass + derivative UBP-like baseline','homogeneous TOF'),
    MethodSpec('fbp','analytic','Ram-Lak filtered delay and sum (exploratory, geometry-dependent)','homogeneous TOF'),
    MethodSpec('cf_das','beamforming','Coherence-factor weighted DAS','straight-ray TOF'),
    MethodSpec('scf_das','beamforming','Sign coherence-factor weighted DAS','straight-ray TOF'),
    MethodSpec('dmas','beamforming','Signed square-root delay-multiply-and-sum','straight-ray TOF'),
    MethodSpec('ds_dmas','beamforming','Double-stage DMAS (nonlinear qualitative)','straight-ray TOF'),
    MethodSpec('vdas','beamforming','Variance-weighted DAS (exploratory)','straight-ray TOF'),
    MethodSpec('mv_das','beamforming','Globally covariance-weighted MV DAS (NOT pixelwise Capon)','straight-ray TOF'),
    MethodSpec('cfmv_das','beamforming','Coherence + globally covariance-weighted DAS','straight-ray TOF'),
    MethodSpec('dmas_ubp','beamforming','DMAS on derivative-weighted arrivals','homogeneous TOF'),
    MethodSpec('vdas_ubp','beamforming','Variance-weighted derivative arrivals','homogeneous TOF'),
    MethodSpec('adjoint_fdtd','wave','Discrete adjoint of the PA-FORGE 2D FDTD forward operator','homogeneous/heterogeneous 2D FDTD',requires_fdtd=True),
    MethodSpec('fista','wave','Nonnegative L2-regularized FISTA of 2D FDTD','homogeneous/heterogeneous 2D FDTD',requires_fdtd=True),
    MethodSpec('time_reversal','wave','Reverse-time boundary-source injection; exploratory not exact TR','2D FDTD',requires_fdtd=True),
    MethodSpec('iterative_time_reversal','wave','Residual-driven repeated reverse-time source injection','2D FDTD',requires_fdtd=True),
    MethodSpec('cg_tikhonov','iterative','Conjugate gradient on regularized normal equations','approximate travel-time operator'),
    MethodSpec('fista_ray','iterative','Nonnegative FISTA with L2 ridge penalty','approximate travel-time operator'),
    MethodSpec('landweber','iterative','Landweber projected gradient descent','approximate travel-time operator'),
    MethodSpec('sirt','iterative','Row/column normalized simultaneous iterative reconstruction','approximate travel-time operator'),
    MethodSpec('iterative_das','iterative','Residual-feedback DAS with approximate line search','approximate travel-time operator'),
    MethodSpec('lsqr','iterative','LSQR on matrix-free travel-time operator','approximate travel-time operator'),
    MethodSpec('tv_fista','iterative','FISTA with TV prox (Chambolle) and nonnegativity','approximate travel-time operator'),
    MethodSpec('admm_tv','iterative','Split ADMM using gradient shrinkage and CG','approximate travel-time operator'),
    MethodSpec('learned_unet','learned','Trained U-Net refinement of baseline backprojection','supervised synthetic paired data',requires_weights=True),
    MethodSpec('learned_rescnn','learned','Trained residual CNN refinement of baseline backprojection','supervised synthetic paired data',requires_weights=True),
    MethodSpec('pnp_fista','learned','Experimental plug-and-play denoiser after ray-model FISTA','ray model + trained denoiser',requires_weights=True),
    MethodSpec('deep_image_prior','learned','Per-acquisition self-supervised CNN fit to detector traces','ray-model signal consistency; no pretraining'),
    MethodSpec('patato_bp','external','Official PATATO Python reference backprojection, optional installation','PATATO external backend'),
    MethodSpec('patato_model','external','Official PATATO model-based reconstruction, optional installation','PATATO external backend'),
    MethodSpec('patato_opencl_bp','external','Official PATATO OpenCL backprojection, requires OpenCL runtime','PATATO external backend'),
    MethodSpec('patato_slow_bp','external','Official PATATO Numpy slow backprojection, optional installation','PATATO external backend'),
]
SPECS = {m.name:m for m in _SPECS}
METHODS=tuple(SPECS)
ALIASES={'adjoint':'adjoint_fdtd','iterative':'fista'}


def method_catalog():
    return [s.__dict__.copy() for s in _SPECS]


@dataclass
class Acquisition:
    traces: np.ndarray
    dt_s: float
    positions_m: np.ndarray
    fov_m: float
    grid_size: int
    sound_speed_m_s: float = 1500.
    sound_map_m_s: np.ndarray | None = None
    wavelength_nm: float | None = None

    def __post_init__(self):
        self.traces=np.asarray(self.traces,dtype=np.float64)
        self.positions_m=np.asarray(self.positions_m,dtype=np.float64)
        if self.traces.ndim!=2 or self.traces.shape[0]<2 or self.traces.shape[1]<16:
            raise ValueError('Expected signals [>=2 detectors, >=16 time samples]')
        if not np.isfinite(self.traces).all():raise ValueError('Nonfinite traces')
        if self.positions_m.shape!=(len(self.traces),3) or not np.isfinite(self.positions_m).all():
            raise ValueError('Measured/simulated detector locations [detectors,3] in metres are required')
        if np.ptp(self.positions_m[:,2])>0.002:
            raise ValueError('3D detector coordinates cannot be silently projected into a 2D reconstruction')
        if not (np.isfinite(self.dt_s) and self.dt_s>0 and 0.005<=self.fov_m<=0.3):
            raise ValueError('Positive dt and reconstruction FOV from 5 to 300 mm required')
        if not (32<=self.grid_size<=256 and 1000<=self.sound_speed_m_s<=2200):
            raise ValueError('Grid 32..256, sound speed 1000..2200 m/s required')
        if self.wavelength_nm is not None and (not np.isfinite(self.wavelength_nm) or not 100<=self.wavelength_nm<=2500):
            raise ValueError('Wavelength metadata must be documented in nm')
        if self.sound_map_m_s is not None:
            self.sound_map_m_s=np.asarray(self.sound_map_m_s,dtype=float)
            if self.sound_map_m_s.shape!=(self.grid_size,)*2 or np.min(self.sound_map_m_s)<=0:
                raise ValueError('Sound map must match the reconstruction grid')


def _coordinates(acq):
    n=acq.grid_size; d=acq.fov_m/n
    y,x=np.mgrid[:n,:n];return (x-(n-1)/2)*d,(y-(n-1)/2)*d


def _arrivals(acq, data=None, derivative=False):
    x,y=_coordinates(acq)
    b=acq.traces if data is None else np.asarray(data,dtype=float)
    if derivative:
        smooth=gaussian_filter1d(b,1.5,axis=1)
        grad=np.gradient(smooth,acq.dt_s,axis=1)
        signals=smooth
    else:signals=b
    sample_axis=np.arange(b.shape[1],dtype=float)
    for i,xyz in enumerate(acq.positions_m):
        t=np.hypot(x-xyz[0],y-xyz[1])/acq.sound_speed_m_s
        offset=t/acq.dt_s
        u=np.interp(offset.ravel(),sample_axis,signals[i],left=0,right=0).reshape(t.shape)
        if derivative:
            v=np.interp(offset.ravel(),sample_axis,grad[i],left=0,right=0).reshape(t.shape)
            u=u-t*v
        yield u


def _beamform(acq,method,low_hz=0.3e6,high_hz=6e6):
    b=acq.traces
    kind=method
    if method=='filtered_ubp':
        fs=1/acq.dt_s
        if not(0<low_hz<high_hz<fs/2):raise ValueError('Invalid bandpass cutoff vs Nyquist')
        if b.shape[1]<48:raise ValueError('Need at least 48 samples for zero-phase bandpass')
        filt=sosfiltfilt(butter(3,[low_hz,high_hz],fs=fs,btype='bandpass',output='sos'),b,axis=1)
        acq=Acquisition(filt,acq.dt_s,acq.positions_m,acq.fov_m,acq.grid_size,acq.sound_speed_m_s)
        kind='ubp'
    if method=='fbp':
        freqs=np.fft.rfftfreq(b.shape[1],acq.dt_s)
        # Ram-Lak temporal ramp; normalized to avoid numeric range explosion.
        weights=freqs/max(float(freqs.max()),1)
        filt=np.fft.irfft(np.fft.rfft(b,axis=1)*weights[None,:],n=b.shape[1],axis=1)
        acq=Acquisition(filt,acq.dt_s,acq.positions_m,acq.fov_m,acq.grid_size,acq.sound_speed_m_s)
    deriv=kind in ('ubp','dmas_ubp','vdas_ubp')
    # Beamforming uses full detector participation, not invented sensor positions.
    aligned=np.stack(list(_arrivals(acq,derivative=deriv)),axis=0)
    n=aligned.shape[0];eps=1e-15+1e-8*np.max(aligned*aligned)
    sm=np.sum(aligned,axis=0)
    mean=sm/n
    if kind in ('das','ubp','filtered_ubp','fbp'):return mean.astype('float32')
    energy=np.sum(aligned**2,axis=0)
    cf=np.minimum(1.,np.maximum(0.,sm*sm/(n*energy+eps)))
    if kind=='cf_das':out=mean*cf
    elif kind=='scf_das':
        sign_mean=np.mean(np.sign(aligned),axis=0)
        out=mean*(1.-np.sqrt(np.maximum(0.,1.-sign_mean**2)))
    elif kind in ('dmas','dmas_ubp','ds_dmas'):
        signed=np.sign(aligned)*np.sqrt(np.abs(aligned))
        dmas=(np.sum(signed,axis=0)**2-np.sum(np.abs(aligned),axis=0))/(max(1,n*(n-1)))
        if kind=='ds_dmas':
            # Second-stage sign-preserving product with first-stage coherent energy.
            out=np.sign(dmas)*np.sqrt(np.abs(dmas))*np.sign(mean)*np.sqrt(np.abs(mean))
        else:out=dmas
    elif kind in ('vdas','vdas_ubp'):
        # Detector-to-detector variance is a heuristic coherence reliability map.
        out=mean/(1.+np.var(aligned,axis=0)/(mean*mean+eps))
    elif kind in ('mv_das','cfmv_das'):
        # Global (rather than pixelwise) Capon weights: regularized trace covariance.
        demeaned=b-np.mean(b,axis=1,keepdims=True)
        cov=demeaned@demeaned.T/b.shape[1]
        cov+=np.eye(n)*(max(np.trace(cov)/n,eps)*.1)
        w=np.linalg.solve(cov,np.ones(n))
        w/=np.sum(w) if abs(np.sum(w))>eps else 1.
        out=np.tensordot(w,aligned,axes=(0,0))
        if kind=='cfmv_das':out*=cf
    else:raise ValueError('Unknown beamforming method')
    return np.asarray(out,dtype=np.float32)


class RayOperator:
    """Matrix-free TOF interpolation surrogate, NOT the PA wave equation.

    Sampling weights are the exact transpose in forward and adjoint, enabling
    reproducible numerical linear algebra. Does not model detector frequency,
    aperture, attenuation, or 3D propagation.
    """
    def __init__(self,acq):
        self.acq=acq
        self.shape=(len(acq.traces)*acq.traces.shape[1], acq.grid_size**2)
        x,y=_coordinates(acq)
        time=np.hypot(x[None,:,:]-acq.positions_m[:,0,None,None],
                      y[None,:,:]-acq.positions_m[:,1,None,None])/(acq.sound_speed_m_s*acq.dt_s)
        t=np.floor(time).astype('int64')
        w=time-t
        nt=acq.traces.shape[1]
        valid=(t>=0)&(t+1<nt)
        self.idx=t.reshape(len(acq.traces),-1)
        self.w=w.reshape(len(acq.traces),-1)
        self.valid=valid.reshape(len(acq.traces),-1)
        self.nt=nt;self.ns=len(acq.traces);self.n=acq.grid_size
        self.scale=1./max(1,self.n)  # Discrete ray-sum scale, not physical pressure.

    def forward(self,img):
        q=np.asarray(img).ravel()
        if q.size!=self.n**2:raise ValueError('Wrong image size')
        out=np.zeros((self.ns,self.nt))
        for k in range(self.ns):
            good=self.valid[k];ii=self.idx[k,good];v=q[good]*self.scale
            np.add.at(out[k],ii,v*(1-self.w[k,good]))
            np.add.at(out[k],ii+1,v*self.w[k,good])
        return out

    def adjoint(self,b):
        b=np.asarray(b,dtype=float).reshape(self.ns,self.nt)
        out=np.zeros(self.n**2)
        for k in range(self.ns):
            good=self.valid[k];ii=self.idx[k,good]
            out[good]+=self.scale*((1-self.w[k,good])*b[k,ii]+self.w[k,good]*b[k,ii+1])
        return out.reshape(self.n,self.n)

    def norm_squared(self):
        x=np.ones((self.n,self.n));x/=np.linalg.norm(x)
        for _ in range(8):
            y=self.adjoint(self.forward(x));m=np.linalg.norm(y)
            if m<1e-16:raise ValueError('Source signals do not overlap reconstruction grid and time window')
            x=y/m
        return max(float(np.sum(x*self.adjoint(self.forward(x)))),1e-16)


def _grad(x):
    return np.stack((np.diff(x,axis=0,append=x[-1:]),np.diff(x,axis=1,append=x[:,-1:])))


def _div(g):
    out=np.zeros_like(g[0]);out[:-1]-=g[0,:-1];out[1:]+=g[0,:-1]
    out[:,:-1]-=g[1,:,:-1];out[:,1:]+=g[1,:,:-1]
    return out


def _tv_prox(image,weight,n_iter=35):
    """Chambolle projection prox for isotropic total variation, then positivity."""
    if weight<=0:return np.maximum(image,0)
    p=np.zeros((2,)+image.shape)
    tau=.24
    for _ in range(n_iter):
        d=_div(p)-image/weight
        g=_grad(d)
        q=p+tau*g
        p=q/np.maximum(1.,np.linalg.norm(q,axis=0,keepdims=True))
    return np.maximum(image-weight*_div(p),0)


def _ray_reconstruct(acq,method,iterations=10,reg=.001,weights=None):
    from scipy.sparse.linalg import LinearOperator,lsqr,cg
    op=RayOperator(acq); rhs=op.adjoint(acq.traces)
    lam=reg*op.norm_squared()
    L=op.norm_squared()+lam
    if method=='lsqr':
        A=LinearOperator(op.shape,matvec=lambda z:op.forward(z.reshape(op.n,op.n)).ravel(),
                         rmatvec=lambda z:op.adjoint(z.reshape(op.ns,op.nt)).ravel(),dtype=float)
        return lsqr(A,acq.traces.ravel(),damp=np.sqrt(lam),iter_lim=iterations,atol=1e-7,btol=1e-7)[0].reshape(op.n,op.n).astype('float32')
    if method=='cg_tikhonov':
        A=LinearOperator((op.n**2,op.n**2),matvec=lambda z:(op.adjoint(op.forward(z.reshape(op.n,op.n)))+lam*z.reshape(op.n,op.n)).ravel(),dtype=float)
        out,info=cg(A,rhs.ravel(),rtol=1e-5,maxiter=iterations)
        if info<0:raise RuntimeError('CG numerical breakdown')
        return np.maximum(out.reshape(op.n,op.n),0).astype('float32')
    x=np.zeros_like(rhs);z=x.copy();t=1.
    if method=='admm_tv':
        rho=max(.02*L,1e-14)
        u=np.zeros_like(_grad(x));d=u.copy()
        def matvec(v):
            v=v.reshape(op.n,op.n)
            return (op.adjoint(op.forward(v))+lam*v+rho*_div(_grad(v))).ravel()
        A=LinearOperator((op.n**2,op.n**2),matvec=matvec,dtype=float)
        for _ in range(iterations):
            b=rhs+rho*_div(d-u)
            sol,info=cg(A,b.ravel(),x0=x.ravel(),rtol=1e-3,maxiter=12)
            if info<0:raise RuntimeError('ADMM inner CG breakdown')
            x=np.maximum(sol.reshape(x.shape),0)
            gx=_grad(x);q=gx+u
            d=q*np.maximum(0,1-(reg*rho)/(rho*np.maximum(np.linalg.norm(q,axis=0,keepdims=True),1e-12)))
            u+=gx-d
        return x.astype('float32')
    if method=='iterative_das':
        x=np.zeros_like(rhs)
        for _ in range(iterations):
            residual=acq.traces-op.forward(x)
            source=Acquisition(residual,acq.dt_s,acq.positions_m,acq.fov_m,acq.grid_size,acq.sound_speed_m_s)
            correction=np.maximum(_beamform(source,'das').astype(float),0)
            ac=op.forward(correction)
            alpha=max(0.,float(np.sum(residual*ac)/(np.sum(ac*ac)+lam*np.sum(correction**2)+1e-16)))
            x=np.maximum(0,x+alpha*correction)
        return x.astype('float32')
    if method=='sirt':
        rows=op.forward(np.ones_like(x));cols=op.adjoint(np.ones_like(acq.traces))
        rweight=1/np.maximum(rows,1e-9);cweight=1/np.maximum(cols,1e-9)
        for _ in range(iterations):
            x=np.maximum(0,x+.75*cweight*op.adjoint((acq.traces-op.forward(x))*rweight)-reg*x)
        return x.astype('float32')
    for _ in range(iterations):
        g=op.adjoint(op.forward(z))-rhs+lam*z
        xn=np.maximum(0,z-g/L)
        if method=='landweber':
            x=xn;z=x
        else:
            if method=='tv_fista':xn=_tv_prox(z-g/L,reg*np.max(np.abs(rhs))/(L+1e-12))
            if method=='pnp_fista':
                from .learned_reconstruction import apply_checkpoint
                xn=apply_checkpoint(xn,weights,expected_family='rescnn')
            tn=(1+np.sqrt(1+4*t*t))/2
            z=xn+(t-1)/tn*(xn-x);x=xn;t=tn
    return x.astype('float32')


def _fdtd(acq,method,iterations=10,reg=.001,decimate_factor=4):
    from .inverse import ForwardOperator
    from scipy.signal import decimate
    if acq.grid_size>128:raise ValueError('FDTD inverse is capped at grid <=128; use analytic/ray methods for larger grids')
    if decimate_factor not in (1,2,4,8,16):raise ValueError('FDTD decimation factor must be 1,2,4,8,16')
    dx=acq.fov_m/acq.grid_size
    positions=np.column_stack((acq.positions_m[:,1]/dx+(acq.grid_size-1)/2,
                               acq.positions_m[:,0]/dx+(acq.grid_size-1)/2))
    if np.min(positions)<1 or np.max(positions)>=acq.grid_size-2:
        req=2*np.max(np.abs(acq.positions_m[:,:2]))*1.15
        raise ValueError(f'FDTD must contain all actual sensors; increase --fov-m to at least {req:.3f} m')
    b=decimate(acq.traces,decimate_factor,axis=1,ftype='fir',zero_phase=True) if decimate_factor>1 else acq.traces
    dt=acq.dt_s*decimate_factor
    sound=acq.sound_map_m_s if acq.sound_map_m_s is not None else np.full((acq.grid_size,)*2,acq.sound_speed_m_s)
    op=ForwardOperator(sound,positions,acq.fov_m,dt,b.shape[1])
    if method=='adjoint_fdtd':return op.adjoint(b).astype('float32')
    if method=='fista':return op.reconstruct(b,iterations=iterations,reg=reg)[0].astype('float32')
    # Reverse-time injection is not equivalent to enforcing pressure boundary
    # time-reversal with a calibrated complete measurement surface.
    from .acoustic import _acceleration,_sponge
    from .inverse import _detector_backproject
    n=op.n;rho=op.rho;damp=_sponge(n)
    def reverse_inject(signals):
        p=np.zeros((n,n));prev=p.copy()
        for j in range(signals.shape[1]-1,-1,-1):
            injection=_detector_backproject(signals[:,j],op.pos,(n,n))
            nxt=(2*p-prev+dt*dt*_acceleration(p,sound,rho,op.dx)+injection)*damp
            prev,p=p,nxt
        return p
    if method=='time_reversal':return reverse_inject(b).astype('float32')
    # Iteratively match *the same* forward waveform with reverse-injection
    # directions and a measured-data residual line search.
    x=np.zeros((n,n))
    for _ in range(iterations):
        residual=b-op.forward(x)
        correction=reverse_inject(residual)
        response=op.forward(correction)
        denominator=np.sum(response*response)+reg*np.sum(correction*correction)+1e-12
        alpha=float(np.sum(response*residual)/denominator)
        x=np.maximum(0,x+max(0,alpha)*correction)
    return x.astype('float32')


def reconstruct(acq:Acquisition,method='ubp',iterations=10,reg=.001,
                time_decimate=4,weights=None,filter_low_hz=.3e6,filter_high_hz=6e6):
    """Shared synthetic + experimental method dispatch; returns finite 2D image."""
    method=ALIASES.get(method,method)
    if method not in SPECS:raise ValueError(f'Unknown method {method}; valid methods: {METHODS}')
    if not (1<=int(iterations)<=100 and 0<=float(reg)<=1):raise ValueError('Iterations 1..100 and reg 0..1 required')
    if SPECS[method].requires_weights and not weights:
        raise ValueError(f'{method} requires a locally trained compatible checkpoint; no untrained network is silently used')
    if method in ('patato_bp','patato_model','patato_opencl_bp','patato_slow_bp'):
        from .patato_native import reconstruct_patato
        img=reconstruct_patato(acq,method)
    elif method in ('adjoint_fdtd','fista','time_reversal','iterative_time_reversal'):
        img=_fdtd(acq,method,int(iterations),float(reg),int(time_decimate))
    elif method in ('cg_tikhonov','fista_ray','landweber','sirt','iterative_das','lsqr','tv_fista','admm_tv','pnp_fista'):
        img=_ray_reconstruct(acq,method,int(iterations),float(reg),weights)
    elif method=='deep_image_prior':
        from .deep_image_prior import reconstruct_dip
        img=reconstruct_dip(acq,iterations=iterations,regularization=reg)
    elif method in ('learned_unet','learned_rescnn'):
        from .learned_reconstruction import apply_checkpoint
        # The checkpoint specifies how input baselines were generated.
        img=apply_checkpoint(_beamform(acq,'das'),weights,
                             expected_family='unet' if method=='learned_unet' else 'rescnn')
    else:
        img=_beamform(acq,method,filter_low_hz,filter_high_hz)
    if img.shape!=(acq.grid_size,acq.grid_size) or not np.isfinite(img).all():
        raise FloatingPointError(f'{method} produced an invalid image')
    return np.asarray(img,dtype=np.float32)
