"""Discrete-adjoint acoustic inversion for PA-FORGE's 2D FDTD simulator.

A physical *numerical* matched forward/adjoint pair: the gradients are checked
using dot products. Inverse is not calibrated optical absorption or 3D imaging.
Requires unfiltered detector measurements, with known detector geometry and dt.
"""
import numpy as np
from .acoustic import _acceleration, _sponge, _sample_bilinear


def _transpose_acceleration(z, sound, rho, dx):
    """Adjoint of L=diag(rho*c²)*div((1/rho)*grad)/dx²."""
    # _acceleration(q, 1/sqrt(rho), rho, dx) has row multiplier 1.
    return _acceleration(z*rho*sound**2,1/np.sqrt(rho),rho,dx)


def _detector_backproject(values, positions, shape):
    """Exactly transpose of bilinear sampling on a pressure grid."""
    y,x=positions[:,0],positions[:,1]
    y0=np.floor(y).astype(int);x0=np.floor(x).astype(int)
    wy=y-y0;wx=x-x0
    out=np.zeros(shape,dtype=np.float64)
    for dy,dx,w in ((0,0,(1-wy)*(1-wx)),(0,1,(1-wy)*wx),
                    (1,0,wy*(1-wx)),(1,1,wy*wx)):
        np.add.at(out,(y0+dy,x0+dx),w*values)
    return out



class ForwardOperator:
    def __init__(self, sound, sensor_positions_px, fov_m, dt_s, n_times, rho=None):
        self.sound=np.asarray(sound,dtype=np.float64)
        self.n=self.sound.shape[0]
        if self.sound.shape!=(self.n,self.n):raise ValueError("sound map must be square")
        self.rho=np.full_like(self.sound,1000.) if rho is None else np.asarray(rho,dtype=np.float64)
        self.pos=np.asarray(sensor_positions_px,dtype=float)
        if self.pos.ndim!=2 or self.pos.shape[1]!=2 or np.min(self.pos)<0 or np.max(self.pos)>=self.n-1:
            raise ValueError("Sensor pixel positions must be within interpolable grid")
        self.dx=fov_m/self.n
        self.dt=float(dt_s);self.nt=int(n_times)
        if self.dt<=0 or self.nt<2 or np.any(self.sound<=0):raise ValueError("Invalid acoustic model")
        if self.dt*np.max(self.sound)*np.sqrt(2)/self.dx>.5+1e-12:
            raise ValueError("Unstable CFL: operator uses CFL <= 0.5 in this codebase")
        self.damp=_sponge(self.n)

    def _L(self,x):
        return _acceleration(x,self.sound,self.rho,self.dx)

    def _Lt(self,x):
        return _transpose_acceleration(x,self.sound,self.rho,self.dx)

    def forward(self,p0):
        p0=np.asarray(p0,dtype=np.float64)
        if p0.shape!=(self.n,self.n):raise ValueError("Wrong pressure shape")
        current=p0.copy()
        previous=p0+.5*self.dt**2*self._L(p0)
        traces=np.empty((len(self.pos),self.nt),dtype=np.float64)
        for t in range(self.nt):
            traces[:,t]=_sample_bilinear(current,self.pos)
            nxt=(2*current-previous+self.dt**2*self._L(current))*self.damp
            previous,current=current,nxt
        return traces

    def adjoint(self,traces):
        traces=np.asarray(traces,dtype=np.float64)
        if traces.shape!=(len(self.pos),self.nt):raise ValueError("Wrong traces shape")
        # adjoints of (current_t, previous_t) after each step
        next_current=np.zeros((self.n,self.n),dtype=np.float64)
        next_previous=np.zeros_like(next_current)
        for t in range(self.nt-1,-1,-1):
            d=self.damp*next_current
            g_current=(_detector_backproject(traces[:,t],self.pos,(self.n,self.n))
                       +next_previous+2*d+self.dt**2*self._Lt(d))
            g_previous=-d
            next_current,next_previous=g_current,g_previous
        # previous_0 = p0 + 0.5*dt²*L(p0), current_0 = p0
        return next_current+next_previous+.5*self.dt**2*self._Lt(next_previous)

    def reconstruct(self,measurements,iterations=8,reg=0.001):
        """Projected FISTA, nonnegative p0, with L2 Tikhonov penalty.

        reg is relative to estimated ||A||² (normalization-invariant).
        Output is a numerical inverse of the *same* simple wave model.
        """
        b=np.asarray(measurements,dtype=np.float64)
        rng=np.random.default_rng(19)
        v=rng.standard_normal((self.n,self.n));v/=np.linalg.norm(v)
        eig=1.
        for _ in range(8):
            z=self.adjoint(self.forward(v))
            eig=max(float(np.linalg.norm(z)),1e-18)
            v=z/eig
        lam=reg*eig
        step=.95/(eig+lam)
        x=np.zeros((self.n,self.n),dtype=float)
        z=x.copy();alpha=1.
        rhs=self.adjoint(b)
        losses=[]
        for _ in range(iterations):
            g=self.adjoint(self.forward(z)) +lam*z-rhs
            xnew=np.maximum(0.,z-step*g)
            alpha_new=(1+np.sqrt(1+4*alpha*alpha))/2
            z=xnew+((alpha-1)/alpha_new)*(xnew-x)
            x=xnew;alpha=alpha_new
            resid=self.forward(x)-b
            losses.append(float(.5*np.sum(resid**2)+.5*lam*np.sum(x**2)))
        return x.astype(np.float32),losses


def from_simulation(config,sound,dt,detectors_m):
    """Build operator with exact sensor locations and spatial grid."""
    n=config.grid_size;dx=config.fov_m/n
    xy=np.asarray(detectors_m)
    ypx=xy[:,1]/dx+(n-1)/2
    xpx=xy[:,0]/dx+(n-1)/2
    return ForwardOperator(sound,np.column_stack([ypx,xpx]),config.fov_m,dt,config.time_steps)
