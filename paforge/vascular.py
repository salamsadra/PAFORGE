"""Reproducible *synthetic* arterial/venous 2D networks (NOT segmented patient anatomy).

Each tube is rasterized by distance to a line segment; child branches taper.
The 2D plane is a synthetic mesoscopic vascular cross-section, not an
anatomically validated 3D human circulatory network.
"""
import numpy as np


def _tube(x, y, a, b, radius_start, radius_end):
    vx, vy = b[0]-a[0], b[1]-a[1]
    q = np.clip(((x-a[0])*vx+(y-a[1])*vy)/max(vx*vx+vy*vy, 1.e-12),0.,1.)
    d2 = (x-(a[0]+q*vx))**2+(y-(a[1]+q*vy))**2
    r=radius_start+(radius_end-radius_start)*q
    return d2 <= r*r


def _grow_tree(x,y,rng,start,angle,length,width,depth,spread):
    """Return binary mask plus terminal tips for a curving, bifurcating tree."""
    mask=np.zeros_like(x,dtype=bool)
    tips=[]
    def grow(a,theta,l,w,level):
        theta += rng.normal(0,.075)
        b=(a[0]+l*np.cos(theta),a[1]+l*np.sin(theta))
        mask[:] |= _tube(x,y,a,b,w,max(w*.72,.007))
        if level<=0:
            tips.append(b)
            return
        for sign in (-1,1):
            grow(b,theta+sign*(spread+rng.uniform(-.13,.13)),l*rng.uniform(.67,.80),w*.72,level-1)
    grow(start,angle,length,width,depth)
    return mask,tips


def make_vascular_network(n,fov_m,rng,branch_levels=3,pattern="branching",density=1.):
    """Artery (1), vein (2), capillary (3) masks and derived union.

    All coordinates dimensionless in a [-1,1] square. Pixel-scale vessel
    diameters depend on grid and FOV, and should be convergence-tested.
    """
    gy,gx=np.mgrid[:n,:n]
    x=(gx-(n-1)/2)/(n/2);y=(gy-(n-1)/2)/(n/2)
    artery=np.zeros((n,n),bool);vein=artery.copy();cap=artery.copy()
    if pattern=="paired":
        positions=[(-.24,1),(.24,2)]
        for x0,typ in positions:
            center=x0+.065*np.sin(6*y+rng.uniform(-1,1))+.025*y
            mask=(np.abs(x-center)<(.042 if typ==1 else .055)) & (np.abs(y)<.77)
            if typ==1: artery |= mask
            else: vein |=mask
        for i in range(max(3,int(5*density))):
            yy=rng.uniform(-.5,.5);a=(-.2,yy);b=(.2,yy+rng.uniform(-.1,.1))
            cap|=_tube(x,y,a,b,.012,.010)
    else:
        levels=branch_levels if pattern=="branching" else branch_levels+1
        for root,theta,typ in [((-.62,.55),-.95,1),((.62,.53),-2.19,2)]:
            mask,tips=_grow_tree(x,y,rng,root,theta,.36,.058 if typ==1 else .067,levels,
                                  spread=.47 if pattern=="branching" else .62)
            if typ==1:artery |= mask
            else:vein |=mask
        if pattern=="plexus":
            for i in range(max(6,int(16*density))):
                p=rng.uniform(-.65,.65,size=2);q=p+rng.normal(0,.17,size=2)
                cap|=_tube(x,y,p,q,.010,.008)
        else:
            for i in range(max(4,int(10*density))):
                p=rng.uniform(-.62,.62,size=2);q=p+rng.normal(0,.14,size=2)
                cap|=_tube(x,y,p,q,.009,.007)
    # Deterministic precedence: arterial crossing vein is labelled arterial.
    vein &=~artery
    cap &=~(artery|vein)
    types=np.zeros((n,n),dtype=np.uint8)
    types[cap]=3;types[vein]=2;types[artery]=1
    return types
