"""Untrained convolutional Deep Image Prior (DIP) constrained by ray-model data.

A self-supervised optimization on a *simplified travel-time forward operator*;
not a full-wave differentiable PAT solver. The neural weights are optimized
for each acquisition. Device calibration and domain shift remain unvalidated.
"""
import numpy as np
from .reconstruction_engine import RayOperator


def reconstruct_dip(acq,iterations=30,regularization=.0001,device='auto',seed=42):
    try:import torch
    except ImportError as exc:raise RuntimeError('Deep Image Prior requires PyTorch; install the train extra') from exc
    if acq.grid_size>128:raise ValueError('DIP limited to grid <=128 for available GPU/CPU memory')
    if device=='auto':device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(int(seed))
    op=RayOperator(acq)
    # Store sparse ray-travel mapping as vectors; linear and differentiable.
    ii=[];jj=[];weights=[]
    for detector in range(op.ns):
        pixel=np.flatnonzero(op.valid[detector]);time=op.idx[detector,pixel]
        w=op.w[detector,pixel]
        ii.append(detector*op.nt+time);ii.append(detector*op.nt+time+1)
        jj.append(pixel);jj.append(pixel)
        weights.append(op.scale*(1-w));weights.append(op.scale*w)
    ti=torch.tensor(np.concatenate(ii),dtype=torch.int64,device=device)
    pj=torch.tensor(np.concatenate(jj),dtype=torch.int64,device=device)
    wt=torch.tensor(np.concatenate(weights),dtype=torch.float32,device=device)
    bscale=max(float(np.max(np.abs(acq.traces))),1e-9)
    target=torch.tensor((acq.traces/bscale).ravel(),dtype=torch.float32,device=device)
    from torch import nn
    net=nn.Sequential(nn.Conv2d(1,24,3,padding=1),nn.LeakyReLU(.1),
                      nn.Conv2d(24,24,3,padding=1),nn.LeakyReLU(.1),
                      nn.Conv2d(24,1,3,padding=1),nn.Softplus()).to(device)
    z=torch.randn((1,1,op.n,op.n),dtype=torch.float32,device=device)
    opt=torch.optim.Adam(net.parameters(),lr=.02)
    for _ in range(int(iterations)):
        opt.zero_grad()
        img=net(z)
        prediction=torch.zeros(op.ns*op.nt,device=device,dtype=torch.float32)
        prediction=prediction.index_add(0,ti,img.reshape(-1)[pj]*wt)
        # Normalized signal discrepancy and a weak isotropic variation penalty.
        tv=(torch.mean(torch.abs(img[:,:,:,1:]-img[:,:,:,:-1]))+
            torch.mean(torch.abs(img[:,:,1:,:]-img[:,:,:-1,:])))
        loss=torch.mean((prediction-target)**2)+float(regularization)*tv
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(),5.)
        opt.step()
    with torch.no_grad():img=net(z).cpu().numpy()[0,0]*bscale
    return np.asarray(img,dtype='float32')
