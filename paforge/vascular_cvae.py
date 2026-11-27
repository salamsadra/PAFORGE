"""Multiclass arterial/venous/capillary CVAE.

Requires *explicitly curated* 2D type-label maps: 0 background, 1 artery,
2 vein, 3 capillary. The built-in demo uses only procedural phantoms.
No real-human trained weights ship with PA-FORGE.
"""
from pathlib import Path
import numpy as np

LABELS=('background','artery','vein','capillary')


def _torch():
    try:import torch
    except ImportError as e:raise ImportError('Install pa-forge[train] for vascular CVAE') from e
    return torch


def model(latent=24):
    torch=_torch()
    import torch.nn as nn
    class VascularCVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder=nn.Sequential(nn.Conv2d(4,16,4,2,1),nn.GELU(),
                nn.Conv2d(16,32,4,2,1),nn.GELU(),
                nn.Conv2d(32,48,4,2,1),nn.GELU(),
                nn.Conv2d(48,64,4,2,1),nn.GELU())
            self.mu=nn.Linear(64*4*4,latent)
            self.logvar=nn.Linear(64*4*4,latent)
            self.fc=nn.Linear(latent,64*4*4)
            self.decoder=nn.Sequential(nn.ConvTranspose2d(64,48,4,2,1),nn.GELU(),
                nn.ConvTranspose2d(48,32,4,2,1),nn.GELU(),
                nn.ConvTranspose2d(32,16,4,2,1),nn.GELU(),
                nn.ConvTranspose2d(16,4,4,2,1))
        def decode(self,z):
            return self.decoder(self.fc(z).reshape(-1,64,4,4))
        def forward(self,one_hot):
            h=self.encoder(one_hot).flatten(1)
            mu=self.mu(h);lv=self.logvar(h).clamp(-10,10)
            latent=mu+torch.randn_like(mu)*torch.exp(.5*lv)
            return self.decode(latent),mu,lv
    return VascularCVAE()


def create_demo_masks(output,n=60,seed=42):
    from .vascular import make_vascular_network
    rng=np.random.default_rng(seed)
    data=[]
    for i in range(n):
        pattern=('branching','paired','plexus')[i%3]
        data.append(make_vascular_network(64,.025,rng,
              branch_levels=int(rng.integers(2,5)),pattern=pattern,
              density=float(rng.uniform(.7,1.8))))
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,types=np.stack(data))
    return output


def train(dataset,weights,epochs=4,batch_size=12,seed=42,lr=.001):
    torch=_torch()
    import torch.nn.functional as F
    from torch.utils.data import DataLoader,TensorDataset
    torch.manual_seed(seed);np.random.seed(seed)
    with np.load(dataset,allow_pickle=False) as d:
        labels=d['types'].astype(np.int64)
    if labels.ndim!=3 or labels.shape[1:]!=(64,64) or len(labels)<8:
        raise ValueError('Dataset must contain types [N,64,64] with N>=8')
    if np.any(labels<0) or np.any(labels>3):raise ValueError('type labels must be 0..3')
    if not all(np.any(labels==i) for i in (1,2,3)):
        raise ValueError('Training corpus must contain arterial, venous and capillary class examples')
    ds=TensorDataset(torch.tensor(labels,dtype=torch.long))
    dl=DataLoader(ds,batch_size=batch_size,shuffle=True,
        generator=torch.Generator().manual_seed(seed))
    network=model();optimizer=torch.optim.Adam(network.parameters(),lr=lr)
    weight=torch.tensor([.12,1.5,1.6,3.],dtype=torch.float32)
    history=[]
    for epoch in range(epochs):
        losses=[]
        for (y,) in dl:
            x=F.one_hot(y,num_classes=4).permute(0,3,1,2).float()
            optimizer.zero_grad()
            logits,mu,logvar=network(x)
            ce=F.cross_entropy(logits,y,weight=weight)
            kl=-.5*(1+logvar-mu.square()-logvar.exp()).sum(dim=1).mean()
            loss=ce+1.e-4*kl
            loss.backward();optimizer.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
    weights=Path(weights);weights.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':network.state_dict(),'latent':24,
                'classes':LABELS,'seed':seed,'epochs':epochs,
                'training_data':str(dataset),'training_claim':'user supplied or synthetic, not verified real data'},weights)
    return history


def sample(weights,rng,grid_size):
    torch=_torch()
    from scipy.ndimage import zoom
    saved=torch.load(weights,map_location='cpu',weights_only=True)
    if tuple(saved.get('classes',()))!=LABELS:
        raise ValueError('Incompatible vascular model class schema')
    network=model(saved['latent']);network.load_state_dict(saved['state_dict']);network.eval()
    z=torch.from_numpy(rng.normal(size=(1,saved['latent'])).astype('float32'))
    with torch.no_grad():
        logits=network.decode(z)[0]
        types=logits.argmax(dim=0).cpu().numpy().astype('uint8')
    if not (np.any(types==1) and np.any(types==2)):
        raise ValueError("Vascular CVAE sample collapsed to background or omitted a blood vessel class. "
                         "Train longer / use a larger curated dataset; no fabricated fallback is used.")
    if grid_size!=64:
        types=zoom(types,grid_size/64,order=0)
        types=types[:grid_size,:grid_size]
    return types
