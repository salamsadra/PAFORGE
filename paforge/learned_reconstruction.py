"""Explicitly trained image-domain PA reconstruction refiners.

Not end-to-end learned raw-to-image inverses: they learn correction of a DAS
baseline from matched synthetic image pairs. No checkpoint or ground-truth
pairs are bundled, and applying synthetic-trained weights to experiment is OOD.
"""
from pathlib import Path
import json
import numpy as np
import h5py


def _torch():
    try:import torch
    except ImportError as exc:raise RuntimeError('Install the train extra: pip install -e ".[train]"') from exc
    return torch


def _model(family):
    torch=_torch()
    nn=torch.nn
    class ResCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net=nn.Sequential(nn.Conv2d(1,24,3,padding=1),nn.ReLU(),nn.Conv2d(24,24,3,padding=1),nn.ReLU(),nn.Conv2d(24,1,3,padding=1))
        def forward(self,x):return x+self.net(x)
    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.a=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.Conv2d(16,16,3,padding=1),nn.ReLU())
            self.b=nn.Sequential(nn.Conv2d(16,32,3,padding=1),nn.ReLU())
            self.up=nn.Sequential(nn.Conv2d(48,16,3,padding=1),nn.ReLU(),nn.Conv2d(16,1,3,padding=1))
        def forward(self,x):
            from torch.nn.functional import avg_pool2d,interpolate
            a=self.a(x)
            b=self.b(avg_pool2d(a,2))
            b=interpolate(b,size=a.shape[-2:],mode='bilinear',align_corners=False)
            return x+self.up(torch.cat([a,b],1))
    if family=='unet':return UNet()
    if family=='rescnn':return ResCNN()
    raise ValueError('Only unet or rescnn are supported')


def _normalize(img):
    img=np.asarray(img,dtype=np.float32)
    scale=max(float(np.percentile(np.abs(img),99.)),1e-8)
    return np.clip(img/scale,-8.,8.).astype('float32'),scale


def _scan_pairs(root):
    root=Path(root)
    paths=sorted(root.rglob('reconstruction.h5'))
    output=[]
    for rec in paths:
        truth=rec.with_name('ground_truth.h5')
        if not truth.is_file():continue
        with h5py.File(rec,'r') as r,h5py.File(truth,'r') as t:
            method=str(r.attrs.get('method','')).split()[0]
            if 'image' not in r or 'initial_pressure_pa' not in t:continue
            if method!='das':continue  # inference uses DAS, not an arbitrary unmatched baseline
            a=np.asarray(r['image'],dtype='float32');b=np.asarray(t['initial_pressure_pa'],dtype='float32')
        if a.shape!=b.shape or a.ndim!=2 or min(a.shape)<32 or not np.isfinite(a).all() or not np.isfinite(b).all():continue
        group=rec.parent.name
        manifest=rec.with_name('manifest.json')
        if manifest.is_file():
            try:
                obj=json.loads(manifest.read_text(encoding='utf-8'))
                prov=obj.get('information',{}).get('material_provenance') or {}
                if isinstance(prov,dict):group=str(prov.get('source_group_id') or prov.get('source_group') or group)
            except (ValueError,TypeError):pass
        output.append((rec.parent,a,b,group))
    return output


def train_paired(root,output,architecture='unet',epochs=5,batch_size=4,seed=42,device='auto',max_samples=0):
    """Train on paired synthetic DAS and synthetic p0; splits by source group.

    Requires >=2 independent groups, so derived frames from the same source
    phantom do not leak between train and validation.
    """
    torch=_torch()
    pairs=_scan_pairs(root)
    if max_samples:pairs=pairs[:int(max_samples)]
    if len(pairs)<2:raise ValueError('Need >=2 synthetic paired samples whose stored baseline method is DAS')
    groups=sorted(set(p[3] for p in pairs))
    if len(groups)<2:raise ValueError('Need >=2 distinct groups for group-disjoint train/validation')
    if len(set(p[1].shape for p in pairs))>1:raise ValueError('All training images must have the same grid size')
    rng=np.random.default_rng(seed)
    rng.shuffle(groups)
    nv=max(1,int(np.ceil(len(groups)*.2)))
    valgroups=set(groups[:nv]);trainpairs=[p for p in pairs if p[3] not in valgroups];valpairs=[p for p in pairs if p[3] in valgroups]
    if not trainpairs or not valpairs:raise ValueError('Nonempty disjoint partitions required')
    if device=='auto':device='cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA not available in this PyTorch installation')
    if device=='mps' and not torch.backends.mps.is_available():raise ValueError('MPS unavailable')
    torch.manual_seed(int(seed))
    net=_model(architecture).to(device)
    opt=torch.optim.Adam(net.parameters(),lr=8e-4)
    def stack(which):
        a=[];b=[]
        for _,img,truth,_group in which:
            scaled,scale=_normalize(img)
            a.append(scaled)
            b.append(np.clip(truth/scale,-8,8).astype('float32'))
        return torch.from_numpy(np.stack(a)[:,None]),torch.from_numpy(np.stack(b)[:,None])
    x,y=stack(trainpairs);vx,vy=stack(valpairs)
    hist=[]
    for epoch in range(int(epochs)):
        net.train();order=rng.permutation(len(x));losses=[]
        for start in range(0,len(x),int(batch_size)):
            ids=order[start:start+int(batch_size)];inp=x[ids].to(device);targ=y[ids].to(device)
            opt.zero_grad();loss=torch.mean((net(inp)-targ)**2);loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step()
            losses.append(float(loss.item()))
        net.eval()
        with torch.no_grad():val=float(torch.mean((net(vx.to(device))-vy.to(device))**2).item())
        hist.append({'epoch':epoch+1,'train_mse':float(np.mean(losses)),'val_mse':val})
    destination=Path(output);destination.parent.mkdir(parents=True,exist_ok=True)
    # Safe loading is available via torch.load(weights_only=True).
    torch.save({'format':'paforge_supervised_das_v1','architecture':architecture,
                'state_dict':net.cpu().state_dict(),'training_groups':[p[3] for p in trainpairs],
                'validation_groups':sorted(valgroups),'supervision':'synthetic_initial_pressure_pa',
                'baseline_method':'das','training_history':hist},destination)
    report={'weights':str(destination),'architecture':architecture,'training_samples':len(trainpairs),
            'validation_samples':len(valpairs),'train_groups':sorted(set(p[3] for p in trainpairs)),
            'validation_groups':sorted(valgroups),'history':hist,'device':device,
            'limitations':'Image-domain supervised refiner trained on synthetic DAS/p0 pairs. Unvalidated on measured signals; not an end-to-end learned wave inverse.'}
    destination.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def apply_checkpoint(img,weights,expected_family=None):
    torch=_torch()
    if not weights or not Path(weights).is_file():raise FileNotFoundError('Please provide a trained local reconstruction checkpoint')
    obj=torch.load(weights,map_location='cpu',weights_only=True)
    if obj.get('format')!='paforge_supervised_das_v1' or obj.get('baseline_method')!='das':
        raise ValueError('Unsupported or incompatible learned reconstruction checkpoint')
    family=obj['architecture']
    if expected_family is not None and expected_family!=family:
        raise ValueError(f'Checkpoint architecture {family} != expected {expected_family}')
    scaled,scale=_normalize(img)
    net=_model(family);net.load_state_dict(obj['state_dict']);net.eval()
    with torch.no_grad():out=net(torch.from_numpy(scaled[None,None])).numpy()[0,0]*scale
    return np.asarray(out,dtype=np.float32)
