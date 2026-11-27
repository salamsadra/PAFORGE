"""A separate categorical CVAE for 2D *phantom material* labels.

NOT the vascular-type CVAE. Uses leakage-free phantom-group validation and
explicit provenance. The model is trained locally on extracted user data.
"""
from pathlib import Path
import json
import numpy as np
from .material_contracts import validate_partition, validate_optical_rows, file_sha256


def _torch():
    import torch
    return torch


def model(n_classes,latent=16):
    import torch
    from torch import nn
    class MaterialCVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder=nn.Sequential(nn.Conv2d(2*n_classes,24,4,2,1),nn.GELU(),
                 nn.Conv2d(24,40,4,2,1),nn.GELU(),nn.Conv2d(40,64,4,2,1),nn.GELU(),
                 nn.Conv2d(64,80,4,2,1),nn.GELU())
            self.mu=nn.Linear(80*4*4,latent)
            self.lv=nn.Linear(80*4*4,latent)
            self.fc=nn.Linear(latent+n_classes,80*4*4)
            self.decoder=nn.Sequential(nn.ConvTranspose2d(80,64,4,2,1),nn.GELU(),
                 nn.ConvTranspose2d(64,40,4,2,1),nn.GELU(),
                 nn.ConvTranspose2d(40,24,4,2,1),nn.GELU(),
                 nn.ConvTranspose2d(24,n_classes,4,2,1))
        def decode(self,z,composition):
            if composition.shape!=(z.shape[0],n_classes):
                raise ValueError('Condition must be [batch,n_materials] material proportions')
            return self.decoder(self.fc(torch.cat([z,composition],dim=1)).view(-1,80,4,4))
        def forward(self,one_hot):
            composition=one_hot.mean(dim=(2,3))
            context=composition[:,:,None,None].expand_as(one_hot)
            h=self.encoder(torch.cat([one_hot,context],dim=1)).flatten(1)
            mu=self.mu(h);lv=self.lv(h).clamp(-12,10)
            z=mu+torch.randn_like(mu)*torch.exp(.5*lv)
            return self.decode(z,composition),mu,lv
    return MaterialCVAE()


def train(corpus_path,profile_path,weights_path,epochs=20,batch_size=8,seed=42,lr=0.001,latent=16,device='auto',evaluate_test=False,kl_weight=1e-4):
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader,TensorDataset,WeightedRandomSampler
    if not np.isfinite(kl_weight) or kl_weight<0:raise ValueError('KL weight must be finite and nonnegative')
    if epochs<1 or epochs>1000 or batch_size<1:raise ValueError('Invalid training hyperparameters')
    # Limit CPU oversubscription for small research datasets.
    if device not in ('auto','cpu','cuda','mps'):
        raise ValueError('device must be auto/cpu/cuda/mps')
    if device=='auto':
        device='cuda' if torch.cuda.is_available() else ('mps' if hasattr(torch.backends,'mps') and torch.backends.mps.is_available() else 'cpu')
    if device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA explicitly requested, but PyTorch CUDA GPU is unavailable. Use CPU/auto or install a compatible CUDA build.')
    if device=='mps' and not (hasattr(torch.backends,'mps') and torch.backends.mps.is_available()):
        raise RuntimeError('Apple MPS explicitly requested but unavailable')
    if device=='cpu':torch.set_num_threads(min(4,torch.get_num_threads()))
    torch.manual_seed(seed);np.random.seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    with np.load(corpus_path,allow_pickle=False) as a:
        masks=a['masks'].astype('int64');groups=a['groups'].astype(str)
        nc=int(a['classes'])
    profile=json.loads(Path(profile_path).read_text(encoding='utf-8'))
    partition=profile['groups'];parts=validate_partition(partition)
    tr,va,te=parts['train'],parts['validation'],parts['test']
    if 'measurements' in profile:validate_optical_rows(profile)
    if not tr | va | te >= set(groups.tolist()):
        raise ValueError('Corpus contains group IDs not assigned to a declared split')
    ti=np.asarray([i for i,g in enumerate(groups) if g in tr],dtype=int)
    vi=np.asarray([i for i,g in enumerate(groups) if g in va],dtype=int)
    if len(ti)<1 or len(vi)<1:raise ValueError('Train and validation groups required')
    if set(groups[ti])&set(groups[vi]):raise ValueError('Data leakage: shared group IDs')
    if masks.ndim!=3 or masks.shape[1:]!=(64,64) or np.min(masks)<0 or np.max(masks)>=nc:
        raise ValueError('Material training masks must be [N,64,64] categorical indices')
    network=model(nc,latent).to(device)
    opt=torch.optim.Adam(network.parameters(),lr=lr)
    # Balance independent phantom groups so multi-device/SVOT slice counts do
    # not make a heavily sampled phantom dominate the CVAE objective.
    train_group_counts={g:int(np.sum(groups[ti]==g)) for g in set(groups[ti])}
    sample_weights=torch.tensor([1.0/train_group_counts[groups[i]] for i in ti],dtype=torch.double)
    sampler=WeightedRandomSampler(sample_weights,num_samples=len(ti),replacement=True,
                                  generator=torch.Generator().manual_seed(seed))
    trainloader=DataLoader(TensorDataset(torch.tensor(masks[ti])),batch_size=batch_size,sampler=sampler)
    valloader=DataLoader(TensorDataset(torch.tensor(masks[vi])),batch_size=batch_size,shuffle=False)
    best=np.inf;history=[];best_state=None
    def objective(labels):
        hot=F.one_hot(labels,nc).permute(0,3,1,2).float()
        logits,mu,lv=network(hot)
        ce=F.cross_entropy(logits,labels)
        kl=-.5*(1+lv-mu.square()-lv.exp()).sum(1).mean()
        return ce+kl_weight*kl
    for epoch in range(epochs):
        network.train();losses=[]
        for (batch,) in trainloader:
            batch=batch.to(device)
            opt.zero_grad();loss=objective(batch);loss.backward();opt.step()
            losses.append(float(loss.detach()))
        network.eval()
        with torch.no_grad():
            val_losses=[]
            for (batch,) in valloader:
                val_losses.append((float(objective(batch.to(device))),len(batch)))
            vl=sum(x*n for x,n in val_losses)/sum(n for _,n in val_losses)
        trloss=float(np.mean(losses));history.append({'epoch':epoch+1,'train_loss':trloss,'validation_loss':vl})
        if vl<best:
            best=vl;best_state={k:v.detach().cpu().clone() for k,v in network.state_dict().items()}
    w=Path(weights_path);w.parent.mkdir(parents=True,exist_ok=True)
    # Holdout evaluation must be an explicit final-stage action; never implicit in training.
    network.load_state_dict(best_state);network.eval()
    test_inds=(np.asarray([i for i,g in enumerate(groups) if g in set(partition['test'])],dtype=int)
               if evaluate_test else np.asarray([],dtype=int))
    heldout=None
    if len(test_inds):
        testloader=DataLoader(TensorDataset(torch.tensor(masks[test_inds])),batch_size=batch_size,shuffle=False)
        total_images=0;sum_loss=0.;correct=0;total_pixels=0
        intersections=np.zeros(nc,dtype=np.float64)
        unions=np.zeros(nc,dtype=np.float64)
        with torch.no_grad():
            for (batch,) in testloader:
                y=batch.to(device)
                hot=F.one_hot(y,nc).permute(0,3,1,2).float()
                composition=hot.mean(dim=(2,3))
                context=composition[:,:,None,None].expand_as(hot)
                features=network.encoder(torch.cat([hot,context],dim=1)).flatten(1)
                logits=network.decode(network.mu(features),composition)
                sum_loss+=float(F.cross_entropy(logits,y))*len(batch)
                pred=logits.argmax(dim=1)
                correct+=int((pred==y).sum().item());total_pixels+=int(y.numel())
                total_images+=len(batch)
                for cls in range(nc):
                    truth=(y==cls);estimate=(pred==cls)
                    intersections[cls]+=float((truth&estimate).sum().item())
                    unions[cls]+=float((truth|estimate).sum().item())
        ious=[float(intersections[i]/unions[i]) for i in range(nc) if unions[i]>0]
        heldout={'test_phantom_count':int(total_images),
                 'autoencoding_cross_entropy':sum_loss/total_images,
                 'autoencoding_pixel_accuracy':correct/total_pixels,
                 'autoencoding_mean_iou':float(np.mean(ious)) if ious else None,
                 'caveat':'Test metrics are deterministic-mean reconstruction of labels, NOT realism of novel prior samples'}
    # One conditioning composition per independent training phantom.
    compositions=[]
    for group in sorted(set(groups[ti])):
        gi=[i for i in ti if groups[i]==group]
        compositions.append(np.mean([np.bincount(masks[i].ravel(),minlength=nc)/masks[i].size for i in gi],axis=0))
    compositions=np.stack(compositions)
    torch.save({'state_dict':best_state,'n_classes':nc,'latent':latent,
                'model_family':'material_cvae','conditioning':'material area fractions',
                'train_compositions':compositions.tolist(),'training_corpus':str(corpus_path),
                'optical_profile':str(profile_path),'train_groups':sorted(tr),
                'validation_groups':sorted(va),'test_groups':partition['test'],
                'seed':seed,'epochs':epochs,'kl_weight':float(kl_weight),
                'corpus_sha256':file_sha256(corpus_path),'profile_sha256':file_sha256(profile_path),'best_validation_loss':best,'training_device':device,
                'test_evaluation_opt_in':bool(evaluate_test),
                'data_claim':'OpenQPAIData material phantom labels; not vascular/human anatomical truth'},w)
    report={'weights':str(w),'kl_weight':float(kl_weight),'corpus_sha256':file_sha256(corpus_path),
            'profile_sha256':file_sha256(profile_path),'history':history,'best_validation_loss':best,'heldout_test':heldout,
            'test_evaluation_opt_in':bool(evaluate_test),'training_device':device,
            'cuda_name':torch.cuda.get_device_name(0) if device=='cuda' else None,
            'train_groups':sorted(tr),'validation_groups':sorted(va),'test_groups':partition['test'],
            'reality_check':'Training convergence and real-world realism are different claims. Test groups were held out.'}
    (w.parent/(w.stem+'_training.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    ax.plot([x['epoch'] for x in history],[x['train_loss'] for x in history],label='Training (phantom groups)')
    ax.plot([x['epoch'] for x in history],[x['validation_loss'] for x in history],label='Validation (held-out groups)')
    ax.set(xlabel='Epoch',ylabel='CVAE ELBO loss',title='Categorical phantom-material CVAE training')
    ax.legend();fig.tight_layout()
    fig.savefig(w.parent/(w.stem+'_loss.png'),dpi=145);plt.close(fig)
    report['training_plot']=str(w.parent/(w.stem+'_loss.png'))
    return report


def sample(weights_path,rng,grid_size=64,temperature=0.9,composition=None):
    import torch
    from scipy.ndimage import zoom
    if not .1<=temperature<=2:raise ValueError('temperature out of range 0.1..2')
    saved=torch.load(weights_path,map_location='cpu',weights_only=True)
    if saved.get('model_family')!='material_cvae':raise ValueError('Checkpoint is not a phantom material CVAE')
    net=model(int(saved['n_classes']),int(saved['latent']))
    net.load_state_dict(saved['state_dict']);net.eval()
    z=torch.tensor(rng.normal(size=(1,int(saved['latent']))).astype('float32'))
    if composition is None:
        observed=saved.get('train_compositions',[])
        if not observed:raise ValueError('CVAE checkpoint lacks conditioning distributions')
        fractions=np.asarray(observed[int(rng.integers(len(observed)))],dtype=np.float32)
    else:
        fractions=np.asarray(composition,dtype=np.float32)
    if fractions.shape!=(int(saved['n_classes']),) or not np.isfinite(fractions).all() or (fractions<0).any() or fractions.sum()<=0:
        raise ValueError('Explicit material fractions must match n_classes, be nonnegative, finite and sum >0')
    fractions=fractions/fractions.sum()
    with torch.no_grad():
        logits=net.decode(z,torch.tensor(fractions[None,:]))[0].numpy()/temperature
    # Spatially correlated categorical sampling: avoid independent per-pixel
    # draws that produce unrealistic salt-and-pepper material maps.
    from scipy.ndimage import gaussian_filter
    gumbel=-np.log(-np.log(np.clip(rng.random(logits.shape),1e-7,1-1e-7)))
    correlated=gaussian_filter(gumbel,sigma=(0,2.0,2.0),mode='nearest')
    correlated=(correlated-correlated.mean(axis=(1,2),keepdims=True))/np.maximum(
        correlated.std(axis=(1,2),keepdims=True),1e-6)
    labels=np.argmax(logits+0.18*correlated,axis=0).astype('uint8')
    if len(np.unique(labels))<2:
        raise ValueError('CVAE generated a single class; inspect training, no fabricated structures added')
    if grid_size!=64:
        labels=zoom(labels,grid_size/64,order=0)[:grid_size,:grid_size]
    return labels


def sample_empirical(corpus_path,profile_path,rng,grid_size=64,distort_pixels=1.0,return_provenance=False):
    """Train-only smooth deformation of a *real measured-phantom* material mask."""
    from scipy.ndimage import map_coordinates,gaussian_filter,zoom
    profile=json.loads(Path(profile_path).read_text(encoding='utf-8'))
    parts=validate_partition(profile['groups']);tr=parts['train']
    if 'measurements' in profile:validate_optical_rows(profile)
    if not np.isfinite(distort_pixels) or distort_pixels<0:raise ValueError('Invalid deformation scale')
    with np.load(corpus_path,allow_pickle=False) as f:
        groups=f['groups'].astype(str)
        if not set(groups)<=set.union(*parts.values()):raise ValueError('Corpus contains group IDs not assigned to a declared split')
        inds=[i for i,g in enumerate(groups) if g in tr]
        if not inds:raise ValueError('No training groups')
        # Select an independent phantom uniformly, then one of its device/slice masks.
        available=sorted(set(groups[inds]));chosen=available[int(rng.integers(len(available)))]
        group_inds=[i for i in inds if groups[i]==chosen]
        index=int(rng.choice(group_inds))
        raw=f['masks'][index]
        if raw.shape!=(64,64) or not np.isfinite(raw).all() or np.any(raw<0) or np.any(raw>=int(f['classes'])) or not np.all(raw==np.round(raw)):
            raise ValueError('Invalid categorical training mask')
        labels=raw.astype('uint8')
    n=labels.shape[0]
    yy,xx=np.mgrid[:n,:n]
    if distort_pixels>0:
        def field():
            a=gaussian_filter(rng.normal(size=(n,n)),sigma=5)
            return (a/(np.std(a)+1e-9))*distort_pixels
        yy=yy+field();xx=xx+field()
    mask=map_coordinates(labels,[yy,xx],order=0,mode='nearest').astype('uint8')
    if grid_size!=64:mask=zoom(mask,grid_size/64,order=0)[:grid_size,:grid_size]
    if return_provenance:
        return mask, {'source_group':chosen,'source_mask_index':index,
                      'source_mask_sha256':__import__('hashlib').sha256(labels.tobytes()).hexdigest()}
    return mask
