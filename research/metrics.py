"""Unmodified metric/formula functions extracted from the saved experiments."""
import numpy as np
import torch
import torch.nn.functional as F

# Source: experiment.py
def metrics(y):
    # Fixed-pattern phase variation after subtracting smooth local structure.
    t=torch.from_numpy(y).permute(0,3,1,2)
    smooth=F.avg_pool2d(F.pad(t,(2,2,2,2),mode='reflect'),5,stride=1)
    residual=(t-smooth).numpy()[:,:,8:-8,8:-8]
    phase={}
    for p in (2,4,8):
        vals=np.stack([residual[:,:,i::p,j::p].mean(axis=(0,2,3)) for i in range(p) for j in range(p)])
        phase[str(p)]=float(np.sqrt(np.mean((vals-vals.mean(0))**2)))
    # Low-resolution distances deliberately exclude pixel checkerboard noise.
    feat=F.avg_pool2d(t,8).flatten(1).numpy()
    d=np.sqrt(np.mean((feat[:,None,:]-feat[None,:,:])**2,axis=-1))
    mask=~np.eye(len(d),dtype=bool)
    nearest=np.where(mask,d,np.inf).min(1)
    centered=feat-feat.mean(0)
    sv=np.linalg.svd(centered,compute_uv=False)**2; prob=sv/(sv.sum()+1e-20)
    # Remove per-image channel means and normalize contrast before shape comparison.
    shape=F.avg_pool2d(t,8).numpy(); shape-=shape.mean((2,3),keepdims=True)
    shape=shape.reshape(len(y),-1); shape/=np.maximum(np.linalg.norm(shape,axis=1,keepdims=True),1e-12)
    cosine=np.clip(1-shape@shape.T,0,2)
    shape_sv=np.linalg.svd(shape-shape.mean(0),compute_uv=False)**2; shape_prob=shape_sv/(shape_sv.sum()+1e-20)
    return dict(phase_rms=phase,pairwise_rms=float(d[mask].mean()),nearest_rms=float(nearest.mean()),effective_rank=float(np.exp(-np.sum(prob*np.log(prob+1e-20)))),shape_cosine_distance=float(cosine[mask].mean()),nearest_shape_distance=float(np.where(mask,cosine,np.inf).min(1).mean()),shape_effective_rank=float(np.exp(-np.sum(shape_prob*np.log(shape_prob+1e-20)))),clipped_fraction=float(((y<.01)|(y>.99)).mean()),pixel_std=float(y.std()),edge_energy=float(np.abs(np.diff(y,axis=1)).mean()+np.abs(np.diff(y,axis=2)).mean()))

# Source: round2/run_experiments.py
def features(y):
    t=F.avg_pool2d(torch.from_numpy(y).permute(0,3,1,2),8).numpy()
    t-=t.mean((2,3),keepdims=True)
    t=t.reshape(len(y),-1); t/=np.maximum(np.linalg.norm(t,axis=1,keepdims=True),1e-12)
    return t

# Source: round2/run_experiments.py
def repetition(y):
    a=features(y); b=features(y[:,:,::-1,:].copy())
    dist=np.minimum(np.clip(1-a@a.T,0,2),np.clip(1-a@b.T,0,2))
    np.fill_diagonal(dist,np.inf)
    return {'mirror_invariant_nearest_shape_distance':float(dist.min(1).mean()),'fraction_with_neighbor_below_cosine_005':float((dist.min(1)<.05).mean()),'fraction_with_neighbor_below_cosine_010':float((dist.min(1)<.10).mean())}

# Source: round4/refine_structure.py
def metrics_pair(base,y):
    r={**metrics(y),**repetition(y)}
    r['paired_rgb_rms']=float(np.sqrt(np.mean((base-y)**2)))
    r['max_channel_mean_shift']=float(np.max(np.abs(base.mean((0,1,2))-y.mean((0,1,2)))))
    return r

# Source: round7/direct_detail.py
def measure(base,y):
    m=metrics_pair(base,y)
    t=torch.from_numpy(y).permute(0,3,1,2)
    # Broad structure gradients: do not count raw pixel noise as detail gain.
    k=torch.tensor([1,4,6,4,1],dtype=torch.float32);kernel=torch.outer(k,k)/256
    s=F.conv2d(F.pad(t,(2,2,2,2),mode='reflect'),kernel[None,None].repeat(3,1,1,1),groups=3).numpy()[:,:,4:-4,4:-4]
    m['smoothed_edge_energy']=float(np.abs(np.diff(s,axis=2)).mean()+np.abs(np.diff(s,axis=3)).mean())
    return m

# Source: round5/portrait_sampler.py
def artifacts(y):
    t=torch.from_numpy(y).permute(0,3,1,2)
    smooth=F.avg_pool2d(F.pad(t,(2,2,2,2),mode='reflect'),5,stride=1)
    r=(t-smooth)[:,:,8:-8,8:-8]
    phase=torch.stack([r[:,:,dy::4,dx::4].mean((2,3)) for dy in range(4) for dx in range(4)],dim=1)
    phase=phase-phase.mean(1,keepdim=True)
    return {'grid4':torch.sqrt((phase**2).mean((1,2))).numpy(),'clipped':((y<.01)|(y>.99)).mean((1,2,3)),'coarse_contrast':F.avg_pool2d(t,8).flatten(1).std(1,unbiased=False).numpy()}

# Source: round8/selective_detail.py
def measurements(base,y):
    m=measure(base,y);m['mean_per_image_grid4']=float(artifacts(y)['grid4'].mean());return m

# Source: round10/overlap.py
def expanded(w,a,b):
    """CRD weights: Wp'=(Wp-b*(Wp-M))*delta + a*M*(Bp-delta)."""
    co4,ci,_,_=w.shape;co=co4//4
    p=w.reshape(co,4,ci);m=p.mean(1)
    out=torch.zeros((co,4,ci,3,3),dtype=w.dtype)
    out[:,:,:,1,1]=p-b*(p-m[:,None])-a*m[:,None]
    taps=[torch.tensor([.25,.75,0]),torch.tensor([0,.75,.25])]
    for py in range(2):
        for px in range(2):
            out[:,py*2+px]+=a*m[:,:,None,None]*torch.outer(taps[py],taps[px])
    return out.reshape(co4,ci,3,3)

# Source: round10/overlap.py
def edge_grid(y):
    """Per-image RGB edge magnitudes, phase-folded in native coordinates.

    CV divides periodic edge amplitude by total edge energy, so uniform
    attenuation alone cannot win. This is a grid proxy, not a quality score.
    """
    edges=[np.abs(np.diff(y,axis=2)).mean(3),np.abs(np.diff(y,axis=1)).mean(3)]
    out={};profiles={}
    for period in [2,4,8,16]:
        cvs=[];amps=[];allp=[]
        for axis,e in enumerate(edges):
            e=e[:,8:120,8:120]
            prof=e.mean(1 if axis==0 else 2)
            folds=np.stack([prof[:,i::period].mean(1) for i in range(period)],1)
            amp=folds.std(1);cvs.append(amp/np.maximum(folds.mean(1),1e-8));amps.append(amp)
            allp.append(folds.mean(0).tolist())
        out[str(period)]=dict(cv=float(np.mean(cvs)),amplitude=float(np.mean(amps)))
        profiles[str(period)]=allp
    return dict(periods=out,score=float(np.mean([out[str(p)]['cv'] for p in [2,4,8]])),profiles=profiles)

# Source: round10/overlap.py
def stats(base,y):
    m=measurements(base,y);m['edge_grid']=edge_grid(y)
    a=torch.from_numpy(base).permute(0,3,1,2);b=torch.from_numpy(y).permute(0,3,1,2)
    m['coarse_rgb_rms']=float((F.avg_pool2d(a,8)-F.avg_pool2d(b,8)).square().mean().sqrt())
    return m
