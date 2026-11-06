"""Analytical weight edits and paired evaluation; no optimizer or backpropagation."""
import json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from generate import Generator, grid

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

def samples(g,z,batch=8): return np.concatenate([g(z[i:i+batch]) for i in range(0,len(z),batch)])

def make_edits(g, strength=0, latent_gain=1):
    edits={}
    # Blend non-overlapping upsampling phase kernels toward their common response.
    # Identical response for spatially constant incoming features at strength=1.
    for wi in (21,23,25):
        w=g.model.array(wi)
        edits[wi]=(w+strength*(w.mean(axis=(1,2),keepdims=True)-w)).astype('float32')
    if latent_gain!=1:
        w=g.model.array(13).copy()
        # SAME padding discards all but [1:3,1:3] for the initial 1x1 input.
        w[:,1:3,1:3,:]*=latent_gain
        edits[13]=w
    return edits

if __name__=='__main__':
    Path('results').mkdir(exist_ok=True)
    base=Generator(); z=np.random.default_rng(42).standard_normal((64,512)).astype('float32')
    configs=[('original',0,1),('phase15',.15,1),('phase30',.3,1),('phase50',.5,1),('gain110',0,1.1),('gain120',0,1.2),('balanced',.3,1.1),('diverse',.3,1.2)]
    results={}; previews=[]; labels=[]
    for name,strength,gain in configs:
        start=time.time(); edits=make_edits(base,strength,gain)
        g=Generator(edits=edits); y=samples(g,z)
        np.save(f'results/{name}.npy',y)
        results[name]={'phase_strength':strength,'latent_gain':gain,**metrics(y)}
        print(name,json.dumps(results[name]),'seconds',round(time.time()-start,2),flush=True)
        grid(y[:24],f'results/{name}.png',labels=[f'{name} {i}' for i in range(24)])
        previews.extend(y[:8]); labels.extend([f'{name} {i}' for i in range(8)])
    grid(np.array(previews),'results/comparison.png',labels=labels)
    Path('results/metrics.json').write_text(json.dumps(results,indent=2))
