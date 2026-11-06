import json
from pathlib import Path
import numpy as np
from generate import Generator, grid
from experiment import samples, metrics

def calibration(g,z):
    means={wi:[] for wi in (21,23,25)}
    for start in range(0,len(z),8):
        _,acts=g(z[start:start+8],trace=True)
        for wi,k in ((21,3),(23,4),(25,5)):
            means[wi].append(acts[k].mean((0,2,3)).numpy())
    return {wi:np.mean(a,axis=0) for wi,a in means.items()}

def correction(g,means,strength):
    out={}
    for wi,mu in means.items():
        w=g.model.array(wi)
        dc=np.einsum('ohwi,i->ohw',w,mu)
        delta=dc.mean((1,2),keepdims=True)-dc
        out[wi]=(w+strength*delta[:,:,:,None]*mu[None,None,None,:]/np.dot(mu,mu)).astype('float32')
    return out

def spectral(g,alpha,gain=1):
    w=g.model.array(13).copy(); a=w[:,1:3,1:3,:].reshape(-1,512).astype('float64')
    val,vec=np.linalg.eigh(a.T@a)
    factors=(np.maximum(val,1e-10)/np.exp(np.log(np.maximum(val,1e-10)).mean()))**(-alpha/2)
    # Keep total linear-response energy fixed, so change isn't merely gain.
    factors*=np.sqrt(val.sum()/np.dot(val,factors**2))*gain
    a=(a@vec*factors)@vec.T
    w[:,1:3,1:3,:]=a.reshape(w[:,1:3,1:3,:].shape)
    return w,{'eigen_min':float(val.min()),'eigen_max':float(val.max()),'top8_fraction':float(val[-8:].sum()/val.sum()),'multiplier_min':float(factors.min()),'multiplier_max':float(factors.max())}

if __name__=='__main__':
    g=Generator(); z=np.random.default_rng(42).standard_normal((64,512)).astype('float32')
    # Calibration seeds disjoint from both candidate selection and held-out tests.
    means=calibration(g,np.random.default_rng(7391).standard_normal((128,512)).astype('float32'))
    np.savez('results/channel_means.npz',**{str(k):v for k,v in means.items()})
    configs=[('dc50',.5,0,1),('dc100',1,0,1),('dc50gain',.5,0,1.1),('spectral15',.5,.15,1),('spectral30',.5,.3,1),('spectral50',.5,.5,1),('spectral30gain',.5,.3,1.1)]
    results={}; pics=[]; labels=[]
    for name,s,a,gain in configs:
        edits=correction(g,means,s); eig={}
        if a or gain!=1: edits[13],eig=spectral(g,a,gain)
        y=samples(Generator(edits=edits),z)
        np.savez(f'results/{name}_edits.npz',**{str(k):v for k,v in edits.items()})
        np.save(f'results/{name}.npy',y)
        results[name]={'dc_strength':s,'spectral_alpha':a,'gain':gain,'spectrum':eig,**metrics(y)}
        print(name,json.dumps(results[name]),flush=True)
        pics.extend(y[:8]); labels.extend([f'{name} {i}' for i in range(8)])
    grid(np.array(pics),'results/refined_comparison.png',labels=labels)
    Path('results/refined_metrics.json').write_text(json.dumps(results,indent=2))
