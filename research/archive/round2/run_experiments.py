"""Logged, checkpointed, gradient-free experiments branching from preserved v1.

Run from workspace root with Python 3.11. No source model is overwritten.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import datetime,hashlib,json,shutil,time
import numpy as np
import torch
import torch.nn.functional as F
from generate import Generator,grid
from experiment import samples,metrics
from refine import spectral

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'checkpoints/000_preserved_v1/model_improved.tflite'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def event(event_name,**kw):
    record={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'event':event_name,**kw}
    with (ROOT/'logs/events.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    print(json.dumps(record),flush=True)
def log(title,text):
    with (ROOT/'WORK_LOG.md').open('a') as f:f.write('\n## '+title+'\n\n'+text+'\n')

def features(y):
    t=F.avg_pool2d(torch.from_numpy(y).permute(0,3,1,2),8).numpy()
    t-=t.mean((2,3),keepdims=True)
    t=t.reshape(len(y),-1); t/=np.maximum(np.linalg.norm(t,axis=1,keepdims=True),1e-12)
    return t

def repetition(y):
    a=features(y); b=features(y[:,:,::-1,:].copy())
    dist=np.minimum(np.clip(1-a@a.T,0,2),np.clip(1-a@b.T,0,2))
    np.fill_diagonal(dist,np.inf)
    return {'mirror_invariant_nearest_shape_distance':float(dist.min(1).mean()),'fraction_with_neighbor_below_cosine_005':float((dist.min(1)<.05).mean()),'fraction_with_neighbor_below_cosine_010':float((dist.min(1)<.10).mean())}

def compact(edits):
    return {str(k)+('_active' if k==13 else ''):(a[:,1:3,1:3,:] if k==13 else a) for k,a in edits.items()}

def load_edits(path,g=None):
    g=g or Generator(str(SOURCE)); out={}
    for key,v in np.load(path).items():
        if key.endswith('_active'):
            wi=int(key.split('_')[0]); w=g.model.array(wi).copy();w[:,1:3,1:3,:]=v;out[wi]=w
        else:out[int(key)]=v
    return out

def candidates(g):
    yield 'baseline',{},'Preserved improved model; no edits.'
    for factor in (.9,.75,.5):
        yield f'bias14_{str(factor).replace(".","p")}',{14:g.model.array(14)*factor},f'Scale first-layer bias by {factor}; lower activation thresholds.'
    for factor in (.9,.75):
        yield f'bias16_{str(factor).replace(".","p")}',{16:g.model.array(16)*factor},f'Scale second-layer bias by {factor}.'
    for alpha,gain in ((.1,1.1),(.2,1.2),(.3,1.3)):
        w,_=spectral(g,alpha,gain)
        yield f'spectrum_{alpha}_{gain}',{13:w},f'Additional spectral flattening alpha {alpha}, gain {gain}, relative to v1.'
    w,_=spectral(g,.1,1.1)
    yield 'spectrum_bias90',{13:w,14:g.model.array(14)*.9},'Combine mild additional spectral flattening with first bias ×0.9.'
    for amount in (.25,.5,1):
        w=g.model.array(13).copy();a=w[:,1:3,1:3,:].copy()
        w[:,1:3,1:3,:]=(1-amount)*a+amount*a[:,:,::-1,:]
        yield f'coarse_reflection_{amount}',{13:w},f'Blend first 2×2 feature lattice with its horizontal reflection at strength {amount}; pose hypothesis, not an exact output mirror.'
    edits={}
    for wt,bt,s,pad,wi,bi in g.layers:
        w=g.model.array(wi).copy()
        if wi==13:w[:,1:3,1:3,:]=w[:,1:3,1:3,:][:,:,::-1,:]
        elif w.shape[2]>1:w=w[:,:,::-1,:].copy()
        else:continue
        edits[wi]=w
    yield 'naive_kernel_reflection',edits,'Reflect all active spatial kernels; expected to fail exact reflection due to asymmetric cropping.'
    e={k:v.copy() for k,v in edits.items()}
    e[15]=g.model.array(15)[:,:,[1,0,2],:].copy()
    yield 'phase_kernel_reflection',e,'Use phase swap [1,0,2] in the 3×3/stride-2 layer; approximate mirror hypothesis.'

def run():
    event('experiment_start',source_sha256=sha(SOURCE),script_sha256=sha(__file__),seed=88421,count=128)
    shutil.copy2(__file__,ROOT/'source'/('run_experiments.'+sha(__file__)[:12]+'.py'))
    log('Step 1 Declare hypotheses and selection data','All candidates branch from v1. Development inputs use seed 88421, 128 standard-normal vectors. Bias changes test whether thresholding restricts mixtures. Spectral edits test underrepresented latent directions. Coarse and kernel reflections test geometry; they must be compared to a true pixel reflection before calling them mirrors. Repetition is also measured after matching reflected images, so mirroring cannot earn false novelty credit. No optimizer, gradients or external image data are used.')
    g=Generator(str(SOURCE)); rng=np.random.default_rng(88421);z=rng.standard_normal((128,512)).astype('float32')
    base_y=samples(g,z)
    all_results=[]
    for idx,(name,edits,hypothesis) in enumerate(candidates(g),1):
        start=time.time(); folder=ROOT/'checkpoints'/f'{idx:03}_{name}'
        if (folder/'manifest.json').exists():
            all_results.append(json.loads((folder/'manifest.json').read_text()))
            event('resume_completed_checkpoint',name=name,checkpoint=str(folder))
            continue
        folder.mkdir(exist_ok=True)
        np.savez_compressed(folder/'edits.npz',**compact(edits))
        restored=load_edits(folder/'edits.npz',g)
        for k in edits:assert np.array_equal(edits[k],restored[k])
        y=samples(Generator(str(SOURCE),edits=restored),z)
        result={'name':name,'checkpoint':str(folder.relative_to(ROOT)),'hypothesis':hypothesis,'source_sha256':sha(SOURCE),'patch_sha256':sha(folder/'edits.npz'),'seed':88421,'count':128,**metrics(y),**repetition(y),'rms_to_exact_mirror':float(np.sqrt(np.mean((y-base_y[:,:,::-1,:])**2))),'rms_to_base':float(np.sqrt(np.mean((y-base_y)**2))),'finite':bool(np.isfinite(y).all())}
        assert result['finite'] and y.shape==base_y.shape
        (folder/'manifest.json').write_text(json.dumps(result,indent=2))
        grid(y[:16],folder/'preview.png',labels=[f'{name} {i}' for i in range(16)],cols=8)
        np.savez_compressed(folder/'preview.npz',images=y[:16],latents=z[:16])
        all_results.append(result)
        event('candidate_checkpoint',name=name,checkpoint=result['checkpoint'],seconds=round(time.time()-start,2),metrics=result)
    (ROOT/'evaluation/development_metrics.json').write_text(json.dumps(all_results,indent=2))
    # Traversals explain what the weight-derived leading directions control.
    w=g.model.array(13)[:,1:3,1:3,:].reshape(-1,512).astype('float64')
    val,vec=np.linalg.eigh(w.T@w);pcs=vec[:,::-1]
    zz=np.zeros((8*7,512),dtype='float32');labels=[]
    levels=[-3,-2,-1,0,1,2,3]
    for k in range(8):
        for j,a in enumerate(levels):zz[k*7+j]=pcs[:,k]*a;labels.append(f'PC{k+1} {a:+d}')
    grid(samples(g,zz),ROOT/'figures/principal_direction_traversals.png',labels=labels,cols=7)
    np.savez_compressed(ROOT/'evaluation/principal_directions.npz',eigenvalues=val[::-1],vectors=pcs,latents=zz)
    log('Step 2 Checkpoint candidate edits and traversals','Saved every candidate as a compressed patch with source hash, patch hash, hypothesis, full development metrics and the first 16 images in seed order. Saved leading latent principal-direction traversals at coefficients −3 through +3. Results are in evaluation/development_metrics.json. Selection and visual review happen after this checkpoint; no holdout images have been evaluated yet.')
    event('experiment_complete',candidates=len(all_results),source_sha256=sha(SOURCE))

if __name__=='__main__':run()
