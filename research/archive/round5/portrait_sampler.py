"""Generate portraits with artifact screening and persistent recent-output memory.

The model consumes random noise. This policy selects existing outputs; it does
not learn new identities. Uses the decoded graph backend pending native runtime
acceptance. History stores descriptors, not generated images or latent vectors.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,hashlib,json,secrets
import numpy as np
import torch
import torch.nn.functional as F
from generate import grid
from round4.check_graph import GraphRunner
from round2.run_experiments import features,repetition

ROOT=Path(__file__).resolve().parent
DEFAULT_MODEL=ROOT/'candidates/model_clean_diverse.tflite'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def noise(rng,count,distribution):
    if distribution=='gaussian':z=rng.standard_normal((count,512))
    elif distribution=='uniform':z=rng.uniform(-1,1,(count,512))
    else:raise ValueError(distribution)
    return z.astype('float32')

def artifacts(y):
    t=torch.from_numpy(y).permute(0,3,1,2)
    smooth=F.avg_pool2d(F.pad(t,(2,2,2,2),mode='reflect'),5,stride=1)
    r=(t-smooth)[:,:,8:-8,8:-8]
    phase=torch.stack([r[:,:,dy::4,dx::4].mean((2,3)) for dy in range(4) for dx in range(4)],dim=1)
    phase=phase-phase.mean(1,keepdim=True)
    return {'grid4':torch.sqrt((phase**2).mean((1,2))).numpy(),'clipped':((y<.01)|(y>.99)).mean((1,2,3)),'coarse_contrast':F.avg_pool2d(t,8).flatten(1).std(1,unbiased=False).numpy()}

class PortraitSampler:
    def __init__(self,model=DEFAULT_MODEL,history=None,screen=None,distribution='gaussian',max_history=128,min_distance=.10):
        self.model=Path(model);self.model_sha256=sha(model);self.runner=GraphRunner(model);self.distribution=distribution
        if max_history<1 or min_distance<0:raise ValueError('Require a positive history size and nonnegative distance threshold')
        self.max_history=max_history;self.min_distance=min_distance;self.history_path=Path(history) if history else None
        self.history=np.empty((0,768),dtype='float32');self.history_mirrored=self.history.copy();self.history_reset_reason=None
        if self.history_path and self.history_path.exists():
            with np.load(self.history_path,allow_pickle=False) as saved:
                if str(saved['model_sha256'])==self.model_sha256 and str(saved['distribution'])==distribution:
                    self.history=saved['features'].astype('float32')[-max_history:];self.history_mirrored=saved['mirrored_features'].astype('float32')[-max_history:]
                    assert self.history.shape==self.history_mirrored.shape and self.history.shape[1:]==(768,)
                else:self.history_reset_reason='Model or noise distribution changed; old history was not reused.'
        if screen is None:screen=ROOT/'evaluation/artifact_screen.json'
        record=json.loads(Path(screen).read_text())
        if record['model_sha256']!=self.model_sha256:raise ValueError('Artifact-screen calibration belongs to a different model; recalibrate first.')
        self.limits=record['distributions'][distribution]['limits']
    def render(self,z):return np.concatenate([self.runner(z[i:i+8]) for i in range(0,len(z),8)])
    def save_history(self):
        if not self.history_path:return
        self.history_path.parent.mkdir(parents=True,exist_ok=True)
        # Replace only this sampler's recoverable descriptor state, never a model.
        temp=self.history_path.with_suffix('.pending.npz')
        np.savez_compressed(temp,features=self.history,mirrored_features=self.history_mirrored,model_sha256=np.array(self.model_sha256),distribution=np.array(self.distribution))
        temp.replace(self.history_path)
    def generate(self,count=16,pool=128,seed=None):
        if not 1<=count<=pool:raise ValueError('Require 1 <= count <= pool')
        if seed is None:seed=secrets.randbits(63)
        z=noise(np.random.default_rng(seed),pool,self.distribution);y=self.render(z);stats=artifacts(y)
        eligible=(stats['grid4']<=self.limits['grid4_max'])&(stats['clipped']<=self.limits['clipped_max'])&(stats['coarse_contrast']>=self.limits['coarse_contrast_min'])
        a=features(y);b=features(y[:,:,::-1,:].copy());h=self.history.copy();hm=self.history_mirrored.copy();selected=[];events=[]
        if len(h):nearest=np.minimum(1-a@h.T,1-a@hm.T).min(1)
        else:nearest=np.full(pool,np.inf,dtype='float32')
        available=np.ones(pool,dtype=bool)
        for _ in range(count):
            screened=np.where(available&eligible)[0];screen_fallback=len(screened)==0
            candidates=np.where(available)[0] if screen_fallback else screened
            novel=candidates[nearest[candidates]>=self.min_distance]
            exhausted=len(novel)==0
            if len(h)==0 and len(selected)==0:
                d=np.minimum(1-a[candidates]@a[candidates].T,1-a[candidates]@b[candidates].T)
                idx=int(candidates[np.argmin(d.mean(1))]);distance=None
            else:
                choices=novel if len(novel) else candidates
                idx=int(choices[np.argmax(nearest[choices])]);distance=float(max(nearest[idx],0))
            selected.append(idx);events.append({'pool_index':idx,'nearest_recent_or_batch_distance':distance,'artifact_screen_passed':bool(eligible[idx]),'artifact_screen_fallback':screen_fallback,'novelty_threshold_unmet':exhausted})
            available[idx]=False;nearest=np.minimum(nearest,np.minimum(1-a@a[idx],1-a@b[idx]))
        ids=np.array(selected,dtype='int32')
        self.history=np.concatenate([h,a[ids]])[-self.max_history:];self.history_mirrored=np.concatenate([hm,b[ids]])[-self.max_history:];self.save_history()
        report={'model_sha256':self.model_sha256,'distribution':self.distribution,'seed':seed,'count':count,'pool':pool,'eligible_pool_count':int(eligible.sum()),'history_size_before':len(h),'history_size_after':len(self.history),'history_reset_reason':self.history_reset_reason,'min_shape_distance':self.min_distance,'selected':events,'first_random_batch':repetition(y[:count]) if count>1 else None,'selected_batch':repetition(y[ids]) if count>1 else None,'raw_artifact_averages':{k:float(v[:count].mean()) for k,v in stats.items()},'selected_artifact_averages':{k:float(v[ids].mean()) for k,v in stats.items()},'backend':'decoded TFLite graph with Torch arithmetic','native_litert':False,'limits':'Coarse shape distances do not identify people. Screening is an engineering proxy and can reject valid portraits or retain defects. A finite pool and limited model cannot guarantee unseen identities.'}
        return y[ids],z[ids],report,y[:count]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default=str(DEFAULT_MODEL));p.add_argument('--count',type=int,default=16);p.add_argument('--pool',type=int,default=128);p.add_argument('--seed',type=int,default=None,help='Omit for fresh randomness; an explicit seed reproduces a candidate pool');p.add_argument('--distribution',choices=['gaussian','uniform'],default='gaussian');p.add_argument('--history',default='round5/runtime/recent_history.npz');p.add_argument('--screen',default='round5/evaluation/artifact_screen.json');p.add_argument('--output',default='round5/figures/portraits.png');args=p.parse_args()
    sampler=PortraitSampler(args.model,args.history,args.screen,args.distribution);y,z,report,_=sampler.generate(args.count,args.pool,args.seed)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);grid(y,out,labels=[str(i) for i in range(len(y))])
    np.savez_compressed(out.with_suffix('.npz'),images=y,latents=z)
    out.with_suffix('.json').write_text(json.dumps(report,indent=2));print(json.dumps({'output':str(out),'history_size':len(sampler.history),'threshold_fallbacks':sum(e['novelty_threshold_unmet'] for e in report['selected'])}))

if __name__=='__main__':main()
