"""Analytically initialize the new spatial branch and phase-offset layers.

No training or optimizer. Every candidate branches from the frozen v1 compact
checkpoint. Selection uses declared guards, with unseen evaluation later.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import datetime,hashlib,json,shutil,time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from round3.restructure import RestructuredGenerator
from generate import grid
from experiment import metrics
from round2.run_experiments import repetition

ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent/'round3/checkpoints/001_compact_identity/model_compact_identity.pt'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def event(name,**kw):
    r={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'event':name,**kw}
    with (ROOT/'logs/events.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
    print(json.dumps(r),flush=True)

class RefinedGenerator(nn.Module):
    def __init__(self,smooth32=0.0,offsets=None):
        super().__init__();self.base=RestructuredGenerator()
        self.base.load_state_dict(torch.load(PARENT,weights_only=True))
        self.register_buffer('phase32',torch.zeros(1,64,2,2))
        self.register_buffer('phase64',torch.zeros(1,64,2,2))
        if offsets:
            for name,value in offsets.items():getattr(self,name).copy_(torch.as_tensor(value))
        if smooth32:
            with torch.no_grad():
                w=self.base.refiner.conv2.weight;w.zero_()
                for c in range(64):
                    w[c,c,1,1]=-smooth32
                    for y,x in [(0,1),(1,0),(1,2),(2,1)]:w[c,c,y,x]=smooth32/4
        for p in self.parameters():p.requires_grad_(False)
    @staticmethod
    def add_phase(x,offset):return x+offset.repeat(1,1,x.shape[2]//2,x.shape[3]//2)
    def forward(self,z,trace=False):
        b=self.base;traces={}
        x=F.leaky_relu(b.stem(z).reshape(-1,1024,2,2),.3)
        x=F.leaky_relu(b.to4(x),.3)
        for k,layer in enumerate(b.upsamplers):
            x=F.leaky_relu(layer(x),.3)
            if k==2:
                x=b.refiner(x);traces['phase32']=x
                x=self.add_phase(x,self.phase32)
            if k==3:
                traces['phase64']=x
                x=self.add_phase(x,self.phase64)
        for k,layer in enumerate(b.color):
            x=layer(x);x=F.leaky_relu(x,.3) if k<2 else torch.sigmoid(x)
        y=x.permute(0,2,3,1)
        return (y,traces) if trace else y

@torch.inference_mode()
def sample(g,z):
    return np.concatenate([g(torch.from_numpy(z[i:i+8])).numpy() for i in range(0,len(z),8)])

@torch.inference_mode()
def calibrate(g,z):
    pieces={'phase32':[],'phase64':[]}
    for i in range(0,len(z),8):
        _,acts=g(torch.from_numpy(z[i:i+8]),trace=True)
        for name,x in acts.items():
            smooth=F.avg_pool2d(F.pad(x,(2,2,2,2),mode='reflect'),5,stride=1)
            r=(x-smooth)[:,:,4:-4,4:-4]
            phase=torch.stack([r[:,:,dy::2,dx::2].mean((0,2,3)) for dy in range(2) for dx in range(2)]).T.reshape(1,64,2,2)
            pieces[name].append(phase.numpy())
    return {k:-(a:=np.mean(v,axis=0))+a.mean((2,3),keepdims=True) for k,v in pieces.items()}

def metrics_pair(base,y):
    r={**metrics(y),**repetition(y)}
    r['paired_rgb_rms']=float(np.sqrt(np.mean((base-y)**2)))
    r['max_channel_mean_shift']=float(np.max(np.abs(base.mean((0,1,2))-y.mean((0,1,2)))))
    return r

def configs():
    yield 'baseline',0,0,0
    for s in [.025,.05,.1]:yield f'neighbor32_{s}',s,0,0
    for s in [.25,.5,.75]:yield f'phase64_{s}',0,0,s
    for s in [.25,.5,.75]:yield f'phase32and64_{s}',0,s,s
    yield 'neighbor32_phase64',.025,0,.5

def main():
    for d in ['logs','checkpoints','evaluation','figures','source','candidates']:(ROOT/d).mkdir(parents=True,exist_ok=True)
    assert sha(PARENT)=='73c66d02ae1a51b8cbf2d62bd9bb189a71c41f5ea8a862e1d960cb1919688777'
    shutil.copy2(__file__,ROOT/'source/refine_structure.frozen.py')
    plan={'calibration_seed':88500,'calibration_count':128,'development_seed':88501,'development_count':128,'holdout_seeds':[88502,88503],'holdout_count_per_distribution':512,'quality_guards':{'phase4_reduction_min':.1,'edge_retention_min':.95,'nearest_shape_retention_min':.95,'paired_rgb_rms_max':.03,'channel_mean_shift_max':.01,'clipped_fraction_increase_max':.002},'training':False,'parent_sha256':sha(PARENT)}
    (ROOT/'evaluation/plan.json').write_text(json.dumps(plan,indent=2));event('plan_frozen',plan=plan)
    g=RefinedGenerator().eval()
    calibration=np.random.default_rng(88500).standard_normal((128,512)).astype('float32')
    offsets=calibrate(g,calibration)
    np.savez_compressed(ROOT/'evaluation/calibration_offsets.npz',**offsets)
    z=np.random.default_rng(88501).standard_normal((128,512)).astype('float32');base=sample(g,z)
    baseline=metrics_pair(base,base);results=[];pics=[];labels=[]
    for idx,(name,s,p32,p64) in enumerate(configs(),1):
        folder=ROOT/'checkpoints'/f'{idx:03}_{name}';folder.mkdir(exist_ok=False)
        delta={k:offsets[k]*v for k,v in [('phase32',p32),('phase64',p64)]}
        model=RefinedGenerator(smooth32=s,offsets=delta).eval();y=sample(model,z)
        m=metrics_pair(base,y)
        checks={'phase4':m['phase_rms']['4']<=baseline['phase_rms']['4']*.9,'edge':m['edge_energy']>=baseline['edge_energy']*.95,'shape':m['mirror_invariant_nearest_shape_distance']>=baseline['mirror_invariant_nearest_shape_distance']*.95,'paired_rms':m['paired_rgb_rms']<=.03,'mean_shift':m['max_channel_mean_shift']<=.01,'clipping':m['clipped_fraction']<=baseline['clipped_fraction']+.002}
        config={'name':name,'smooth32':s,'phase32_strength':p32,'phase64_strength':p64}
        np.savez_compressed(folder/'edits.npz',smooth32=np.array(s),**delta)
        r={'config':config,'metrics':m,'guards':checks,'eligible':all(checks.values()),'patch_sha256':sha(folder/'edits.npz'),'checkpoint':str(folder.relative_to(ROOT))}
        (folder/'manifest.json').write_text(json.dumps(r,indent=2))
        np.savez_compressed(folder/'preview.npz',images=y[:16],latents=z[:16])
        grid(y[:16],folder/'preview.png',labels=[f'{name} {i}' for i in range(16)])
        pics.extend(y[:6]);labels.extend([f'{name} {i}' for i in range(6)])
        results.append(r);event('candidate_evaluated',result=r)
    grid(np.array(pics),ROOT/'figures/development.png',labels=labels,cols=6)
    (ROOT/'evaluation/development.json').write_text(json.dumps(results,indent=2))
    event('development_complete',eligible=[r['config']['name'] for r in results if r['eligible']])

if __name__=='__main__':main()
