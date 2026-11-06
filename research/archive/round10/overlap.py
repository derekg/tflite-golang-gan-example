"""Analytical overlapping polyphase projections; no training or external data.

Run from the repository with the Torch Python. Old artifacts are immutable.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import datetime, hashlib, json, shutil, subprocess
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from round9.spatial_cleanup import SpatialGenerator, validate
from round8.selective_detail import noise, measurements
from round4.refine_structure import sample
from round4.export_compact import Graph
from round5.combine import load_model
from round6.upscale import head
from inspect_model import Model

ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent/'round9/candidates/model_spatial_experimental.tflite'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def event(event_name,**kw):
    v=dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),event=event_name,**kw)
    with (ROOT/'logs/events.jsonl').open('a') as f:f.write(json.dumps(v)+'\n')
    print(json.dumps(v),flush=True)
def parent():
    cfg=json.loads((ROOT.parent/'round9/evaluation/result.json').read_text())['chosen_config']
    ranks=json.loads((ROOT.parent/'round9/evaluation/channel_ranking.json').read_text())
    g=SpatialGenerator(cfg,ranks).eval()
    g.load_state_dict(torch.load(PARENT.with_suffix('.pt'),weights_only=True))
    return g

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

class OverlapGenerator(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.model=parent();self.cfg=cfg
        for p in self.parameters():p.requires_grad_(False)
    def forward(self,z,trace=False):
        g=self.model;p=g.parent;b=p.base;traces={}
        x=F.leaky_relu(b.stem(z).reshape(-1,1024,2,2),.3)
        x=F.leaky_relu(b.to4(x),.3)
        for k,layer in enumerate(b.upsamplers):
            raw=F.pixel_shuffle(layer.projection(x),2)
            if k in self.cfg['stages']:
                w=layer.projection.weight;co=w.shape[0]//4
                common=F.conv2d(x,w.reshape(co,4,w.shape[1],1,1).mean(1))
                nearest=F.interpolate(common,scale_factor=2,mode='nearest')
                smooth=F.interpolate(common,scale_factor=2,mode='bilinear',align_corners=False)
                raw=raw+self.cfg['a']*(smooth-nearest)-self.cfg['b']*(raw-nearest)
            x=F.leaky_relu(raw+layer.bias[None,:,None,None],.3)
            if k==2:x=p.add_phase(b.refiner(x),p.phase32)
            if k==3:x=p.add_phase(x,p.phase64)
            if trace:traces[str(x.shape[-1])]=x.detach().clone()
        x=F.conv2d(F.pad(x,(1,1,1,1),mode='replicate'),g.spatial_kernel,groups=64)
        if self.cfg.get('blur',0):
            s=self.cfg['blur'];x=(1-s)*x+s*F.avg_pool2d(F.pad(x,(1,1,1,1),mode='replicate'),3,1)
        if trace:traces['128_after_filter']=x.detach().clone()
        for k,layer in enumerate(b.color):
            x=layer(x);x=F.leaky_relu(x,.3) if k<2 else torch.sigmoid(x)
        y=x.permute(0,2,3,1)
        return (y,traces) if trace else y

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

def stats(base,y):
    m=measurements(base,y);m['edge_grid']=edge_grid(y)
    a=torch.from_numpy(base).permute(0,3,1,2);b=torch.from_numpy(y).permute(0,3,1,2)
    m['coarse_rgb_rms']=float((F.avg_pool2d(a,8)-F.avg_pool2d(b,8)).square().mean().sqrt())
    return m

def guards(m,b):
    return dict(edge_grid=m['edge_grid']['score']<=b['edge_grid']['score']*.85,
                phase4=m['mean_per_image_grid4']<=b['mean_per_image_grid4']*.80,
                broad_edges=m['smoothed_edge_energy']>=b['smoothed_edge_energy']*.95,
                shape=m['mirror_invariant_nearest_shape_distance']>=b['mirror_invariant_nearest_shape_distance']*.97,
                coarse_drift=m['coarse_rgb_rms']<=.02,rgb_drift=m['paired_rgb_rms']<=.035,
                color=m['max_channel_mean_shift']<=.008,
                clipping=m['clipped_fraction']<=b['clipped_fraction']+.002)

def export(g,path):
    assert not g.cfg.get('blur',0),'Blur controls are not exported.'
    spec=json.loads(Path(str(PARENT)+'.graph.json').read_text());m=Model(PARENT);graph=Graph();replacements={}
    for k in g.cfg['stages']:
        w=expanded(g.model.parent.base.upsamplers[k].projection.weight,g.cfg['a'],g.cfg['b']).numpy()
        co4,ci,kh,kw=w.shape
        replacements[f'upsample{k}/weights']=w.reshape(co4//4,4,ci,kh,kw).transpose(1,0,3,4,2).reshape(co4,kh,kw,ci)
    for i,t in enumerate(spec['tensors']):
        value=replacements.get(t['name'],m.array(i));shape=list(value.shape) if t['name'] in replacements else t['shape']
        assert graph.tensor(t['name'],shape,value,np.float32 if t['type']==0 else np.int32,t['dynamic'])==i
    for op in spec['ops']:
        args=op['inputs'].copy();opts={int(k):tuple(v) for k,v in op['opts'].items()}
        if op['code']==3 and spec['tensors'][args[1]]['name'] in replacements:
            name=spec['tensors'][args[1]]['name'].replace('/weights','/overlap')
            n,h,w,c=graph.tensors[args[0]]['shape']
            pads=graph.constant(name+'/paddings',[[0,0],[1,1],[1,1],[0,0]],np.int32)
            padded=graph.tensor(name+'/edge_pad',[n,h+2,w+2,c],dynamic=True)
            graph.op(100,[args[0],pads],padded,77,{0:('b',1)});args[0]=padded;opts[0]=('b',1)
        graph.op(op['code'],args,op['outputs'][0],op['type'],opts,op['version'])
    info=graph.build(path,int(m.graph.vector(2)[0]));write(str(path)+'.graph.json',dict(export=info,tensors=graph.tensors,ops=graph.ops))
    new=Model(path);changed=[]
    for i,t in enumerate(m.tensors):
        a,c=m.array(i),new.array(i)
        if a is not None and not np.array_equal(a,c):changed.append(t.string(3))
    assert set(changed)==set(replacements)
    return dict(**info,changed_inherited_tensors=changed)

def configs():
    yield dict(name='baseline',stages=[],a=0.,b=0.)
    for k in [1,2,3,4]:
        for b in [0.,.35]:yield dict(name=f'stage{2**(k+3)}_a0.35_b{b:g}',stages=[k],a=.35,b=b)
    for stages,label in [([2,3,4],'late3'),([1,2,3,4],'last4')]:
        for a,b in [(.15,.15),(.35,.15),(.35,.35),(.6,.3),(.6,.6)]:
            yield dict(name=f'{label}_a{a:g}_b{b:g}',stages=stages,a=a,b=b)
    yield dict(name='phase_only_0.35',stages=[2,3,4],a=0.,b=.35)
    for s in [.15,.35,.6]:yield dict(name=f'blur_control_{s:g}',stages=[],a=0.,b=0.,blur=s)

@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    for d in ['evaluation','checkpoints','logs','candidates','figures']:(ROOT/d).mkdir(parents=True,exist_ok=True)
    assert not (ROOT/'evaluation/result.json').exists(),'Preserve finished experiments.'
    protected={str(p):sha(p) for p in [PARENT,PARENT.with_suffix('.pt'),ROOT.parent/'model.tflite',ROOT.parent/'round8/candidates/model_selective_experimental.tflite',ROOT.parent/'portrait-morph/dist/index.html']}
    cs=list(configs());plan=dict(configs=cs,development=dict(gaussian_seed=89100,uniform_seed=89101,count_each=32),holdout=dict(gaussian_seed=89102,uniform_seed=89103,count_each=128),training=False,gradients=False,external_data=False,formula='Wprime_p=(Wp-b*(Wp-M))*delta+a*M*(Bp-delta); M=mean_p(Wp); Bp=half-pixel bilinear phase taps',criteria=dict(edge_grid_ratio_max=.85,per_image_phase4_ratio_max=.8,broad_edge_ratio_min=.95,nearest_shape_ratio_min=.97,coarse_rgb_rms_max=.02,rgb_rms_max=.035,color_shift_max=.008,clip_increase_max=.002),selection='Pass both distributions; minimize worst normalized edge-grid ratio. If none pass, diagnostic minimizes grid ratio among structure-safe non-blur candidates; no promotion.',protected=protected)
    if (ROOT/'evaluation/plan.json').exists():
        assert json.loads((ROOT/'evaluation/plan.json').read_text())==plan
        shutil.copy2(__file__,ROOT/'evaluation/implementation.resume.py')
        event('resume',reason='Fixed event logger argument collision; plan and candidate formulas unchanged.')
    else:
        write(ROOT/'evaluation/plan.json',plan);shutil.copy2(__file__,ROOT/'evaluation/implementation.frozen.py');event('plan_frozen',candidates=len(cs))
    # Independent algebra test: direct interpolation vs exported 3x3 kernels,
    # including borders, DC phase means, and exact zero-strength behavior.
    algebra=[]
    g=OverlapGenerator(cs[0]).eval()
    for k in [1,2,3,4]:
        w=g.model.parent.base.upsamplers[k].projection.weight;co=w.shape[0]//4;ci=w.shape[1]
        x=torch.from_numpy(noise(89120+k,2)[:,:ci]).reshape(2,ci,1,1).expand(2,ci,7,9).clone()
        x+=torch.randn(x.shape,generator=torch.Generator().manual_seed(89130+k))*.1
        for a,b in [(0.,0.),(.35,.15),(1.,1.)]:
            wt=expanded(w,a,b);raw=F.pixel_shuffle(F.conv2d(x,w),2)
            common=F.conv2d(x,w.reshape(co,4,ci,1,1).mean(1))
            nnx=F.interpolate(common,scale_factor=2,mode='nearest')
            expected=raw+a*(F.interpolate(common,scale_factor=2,mode='bilinear',align_corners=False)-nnx)-b*(raw-nnx)
            actual=F.pixel_shuffle(F.conv2d(F.pad(x,(1,1,1,1),mode='replicate'),wt),2)
            err=float((actual-expected).abs().max());assert err<2e-5,err
            dc=float((wt.reshape(co,4,ci,3,3).sum((3,4)).mean(1)-w.reshape(co,4,ci).mean(1)).abs().max());assert dc<1e-6
            algebra.append(dict(stage=k,a=a,b=b,max_error=err,phase_mean_dc_error=dc))
    z=noise(89121,3);identity=float(np.max(np.abs(sample(g,z)-sample(parent(),z))));assert identity<2e-6
    write(ROOT/'evaluation/algebra.json',dict(identity_error=identity,tests=algebra));event('algebra_passed',identity_error=identity)
    inputs={key:noise(seed,32,key=='uniform') for key,seed in [('gaussian',89100),('uniform',89101)]}
    bases={key:sample(g,z) for key,z in inputs.items()};bm={key:stats(y,y) for key,y in bases.items()}
    # Original/round8/round9 edge diagnosis uses matched noise.
    from generate import Generator
    diagnostic={};original=Generator(ROOT.parent/'model.tflite');r8=load_model(ROOT.parent/'round8/candidates/model_selective_experimental.pt')
    for key,z in inputs.items():
        diagnostic[key]={}
        for name,model in [('original',original),('round8',r8),('round9',g)]:
            y=np.concatenate([model(zz) for zz in np.array_split(z,8)]) if name=='original' else sample(model,z)
            diagnostic[key][name]=edge_grid(y)
    write(ROOT/'evaluation/edge_diagnosis.json',diagnostic)
    # Feature-space edge phase profiles at each scale (channels are not RGB).
    _,acts=g(torch.from_numpy(inputs['gaussian'][:8]),trace=True);features={}
    for key,x in acts.items():
        h=x.shape[-1];arr=x.permute(0,2,3,1).numpy();p=[]
        for axis in [1,2]:
            e=np.abs(np.diff(arr,axis=axis)).mean(3);profile=e.mean(2 if axis==1 else 1)
            folds=np.stack([profile[:,i::2].mean(1) for i in range(2)],1)
            p.append(float((folds.std(1)/np.maximum(folds.mean(1),1e-8)).mean()))
        features[key]=dict(shape=list(x.shape),edge_phase2_cv_axes=p)
    write(ROOT/'evaluation/feature_diagnosis.json',features)
    records=[]
    for idx,cfg in enumerate(cs):
        cp=ROOT/'checkpoints'/f'{idx:03}_{cfg["name"]}'
        if (cp/'manifest.json').exists():
            rec=json.loads((cp/'manifest.json').read_text());assert rec['config']==cfg and rec['patch_sha256']==sha(cp/'weights.npz')
            records.append(rec);event('checkpoint_restored',name=cfg['name']);continue
        model=OverlapGenerator(cfg).eval();res={};preview={}
        for key,z in inputs.items():
            y=sample(model,z);m=stats(bases[key],y);res[key]=dict(metrics=m,guards=guards(m,bm[key]));preview[key]=y[:4]
        eligible=all(all(v['guards'].values()) for v in res.values())
        structural=all(all(v for k,v in r['guards'].items() if k not in ['edge_grid','phase4']) for r in res.values())
        score=max(r['metrics']['edge_grid']['score']/bm[key]['edge_grid']['score'] for key,r in res.items())
        cp.mkdir(exist_ok=False)
        patches={str(k):expanded(model.model.parent.base.upsamplers[k].projection.weight,cfg['a'],cfg['b']).numpy() for k in cfg['stages']}
        np.savez_compressed(cp/'weights.npz',**patches);np.savez_compressed(cp/'preview.npz',**preview)
        rec=dict(config=cfg,statistics=res,eligible=eligible,structure_safe=structural,score=score,checkpoint=str(cp.relative_to(ROOT)),patch_sha256=sha(cp/'weights.npz'))
        records.append(rec);write(cp/'manifest.json',rec);event('candidate',name=cfg['name'],eligible=eligible,structure_safe=structural,grid_ratio=score)
    write(ROOT/'evaluation/development.json',dict(baseline=bm,candidates=records))
    pool=[r for r in records if r['eligible'] and r['config']['stages'] and not r['config'].get('blur',0)]
    diagnostic=not bool(pool)
    if not pool:pool=[r for r in records if r['structure_safe'] and r['config']['stages']]
    assert pool,'No structure-safe overlapping model: inspect checkpoints.'
    chosen=min(pool,key=lambda r:r['score']);write(ROOT/'evaluation/frozen_selection.json',dict(chosen=chosen,diagnostic=diagnostic));event('selection_frozen',config=chosen['config'],diagnostic=diagnostic)
    cfg=chosen['config'];model=OverlapGenerator(cfg).eval();holdout={}
    for key,seed in [('gaussian',89102),('uniform',89103)]:
        z=noise(seed,128,key=='uniform');base=sample(g,z);y=sample(model,z);b=stats(base,base);m=stats(base,y)
        holdout[key]=dict(baseline=b,candidate=m,guards=guards(m,b));event('holdout',distribution=key,guards=holdout[key]['guards'])
        np.savez_compressed(ROOT/'figures'/f'{key}_holdout.npz',latents=z[:8],round9=base[:8],round10=y[:8])
    write(ROOT/'evaluation/holdout.json',holdout)
    p=ROOT/'candidates/model_overlap_experimental.tflite';info=export(model,p);v=validate(p,model)
    torch.save(dict(config=cfg,state_dict=model.state_dict()),p.with_suffix('.pt'))
    import round6.upscale as up
    up.PARENT=p;q=ROOT/'candidates/model_overlap_experimental_2x.tflite';qi=up.export(q);qv=validate(q,model,True)
    write(ROOT/'evaluation/graph_validation.json',dict(native128=v,upscale256=qv))
    for key,seed in [('gaussian',88920),('uniform',88921)]:
        z=noise(seed,8,key=='uniform')
        np.savez_compressed(ROOT/'figures'/f'{key}_comparison.npz',latents=z,original=original(z),round8=sample(r8,z),round9=sample(g,z),round10=sample(model,z))
    for path,h in protected.items():assert sha(path)==h
    result=dict(config=cfg,criteria_passed=not diagnostic and all(all(h['guards'].values()) for h in holdout.values()),promoted=False,visual_review_pending=True,models=[dict(path=str(p),sha256=sha(p),export=info),dict(path=str(q),sha256=sha(q),export=qi)],protected_files=protected,native_litert_tested=False,training=False,external_data=False)
    write(ROOT/'evaluation/result.json',result);cp=ROOT/'checkpoints/023_delivery';cp.mkdir(exist_ok=False)
    shutil.copy2(__file__,cp/'overlap.py');write(cp/'manifest.json',result);event('delivery_complete',models=result['models'])

if __name__=='__main__':main()
