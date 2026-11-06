"""Add a channelwise spatial mixer before RGB; fixed analytical edits, no training."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import datetime,hashlib,json,shutil,subprocess
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from inspect_model import Model
from round5.combine import load_model
from round4.refine_structure import sample
from round4.export_compact import Graph
from round8.selective_detail import measurements,noise
from round6.upscale import head

ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent/'round8/candidates/model_selective_experimental.tflite'
PARENT_PT=PARENT.with_suffix('.pt')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def event(name,**fields):
    r=dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),event=name,**fields)
    with (ROOT/'logs/events.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
    print(json.dumps(r),flush=True)

@torch.inference_mode()
def rank_channels():
    g=load_model(PARENT_PT);features=[]
    hook=g.base.color[0].register_forward_pre_hook(lambda mod,args:features.append(args[0]))
    phase=[];broad=[];std=[]
    try:
        z=np.concatenate([noise(89000,64),noise(89004,64,True)])
        for zz in np.array_split(z,16):
            g(torch.from_numpy(zz));x=features.pop()
            smooth=F.avg_pool2d(F.pad(x,(2,2,2,2),mode='reflect'),5,stride=1)
            r=(x-smooth)[:,:,8:-8,8:-8]
            p=torch.stack([r[:,:,dy::4,dx::4].mean((2,3)) for dy in range(4) for dx in range(4)],1)
            phase.append(((p-p.mean(1,keepdim=True))**2).mean(1).sqrt().mean(0).numpy())
            s=smooth[:,:,8:-8,8:-8]
            broad.append((torch.diff(s,dim=2).abs().mean((0,2,3))+torch.diff(s,dim=3).abs().mean((0,2,3))).numpy())
            std.append(x.std((0,2,3),unbiased=False).numpy())
    finally:hook.remove()
    color=[layer.weight.detach().numpy()[:,:,0,0] for layer in g.base.color]
    sensitivity=(np.abs(color[2])@np.abs(color[1])@np.abs(color[0])).sum(0)
    p,b,s=map(lambda a:np.mean(a,axis=0),[phase,broad,std])
    score=p*sensitivity
    return dict(artifact_rank=np.argsort(-score,kind='stable').tolist(),score=score.tolist(),per_image_phase4=p.tolist(),smoothed_gradient=b.tolist(),activation_std=s.tolist(),absolute_color_path_sensitivity=sensitivity.tolist())

def kernel(cfg,ranks):
    w=np.zeros((64,1,3,3),dtype='float32');w[:,0,1,1]=1
    if cfg['kind']!='layer':return w
    s=cfg['strength'];t=np.zeros((3,3),dtype='float32')
    if cfg['filter']=='cross':
        for y,x in [(0,1),(1,0),(1,2),(2,1)]:t[y,x]=.25
    else:t=np.outer([.25,.5,.25],[.25,.5,.25]).astype('float32')
    for c in ranks['artifact_rank'][:cfg['channels']]:
        w[c,0]*=1-s;w[c,0]+=s*t
    assert np.all(w>=0) and np.allclose(w.sum((1,2,3)),1,atol=1e-7)
    return w

class SpatialGenerator(nn.Module):
    def __init__(self,cfg,ranks):
        super().__init__();self.parent=load_model(PARENT_PT);self.cfg=cfg
        self.register_buffer('spatial_kernel',torch.from_numpy(kernel(cfg,ranks)))
        if cfg['kind']=='weights':
            oldranks=json.loads((ROOT.parent/'round8/evaluation/channel_rankings.json').read_text())
            with torch.no_grad():
                for i,key in [(3,'64'),(4,'128')]:
                    phase=self.parent.base.upsamplers[i].projection.weight.reshape(64,4,64,1,1)
                    for c in oldranks[key]['artifact_rank'][:cfg['channels']]:
                        phase[c].lerp_(phase[c].mean(0,keepdim=True).expand_as(phase[c]),cfg['strength'])
        for p in self.parameters():p.requires_grad_(False)
    def forward(self,z):
        b=self.parent.base
        x=F.leaky_relu(b.stem(z).reshape(-1,1024,2,2),.3)
        x=F.leaky_relu(b.to4(x),.3)
        for k,layer in enumerate(b.upsamplers):
            x=F.leaky_relu(layer(x),.3)
            if k==2:x=self.parent.add_phase(b.refiner(x),self.parent.phase32)
            if k==3:x=self.parent.add_phase(x,self.parent.phase64)
        if self.cfg['kind']=='layer':
            x=F.conv2d(F.pad(x,(1,1,1,1),mode='replicate'),self.spatial_kernel,groups=64)
        for k,layer in enumerate(b.color):
            x=layer(x);x=F.leaky_relu(x,.3) if k<2 else torch.sigmoid(x)
        return x.permute(0,2,3,1)

def checks(m,b):
    # Cleanup seeks lower periodic artifacts with retained structure. It is not
    # the round8 sharpening experiment, so these new criteria are frozen first.
    return dict(grid4=m['phase_rms']['4']<=b['phase_rms']['4']*.95,
                per_image_grid4=m['mean_per_image_grid4']<=b['mean_per_image_grid4']*.97,
                broad_edges=m['smoothed_edge_energy']>=b['smoothed_edge_energy']*.995,
                raw_edges=m['edge_energy']>=b['edge_energy']*.94,
                shape=m['mirror_invariant_nearest_shape_distance']>=b['mirror_invariant_nearest_shape_distance']*.98,
                drift=m['paired_rgb_rms']<=.02,color=m['max_channel_mean_shift']<=.005,
                clipping=m['clipped_fraction']<=b['clipped_fraction']+.002,
                grid2=m['phase_rms']['2']<=b['phase_rms']['2']*1.05,
                grid8=m['phase_rms']['8']<=b['phase_rms']['8']*1.05)

def export(g,path,force_layer=False):
    spec=json.loads(Path(str(PARENT)+'.graph.json').read_text());m=Model(PARENT);graph=Graph()
    replace={}
    if g.cfg['kind']=='weights':
        for k in [3,4]:
            w=g.parent.base.upsamplers[k].projection.weight.detach().numpy()
            replace[f'upsample{k}/weights']=w.reshape(64,2,2,64).transpose(1,2,0,3).reshape(256,1,1,64)
    for i,t in enumerate(spec['tensors']):
        assert graph.tensor(t['name'],t['shape'],replace.get(t['name'],m.array(i)),np.float32 if t['type']==0 else np.int32,t['dynamic'])==i
    added=False
    for op in spec['ops']:
        args=op['inputs'].copy()
        if spec['tensors'][op['outputs'][0]]['name']=='color0' and (g.cfg['kind']=='layer' or force_layer):
            x=args[0];pad=graph.constant('spatial128/paddings',[[0,0],[1,1],[1,1],[0,0]],np.int32)
            padded=graph.tensor('spatial128/edge_pad',[1,130,130,64],dynamic=True);graph.op(100,[x,pad],padded,77,{0:('b',1)})
            w=graph.constant('spatial128/weights',g.spatial_kernel.numpy().transpose(1,2,3,0))
            bias=graph.constant('spatial128/bias',np.zeros(64,dtype='float32'))
            y=graph.tensor('spatial128/depthwise',[1,128,128,64],dynamic=True)
            graph.op(4,[padded,w,bias],y,2,{0:('b',1),1:('i',1),2:('i',1),3:('i',1)})
            args[0]=y;added=True
        graph.op(op['code'],args,op['outputs'][0],op['type'],{int(k):tuple(v) for k,v in op['opts'].items()},op['version'])
    info=graph.build(path,int(m.graph.vector(2)[0]));write(str(path)+'.graph.json',dict(export=info,tensors=graph.tensors,ops=graph.ops))
    out=Model(path);changed=[]
    for i,t in enumerate(m.tensors):
        a,b=m.array(i),out.array(i)
        if a is not None and not np.array_equal(a,b):changed.append(t.string(3))
    assert set(changed)==set(replace)
    return dict(**info,inherited_constants_changed=changed,spatial_layer_added=added)

def validate(path,module,upscale=False):
    from round9.check_graph import GraphRunner
    tests=[]
    for name,z in [('zero',np.zeros((1,512),dtype='float32')),('gaussian',noise(89010,3)),('uniform',noise(89011,2,True))]:
        expected=sample(module,z)
        if upscale:expected=head(expected)
        for backend in ['torch','numpy']:
            y=GraphRunner(path,backend)(z);error=float(np.max(np.abs(y-expected)))
            assert y.shape==expected.shape and np.isfinite(y).all() and np.allclose(y,expected,atol=2e-5,rtol=2e-5),(backend,error)
            tests.append(dict(input=name,batch=len(z),backend=backend,shape=list(y.shape),max_error=error))
    return dict(tests=tests,native_litert_tested=False)

def main():
    torch.set_num_threads(4)
    for d in ['evaluation','checkpoints','logs','candidates','figures']:(ROOT/d).mkdir(parents=True,exist_ok=True)
    assert not (ROOT/'evaluation/plan.json').exists(),'Use a new round, never overwrite this experiment.'
    protected={str(p):sha(p) for p in [PARENT,PARENT_PT,PARENT.with_name('model_selective_experimental_2x.tflite'),ROOT.parent/'model.tflite',ROOT.parent/'round5/candidates/model_clean_diverse.tflite']}
    assert sha(PARENT)=='5f899d68b63bf194197eb0d1dcf1e98f7002d6f0b8340f816e2f20b1132e0539'
    configs=[dict(name='baseline',kind='baseline',channels=0,strength=0.,filter=None)]
    for filt in ['cross','binomial']:
        for k in [8,16,32,64]:
            for s in [.05,.1,.2,.35]:configs.append(dict(name=f'{filt}_channels{k}_strength{s:g}',kind='layer',channels=k,strength=s,filter=filt))
    for k in [8,16,32]:
        for s in [.05,.1,.2]:configs.append(dict(name=f'weights_channels{k}_strength{s:g}',kind='weights',channels=k,strength=s,filter=None))
    plan=dict(parent_sha256=sha(PARENT),training=False,external_images=False,gradients=False,optimizer=False,architecture='Replicated-edge 3x3 depthwise filter at 128x128x64, after final upsampling activation and before first RGB color projection.',initialization='Identity for all channels, then convex neighbor blending in selected artifact-ranked channels.',calibration=[dict(seed=89000,count=64,distribution='gaussian'),dict(seed=89004,count=64,distribution='uniform')],development=[dict(seed=89001,count=64,distribution='gaussian'),dict(seed=89005,count=64,distribution='uniform')],holdout=[dict(seed=89002,count=256,distribution='gaussian'),dict(seed=89003,count=256,distribution='uniform')],configs=configs,criteria=dict(fixed_grid4_max_ratio=.95,per_image_grid4_max_ratio=.97,smoothed_edge_min_ratio=.995,raw_edge_min_ratio=.94,nearest_shape_min_ratio=.98,rgb_rms_max=.02,color_shift_max=.005,clip_increase_max=.002,grid2_and8_max_ratio=1.05),selection='Must pass all criteria on BOTH development distributions. Minimize worst per-image-grid4 ratio. If none pass, choose diagnostic with fewest total failed criteria then lowest worst per-image-grid4 ratio. Freeze before holdout.',promotion='Require all independent criteria on both noise laws and visual review; do not overwrite parent.',limits='Cleanup proxies are not anatomical quality or recovered high-resolution detail.')
    write(ROOT/'evaluation/plan.json',plan);shutil.copy2(__file__,ROOT/'evaluation/implementation.frozen.py');event('plan_frozen',candidates=len(configs))
    ranks=rank_channels();write(ROOT/'evaluation/channel_ranking.json',ranks);event('rank_frozen',top16=ranks['artifact_rank'][:16])
    identity=dict(name='identity',kind='layer',channels=0,strength=0.,filter='cross')
    g=SpatialGenerator(identity,ranks).eval();z=noise(89009,3)
    a=sample(load_model(PARENT_PT),z);b=sample(g,z);identity_error=float(np.max(np.abs(a-b)));assert identity_error<2e-6
    c=ROOT/'checkpoints/000_identity';c.mkdir(exist_ok=False);p=c/'model_identity.tflite';info=export(g,p);v=validate(p,g)
    write(c/'manifest.json',dict(module_identity_max_error=identity_error,export=info,validation=v))
    # Kernel arithmetic tests independently exercise borders and channel layout.
    from round9.check_graph import GraphRunner
    runner=GraphRunner(p,'numpy');probe=np.random.default_rng(89012).random((2,7,9,64),dtype=np.float32)
    probe_cfg=dict(kind='layer',channels=16,strength=.2,filter='binomial');w=kernel(probe_cfg,ranks)
    expected=F.conv2d(F.pad(torch.from_numpy(probe.transpose(0,3,1,2).copy()),(1,1,1,1),mode='replicate'),torch.from_numpy(w),groups=64).permute(0,2,3,1).numpy()
    actual=runner.depthwise(np.pad(probe,((0,0),(1,1),(1,1),(0,0)),mode='symmetric'),w.transpose(1,2,3,0),np.zeros(64,dtype='float32'))
    assert np.max(np.abs(expected-actual))<3e-7
    write(ROOT/'evaluation/identity_checks.json',dict(module_identity_max_error=identity_error,depthwise_random_max_error=float(np.max(np.abs(expected-actual))),constant_preservation='All channel kernels nonnegative and sum to one; replicated borders preserve constants.'))
    bases={};inputs={};base_metrics={}
    for h in plan['development']:
        key=h['distribution'];inputs[key]=noise(h['seed'],h['count'],key=='uniform');bases[key]=sample(load_model(PARENT_PT),inputs[key]);base_metrics[key]=measurements(bases[key],bases[key])
    records=[]
    for i,cfg in enumerate(configs):
        g=SpatialGenerator(cfg,ranks).eval();stats={}
        for key in ['gaussian','uniform']:
            y=sample(g,inputs[key]);m=measurements(bases[key],y);stats[key]=dict(metrics=m,guards=checks(m,base_metrics[key]))
        c=ROOT/'checkpoints'/f'{i+1:03}_{cfg["name"]}';c.mkdir(exist_ok=False)
        np.savez_compressed(c/'edits.npz',spatial_kernel=g.spatial_kernel.numpy(),projection3=g.parent.base.upsamplers[3].projection.weight.detach().numpy(),projection4=g.parent.base.upsamplers[4].projection.weight.detach().numpy())
        eligible=all(all(v['guards'].values()) for v in stats.values())
        failed=sum(sum(not x for x in v['guards'].values()) for v in stats.values())
        score=max(v['metrics']['mean_per_image_grid4']/base_metrics[key]['mean_per_image_grid4'] for key,v in stats.items())
        rec=dict(config=cfg,eligible=eligible,failed_criteria=failed,worst_per_image_grid4_ratio=score,statistics=stats,checkpoint=str(c.relative_to(ROOT)),patch_sha256=sha(c/'edits.npz'))
        write(c/'manifest.json',rec);records.append(rec);event('candidate_evaluated',candidate=cfg['name'],eligible=eligible,failed=failed,worst_grid_ratio=score)
    write(ROOT/'evaluation/development.json',dict(baseline=base_metrics,candidates=records))
    eligible=[r for r in records if r['eligible']]
    chosen=min(eligible,key=lambda r:r['worst_per_image_grid4_ratio']) if eligible else min(records[1:],key=lambda r:(r['failed_criteria'],r['worst_per_image_grid4_ratio']))
    write(ROOT/'evaluation/frozen_selection.json',dict(chosen=chosen,eligible_count=len(eligible),diagnostic=not bool(eligible)));event('selection_frozen',config=chosen['config'],eligible_count=len(eligible))
    g=SpatialGenerator(chosen['config'],ranks).eval();holdout={};passed=bool(eligible)
    for h in plan['holdout']:
        key=h['distribution'];zz=noise(h['seed'],h['count'],key=='uniform');a=sample(load_model(PARENT_PT),zz);y=sample(g,zz);bm=measurements(a,a);m=measurements(a,y);ch=checks(m,bm);passed=passed and all(ch.values());holdout[key]=dict(baseline=bm,candidate=m,guards=ch)
        event('holdout_evaluated',distribution=key,passed=all(ch.values()),failed=[k for k,v in ch.items() if not v])
    write(ROOT/'evaluation/holdout.json',holdout)
    name='model_spatial_cleanup' if passed else 'model_spatial_experimental';p=ROOT/'candidates'/f'{name}.tflite'
    info=export(g,p);v=validate(p,g);torch.save(g.state_dict(),p.with_suffix('.pt'))
    import round6.upscale as up
    up.PARENT=p;q=ROOT/'candidates'/f'{name}_2x.tflite';upinfo=up.export(q);uv=validate(q,g,True)
    write(ROOT/'evaluation/graph_validation.json',dict(native128=v,upscale256=uv))
    # Save the exact prior comparison inputs and add round9, so all three stages
    # share latent vectors and the same 256px display interpolation.
    from generate import Generator
    original=Generator(ROOT.parent/'model.tflite')
    for key,seed in [('gaussian',88920),('uniform',88921)]:
        zz=noise(seed,8,key=='uniform');a=original(zz);b=sample(load_model(PARENT_PT),zz);y=sample(g,zz)
        np.savez_compressed(ROOT/'figures'/f'{key}_comparison.npz',latents=zz,original=head(a),round8=head(b),round9=head(y))
    subprocess.run(['python3',str(ROOT/'render_comparison.py')],check=True)
    for path,h in protected.items():assert sha(path)==h
    models=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p),export=info),dict(path=str(q.relative_to(ROOT)),sha256=sha(q),export=upinfo)]
    result=dict(criteria_passed=passed,promoted=False,visual_review_pending=True,chosen_config=chosen['config'],models=models,protected_files=protected,native_litert_tested=False,training_performed=False,external_images_used=False,new_detail_recovery_claimed=False)
    write(ROOT/'evaluation/result.json',result);c=ROOT/'checkpoints/043_delivery';c.mkdir(exist_ok=False)
    for f in [Path(__file__),ROOT/'check_graph.py',ROOT/'evaluation/result.json',ROOT/'evaluation/frozen_selection.json',ROOT/'evaluation/channel_ranking.json']:shutil.copy2(f,c/f.name)
    event('delivery_frozen',criteria_passed=passed,models=models)

if __name__=='__main__':main()
