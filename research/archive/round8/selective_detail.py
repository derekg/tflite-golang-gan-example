"""Channel-selective analytical edits; finite sweep, no optimizer or outside data.

Requires the preserved round5 parent and previous rounds' helper modules.
Completed runs cannot be overwritten. All selection precedes fresh-seed evaluation.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import datetime, hashlib, json, shutil
import numpy as np
import torch
import torch.nn.functional as F
from round5.combine import load_model
from round4.refine_structure import sample
from round7.direct_detail import measure, guards
from round5.portrait_sampler import artifacts
from generate import grid

ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent/'round5/candidates/model_clean_diverse.pt'
EDIT_KEYS=['base.upsamplers.3.projection.weight','base.upsamplers.4.projection.weight','base.refiner.conv2.weight']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def event(event_name,**fields):
    r=dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),event=event_name,**fields)
    with (ROOT/'logs/events.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
    print(json.dumps(r),flush=True)

def noise(seed,n,uniform=False):
    r=np.random.default_rng(seed)
    return (r.uniform(-1,1,(n,512)) if uniform else r.standard_normal((n,512))).astype('float32')

@torch.inference_mode()
def rank_channels(model):
    stats={key:[] for key in ['32','64','128']}
    # Independent calibration probes, equally split across the two noise laws.
    z=np.concatenate([noise(88900,64),noise(88904,64,True)])
    last=[]
    hook=model.base.upsamplers[4].register_forward_hook(lambda mod,inp,out:last.append(F.leaky_relu(out,.3)))
    try:
        for zz in np.array_split(z,16):
            _,trace=model(torch.from_numpy(zz),trace=True)
            acts={'32':trace['phase32'],'64':model.add_phase(trace['phase64'],model.phase64),'128':last.pop()}
            for key,x in acts.items():
                sm=F.avg_pool2d(F.pad(x,(2,2,2,2),mode='reflect'),5,stride=1)
                high=(x-sm)[:,:,4:-4,4:-4]
                p=torch.stack([high[:,:,dy::2,dx::2].mean((2,3)) for dy in range(2) for dx in range(2)],1)
                phase=((p-p.mean(1,keepdim=True))**2).mean(1).sqrt().mean(0)
                s=sm[:,:,4:-4,4:-4]
                broad=(torch.diff(s,dim=2).abs().mean((0,2,3))+torch.diff(s,dim=3).abs().mean((0,2,3)))
                std=x.std((0,2,3),unbiased=False)
                stats[key].append(np.stack([phase.numpy(),broad.numpy(),std.numpy()]))
    finally:hook.remove()
    # Absolute-weight path sensitivity is a heuristic, not a semantic label or
    # exact derivative: activations and cancellation make the true effect differ.
    color=[layer.weight.detach().numpy()[:,:,0,0] for layer in model.base.color]
    sensitivity=np.abs(color[2])@np.abs(color[1])@np.abs(color[0]);s128=sensitivity.sum(0)
    def projection(i):
        w=model.base.upsamplers[i].projection.weight.detach().numpy()
        return np.abs(w.reshape(64,4,64)).mean(1)
    s64=s128@projection(4);s32=s64@projection(3)
    result={}
    for key,sens in [('32',s32),('64',s64),('128',s128)]:
        phase,broad,std=np.mean(stats[key],axis=0)
        artifact=phase*sens
        structure=broad*sens/(1+10*phase/np.maximum(std,1e-8))
        result[key]=dict(phase=phase.tolist(),broad_edge=broad.tolist(),activation_std=std.tolist(),absolute_path_sensitivity=sens.tolist(),artifact_score=artifact.tolist(),structure_score=structure.tolist(),artifact_rank=np.argsort(-artifact,kind='stable').tolist(),structure_rank=np.argsort(-structure,kind='stable').tolist())
    return result

def edit(cfg,ranks):
    g=load_model(PARENT);k=cfg['channels'];balance=cfg['balance'];detail=cfg['detail']
    with torch.no_grad():
        for layer,key in [(3,'64'),(4,'128')]:
            w=g.base.upsamplers[layer].projection.weight
            p=w.reshape(64,4,64,1,1)
            for c in ranks[key]['artifact_rank'][:k]:
                p[c].lerp_(p[c].mean(0,keepdim=True).expand_as(p[c]),balance)
        w=g.base.refiner.conv2.weight;assert torch.count_nonzero(w)==0
        for c in ranks['32']['structure_rank'][:k]:
            w[c,c,1,1]=detail
            for y,x in [(0,1),(1,0),(1,2),(2,1)]:w[c,c,y,x]=-detail/4
    for p in g.parameters():p.requires_grad_(False)
    return g

def measurements(base,y):
    m=measure(base,y);m['mean_per_image_grid4']=float(artifacts(y)['grid4'].mean());return m

def checks(m,b):
    out=guards(m,b)
    out.update(grid2=m['phase_rms']['2']<=b['phase_rms']['2']*1.05,
               grid8=m['phase_rms']['8']<=b['phase_rms']['8']*1.05,
               per_image_grid4=m['mean_per_image_grid4']<=b['mean_per_image_grid4'])
    return out

def export_candidate(g,z,status):
    from round4.export_compact import export
    from round4.check_graph import validate
    from inspect_model import Model
    prefix='model_selective_detail' if status else 'model_selective_experimental'
    p=ROOT/'candidates'/f'{prefix}.tflite';info=export(g,p)
    torch.save(g.state_dict(),p.with_suffix('.pt'))
    test128=validate(p,g)
    # Assert all state changes are restricted to the declared three tensors.
    old=load_model(PARENT).state_dict();new=g.state_dict()
    changed=[k for k in old if not torch.equal(old[k],new[k])]
    assert set(changed)<=set(EDIT_KEYS)
    # The graph constants must agree with the generated source module; validate
    # both decoded arithmetic backends, then append the previously checked head.
    import round6.upscale as up
    from round6.check_graph import GraphRunner
    up.PARENT=p
    target=ROOT/'candidates'/f'{prefix}_2x.tflite';upinfo=up.export(target)
    tests=[]
    for name,zz in [('zero',np.zeros((1,512),dtype='float32')),('gaussian',z[:3]),('uniform',noise(88910,2,True))]:
        expected=up.head(sample(g,zz))
        for backend in ['torch','numpy']:
            actual=GraphRunner(target,backend)(zz)
            assert actual.shape==(len(zz),256,256,3) and np.isfinite(actual).all()
            err=float(np.max(np.abs(actual-expected)))
            assert np.allclose(actual,expected,atol=2e-5,rtol=2e-5)
            tests.append(dict(input=name,batch=len(zz),backend=backend,max_error=err))
    result=dict(tests128=test128,tests256=tests,changed_state_tensors=changed,native_litert_tested=False)
    write(ROOT/'evaluation/graph_validation.json',result)
    return [dict(path=str(p.relative_to(ROOT)),sha256=sha(p),graph=info),dict(path=str(target.relative_to(ROOT)),sha256=sha(target),graph=upinfo)]

def main():
    torch.set_num_threads(4)
    for d in ['evaluation','checkpoints','figures','logs','candidates']:(ROOT/d).mkdir(parents=True,exist_ok=True)
    assert not (ROOT/'evaluation/plan.json').exists(),'Preserve this round; use a new output directory.'
    paths=[PARENT,PARENT.with_suffix('.tflite'),ROOT.parent/'round6/candidates/model_upscale_2x.tflite',ROOT.parent/'round6/candidates/model_upscale_2x_sharp.tflite',ROOT.parent/'model.tflite',ROOT.parent/'model_improved.tflite']
    protected={str(p):sha(p) for p in paths}
    assert sha(PARENT)=='2b899582c34e853dadd4be17dbb5fcd347e719d8d421d0afe516a200531b95c0'
    configs=[dict(name='baseline',channels=0,balance=0.,detail=0.)]
    for k in [1,4,8,16]:
        for b in [0.,.1,.25]:
            for d in [0.,.1,.25]:
                if b or d:configs.append(dict(name=f'channels{k}_phase{b:g}_detail{d:g}',channels=k,balance=b,detail=d))
    plan=dict(training=False,external_images=False,optimizer=False,gradients=False,parent_sha256=sha(PARENT),calibration=[dict(seed=88900,distribution='gaussian',count=64),dict(seed=88904,distribution='uniform',count=64)],development=dict(seed=88901,count=96,distribution='gaussian'),holdout=[dict(seed=88902,count=256,distribution='gaussian'),dict(seed=88903,count=256,distribution='uniform')],configs=configs,ranking='Per-channel phase variation and broad gradients, weighted by heuristic absolute downstream weight-path sensitivity. Frozen from calibration inputs only.',criteria=dict(grid4_max_ratio=.95,broad_edge_min_ratio=1.005,raw_edge_min_ratio=.98,nearest_shape_min_ratio=.98,rgb_rms_max=.015,color_shift_max=.005,clip_increase_max=.002,grid2_and_grid8_max_ratio=1.05,mean_per_image_grid4_max_ratio=1.),selection='Lowest phase4 RMS among eligible development candidates. If none eligible, diagnostic with fewest failed criteria then lowest phase4 RMS. No coefficient or channel changes after development.',export_policy='Export selected coefficients even if rejected, with experimental filename. Never replace parent. Promotion requires every criterion on both heldout distributions, plus separate visual review.',claim_limit='Metrics are proxies, not evidence of newly reconstructed facial detail.')
    write(ROOT/'evaluation/plan.json',plan);event('plan_frozen',candidates=len(configs))
    shutil.copy2(__file__,ROOT/'evaluation/implementation.frozen.py')
    ranks=rank_channels(load_model(PARENT));write(ROOT/'evaluation/channel_rankings.json',ranks)
    for key in ['32','64','128']:event('channels_ranked',stage=key,structure_top8=ranks[key]['structure_rank'][:8],artifact_top8=ranks[key]['artifact_rank'][:8])
    c=ROOT/'checkpoints/000_channel_ranking';c.mkdir(exist_ok=False);shutil.copy2(ROOT/'evaluation/channel_rankings.json',c/'channel_rankings.json');shutil.copy2(ROOT/'evaluation/plan.json',c/'plan.json')
    z=noise(88901,96);base=sample(load_model(PARENT),z);bm=measurements(base,base)
    results=[];pics=[]
    for i,cfg in enumerate(configs):
        g=edit(cfg,ranks);y=sample(g,z);m=measurements(base,y);ch=checks(m,bm)
        c=ROOT/'checkpoints'/f'{i+1:03}_{cfg["name"]}';c.mkdir(exist_ok=False)
        np.savez_compressed(c/'weights.npz',**{k:v.numpy() for k,v in g.state_dict().items() if k in EDIT_KEYS})
        np.savez_compressed(c/'preview.npz',images=y[:4],latents=z[:4])
        record=dict(config=cfg,metrics=m,guards=ch,eligible=all(ch.values()),checkpoint=str(c.relative_to(ROOT)),patch_sha256=sha(c/'weights.npz'))
        write(c/'manifest.json',record);results.append(record);pics.extend(y[:4])
        event('candidate_evaluated',name=cfg['name'],eligible=record['eligible'],failed=[k for k,v in ch.items() if not v],grid4_ratio=m['phase_rms']['4']/bm['phase_rms']['4'],broad_edge_ratio=m['smoothed_edge_energy']/bm['smoothed_edge_energy'])
    write(ROOT/'evaluation/development.json',dict(baseline=bm,candidates=results))
    grid(np.array(pics),ROOT/'figures/development.png',labels=[f'{cfg["name"]}: {i}' for cfg in configs for i in range(4)],cols=4,scale=1)
    eligible=[r for r in results if r['eligible']]
    chosen=min(eligible,key=lambda r:r['metrics']['phase_rms']['4']) if eligible else min(results[1:],key=lambda r:(sum(not v for v in r['guards'].values()),r['metrics']['phase_rms']['4']))
    write(ROOT/'evaluation/frozen_selection.json',dict(chosen=chosen,eligible_count=len(eligible),selection_is_diagnostic=not bool(eligible)))
    event('selection_frozen',config=chosen['config'],eligible_count=len(eligible))
    g=edit(chosen['config'],ranks);holdout={};passed=bool(eligible)
    for h in plan['holdout']:
        zz=noise(h['seed'],h['count'],h['distribution']=='uniform');a=sample(load_model(PARENT),zz);y=sample(g,zz)
        b=measurements(a,a);m=measurements(a,y);ch=checks(m,b);passed=passed and all(ch.values())
        holdout[h['distribution']]=dict(baseline=b,candidate=m,guards=ch)
        np.savez_compressed(ROOT/'figures'/f'{h["distribution"]}_comparison.npz',latents=zz[:8],parent=a[:8],candidate=y[:8])
        from round6.upscale import head
        pictures=np.stack([head(a[:8]),head(y[:8])],axis=1).reshape(-1,256,256,3)
        grid(pictures,ROOT/'figures'/f'{h["distribution"]}_comparison.png',labels=[f'{i}: {label}' for i in range(8) for label in ['preserved parent','selective channel edit']],cols=2,scale=1)
        event('holdout_evaluated',distribution=h['distribution'],passed=all(ch.values()),failed=[k for k,v in ch.items() if not v])
    write(ROOT/'evaluation/holdout.json',holdout)
    models=export_candidate(g,z,passed)
    for p,h in protected.items():assert sha(p)==h
    result=dict(criteria_passed=passed,promoted=False,visual_review_pending=True,chosen_config=chosen['config'],models=models,native_litert_tested=False,training_performed=False,external_images_used=False,protected_files=protected,new_detail_recovery_claimed=False)
    write(ROOT/'evaluation/result.json',result)
    c=ROOT/'checkpoints/034_delivery';c.mkdir(exist_ok=False)
    for p in [Path(__file__),ROOT/'evaluation/result.json',ROOT/'evaluation/frozen_selection.json',ROOT/'evaluation/channel_rankings.json',ROOT/'evaluation/graph_validation.json']:shutil.copy2(p,c/p.name)
    event('delivery_frozen',criteria_passed=passed,models=models)

if __name__=='__main__':main()
