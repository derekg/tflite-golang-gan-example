"""Finite analytical weight edits. No training, gradients, optimizer or external images."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import datetime, hashlib, json, shutil
import numpy as np
import torch
import torch.nn.functional as F
from round5.combine import load_model
from round4.refine_structure import sample, metrics_pair
from generate import grid

ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent/'round5/candidates/model_clean_diverse.pt'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def event(event_name,**v):
    r=dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),event=event_name,**v)
    with (ROOT/'logs/events.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
    print(json.dumps(r),flush=True)

def edit(balance,detail):
    g=load_model(PARENT)
    with torch.no_grad():
        # Preserve phase-average weights exactly in real arithmetic. Alter only
        # the last two spatial projection layers, not latent/identity mapping.
        for i in [3,4]:
            w=g.base.upsamplers[i].projection.weight
            phase=w.reshape(-1,4,w.shape[1],1,1)
            phase.lerp_(phase.mean(1,keepdim=True).expand_as(phase),balance)
        # Existing zero residual branch at 32px: a small channel-diagonal
        # negative Laplacian after its inherited leaky activation. Zero padding
        # is retained deliberately to match this architecture and exporter.
        w=g.base.refiner.conv2.weight
        assert torch.count_nonzero(w)==0
        for c in range(64):
            w[c,c,1,1]=detail
            for y,x in [(0,1),(1,0),(1,2),(2,1)]:w[c,c,y,x]=-detail/4
    return g

def measure(base,y):
    m=metrics_pair(base,y)
    t=torch.from_numpy(y).permute(0,3,1,2)
    # Broad structure gradients: do not count raw pixel noise as detail gain.
    k=torch.tensor([1,4,6,4,1],dtype=torch.float32);kernel=torch.outer(k,k)/256
    s=F.conv2d(F.pad(t,(2,2,2,2),mode='reflect'),kernel[None,None].repeat(3,1,1,1),groups=3).numpy()[:,:,4:-4,4:-4]
    m['smoothed_edge_energy']=float(np.abs(np.diff(s,axis=2)).mean()+np.abs(np.diff(s,axis=3)).mean())
    return m

def guards(m,b):
    return dict(grid4=m['phase_rms']['4']<=b['phase_rms']['4']*.95,
                broad_edge=m['smoothed_edge_energy']>=b['smoothed_edge_energy']*1.005,
                raw_edge=m['edge_energy']>=b['edge_energy']*.98,
                shape=m['mirror_invariant_nearest_shape_distance']>=b['mirror_invariant_nearest_shape_distance']*.98,
                drift=m['paired_rgb_rms']<=.015,
                color=m['max_channel_mean_shift']<=.005,
                clipping=m['clipped_fraction']<=b['clipped_fraction']+.002)

def main():
    torch.set_num_threads(4)
    for d in ['evaluation','figures','checkpoints','logs','candidates']:(ROOT/d).mkdir(parents=True,exist_ok=True)
    assert not (ROOT/'evaluation/result.json').exists(),'Use a new round; preserve this run.'
    protected={str(p):sha(p) for p in [PARENT,PARENT.with_suffix('.tflite'),ROOT.parent/'round6/candidates/model_upscale_2x.tflite',ROOT.parent/'round6/candidates/model_upscale_2x_sharp.tflite']}
    assert sha(PARENT)=='2b899582c34e853dadd4be17dbb5fcd347e719d8d421d0afe516a200531b95c0'
    configs=[dict(name=f'phase{b:g}_detail{d:g}',balance=b,detail=d) for b in [0,.025,.05,.1] for d in [0,.01,.025,.05]]
    plan=dict(training=False,external_images=False,gradients=False,optimizer=False,parent_sha256=sha(PARENT),development=dict(seed=88800,count=64),holdout=[dict(distribution='gaussian',seed=88801,count=128),dict(distribution='uniform',seed=88802,count=128)],configs=configs,guards=dict(grid4_max_ratio=.95,smoothed_edge_min_ratio=1.005,raw_edge_min_ratio=.98,nearest_shape_min_ratio=.98,paired_rgb_rms_max=.015,color_drift_max=.005,clipping_max_increase=.002),selection='Among eligible development candidates, minimize phase4 RMS. Holdout cannot change the selected coefficients.',claim_limit='Structure and phase proxies only; no evidence of recovered missing detail.')
    if (ROOT/'evaluation/plan.json').exists():
        assert json.loads((ROOT/'evaluation/plan.json').read_text())==plan
        event('resume_frozen_plan',reason='Corrected logging argument collision; candidate definitions and guards unchanged.')
    else:
        write(ROOT/'evaluation/plan.json',plan);event('plan_frozen',plan=plan)
    rng=np.random.default_rng(88800);z=rng.standard_normal((64,512)).astype('float32');base=sample(edit(0,0),z);bm=measure(base,base)
    records=[];pics=[]
    for i,cfg in enumerate(configs):
        g=edit(cfg['balance'],cfg['detail']);y=sample(g,z);m=measure(base,y);checks=guards(m,bm)
        c=ROOT/'checkpoints'/f'{i:03}_{cfg["name"]}';c.mkdir(exist_ok=True)
        patches={k:v.numpy() for k,v in g.state_dict().items() if k in ['base.upsamplers.3.projection.weight','base.upsamplers.4.projection.weight','base.refiner.conv2.weight']}
        if (c/'weights.npz').exists():
            restored=np.load(c/'weights.npz')
            assert all(np.array_equal(restored[k],v) for k,v in patches.items())
        else:np.savez_compressed(c/'weights.npz',**patches)
        rec=dict(config=cfg,metrics=m,guards=checks,eligible=all(checks.values()),checkpoint=str(c.relative_to(ROOT)),patch_sha256=sha(c/'weights.npz'))
        if (c/'manifest.json').exists():assert json.loads((c/'manifest.json').read_text())==rec
        else:write(c/'manifest.json',rec)
        records.append(rec)
        if not (c/'preview.npz').exists():np.savez_compressed(c/'preview.npz',images=y[:4],latents=z[:4])
        pics.extend(y[:4]);event('candidate_evaluated',name=cfg['name'],eligible=rec['eligible'],phase4=m['phase_rms']['4'],broad_edge=m['smoothed_edge_energy'])
    write(ROOT/'evaluation/development.json',dict(baseline=bm,candidates=records))
    grid(np.array(pics),ROOT/'figures/development.png',labels=[f'{c["name"]}: {j}' for c in configs for j in range(4)],cols=4,scale=1)
    eligible=[r for r in records if r['eligible']];selected=min(eligible,key=lambda r:r['metrics']['phase_rms']['4']) if eligible else None
    # Even if no candidate qualifies, freeze and evaluate the best phase-cleanup
    # diagnostic. It is explicitly not promoted and cannot replace the parent.
    diagnostic=selected or min(records,key=lambda r:r['metrics']['phase_rms']['4'])
    write(ROOT/'evaluation/frozen_selection.json',dict(selected=selected,diagnostic=diagnostic,promotion_possible=selected is not None))
    cfg=diagnostic['config'];g=edit(cfg['balance'],cfg['detail']);heldout={};all_ok=selected is not None
    for h in plan['holdout']:
        r=np.random.default_rng(h['seed']);n=h['count'];zz=(r.standard_normal((n,512)) if h['distribution']=='gaussian' else r.uniform(-1,1,(n,512))).astype('float32')
        a=sample(edit(0,0),zz);y=sample(g,zz);b=measure(a,a);m=measure(a,y);checks=guards(m,b);all_ok=all_ok and all(checks.values())
        heldout[h['distribution']]=dict(baseline=b,candidate=m,guards=checks)
        if h['distribution']=='gaussian':
            from round6.upscale import head
            comparison=np.stack([head(a[:8]),head(y[:8])],axis=1).reshape(-1,256,256,3)
            grid(comparison,ROOT/'figures/holdout_comparison.png',labels=[f'{i}: {name}' for i in range(8) for name in ['preserved parent','direct-edit diagnostic']],cols=2,scale=1)
            np.savez_compressed(ROOT/'figures/holdout_comparison.npz',latents=zz[:8],parent=a[:8],candidate=y[:8])
    write(ROOT/'evaluation/holdout.json',heldout)
    models=[]
    if all_ok:
        from round4.export_compact import export
        from round4.check_graph import validate
        p=ROOT/'candidates/model_direct_detail.tflite';info=export(g,p);validation=validate(p,g)
        torch.save(g.state_dict(),p.with_suffix('.pt'))
        models.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),graph=info))
        import round6.upscale as up
        up.PARENT=p;out=ROOT/'candidates/model_direct_detail_2x.tflite';info=up.export(out)
        from round6.check_graph import GraphRunner
        errors=[]
        for backend in ['torch','numpy']:
            actual=GraphRunner(out,backend)(z[:3]);expected=up.head(sample(g,z[:3]));err=float(np.max(np.abs(actual-expected)));assert np.allclose(actual,expected,atol=2e-5,rtol=2e-5);errors.append(err)
        models.append(dict(path=str(out.relative_to(ROOT)),sha256=sha(out),graph=info))
        write(ROOT/'evaluation/graph_validation.json',dict(native128=validation,upscale_errors=errors,native_litert_tested=False))
    for p,h in protected.items():assert sha(p)==h
    summary=dict(promoted=bool(all_ok),selected_config=selected['config'] if selected else None,diagnostic_config=cfg,models=models,protected_files=protected,native_litert_tested=False,training_performed=False,new_detail_recovery_claimed=False)
    write(ROOT/'evaluation/result.json',summary)
    c=ROOT/'checkpoints/016_delivery';c.mkdir(exist_ok=False);shutil.copy2(__file__,c/'direct_detail.py');write(c/'manifest.json',summary)
    event('experiment_complete',**summary)

if __name__=='__main__':main()
