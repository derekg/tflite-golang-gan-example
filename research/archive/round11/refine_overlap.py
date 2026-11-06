"""Fresh-seed follow-up: independently tune continuity and phase residual.

All candidates branch from preserved round9. Round10 remains immutable.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import datetime,json,shutil
import numpy as np
import torch
from round10.overlap import OverlapGenerator,expanded,export,stats,guards,sha,write,PARENT
from round9.spatial_cleanup import validate
from round8.selective_detail import noise
from round4.refine_structure import sample
ROOT=Path(__file__).resolve().parent
def event(event_name,**kw):
    v=dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),event=event_name,**kw)
    with (ROOT/'logs/events.jsonl').open('a') as f:f.write(json.dumps(v)+'\n')
    print(json.dumps(v),flush=True)

@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    for d in ['evaluation','checkpoints','logs','candidates','figures']:(ROOT/d).mkdir(parents=True,exist_ok=True)
    assert not (ROOT/'evaluation/plan.json').exists()
    cs=[dict(name=f'overlap{a:g}_phase{b:g}',stages=[4],a=a,b=b) for a in [.35,.5,.65,.8] for b in [0.,.1,.2,.3]]
    protected={str(p):sha(p) for p in [PARENT,PARENT.with_suffix('.pt'),ROOT.parent/'model.tflite',ROOT.parent/'round8/candidates/model_selective_experimental.tflite',ROOT.parent/'round10/candidates/model_overlap_experimental.tflite',ROOT.parent/'portrait-morph/dist/index.html']}
    plan=dict(parent=str(PARENT),protected=protected,configs=cs,development=dict(gaussian_seed=89200,uniform_seed=89201,count_each=64),holdout=dict(gaussian_seed=89202,uniform_seed=89203,count_each=256),reason='Round10 isolated the final upsampler; full phase averaging increased normalized edge periodicity. Tune shared-field overlap and phase residual separately.',criteria=json.loads((ROOT.parent/'round10/evaluation/plan.json').read_text())['criteria'],selection='Same frozen round10 guards on BOTH distributions; minimize worst edge-grid ratio among eligible candidates. If none, diagnostic structure-safe candidate only.',training=False,gradients=False,external_data=False)
    write(ROOT/'evaluation/plan.json',plan);shutil.copy2(__file__,ROOT/'evaluation/implementation.frozen.py');event('plan_frozen',candidates=len(cs))
    baseline=OverlapGenerator(dict(name='round9',stages=[],a=0.,b=0.)).eval()
    inputs={k:noise(s,64,k=='uniform') for k,s in [('gaussian',89200),('uniform',89201)]}
    bases={k:sample(baseline,z) for k,z in inputs.items()};bm={k:stats(y,y) for k,y in bases.items()};records=[]
    for idx,cfg in enumerate(cs):
        model=OverlapGenerator(cfg).eval();res={};preview={}
        for k,z in inputs.items():
            y=sample(model,z);m=stats(bases[k],y);res[k]=dict(metrics=m,guards=guards(m,bm[k]));preview[k]=y[:4]
        eligible=all(all(v['guards'].values()) for v in res.values())
        structural=all(all(v for k,v in r['guards'].items() if k not in ['edge_grid','phase4']) for r in res.values())
        score=max(v['metrics']['edge_grid']['score']/bm[k]['edge_grid']['score'] for k,v in res.items())
        cp=ROOT/'checkpoints'/f'{idx:03}_{cfg["name"]}';cp.mkdir(exist_ok=False)
        np.savez_compressed(cp/'weights.npz',projection4=expanded(model.model.parent.base.upsamplers[4].projection.weight,cfg['a'],cfg['b']).numpy())
        np.savez_compressed(cp/'preview.npz',**preview)
        rec=dict(config=cfg,statistics=res,eligible=eligible,structure_safe=structural,score=score,checkpoint=str(cp.relative_to(ROOT)),patch_sha256=sha(cp/'weights.npz'))
        write(cp/'manifest.json',rec);records.append(rec);event('candidate',name=cfg['name'],eligible=eligible,structure_safe=structural,grid_ratio=score)
    write(ROOT/'evaluation/development.json',dict(baseline=bm,candidates=records))
    pool=[r for r in records if r['eligible']];diagnostic=not bool(pool)
    if not pool:pool=[r for r in records if r['structure_safe']]
    assert pool
    chosen=min(pool,key=lambda r:r['score']);cfg=chosen['config'];write(ROOT/'evaluation/frozen_selection.json',dict(chosen=chosen,diagnostic=diagnostic));event('selection_frozen',config=cfg,diagnostic=diagnostic)
    model=OverlapGenerator(cfg).eval();holdout={}
    for key,seed in [('gaussian',89202),('uniform',89203)]:
        z=noise(seed,256,key=='uniform');base=sample(baseline,z);y=sample(model,z);b=stats(base,base);m=stats(base,y)
        holdout[key]=dict(baseline=b,candidate=m,guards=guards(m,b));event('holdout',distribution=key,guards=holdout[key]['guards'])
        np.savez_compressed(ROOT/'figures'/f'{key}_holdout.npz',latents=z,round9=base,round11=y)
    write(ROOT/'evaluation/holdout.json',holdout)
    p=ROOT/'candidates/model_grid_cleanup_experimental.tflite';info=export(model,p);v=validate(p,model)
    torch.save(dict(config=cfg,state_dict=model.state_dict()),p.with_suffix('.pt'))
    import round6.upscale as up
    up.PARENT=p;q=ROOT/'candidates/model_grid_cleanup_experimental_2x.tflite';qi=up.export(q);qv=validate(q,model,True)
    write(ROOT/'evaluation/graph_validation.json',dict(native128=v,upscale256=qv))
    from generate import Generator
    from round5.combine import load_model
    original=Generator(ROOT.parent/'model.tflite');r8=load_model(ROOT.parent/'round8/candidates/model_selective_experimental.pt')
    c10=json.loads((ROOT.parent/'round10/evaluation/result.json').read_text())['config'];r10=OverlapGenerator(c10).eval()
    for key,seed in [('gaussian',88920),('uniform',88921)]:
        z=noise(seed,8,key=='uniform')
        np.savez_compressed(ROOT/'figures'/f'{key}_comparison.npz',latents=z,original=original(z),round8=sample(r8,z),round9=sample(baseline,z),round10=sample(r10,z),round11=sample(model,z))
    # Paired bootstrap intervals for edge periodicity on untouched holdout.
    from round10.overlap import edge_grid
    intervals={};rng=np.random.default_rng(89210)
    for key in ['gaussian','uniform']:
        d=np.load(ROOT/'figures'/f'{key}_holdout.npz')
        old=np.array([edge_grid(a[None])['score'] for a in d['round9']]);new=np.array([edge_grid(a[None])['score'] for a in d['round11']])
        ii=rng.integers(0,len(old),(2000,len(old)));ratios=new[ii].mean(1)/old[ii].mean(1)
        intervals[key]=dict(ratio=float(new.mean()/old.mean()),paired_bootstrap_ratio_95pct=np.percentile(ratios,[2.5,97.5]).tolist(),replicates=2000)
    write(ROOT/'evaluation/uncertainty.json',intervals)
    for path,h in protected.items():assert sha(path)==h
    result=dict(config=cfg,criteria_passed=not diagnostic and all(all(h['guards'].values()) for h in holdout.values()),promoted=False,visual_review_pending=True,models=[dict(path=str(p),sha256=sha(p),export=info),dict(path=str(q),sha256=sha(q),export=qi)],protected_files=protected,native_litert_tested=False,training=False,external_data=False)
    write(ROOT/'evaluation/result.json',result);cp=ROOT/'checkpoints/016_delivery';cp.mkdir(exist_ok=False)
    shutil.copy2(__file__,cp/'refine_overlap.py');write(cp/'manifest.json',result);event('delivery_complete',models=result['models'])

if __name__=='__main__':main()
