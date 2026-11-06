"""Combine the frozen diversity edit with freshly calibrated compact cleanup.

User confirmed the intended input is random noise. Its exact distribution is
still unspecified, so both Gaussian and uniform noise are evaluated.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,datetime,hashlib,shutil
import numpy as np
import torch
from inspect_model import Model
from generate import grid
from round3.restructure import transfer
from round4.refine_structure import RefinedGenerator,sample,calibrate,metrics_pair
from round4.export_compact import export
from round4.check_graph import validate,GraphRunner

ROOT=Path(__file__).resolve().parent
V1=ROOT.parent/'model_improved.tflite'
V2=ROOT.parent/'round2/candidates/model_diversity_v2.tflite'
CLEAN=ROOT.parent/'round4/candidates/model_compact_refined.pt'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def event(kind,**kw):
    r={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'event':kind,**kw}
    with (ROOT/'logs/events.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
    print(json.dumps(r),flush=True)

def build(mix=0,clean=True):
    g=RefinedGenerator().eval()
    if mix:
        diverse=transfer(V2)
        with torch.no_grad():g.base.stem.weight.lerp_(diverse.stem.weight,mix)
    if clean:
        z=np.random.default_rng(88600).standard_normal((128,512)).astype('float32')
        offset=calibrate(g,z)['phase64']*.75
        g.phase64.copy_(torch.from_numpy(offset))
    return g

def load_model(path):
    g=RefinedGenerator().eval();g.load_state_dict(torch.load(path,weights_only=True));return g

def main():
    for d in ['logs','evaluation','figures','checkpoints','candidates','source']:(ROOT/d).mkdir(exist_ok=True,parents=True)
    plan={'user_confirmed':'Input is random noise; goals are cleaner portraits and fewer repeats','distribution_detail':'Unspecified; use standard normal as primary assumption and uniform [-1,1] as robustness check','calibration_seed':88600,'calibration_count':128,'development_seed':88601,'development_count':128,'holdout_seeds':[88602,88603],'holdout_count':512,'guards':{'phase4_vs_clean_max_ratio':1.15,'edge_min_ratio':.95,'mirror_nearest_min_ratio':1.05,'average_shape_min_ratio':.85,'clipping_max_absolute_increase':.002},'sources':{str(p.relative_to(ROOT.parent)):sha(p) for p in [V1,V2,CLEAN]}}
    (ROOT/'evaluation/plan.json').write_text(json.dumps(plan,indent=2));shutil.copy2(__file__,ROOT/'source/combine.frozen.py');event('plan_frozen',plan=plan)
    a,b=Model(V1),Model(V2)
    for i in range(len(a.tensors)):
        x=a.array(i);y=b.array(i)
        if x is not None and i!=13:assert np.array_equal(x,y),i
    z=np.random.default_rng(88601).standard_normal((128,512)).astype('float32')
    baseline=load_model(CLEAN);base=sample(baseline,z);bm=metrics_pair(base,base)
    results=[]
    configs=[('existing_clean',None,None),('diversity_only',1,False),('combined_half',.5,True),('combined_full',1,True)]
    for idx,(name,mix,clean) in enumerate(configs,1):
        g=baseline if mix is None else build(mix,clean)
        y=sample(g,z);m=metrics_pair(base,y)
        checks={'phase4':m['phase_rms']['4']<=bm['phase_rms']['4']*1.15,'edge':m['edge_energy']>=bm['edge_energy']*.95,'repetition':m['mirror_invariant_nearest_shape_distance']>=bm['mirror_invariant_nearest_shape_distance']*1.05,'shape_spread':m['shape_cosine_distance']>=bm['shape_cosine_distance']*.85,'clipping':m['clipped_fraction']<=bm['clipped_fraction']+.002}
        folder=ROOT/'checkpoints'/f'{idx:03}_{name}';folder.mkdir(exist_ok=False)
        # Store the actually changed stem and phase values, with an immutable parent.
        np.savez_compressed(folder/'edits.npz',stem_weight=g.base.stem.weight.detach().numpy(),phase64=g.phase64.numpy())
        record={'name':name,'mix':mix,'clean':clean,'metrics':m,'guards':checks,'eligible':all(checks.values()),'checkpoint':str(folder.relative_to(ROOT)),'patch_sha256':sha(folder/'edits.npz')}
        (folder/'manifest.json').write_text(json.dumps(record,indent=2));np.savez_compressed(folder/'preview.npz',images=y[:16],latents=z[:16]);grid(y[:16],folder/'preview.png',labels=[f'{name} {i}' for i in range(16)])
        results.append(record);event('development_candidate',record=record)
    (ROOT/'evaluation/development.json').write_text(json.dumps(results,indent=2))
    eligible=[r for r in results if r['eligible']]
    if not eligible:
        event('no_candidate_promoted');return
    chosen=max(eligible,key=lambda r:r['metrics']['mirror_invariant_nearest_shape_distance'])
    frozen=ROOT/'checkpoints/005_frozen_combination';frozen.mkdir(exist_ok=False)
    selection={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'chosen':chosen['name'],'reason':'Highest reflection-invariant nearest shape distance among candidates passing predeclared cleanup, edge, shape-spread and clipping guards','holdout_opened':False,'source_checkpoint':chosen['checkpoint']}
    (frozen/'selection.json').write_text(json.dumps(selection,indent=2));event('selection_frozen',selection=selection)
    shutil.copy2(ROOT/chosen['checkpoint']/'edits.npz',frozen/'edits.npz');patch=dict(np.load(frozen/'edits.npz'));model=RefinedGenerator().eval()
    with torch.no_grad():model.base.stem.weight.copy_(torch.from_numpy(patch['stem_weight']));model.phase64.copy_(torch.from_numpy(patch['phase64']))
    pt=ROOT/'candidates/model_clean_diverse.pt';assert not pt.exists();torch.save(model.state_dict(),pt)
    tflite=ROOT/'candidates/model_clean_diverse.tflite';info=export(model,tflite);graph_check=validate(tflite,model)
    (ROOT/'evaluation/graph_validation.json').write_text(json.dumps(graph_check,indent=2))
    heldout={}
    for name,seed in [('gaussian',88602),('uniform',88603)]:
        rng=np.random.default_rng(seed);z=(rng.standard_normal((512,512)) if name=='gaussian' else rng.uniform(-1,1,(512,512))).astype('float32')
        a=sample(baseline,z);b=sample(model,z);ma=metrics_pair(a,a);mb=metrics_pair(a,b)
        r={'seed':seed,'count':512,'clean_baseline':ma,'combined':mb,'phase_change_percent':{k:100*(mb['phase_rms'][k]/ma['phase_rms'][k]-1) for k in ma['phase_rms']},'relative_change_percent':{k:100*(mb[k]/ma[k]-1) if ma[k] else None for k in ma if isinstance(ma[k],float)}}
        decoded=np.concatenate([GraphRunner(tflite)(z[i:i+8]) for i in range(0,16,8)])
        assert np.allclose(b[:16],decoded,atol=2e-5,rtol=2e-5)
        r['serialized_graph_max_error_first16']=float(np.abs(b[:16]-decoded).max());heldout[name]=r;event('holdout_result',distribution=name,result=r)
        np.savez_compressed(ROOT/f'evaluation/{name}_preview.npz',baseline=a[:16],combined=b[:16],latents=z[:16])
        pics=np.stack([a[:16],b[:16]],axis=1).reshape(-1,128,128,3);grid(pics,ROOT/f'figures/{name}_comparison.png',labels=[f'{i//2} '+['clean baseline','combined'][i%2] for i in range(32)])
    (ROOT/'evaluation/heldout.json').write_text(json.dumps(heldout,indent=2))
    manifest={'sources':plan['sources'],'chosen':chosen['name'],'torch_sha256':sha(pt),'tflite_sha256':sha(tflite),'tflite_bytes':tflite.stat().st_size,'graph':info,'native_litert_tested':False,'training_performed':False,'new_identities_claimed':False}
    (frozen/'manifest.json').write_text(json.dumps(manifest,indent=2));event('complete',manifest=manifest)

if __name__=='__main__':main()
