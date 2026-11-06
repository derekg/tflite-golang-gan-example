"""Reject a novelty policy that raises artifacts; select a stricter screen.

Preserve the first policy and its evidence, use new development pools, freeze a
choice, and evaluate new untouched request seeds. No model weights are changed.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,shutil,datetime
import numpy as np
from generate import grid
from round2.run_experiments import repetition
from round4.check_graph import GraphRunner
from round5.portrait_sampler import ROOT,DEFAULT_MODEL,PortraitSampler,noise,artifacts,sha
from round5.combine import event

def summarize(raw,selected):
    return {'raw_sequence':repetition(raw),'selected_sequence':repetition(selected),'raw_artifacts':{k:float(v.mean()) for k,v in artifacts(raw).items()},'selected_artifacts':{k:float(v.mean()) for k,v in artifacts(selected).items()}}

def main():
    prior=ROOT/'checkpoints/006_sampler_v1';prior.mkdir(exist_ok=False)
    for n in ['artifact_screen.json','sampler_plan.json','sampler_results.json','sampler_validation.json']:shutil.copy2(ROOT/'evaluation'/n,prior/n)
    for n in ['portrait_sampler.py','evaluate_sampler.py']:shutil.copy2(ROOT/n,prior/n)
    shutil.copy2(ROOT/'figures/repeat_aware_batch.png',prior/'repeat_aware_batch.png')
    plan={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reason':'Initial 95th-percentile grid screen increased per-image grid4 average despite reducing descriptor repetition. Preserve and revise it.','grid_quantiles':[.5,.75,.95],'calibration_seeds':{'gaussian':88610,'uniform':88611},'development_seeds':[88612,88613,88614,88615],'holdout_seeds':list(range(88650,88658)),'count_per_request':8,'pool_per_request':128,'guards':{'selected_grid4_max_ratio':1,'nearest_shape_min_ratio':1.25,'coarse_contrast_min_ratio':.85,'clipping_max_absolute_increase':.001},'selection':'Highest nearest shape distance among screens passing all guards on development pools'}
    (ROOT/'evaluation/sampling_revision_plan.json').write_text(json.dumps(plan,indent=2));event('sampling_revision_plan_frozen',plan=plan)
    runner=GraphRunner(DEFAULT_MODEL);screens={q:{'model_sha256':sha(DEFAULT_MODEL),'distributions':{}} for q in plan['grid_quantiles']}
    for dist,seed in plan['calibration_seeds'].items():
        z=noise(np.random.default_rng(seed),128,dist);y=np.concatenate([runner(z[i:i+8]) for i in range(0,128,8)]);s=artifacts(y)
        for q,screen in screens.items():screen['distributions'][dist]={'seed':seed,'count':128,'grid_quantile':q,'limits':{'grid4_max':float(np.quantile(s['grid4'],q)),'clipped_max':float(np.clip(np.quantile(s['clipped'],.95),.0001,.01)),'coarse_contrast_min':float(np.quantile(s['coarse_contrast'],.05))}}
    cached={}
    for seed in plan['development_seeds']:
        z=noise(np.random.default_rng(seed),128,'gaussian');cached[seed]=np.concatenate([runner(z[i:i+8]) for i in range(0,128,8)])
    development=[]
    for q,screen in screens.items():
        path=ROOT/f'evaluation/screen_q{int(q*100)}.json';path.write_text(json.dumps(screen,indent=2));sampler=PortraitSampler(screen=path);raw=[];chosen=[]
        for seed in plan['development_seeds']:
            sampler.render=lambda z,expected=cached[seed]:expected.copy()
            selected,_,report,base=sampler.generate(8,128,seed);raw.extend(base);chosen.extend(selected)
        result=summarize(np.array(raw),np.array(chosen));r=result['raw_artifacts'];s=result['selected_artifacts']
        checks={'grid4':s['grid4']<=r['grid4'],'novelty':result['selected_sequence']['mirror_invariant_nearest_shape_distance']>=result['raw_sequence']['mirror_invariant_nearest_shape_distance']*1.25,'contrast':s['coarse_contrast']>=r['coarse_contrast']*.85,'clipping':s['clipped']<=r['clipped']+.001}
        row={'grid_quantile':q,'screen':str(path.relative_to(ROOT)),'metrics':result,'guards':checks,'eligible':all(checks.values())};development.append(row);event('sampling_revision_development',result=row)
    (ROOT/'evaluation/sampling_revision_development.json').write_text(json.dumps(development,indent=2))
    eligible=[r for r in development if r['eligible']];assert eligible,'No sampler met quality/novelty constraints; retain model-only workflow.'
    selected=max(eligible,key=lambda r:r['metrics']['selected_sequence']['mirror_invariant_nearest_shape_distance'])
    frozen=ROOT/'checkpoints/007_frozen_sampler';frozen.mkdir(exist_ok=False)
    selection={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'chosen_quantile':selected['grid_quantile'],'source_screen':selected['screen'],'holdout_opened':False,'plan':plan}
    (frozen/'selection.json').write_text(json.dumps(selection,indent=2));shutil.copy2(ROOT/selected['screen'],frozen/'artifact_screen.json');shutil.copy2(ROOT/selected['screen'],ROOT/'evaluation/artifact_screen.json');event('sampling_revision_selected',selection=selection)
    sampler=PortraitSampler();raw=[];chosen=[];reports=[]
    for seed in plan['holdout_seeds']:
        y,_,report,base=sampler.generate(8,128,seed);raw.extend(base);chosen.extend(y);reports.append(report)
    raw=np.array(raw);chosen=np.array(chosen);result=summarize(raw,chosen)
    result.update({'plan':plan,'selected_quantile':selected['grid_quantile'],'requests':reports,'inference_work_multiplier':16,'native_litert':False,'threshold_fallbacks':sum(e['novelty_threshold_unmet'] for r in reports for e in r['selected'])})
    (ROOT/'evaluation/sampler_results.json').write_text(json.dumps(result,indent=2));event('sampling_revision_holdout',summary={k:v for k,v in result.items() if k!='requests'})
    grid(raw[:16],ROOT/'figures/random_batch.png',labels=[f'random {i}' for i in range(16)])
    grid(chosen[:16],ROOT/'figures/repeat_aware_batch.png',labels=[f'selected {i}' for i in range(16)])
    np.savez_compressed(ROOT/'evaluation/sampler_preview.npz',random=raw[:16],selected=chosen[:16])
    state=ROOT/'evaluation/test_recent_history_v2.npz';assert not state.exists()
    a=PortraitSampler(history=state);_,_,r1,_=a.generate(4,64,88641)
    b=PortraitSampler(history=state);assert len(b.history)==4
    _,_,r2,_=b.generate(4,64,88641);i1={e['pool_index'] for e in r1['selected']};i2={e['pool_index'] for e in r2['selected']};assert not i1&i2 and len(b.history)==8
    checks={'loaded_history_count':4,'updated_history_count':8,'same_candidate_pool':True,'selected_indices_disjoint':True}
    (ROOT/'evaluation/sampler_validation.json').write_text(json.dumps(checks,indent=2));event('sampling_revision_history_check',checks=checks)

if __name__=='__main__':main()
