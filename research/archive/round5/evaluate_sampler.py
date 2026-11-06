"""Freeze artifact screens and check the repeat-aware policy on unseen batches."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,datetime
import numpy as np
from generate import grid
from round4.check_graph import GraphRunner
from round2.run_experiments import repetition
from round5.portrait_sampler import ROOT,DEFAULT_MODEL,PortraitSampler,noise,artifacts,sha
from round5.combine import event

def main():
    plan={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'artifact_calibration_seeds':{'gaussian':88610,'uniform':88611},'artifact_calibration_count':128,'screen_rules':{'grid4_max':'95th percentile','clipped_max':'95th percentile, with floor 0.0001 and ceiling 0.01','coarse_contrast_min':'5th percentile'},'evaluation_seeds':list(range(88620,88628)),'count_per_request':8,'pool_per_request':128,'history_max':128,'distance_threshold':.10,'evaluation_distribution':'gaussian','no_mirroring':True,'model_sha256':sha(DEFAULT_MODEL)}
    (ROOT/'evaluation/sampler_plan.json').write_text(json.dumps(plan,indent=2));event('sampler_plan_frozen',plan=plan)
    runner=GraphRunner(DEFAULT_MODEL);screen={'model_sha256':sha(DEFAULT_MODEL),'distributions':{}}
    for distribution,seed in plan['artifact_calibration_seeds'].items():
        z=noise(np.random.default_rng(seed),128,distribution);y=np.concatenate([runner(z[i:i+8]) for i in range(0,128,8)]);s=artifacts(y)
        screen['distributions'][distribution]={'seed':seed,'count':128,'limits':{'grid4_max':float(np.quantile(s['grid4'],.95)),'clipped_max':float(np.clip(np.quantile(s['clipped'],.95),.0001,.01)),'coarse_contrast_min':float(np.quantile(s['coarse_contrast'],.05))}}
    (ROOT/'evaluation/artifact_screen.json').write_text(json.dumps(screen,indent=2));event('artifact_screen_frozen',screen=screen)
    sampler=PortraitSampler(history=None);raw=[];selected=[];reports=[]
    for seed in plan['evaluation_seeds']:
        a,z,r,base=sampler.generate(8,128,seed);raw.extend(base);selected.extend(a);reports.append(r)
    raw=np.array(raw);selected=np.array(selected)
    record={'plan':plan,'raw_sequence':repetition(raw),'selected_sequence':repetition(selected),'raw_artifacts':{k:float(v.mean()) for k,v in artifacts(raw).items()},'selected_artifacts':{k:float(v.mean()) for k,v in artifacts(selected).items()},'requests':reports,'inference_work_multiplier':16,'native_litert':False,'claim':'Descriptor-based recent repeat avoidance over this fixed evaluation sequence; not a face identity benchmark or proof of artistic quality.'}
    (ROOT/'evaluation/sampler_results.json').write_text(json.dumps(record,indent=2));event('sampler_evaluated',summary={k:v for k,v in record.items() if k!='requests'})
    grid(raw[:16],ROOT/'figures/random_batch.png',labels=[f'random {i}' for i in range(16)])
    grid(selected[:16],ROOT/'figures/repeat_aware_batch.png',labels=[f'selected {i}' for i in range(16)])
    np.savez_compressed(ROOT/'evaluation/sampler_preview.npz',random=raw[:16],selected=selected[:16])
    # Persistence test: reopen memory and present the identical candidate pool.
    state=ROOT/'evaluation/test_recent_history.npz';assert not state.exists()
    one=PortraitSampler(history=state);_,_,r1,_=one.generate(4,64,88640)
    two=PortraitSampler(history=state);assert len(two.history)==4
    _,_,r2,_=two.generate(4,64,88640)
    i1={e['pool_index'] for e in r1['selected']};i2={e['pool_index'] for e in r2['selected']};assert not i1&i2
    assert len(two.history)==8
    check={'history_round_trip_count':4,'history_after_second_request':8,'identical_candidate_pool_reused':True,'same_pool_selected_indices_disjoint':True,'first_indices':sorted(i1),'second_indices':sorted(i2)}
    (ROOT/'evaluation/sampler_validation.json').write_text(json.dumps(check,indent=2));event('sampler_history_checked',checks=check)

if __name__=='__main__':main()
