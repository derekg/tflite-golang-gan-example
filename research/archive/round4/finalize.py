"""Freeze the development choice, export both formats, and evaluate unseen inputs."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,shutil,datetime
import numpy as np
import torch
from generate import grid
from round4.refine_structure import ROOT,PARENT,RefinedGenerator,sample,metrics_pair,event,sha
from round4.export_compact import export
from round4.check_graph import validate,GraphRunner

def main():
    chosen=ROOT/'checkpoints/007_phase64_0.75';folder=ROOT/'checkpoints/012_frozen_selection';folder.mkdir(exist_ok=False)
    shutil.copy2(chosen/'edits.npz',folder/'edits.npz');shutil.copy2(__file__,ROOT/'source/finalize.frozen.py')
    selection={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'choice':'007_phase64_0.75','reason':'Passes all declared development guards. Its four-pixel metric is within 0.1% of the more complex two-stage phase correction; prefer one offset layer and leave the spatial smoothing branch inactive. Spatial smoothing alone did not meet the minimum phase-reduction guard.','holdout_opened':False,'parent_sha256':sha(PARENT),'patch_sha256':sha(folder/'edits.npz')}
    (folder/'selection.json').write_text(json.dumps(selection,indent=2));event('selection_frozen',selection=selection)
    delta=dict(np.load(folder/'edits.npz'));strength=float(delta.pop('smooth32'))
    model=RefinedGenerator(strength,delta).eval();weights=ROOT/'candidates/model_compact_refined.pt'
    assert not weights.exists();torch.save(model.state_dict(),weights)
    graph=ROOT/'candidates/model_compact_refined.tflite';export_info=export(model,graph)
    validation=validate(graph,model)
    (ROOT/'evaluation/export_validation.json').write_text(json.dumps(validation,indent=2))
    base=RefinedGenerator().eval();results={}
    for name,seed in [('gaussian',88502),('uniform',88503)]:
        rng=np.random.default_rng(seed)
        z=(rng.standard_normal((512,512)) if name=='gaussian' else rng.uniform(-1,1,(512,512))).astype('float32')
        a=sample(base,z);b=sample(model,z);ma=metrics_pair(a,a);mb=metrics_pair(a,b)
        r={'seed':seed,'count':len(z),'v1_compact':ma,'refined':mb,'relative_change_percent':{k:100*(mb[k]/ma[k]-1) if ma[k] else None for k in ma if isinstance(ma[k],float)},'phase_change_percent':{k:100*(mb['phase_rms'][k]/ma['phase_rms'][k]-1) for k in ma['phase_rms']}}
        results[name]=r;event('holdout_evaluated',distribution=name,result=r)
        np.savez_compressed(ROOT/f'evaluation/{name}_preview.npz',baseline=a[:16],refined=b[:16],latents=z[:16])
        pics=np.stack([a[:16],b[:16]],axis=1).reshape(-1,128,128,3)
        grid(pics,ROOT/f'figures/{name}_before_after.png',labels=[f'{i//2} '+['compact v1','refined'][i%2] for i in range(32)],cols=8)
        # Check the serialized graph on all held-out inputs as well as modules.
        runner=GraphRunner(graph);decoded=np.concatenate([runner(z[i:i+8]) for i in range(0,len(z),8)])
        assert np.allclose(b,decoded,atol=2e-5,rtol=2e-5)
        r['module_to_serialized_graph_max_error']=float(np.abs(b-decoded).max())
    (ROOT/'evaluation/heldout.json').write_text(json.dumps(results,indent=2))
    record={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'parent_sha256':sha(PARENT),'torch_checkpoint_sha256':sha(weights),'tflite_sha256':sha(graph),'torch_checkpoint_bytes':weights.stat().st_size,'tflite_bytes':graph.stat().st_size,'tflite_size_reduction_from_v1_percent':100*(1-graph.stat().st_size/74217344),'selected_offsets':'phase64 only, strength 0.75','smooth32':0,'spatial_refinement_branch_active':False,'graph':export_info,'native_litert_tested':False,'training_performed':False,'new_identity_or_pose_claimed':False}
    (folder/'manifest.json').write_text(json.dumps(record,indent=2));event('export_and_holdout_complete',manifest=record)

if __name__=='__main__':main()
