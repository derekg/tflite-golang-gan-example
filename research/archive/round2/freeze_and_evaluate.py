"""Freeze development selection, export patches, then evaluate unseen seeds."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import hashlib,json,shutil,datetime
import numpy as np
from generate import Generator,grid
from experiment import samples,metrics
from verify_forward import verify
from round2.run_experiments import ROOT,SOURCE,event,log,load_edits,repetition,sha
from round2.export_mirror import export

def export_weights(source,target,edits):
    g=Generator(str(source));m=g.model;raw=bytearray(m.data);ranges=[];counts=[]
    assert not target.exists()
    for k,a in edits.items():
        old=m.array(k);assert a.shape==old.shape and a.dtype==old.dtype and np.isfinite(a).all()
        p=m.offset(k);raw[p:p+a.nbytes]=a.tobytes();ranges.append((p,p+a.nbytes))
        counts.append({'tensor':k,'changed':int((a!=old).sum()),'relative_l2':float(np.linalg.norm(a-old)/np.linalg.norm(old))})
    tail=0
    for start,end in sorted(ranges):assert raw[tail:start]==m.data[tail:start];tail=end
    assert raw[tail:]==m.data[tail:]
    target.write_bytes(raw);check=Generator(str(target))
    for k,a in edits.items():assert np.array_equal(check.model.array(k),a)
    return {'source_sha256':sha(source),'target_sha256':sha(target),'unchanged_graph_bytes':True,'tensor_changes':counts,'target_bytes':len(raw)}

def main():
    chosen=ROOT/'checkpoints/008_spectrum_0.2_1.2'
    frozen=ROOT/'checkpoints/016_frozen_selection';frozen.mkdir(exist_ok=False)
    shutil.copy2(chosen/'edits.npz',frozen/'edits.npz')
    selection={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'chosen_checkpoint':str(chosen.relative_to(ROOT)),'reason':'Additional alpha 0.2 and gain 1.2 improves nearest shape distance and shape rank in development with modest detail change. Stronger flattening gives larger rank but lower pairwise separation; reject it. Bias edits give negligible structural gains. Reject coarse and approximate mirror weight edits for visible blur or displacement.','holdout_seeds':[88422,88423],'count_per_distribution':512,'source_sha256':sha(SOURCE),'patch_sha256':sha(frozen/'edits.npz'),'selection_frozen_before_holdout':True}
    (frozen/'selection.json').write_text(json.dumps(selection,indent=2))
    shutil.copy2(__file__,ROOT/'source'/('freeze_and_evaluate.'+sha(__file__)[:12]+'.py'))
    event('selection_frozen',**selection)
    log('Step 3 Review and freeze selection',selection['reason']+' The eight principal-direction traversals mainly trade between recurring compositions, illumination and color; no clean yaw-only direction was identified. Select checkpoint 008 before opening the two holdout sets, seeds 88422 and 88423.')
    target=ROOT/'candidates/model_diversity_v2.tflite'
    patch=export_weights(SOURCE,target,load_edits(frozen/'edits.npz'))
    (frozen/'patch_manifest.json').write_text(json.dumps(patch,indent=2))
    shutil.copy2(target,frozen/target.name)
    reference=verify(str(target));(ROOT/'evaluation/v2_forward_validation.json').write_text(json.dumps(reference,indent=2))
    event('weight_model_exported',**patch)
    mirror=export(str(target),str(ROOT/'candidates/model_diversity_v2_mirrored.tflite'))
    event('mirror_graph_exported',target=mirror['target'],target_sha256=mirror['target_sha256'],decoded_max_error=mirror['decoded_mirror_max_error'])
    base=Generator(str(SOURCE));new=Generator(str(target));results={}
    for name,seed in [('normal',88422),('uniform',88423)]:
        rng=np.random.default_rng(seed);z=(rng.standard_normal((512,512)) if name=='normal' else rng.uniform(-1,1,(512,512))).astype('float32')
        a=samples(base,z);b=samples(new,z)
        ma={**metrics(a),**repetition(a)};mb={**metrics(b),**repetition(b)}
        blocks=[]
        for start in range(0,512,64):
            aa=repetition(a[start:start+64]);bb=repetition(b[start:start+64]);key='mirror_invariant_nearest_shape_distance'
            blocks.append({'start':start,'original':aa[key],'edited':bb[key],'relative_change_percent':100*(bb[key]/aa[key]-1)})
        result={'seed':seed,'count':512,'v1':ma,'v2':mb,'relative_change_percent':{k:100*(mb[k]/ma[k]-1) if ma[k] else None for k in ma if isinstance(ma[k],float)},'phase_change_percent':{k:100*(mb['phase_rms'][k]/ma['phase_rms'][k]-1) for k in ma['phase_rms']},'paired_blocks_of_64':blocks}
        results[name]=result;event('holdout_evaluated',distribution=name,metrics=result)
        # Save sequential first examples, not selected for attractive outcomes.
        np.savez_compressed(ROOT/f'evaluation/{name}_preview.npz',v1=a[:16],v2=b[:16],latents=z[:16])
        pics=np.stack([a[:16],b[:16],b[:16,:,::-1,:]],axis=1).reshape(-1,128,128,3)
        labels=[f'{i//3} '+['v1','v2','v2 mirrored'][i%3] for i in range(48)]
        grid(pics,ROOT/f'figures/{name}_v1_v2_mirror.png',labels=labels,cols=6)
    (ROOT/'evaluation/heldout_metrics.json').write_text(json.dumps(results,indent=2))
    log('Step 4 Evaluate the frozen models','Evaluated 512 matched standard-normal and 512 matched uniform inputs. Saved full metrics, eight paired blocks per distribution, reference comparisons and sequential image triplets. The explicit mirror model adds REVERSE_V2 on NHWC axis 2; it is a graph-level transformation with zero weight changes and does not count as identity diversity. Native LiteRT execution remains unverified.')
    for p,expected in json.loads((ROOT/'checkpoints/000_preserved_v1/manifest.json').read_text())['files'].items():assert sha(p)==expected['sha256']
    event('holdout_complete',source_models_unchanged=True)

if __name__=='__main__':main()
