"""Optional display-time repetition control; no weight edits or new identities.

Generate a larger pool, choose a batch with farthest-first shape coverage, then
optionally mirror half the displayed images. Mirrored matches count as repeats.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,json,hashlib
import numpy as np
from generate import Generator,grid
from experiment import samples
from round2.run_experiments import features,repetition,event,log

def choose(y,count):
    assert 1<=count<=len(y)
    a=features(y);b=features(y[:,:,::-1,:].copy())
    dist=np.minimum(np.clip(1-a@a.T,0,2),np.clip(1-a@b.T,0,2))
    # Start with a medoid, not the most extreme sample.
    selected=[int(np.argmin(dist.mean(1)))];nearest=dist[selected[0]].copy()
    while len(selected)<count:
        nearest[selected]=-np.inf;i=int(np.argmax(nearest));selected.append(i)
        nearest=np.minimum(nearest,dist[i])
    return np.array(selected,dtype='int32')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default='model_improved.tflite');p.add_argument('--count',type=int,default=16);p.add_argument('--pool',type=int,default=256);p.add_argument('--seed',type=int,default=88424);p.add_argument('--balanced-mirror',action='store_true');p.add_argument('--output',default='round2/figures/diverse_batch.png')
    args=p.parse_args();assert 2<=args.count<=args.pool
    rng=np.random.default_rng(args.seed);z=rng.standard_normal((args.pool,512)).astype('float32')
    y=samples(Generator(args.model),z);indices=choose(y,args.count);out=y[indices].copy()
    flags=np.zeros(args.count,dtype=bool)
    if args.balanced_mirror:
        flags[1::2]=True;out[flags]=out[flags,:,::-1,:]
    record={'model':args.model,'model_sha256':hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),'seed':args.seed,'pool':args.pool,'count':args.count,'indices':indices.tolist(),'mirrored':flags.tolist(),'policy':'Farthest-first using reflection-invariant, mean-centered and contrast-normalized 16×16 RGB shape descriptors. Medoid initialization. No aesthetic or face-identity quality model.','random_first_batch_metrics':repetition(y[:args.count]),'selected_batch_metrics_before_mirroring':repetition(y[indices]),'selected_batch_metrics_after_mirroring':repetition(out),'limitations':'Selection increases coverage of the existing pool, can favor defects or unusual colors, and does not create new identities or pose capability.'}
    path=Path(args.output);path.parent.mkdir(exist_ok=True,parents=True)
    grid(out,path,labels=[f'pool {i}'+(' mirrored' if f else '') for i,f in zip(indices,flags)],cols=8)
    np.savez_compressed(path.with_suffix('.npz'),images=out,latents=z[indices],pool_indices=indices,mirrored=flags)
    path.with_suffix('.json').write_text(json.dumps(record,indent=2))
    # Confirm that the display policy never changes the generator's raw images.
    restored=out.copy();restored[flags]=restored[flags,:,::-1,:]
    assert np.array_equal(restored,y[indices])
    event('diverse_display_batch',**record)
    return record

if __name__=='__main__':main()
