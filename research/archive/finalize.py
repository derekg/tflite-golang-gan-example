"""Freeze selected edits, patch constant buffers, and evaluate unseen seeds."""
import hashlib,json
from pathlib import Path
import numpy as np
from inspect_model import Model
from generate import Generator,grid
from experiment import metrics,samples
from verify_forward import verify

def sha(b):return hashlib.sha256(b).hexdigest()

def patch(source,target,edits_path):
    m=Model(source); data=bytearray(m.data); intervals=[]; tensors=[]
    edits={int(k):v for k,v in np.load(edits_path).items()}
    for i,a in sorted(edits.items()):
        old=m.array(i); assert old.shape==a.shape and old.dtype==a.dtype and np.isfinite(a).all()
        pos=m.offset(i); end=pos+a.nbytes
        assert data[pos:end]==old.tobytes()
        data[pos:end]=a.tobytes(); intervals.append((pos,end))
        tensors.append({'index':i,'shape':list(a.shape),'offset':pos,'nbytes':a.nbytes,'changed_parameters':int(np.count_nonzero(a!=old)),'relative_l2_change':float(np.linalg.norm((a-old).ravel())/np.linalg.norm(old.ravel()))})
    # Every byte outside the nominated constant buffers must be untouched.
    last=0
    for a,b in sorted(intervals):
        assert m.data[last:a]==data[last:a]; last=b
    assert m.data[last:]==data[last:]
    Path(target).write_bytes(data)
    loaded=Model(target)
    for i,a in edits.items(): assert np.array_equal(loaded.array(i),a)
    params=sum(m.array(i).size for i in range(len(m.tensors)) if m.array(i) is not None and m.array(i).dtype==np.float32)
    return {'source':source,'target':target,'source_sha256':sha(m.data),'target_sha256':sha(data),'size_bytes':len(data),'stored_float_parameters':params,'changed_float_parameters':sum(t['changed_parameters'] for t in tensors),'tensors':tensors,'configuration':{'dc_strength':.6,'spectral_alpha':.1,'gain':1.15,'selection_seed':42,'selection_count':64,'calibration_seed':7391,'calibration_count':128},'structure_and_other_bytes_unchanged':True}

if __name__=='__main__':
    manifest=patch('model.tflite','model_improved.tflite','results/s10g115_edits.npz')
    Path('results/patch_manifest.json').write_text(json.dumps(manifest,indent=2))
    base=Generator(); edited=Generator('model_improved.tflite')
    verification={'original':verify(),'edited':verify('model_improved.tflite'),'native_litert_tested':False}
    Path('results/forward_validation.json').write_text(json.dumps(verification,indent=2))
    results={}
    for name,seed,distribution in [('normal',20261103,'normal'),('uniform',20261104,'uniform')]:
        rng=np.random.default_rng(seed)
        z=(rng.standard_normal((512,512)) if distribution=='normal' else rng.uniform(-1,1,(512,512))).astype('float32')
        y0=samples(base,z); y1=samples(edited,z)
        a,b=metrics(y0),metrics(y1)
        results[name]={'seed':seed,'count':len(z),'original':a,'edited':b,'relative_change_percent':{k:float(100*(b[k]/a[k]-1)) for k in a if isinstance(a[k],float)},'phase_change_percent':{k:float(100*(b['phase_rms'][k]/a['phase_rms'][k]-1)) for k in a['phase_rms']}}
        print(name,json.dumps(results[name]),flush=True)
        # Matched pairs, left original / right edited, no cherry-picking.
        pics=np.stack([y0[:16],y1[:16]],axis=1).reshape(-1,128,128,3)
        labels=[f'{i//2}: '+('original' if i%2==0 else 'edited') for i in range(32)]
        grid(pics,f'results/{name}_before_after.png',labels=labels,cols=8,scale=2)
        np.savez(f'results/{name}_preview.npz',original=y0[:16],edited=y1[:16],latents=z[:16])
    Path('results/heldout_metrics.json').write_text(json.dumps(results,indent=2))
    assert sha(Path('model.tflite').read_bytes())==manifest['source_sha256']
    print('Original preserved; manifest, model and held-out evaluation written.',flush=True)
