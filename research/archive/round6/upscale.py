"""Append analytically initialized 2x RGB upscaling; no training or new detail claim.

Run with python3 round6/upscale.py
Parent tensors and ops are copied without numerical edits. Never overwrite exports.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import datetime, hashlib, json, shutil
import numpy as np
import torch
import torch.nn.functional as F
from inspect_model import Model
from round4.export_compact import Graph
from round5.combine import load_model
from generate import grid

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent / 'round5/candidates/model_clean_diverse.tflite'
PARENT_PT = PARENT.with_suffix('.pt')
EXPECTED = 'a19b85ceffb1eb0f9f8c515cf5e6cfbd0acf5eda861086d9e7de393594de13c3'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value): Path(path).write_text(json.dumps(value, indent=2) + '\n')
def event(name, **fields):
    record = dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), event=name, **fields)
    with (ROOT/'logs/events.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
    print(json.dumps(record), flush=True)

def weights(nearest=False):
    # TF NHWC DCR phase order: output-channel = (dy*2 + dx)*3 + RGB.
    w = np.zeros((12,3,3,3), dtype='float32')
    taps = [np.array([.25,.75,0]), np.array([0,.75,.25])]
    for py in range(2):
        for px in range(2):
            kernel = np.outer(taps[py], taps[px])
            if nearest: kernel = np.zeros((3,3)); kernel[1,1] = 1
            for c in range(3): w[(py*2+px)*3+c,:,:,c] = kernel
    return w

def sharpen_weights(strength):
    w = np.zeros((3,3,3,3), dtype='float32')
    for c in range(3):
        w[c,1,1,c] = 1+strength
        for y,x in [(0,1),(1,0),(1,2),(2,1)]: w[c,y,x,c] = -strength/4
    return w

def conv_head(x, w):
    return F.conv2d(F.pad(x,(1,1,1,1),mode='replicate'), torch.from_numpy(w.transpose(0,3,1,2).copy()))

@torch.inference_mode()
def head(images, strength=0., nearest=False):
    x = torch.from_numpy(images.transpose(0,3,1,2).copy())
    # DCR -> CRD for PyTorch pixel_shuffle.
    w = weights(nearest).reshape(4,3,3,3,3).transpose(1,0,2,3,4).reshape(12,3,3,3)
    y = F.pixel_shuffle(conv_head(x,w),2)
    if strength: y = conv_head(y,sharpen_weights(strength)).clamp(0,1)
    return y.permute(0,2,3,1).numpy()

def clone_parent():
    spec = json.loads(Path(str(PARENT)+'.graph.json').read_text()); m = Model(PARENT); g = Graph()
    for i,t in enumerate(spec['tensors']):
        assert g.tensor(t['name'],t['shape'],m.array(i),np.float32 if t['type']==0 else np.int32,t['dynamic']) == i
    for op in spec['ops']:
        g.op(op['code'],op['inputs'],op['outputs'][0],op['type'],{int(k):tuple(v) for k,v in op['opts'].items()},op['version'])
    return g, int(m.graph.vector(2)[0])

def export(path, strength=0.):
    g,x = clone_parent()
    def padded_conv(name, x, w):
        n,h,wi,c = g.tensors[x]['shape']
        pad = g.constant(name+'/paddings',[[0,0],[1,1],[1,1],[0,0]],np.int32)
        p = g.tensor(name+'/edge_pad',[n,h+2,wi+2,c],dynamic=True)
        g.op(100,[x,pad],p,77,{0:('b',1)}) # MIRROR_PAD, SYMMETRIC; pad1 duplicates edge.
        wt = g.constant(name+'/weights',w); bt = g.constant(name+'/bias',np.zeros(w.shape[0],dtype='float32'))
        y = g.tensor(name+'/conv',[n,h,wi,w.shape[0]],dynamic=True)
        g.op(3,[p,wt,bt],y,1,{0:('b',1),1:('i',1),2:('i',1),4:('i',1),5:('i',1)})
        return y
    x = padded_conv('upscale2x',x,weights())
    y = g.tensor('upscale2x/rgb',[1,256,256,3],dynamic=True); g.op(5,[x],y,94,{0:('i',2)}); x=y
    if strength:
        x=padded_conv('sharpen',x,sharpen_weights(strength))
        zero=g.constant('clip/zero',np.array(0,dtype='float32')); one=g.constant('clip/one',np.array(1,dtype='float32'))
        y=g.tensor('clip/lower',[1,256,256,3],dynamic=True);g.op(55,[x,zero],y,39)
        x=g.tensor('clip/rgb',[1,256,256,3],dynamic=True);g.op(57,[y,one],x,39)
    info=g.build(path,x)
    write(str(path)+'.graph.json',dict(export=info,tensors=g.tensors,ops=g.ops,upscale_strength=strength))
    # Check every inherited tensor, including constants inaccessible from input.
    a,b=Model(PARENT),Model(path)
    for i in range(len(a.tensors)):
        va,vb=a.array(i),b.array(i)
        assert (va is None and vb is None) or np.array_equal(va,vb),i
    return info

def reconstruction(reference, candidate):
    err=np.mean((reference-candidate)**2)
    return dict(mse=float(err), psnr_db=float(-10*np.log10(max(err,1e-20))))

def evaluate(images):
    x=torch.from_numpy(images.transpose(0,3,1,2).copy())
    low=F.avg_pool2d(x,2).permute(0,2,3,1).numpy()
    results={}
    for name,s,near in [('nearest',0,True),('bilinear',0,False),('sharp_010',.1,False),('sharp_025',.25,False)]:
        y=head(images,s,near); small=head(low,s,near)
        back=F.avg_pool2d(torch.from_numpy(y.transpose(0,3,1,2).copy()),2).permute(0,2,3,1).numpy()
        results[name]=dict(self_reconstruction_64_to_128=reconstruction(images,small), roundtrip_128_to_256_to_128=reconstruction(images,back), range=[float(y.min()),float(y.max())],mean_shift=float(y.mean()-images.mean()),edge_energy=float((np.abs(np.diff(y,axis=1)).mean()+np.abs(np.diff(y,axis=2)).mean())/2))
    return results

def main():
    torch.set_num_threads(4)
    for d in ['candidates','checkpoints','evaluation','figures','logs']: (ROOT/d).mkdir(parents=True,exist_ok=True)
    assert sha(PARENT)==EXPECTED
    assert not (ROOT/'evaluation/final_audit.json').exists(), 'Finished run exists; use a new round.'
    protected={str(p.relative_to(ROOT.parent)):sha(p) for p in [ROOT.parent/'model.tflite',ROOT.parent/'model_improved.tflite',PARENT,PARENT_PT]}
    plan=dict(parent_sha256=EXPECTED,input='float32 [batch,512] random noise',output='float32 [batch,256,256,3] RGB [0,1]',training=False,initialization='Analytic separable half-pixel bilinear phase kernels; optional mild discrete-Laplacian sharpening.',default='bilinear for bounded convex interpolation',development_seed=88700,holdout_seeds=[88701,88702],quality_limit='Self reconstruction is a proxy, not a 256px ground truth or portrait quality benchmark.')
    write(ROOT/'evaluation/plan.json',plan);event('plan_frozen',**plan)
    checkpoint=ROOT/'checkpoints/001_analytic_head';checkpoint.mkdir(exist_ok=False)
    np.savez(checkpoint/'weights.npz',bilinear=weights(),sharp_010=sharpen_weights(.1),sharp_025=sharpen_weights(.25))
    write(checkpoint/'manifest.json',dict(parent_sha256=EXPECTED,weights_sha256=sha(checkpoint/'weights.npz'),new_parameters_default=336))
    base=load_model(PARENT_PT)
    def sample(seed,n,uniform=False):
        r=np.random.default_rng(seed);z=(r.uniform(-1,1,(n,512)) if uniform else r.standard_normal((n,512))).astype('float32')
        with torch.inference_mode(): y=np.concatenate([base(torch.from_numpy(v)).numpy() for v in np.array_split(z,max(1,n//8))])
        return z,y
    z,images=sample(88700,32)
    write(ROOT/'evaluation/development.json',evaluate(images));event('development_complete')
    # Numerical head checks exercise borders, channels, subpixel order and constant preservation.
    rng=np.random.default_rng(88710)
    probes={'random':rng.random((2,9,7,3),dtype=np.float32),'constant':np.full((1,5,6,3),.37,dtype='float32'),'ramp':np.linspace(0,1,11*13*3,dtype='float32').reshape(1,11,13,3),'impulse':np.eye(7,dtype='float32')[None,:,:,None].repeat(3,axis=3)}
    checks=[]
    for name,v in probes.items():
        expected=F.interpolate(torch.from_numpy(v.transpose(0,3,1,2).copy()),scale_factor=2,mode='bilinear',align_corners=False).permute(0,2,3,1).numpy()
        out=head(v);error=float(np.max(np.abs(out-expected)));assert error<2e-7
        checks.append(dict(probe=name,max_error_vs_torch_bilinear=error))
    write(ROOT/'evaluation/head_checks.json',checks)
    results={}
    for distribution,seed in [('gaussian',88701),('uniform',88702)]:
        _,y=sample(seed,64,distribution=='uniform');results[distribution]=evaluate(y)
    write(ROOT/'evaluation/holdout.json',results);event('holdout_complete',counts=dict(gaussian=64,uniform=64))
    np.savez_compressed(ROOT/'figures/comparison.npz',latents=z[:8],original=images[:8],bilinear=head(images[:8]),sharp_010=head(images[:8],.1))
    comparison=np.stack([head(images[:8],nearest=True),head(images[:8]),head(images[:8],.1)],axis=1).reshape(-1,256,256,3)
    grid(comparison,ROOT/'figures/comparison.png',labels=[f'{i}: {name}' for i in range(8) for name in ['nearest baseline','analytic 2x','mild sharpen']],cols=3,scale=1)
    models=[]
    for name,s in [('model_upscale_2x',0.),('model_upscale_2x_sharp',.1)]:
        p=ROOT/'candidates'/f'{name}.tflite';info=export(p,s);models.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),strength=s,graph=info));event('model_exported',**models[-1])
    from round6.check_graph import GraphRunner
    tests=[]
    for item in models:
        p=ROOT/item['path'];fast=GraphRunner(p);independent=GraphRunner(p,'numpy')
        for name,testz in [('zero',np.zeros((1,512),dtype='float32')),('normal',z[:3]),('uniform',np.random.default_rng(88711).uniform(-1,1,(2,512)).astype('float32'))]:
            with torch.inference_mode(): expected=head(base(torch.from_numpy(testz)).numpy(),item['strength'])
            out=fast(testz);ref=independent(testz)
            errors=[float(np.max(np.abs(out-expected))),float(np.max(np.abs(ref-expected)))]
            assert np.allclose(out,expected,atol=2e-5,rtol=2e-5) and np.allclose(ref,expected,atol=2e-5,rtol=2e-5),errors
            tests.append(dict(model=item['path'],input=name,batch=len(testz),shape=list(out.shape),decoded_torch_max_error=errors[0],decoded_numpy_max_error=errors[1]))
        event('graph_validated',model=item['path'])
    write(ROOT/'evaluation/graph_validation.json',dict(tests=tests,native_litert_tested=False))
    for rel,h in protected.items(): assert sha(ROOT.parent/rel)==h
    checkpoint=ROOT/'checkpoints/002_delivery';checkpoint.mkdir(exist_ok=False)
    for p in [Path(__file__),ROOT/'check_graph.py',ROOT/'evaluation/plan.json',ROOT/'evaluation/holdout.json',ROOT/'evaluation/graph_validation.json']:
        shutil.copy2(p,checkpoint/p.name)
    audit=dict(parent_files_preserved=protected,models=models,native_litert_tested=False,training_performed=False,output_resolution=[256,256],added_ops_default=3,new_float_parameters_default=336,new_identity_or_detail_claimed=False)
    write(ROOT/'evaluation/final_audit.json',audit);write(checkpoint/'manifest.json',audit);event('delivery_frozen',models=models)

if __name__=='__main__': main()
