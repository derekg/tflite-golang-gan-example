"""Independent NHWC NumPy scatter implementation validates the decoded forward pass.

This checks reconstruction arithmetic; it does not substitute for a native LiteRT
compatibility run, which needs a separately installed interpreter.
"""
import json
import numpy as np
from generate import Generator

def reference(g,z):
    x=np.array(z,dtype='float32').reshape(-1,1,1,512)
    traces=[]
    for k,(wt,bt,s,pad,wi,bi) in enumerate(g.layers):
        w=wt.numpy().transpose(1,2,3,0); b=bt.numpy()
        n,h,width,ci=x.shape; co,kh,kw,_=w.shape
        oh,ow=(h*s,width*s) if pad==0 else ((h-1)*s+kh,(width-1)*s+kw)
        ph=max((h-1)*s+kh-oh,0)//2; pw=max((width-1)*s+kw-ow,0)//2
        y=np.zeros((n,oh,ow,co),dtype='float32')
        for dy in range(kh):
            for dx in range(kw):
                ys=np.arange(h)*s+dy-ph; xs=np.arange(width)*s+dx-pw
                iy=np.where((ys>=0)&(ys<oh))[0]; ix=np.where((xs>=0)&(xs<ow))[0]
                if not len(iy) or not len(ix):continue
                v=x[:,iy[:,None],ix[None,:],:]@w[:,dy,dx,:].T
                y[:,ys[iy,None],xs[ix][None,:],:]+=v
        y+=b
        x=np.maximum(y,0)+.3*np.minimum(y,0) if k<9 else 1/(1+np.exp(-y))
        traces.append(x.copy())
    if g.reverse_axes: x=np.flip(x,axis=g.reverse_axes).copy()
    return x,traces

def verify(path='model.tflite'):
    g=Generator(path); tests=[]
    rng=np.random.default_rng(126)
    for name,z in [('zero',np.zeros((1,512),dtype='float32')),('normal',rng.standard_normal((1,512)).astype('float32')),('uniform',rng.uniform(-1,1,(1,512)).astype('float32'))]:
        y,ta=g(z,trace=True); yr,tb=reference(g,z)
        errs=[float(np.max(np.abs(a.permute(0,2,3,1).numpy()-b))) for a,b in zip(ta,tb)]
        assert np.allclose(y,yr,atol=2e-5,rtol=2e-5),(name,errs)
        tests.append({'input':name,'max_output_error':float(np.max(np.abs(y-yr))),'layer_max_errors':errs})
    return tests

if __name__=='__main__':
    import sys
    print(json.dumps(verify(sys.argv[1] if len(sys.argv)>1 else 'model.tflite'),indent=2))
