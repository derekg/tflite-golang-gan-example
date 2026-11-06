"""Execute the exported FlatBuffer graph, independently of the module topology.

Frozen round6 executor plus depth-multiplier-one DepthwiseConv2D for round9. Both Torch
and NumPy arithmetic backends are available. This is not a native LiteRT runtime.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import torch
import torch.nn.functional as F
from inspect_model import Model

class GraphRunner:
    def __init__(self,path,backend='torch'):
        self.model=Model(path);self.backend=backend
        self.constants={i:a.copy() for i in range(len(self.model.tensors)) if (a:=self.model.array(i)) is not None}
        assert backend in ('torch','numpy')
    def convolution(self,x,w,b,padding):
        kh,kw=w.shape[1:3];ph,pw=(kh//2,kw//2) if padding==0 else (0,0)
        if self.backend=='torch':
            y=F.conv2d(torch.from_numpy(np.ascontiguousarray(x.transpose(0,3,1,2))),torch.from_numpy(np.ascontiguousarray(w.transpose(0,3,1,2))),torch.from_numpy(b),padding=(ph,pw))
            return y.permute(0,2,3,1).numpy()
        xp=np.pad(x,((0,0),(ph,ph),(pw,pw),(0,0)))
        n,h,wi,c=x.shape;oh=xp.shape[1]-kh+1;ow=xp.shape[2]-kw+1
        y=np.zeros((n,oh,ow,w.shape[0]),dtype='float32')
        for dy in range(kh):
            for dx in range(kw):y+=xp[:,dy:dy+oh,dx:dx+ow,:]@w[:,dy,dx,:].T
        return y+b
    def transpose_conv(self,x,w,b,shape,stride,padding):
        _,oh,ow,co=shape;n,h,wi,ci=x.shape;kh,kw=w.shape[1:3]
        ph=max((h-1)*stride+kh-oh,0)//2 if padding==0 else 0
        pw=max((wi-1)*stride+kw-ow,0)//2 if padding==0 else 0
        if self.backend=='torch':
            y=F.conv_transpose2d(torch.from_numpy(np.ascontiguousarray(x.transpose(0,3,1,2))),torch.from_numpy(np.ascontiguousarray(w.transpose(3,0,1,2))),stride=stride)
            y=y[:,:,ph:ph+oh,pw:pw+ow]+torch.from_numpy(b)[None,:,None,None]
            return y.permute(0,2,3,1).numpy()
        y=np.zeros((n,oh,ow,co),dtype='float32')
        for dy in range(kh):
            for dx in range(kw):
                ys=np.arange(h)*stride+dy-ph;xs=np.arange(wi)*stride+dx-pw
                iy=np.where((ys>=0)&(ys<oh))[0];ix=np.where((xs>=0)&(xs<ow))[0]
                if len(iy) and len(ix):y[:,ys[iy,None],xs[ix][None,:],:]+=x[:,iy[:,None],ix[None,:],:]@w[:,dy,dx,:].T
        return y+b
    def depthwise(self,x,w,b):
        assert w.shape[0]==1 and w.shape[3]==x.shape[3]
        kh,kw=w.shape[1:3];channels=x.shape[3]
        if self.backend=='torch':
            y=F.conv2d(torch.from_numpy(np.ascontiguousarray(x.transpose(0,3,1,2))),torch.from_numpy(np.ascontiguousarray(w.transpose(3,0,1,2))),torch.from_numpy(b),groups=channels)
            return y.permute(0,2,3,1).numpy()
        oh=x.shape[1]-kh+1;ow=x.shape[2]-kw+1
        y=np.zeros((len(x),oh,ow,channels),dtype='float32')
        for dy in range(kh):
            for dx in range(kw):y+=x[:,dy:dy+oh,dx:dx+ow,:]*w[0,dy,dx,:]
        return y+b
    @torch.inference_mode()
    def __call__(self,z,trace=False):
        m=self.model;values=dict(self.constants);values[int(m.graph.vector(1)[0])]=np.asarray(z,dtype='float32');traces=[]
        for i,op in enumerate(m.ops):
            code=m.opcodes[op.scalar(0,'I')];ids=op.vector(1);args=[values[int(j)] for j in ids];opt=op.table(4) if op.field(4) else None
            if code==9:
                x,w,b=args;assert opt.scalar(0,'b')==0
                y=(F.linear(torch.from_numpy(x),torch.from_numpy(w),torch.from_numpy(b)).numpy() if self.backend=='torch' else x@w.T+b)
            elif code==22:y=args[0].reshape(tuple(int(v) for v in args[1]))
            elif code==98:
                alpha=opt.scalar(0,'f');y=np.maximum(args[0],0)+alpha*np.minimum(args[0],0)
            elif code==77:y=np.asarray(args[0].shape,dtype='int32')
            elif code==45:
                assert opt.scalar(4)==1 and len(args[0].shape)==1
                y=np.asarray(args[0][int(args[1][0])],dtype='int32')
            elif code==83:y=np.stack(args,axis=opt.scalar(1))
            elif code==67:
                assert opt.scalar(1)==opt.scalar(2)
                y=self.transpose_conv(args[2],args[1],args[3],tuple(int(v) for v in args[0]),opt.scalar(1),opt.scalar(0,'b'))
            elif code==3:
                assert opt.scalar(1)==opt.scalar(2)==1 and opt.scalar(3,'b')==0
                assert opt.scalar(4,default=1)==opt.scalar(5,default=1)==1
                y=self.convolution(*args,padding=opt.scalar(0,'b'))
            elif code==4:
                assert opt.scalar(0,'b')==1 and opt.scalar(1)==opt.scalar(2)==1
                assert opt.scalar(3)==1 and opt.scalar(4,'b')==0
                assert opt.scalar(5,default=1)==opt.scalar(6,default=1)==1
                y=self.depthwise(*args)
            elif code==5:
                x=args[0];size=opt.scalar(0);n,h,w,c=x.shape;co=c//size**2
                y=x.reshape(n,h,w,size,size,co).transpose(0,1,3,2,4,5).reshape(n,h*size,w*size,co)
            elif code==0:
                assert opt.scalar(0,'b')==0;y=args[0]+args[1]
            elif code==100:
                assert opt.scalar(0,'b')==1
                y=np.pad(args[0],tuple(map(tuple,args[1])),mode='symmetric')
            elif code==55:y=np.maximum(args[0],args[1])
            elif code==57:y=np.minimum(args[0],args[1])
            elif code==14:y=1/(1+np.exp(-args[0]))
            else:raise ValueError((i,code))
            oid=int(op.vector(2)[0]);values[oid]=y
            declared=list(m.tensors[oid].vector(0));actual=list(y.shape)
            assert declared[1:]==actual[1:] if len(actual)>0 else declared==actual,(i,code,declared,actual)
            if trace:traces.append(y.copy())
        output=values[int(m.graph.vector(2)[0])]
        return (output,traces) if trace else output
