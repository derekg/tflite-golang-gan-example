"""Build a compact TFLite graph from explicit float32 tensors and operations.

No TensorFlow converter is installed. Uses standard TFLite schema fields only;
native loading must be checked separately. Unlike the older mirror exporter,
this rebuilds all buffers and does not retain unused source weights as a tail.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import flatbuffers
import numpy as np
from round4.refine_structure import RefinedGenerator

class Graph:
    def __init__(self):self.tensors=[];self.buffers=[None];self.ops=[]
    def tensor(self,name,shape,value=None,dtype=np.float32,dynamic=False):
        bi=0
        if value is not None:
            value=np.array(value,dtype=dtype,copy=True,order='C');assert list(value.shape)==list(shape)
            self.buffers.append(value.tobytes());bi=len(self.buffers)-1
        typecode={np.dtype('float32'):0,np.dtype('int32'):2}[np.dtype(dtype)]
        self.tensors.append({'name':name,'shape':list(shape),'type':typecode,'buffer':bi,'dynamic':dynamic})
        return len(self.tensors)-1
    def constant(self,name,value,dtype=np.float32):
        value=np.asarray(value,dtype=dtype);return self.tensor(name,value.shape,value,dtype)
    def op(self,code,inputs,output,opt_type=0,opt=None,version=1):
        self.ops.append({'code':code,'inputs':list(inputs),'outputs':[output],'type':opt_type,'opts':opt or {},'version':version})
    def build(self,path,output):
        assert not Path(path).exists(),path
        b=flatbuffers.Builder(40_000_000)
        def ivec(vals):return b.CreateNumpyVector(np.array(vals,dtype='<i4'))
        def refs(vals):
            b.StartVector(4,len(vals),4)
            for v in reversed(vals):b.PrependUOffsetTRelative(v)
            return b.EndVector()
        def table(n,offsets=None,fields=None):
            b.StartObject(n)
            for k,v in (offsets or {}).items():
                if v:b.PrependUOffsetTRelativeSlot(k,v,0)
            for k,(kind,v) in (fields or {}).items():
                if kind=='i':b.PrependInt32Slot(k,v,0)
                elif kind=='I':b.PrependUint32Slot(k,v,0)
                elif kind=='b':b.PrependInt8Slot(k,v,0)
                elif kind=='B':b.PrependUint8Slot(k,v,0)
                elif kind=='?':b.PrependBoolSlot(k,v,False)
                elif kind=='f':b.PrependFloat32Slot(k,v,0)
                else:raise ValueError(kind)
            return b.EndObject()
        bufrefs=[]
        for raw in self.buffers:
            if raw is None:bufrefs.append(table(3));continue
            b.StartVector(1,len(raw),16);b.head-=len(raw);b.Bytes[b.head:b.head+len(raw)]=raw;v=b.EndVector()
            bufrefs.append(table(3,{0:v}))
        tensorrefs=[]
        for t in self.tensors:
            shape=ivec(t['shape']);name=b.CreateString(t['name']);sig=ivec([-1]+t['shape'][1:]) if t['dynamic'] else 0
            tensorrefs.append(table(10,{0:shape,3:name,7:sig},{1:('b',t['type']),2:('I',t['buffer']),8:('?',True)}))
        codepairs=list(dict.fromkeys((o['code'],o['version']) for o in self.ops));coderefs=[]
        for code,version in codepairs:
            coderefs.append(table(4,fields={0:('b',min(code,127)),2:('i',version),3:('i',code)}))
        oprefs=[]
        for o in self.ops:
            ins=ivec(o['inputs']);outs=ivec(o['outputs'])
            opts=table(max(o['opts'],default=-1)+1,fields=o['opts']) if o['type'] else 0
            oprefs.append(table(14,{1:ins,2:outs,4:opts},{0:('I',codepairs.index((o['code'],o['version']))),3:('B',o['type'])}))
        tv=refs(tensorrefs);iv=ivec([0]);ov=ivec([output]);ops=refs(oprefs);gname=b.CreateString('compact_portrait')
        graph=table(6,{0:tv,1:iv,2:ov,3:ops,4:gname})
        alias_in=b.CreateString('input_38');imap=table(2,{0:alias_in},{1:('I',0)})
        alias_out=b.CreateString('activation_27');omap=table(2,{0:alias_out},{1:('I',output)})
        imaps=refs([imap]);omaps=refs([omap]);key=b.CreateString('serving_default')
        signature=table(5,{0:imaps,1:omaps,2:key})
        cv=refs(coderefs);gv=refs([graph]);bv=refs(bufrefs);sv=refs([signature])
        desc=b.CreateString('Compact portrait decoder with explicit spatial refinement and calibrated phase offsets. No training.')
        model=table(10,{1:cv,2:gv,3:desc,4:bv,7:sv},{0:('I',3)})
        b.Finish(model,file_identifier=b'TFL3');Path(path).write_bytes(bytes(b.Output()))
        return {'tensors':len(self.tensors),'ops':len(self.ops),'buffers':len(self.buffers),'opcodes':codepairs,'bytes':Path(path).stat().st_size}

def export(model,path):
    b=model.base;g=Graph()
    def a(t):return t.detach().cpu().numpy()
    def act(name,shape,code,x,opts_type,opts):
        y=g.tensor(name,shape,dynamic=True);g.op(code,[x],y,opts_type,opts);return y
    def conv(name,x,layer):
        n,h,w,ci=g.tensors[x]['shape'];co=layer.weight.shape[0]
        wt=g.constant(name+'/weights',a(layer.weight).transpose(0,2,3,1));bt=g.constant(name+'/bias',a(layer.bias))
        y=g.tensor(name,[n,h,w,co],dynamic=True)
        g.op(3,[x,wt,bt],y,1,{0:('b',0),1:('i',1),2:('i',1),4:('i',1),5:('i',1)})
        return y
    def leaky(name,x):return act(name,g.tensors[x]['shape'],98,x,75,{0:('f',.3)})
    def phase(name,x,offset):
        if not np.any(a(offset)):return x
        shape=g.tensors[x]['shape'];_,h,w,c=shape
        template=np.tile(a(offset).transpose(0,2,3,1),(1,h//2,w//2,1))
        t=g.constant(name+'/offset',template);y=g.tensor(name,shape,dynamic=True);g.op(0,[x,t],y,11);return y
    z=g.tensor('serving_default_input_38:0',[1,512],dynamic=True)
    # Reorder the NCHW flattened dense output to NHWC before reshaping.
    w=a(b.stem.weight).reshape(1024,2,2,512).transpose(1,2,0,3).reshape(4096,512)
    bias=a(b.stem.bias).reshape(1024,2,2).transpose(1,2,0).reshape(4096)
    wi=g.constant('compact_stem/weights',w);bi=g.constant('compact_stem/bias',bias)
    dense=g.tensor('compact_stem',[1,4096],dynamic=True);g.op(9,[z,wi,bi],dense,8)
    shape=g.constant('seed_shape',[-1,2,2,1024],np.int32);x=g.tensor('seed',[1,2,2,1024],dynamic=True);g.op(22,[dense,shape],x,17)
    x=leaky('seed_activation',x)
    # Preserve a dynamic batch output shape for the single transposed convolution.
    inp_shape=g.tensor('input_shape',[2],dtype=np.int32);g.op(77,[z],inp_shape,55,{0:('b',2)})
    begin=g.constant('batch_begin',[0],np.int32);end=g.constant('batch_end',[1],np.int32);stride=g.constant('batch_stride',[1],np.int32)
    batch=g.tensor('batch',[],dtype=np.int32);g.op(45,[inp_shape,begin,end,stride],batch,32,{4:('i',1)})
    four=g.constant('size_four',np.array(4,dtype=np.int32),np.int32);channels=g.constant('channels_512',np.array(512,dtype=np.int32),np.int32)
    outshape=g.tensor('to4_output_shape',[4],dtype=np.int32);g.op(83,[batch,four,four,channels],outshape,59,{0:('i',4),1:('i',0)})
    wt=g.constant('to4/weights',a(b.to4.conv.weight).transpose(1,2,3,0));bt=g.constant('to4/bias',a(b.to4.conv.bias))
    y=g.tensor('to4',[1,4,4,512],dynamic=True);g.op(67,[outshape,wt,x,bt],y,49,{0:('b',0),1:('i',2),2:('i',2)},version=3);x=leaky('to4_activation',y)
    for k,layer in enumerate(b.upsamplers):
        _,h,w,ci=g.tensors[x]['shape'];co=layer.bias.numel()
        # TF NHWC DepthToSpace phase order differs from PyTorch PixelShuffle.
        wt=a(layer.projection.weight)[:,:,0,0].reshape(co,2,2,ci).transpose(1,2,0,3).reshape(co*4,1,1,ci)
        wi=g.constant(f'upsample{k}/weights',wt);zero=g.constant(f'upsample{k}/projection_bias',np.zeros(co*4,dtype='float32'))
        projection=g.tensor(f'upsample{k}/projection',[1,h,w,co*4],dynamic=True)
        g.op(3,[x,wi,zero],projection,1,{1:('i',1),2:('i',1),4:('i',1),5:('i',1)})
        y=g.tensor(f'upsample{k}/depth_to_space',[1,h*2,w*2,co],dynamic=True);g.op(5,[projection],y,94,{0:('i',2)})
        bias=g.constant(f'upsample{k}/bias',a(layer.bias));biased=g.tensor(f'upsample{k}/biased',[1,h*2,w*2,co],dynamic=True);g.op(0,[y,bias],biased,11)
        x=leaky(f'upsample{k}/activation',biased)
        if k==2:
            residual=conv('refiner32/conv1',x,b.refiner.conv1);residual=leaky('refiner32/activation',residual);residual=conv('refiner32/conv2',residual,b.refiner.conv2)
            y=g.tensor('refiner32/add',g.tensors[x]['shape'],dynamic=True);g.op(0,[x,residual],y,11);x=phase('phase32',y,model.phase32)
        if k==3:x=phase('phase64',x,model.phase64)
    for k,layer in enumerate(b.color):
        x=conv(f'color{k}',x,layer)
        if k<2:x=leaky(f'color{k}/activation',x)
    y=act('StatefulPartitionedCall:0',[1,128,128,3],14,x,0,{})
    info=g.build(path,y)
    Path(str(path)+'.graph.json').write_text(json.dumps({'export':info,'tensors':g.tensors,'ops':g.ops},indent=2))
    return info

if __name__=='__main__':
    print(json.dumps(export(RefinedGenerator().eval(),sys.argv[1]),indent=2))
