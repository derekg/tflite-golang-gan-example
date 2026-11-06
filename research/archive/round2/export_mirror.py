"""Add an explicit horizontal REVERSE_V2 output operation, without altering weights.

This is a graph transform, not a weight-space discovery or 3D pose change.
Existing immutable tables are reused byte-for-byte in a 16-byte-aligned tail.
Only the new root, graph vectors, output tensor, axis, op and signature are built.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,hashlib
import flatbuffers
import numpy as np
from inspect_model import Model
from generate import Generator
from verify_forward import verify

def export(source,target):
    m=Model(source); b=flatbuffers.Builder(len(m.data)+16384)
    assert len(m.root.tables(2))==1
    assert not m.root.field(8) and not m.root.field(9),'External buffers unsupported'
    for buf in m.buffers:assert not buf.scalar(1,'Q') and not buf.scalar(2,'Q')
    # Preserve all original relative references by keeping the entire source tail.
    b.Prep(16,len(m.data)); b.head-=len(m.data);b.Bytes[b.head:b.head+len(m.data)]=m.data
    def ref(pos):return len(m.data)-pos if pos else 0
    def old(t,n):return ref(t.indirect(n))
    def vec_int(vals):return b.CreateNumpyVector(np.array(vals,dtype='<i4'))
    def vec_ref(vals):
        b.StartVector(4,len(vals),4)
        for v in reversed(vals):b.PrependUOffsetTRelative(v)
        return b.EndVector()
    def obj(n,offsets={},ints={},uints={},bytes_={},bools={}):
        b.StartObject(n)
        for k,v in offsets.items():
            if v:b.PrependUOffsetTRelativeSlot(k,v,0)
        for k,v in ints.items():b.PrependInt32Slot(k,v,0)
        for k,v in uints.items():b.PrependUint32Slot(k,v,0)
        for k,v in bytes_.items():b.PrependInt8Slot(k,v,0)
        for k,v in bools.items():b.PrependBoolSlot(k,v,False)
        return b.EndObject()
    n=len(m.tensors);axis_idx=n;out_idx=n+1;old_out=int(m.graph.vector(2)[0]);ot=m.tensors[old_out]
    axis_data=b.CreateByteVector(np.array([2],dtype='<i4').tobytes())
    axis_buf=obj(3,{0:axis_data})
    axis_shape=vec_int([1]);axis_name=b.CreateString('horizontal_mirror_axis')
    axis_tensor=obj(10,{0:axis_shape,3:axis_name},uints={2:len(m.buffers)},bytes_={1:2})
    output_name=b.CreateString('mirrored_output:0')
    output_tensor=obj(10,{0:old(ot,0),3:output_name,7:old(ot,7)},bytes_={1:ot.scalar(1,'b')},bools={8:bool(ot.scalar(8,'?'))})
    op_inputs=vec_int([old_out,axis_idx]);op_outputs=vec_int([out_idx])
    reverse_op=obj(14,{1:op_inputs,2:op_outputs},uints={0:len(m.opcodes)})
    reverse_code=obj(4,ints={2:1,3:105},bytes_={0:105})
    tensors=vec_ref([ref(t.pos) for t in m.tensors]+[axis_tensor,output_tensor])
    ops=vec_ref([ref(o.pos) for o in m.ops]+[reverse_op]);outputs=vec_int([out_idx])
    graph=obj(6,{0:tensors,1:old(m.graph,1),2:outputs,3:ops,4:old(m.graph,4)})
    codes=vec_ref([ref(t.pos) for t in m.root.tables(1)]+[reverse_code])
    buffers=vec_ref([ref(t.pos) for t in m.buffers]+[axis_buf]);graphs=vec_ref([graph])
    signatures=[]
    for s in m.root.tables(7):
        maps=[]
        for t in s.tables(1):
            assert t.scalar(1,'I')==old_out
            maps.append(obj(2,{0:old(t,0)},uints={1:out_idx}))
        sig_outputs=vec_ref(maps)
        signatures.append(obj(5,{0:old(s,0),1:sig_outputs,2:old(s,2),3:old(s,3)},uints={4:s.scalar(4,'I')}))
    sigs=vec_ref(signatures)
    description=b.CreateString('Preserved portrait generator with explicit horizontal output mirror; no learned pose change.')
    root=obj(10,{1:codes,2:graphs,3:description,4:buffers,5:old(m.root,5),6:old(m.root,6),7:sigs},uints={0:m.root.scalar(0,'I')})
    b.Finish(root,file_identifier=b'TFL3');data=bytes(b.Output())
    assert data.endswith(m.data) and (len(data)-len(m.data))%16==0
    Path(target).write_bytes(data)
    new=Model(target)
    assert new.opcodes[-1]==105 and len(new.ops)==len(m.ops)+1
    assert new.graph.vector(2).tolist()==[out_idx]
    assert new.ops[-1].vector(1).tolist()==[old_out,axis_idx]
    assert new.array(axis_idx).tolist()==[2]
    for i in range(len(m.tensors)):
        a=m.array(i);a1=new.array(i)
        if a is not None:assert np.array_equal(a,a1)
    for s in new.root.tables(7):assert s.tables(1)[0].scalar(1,'I')==out_idx
    rng=np.random.default_rng(55103);z=rng.standard_normal((64,512)).astype('float32')
    original=Generator(source)(z);mirrored=Generator(target)(z)
    assert np.array_equal(mirrored,original[:,:,::-1,:])
    record={'source':str(source),'target':str(target),'source_sha256':hashlib.sha256(m.data).hexdigest(),'target_sha256':hashlib.sha256(data).hexdigest(),'source_bytes':len(m.data),'target_bytes':len(data),'added_bytes':len(data)-len(m.data),'weight_changes':0,'new_operation':'REVERSE_V2, axis 2 NHWC','graph_changed':True,'decoded_mirror_equality_count':64,'decoded_mirror_max_error':float(np.max(np.abs(mirrored-original[:,:,::-1,:]))),'reference_check':verify(target),'native_litert_tested':False}
    Path(str(target)+'.json').write_text(json.dumps(record,indent=2))
    return record

if __name__=='__main__':
    print(json.dumps(export(sys.argv[1],sys.argv[2]),indent=2))
