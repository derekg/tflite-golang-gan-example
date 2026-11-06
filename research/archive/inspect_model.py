"""Minimal read-only TFLite FlatBuffer reader; offsets permit exact weight patches."""
import struct
import json
from pathlib import Path
import numpy as np

class Table:
    def __init__(self, data, pos): self.data, self.pos = data, pos
    def field(self, n):
        v = self.pos - struct.unpack_from('<i', self.data, self.pos)[0]
        q = v + 4 + 2*n
        off = struct.unpack_from('<H', self.data, q)[0] if q < v+struct.unpack_from('<H',self.data,v)[0] else 0
        return self.pos+off if off else 0
    def scalar(self, n, fmt='i', default=0):
        p=self.field(n)
        return struct.unpack_from('<'+fmt,self.data,p)[0] if p else default
    def indirect(self, n):
        p=self.field(n)
        return p+struct.unpack_from('<I',self.data,p)[0] if p else 0
    def table(self,n): return Table(self.data,self.indirect(n))
    def vector(self,n, dtype='<i4'):
        p=self.indirect(n)
        return np.frombuffer(self.data,dtype=dtype,count=struct.unpack_from('<I',self.data,p)[0],offset=p+4) if p else np.array([],dtype=dtype)
    def tables(self,n):
        p=self.indirect(n)
        if not p:return []
        return [Table(self.data,q+struct.unpack_from('<I',self.data,q)[0]) for q in range(p+4,p+4+4*struct.unpack_from('<I',self.data,p)[0],4)]
    def string(self,n):
        p=self.indirect(n)
        return self.data[p+4:p+4+struct.unpack_from('<I',self.data,p)[0]].decode() if p else ''

class Model:
    def __init__(self,path='model.tflite'):
        self.data=Path(path).read_bytes()
        assert self.data[4:8]==b'TFL3'
        self.root=Table(self.data,struct.unpack_from('<I',self.data,0)[0])
        self.graph=self.root.tables(2)[0]
        self.buffers=self.root.tables(4)
        self.tensors=self.graph.tables(0)
        self.opcodes=[x.scalar(3,default=x.scalar(0,'b')) for x in self.root.tables(1)]
        self.ops=self.graph.tables(3)
    def array(self,i):
        t=self.tensors[i]; b=self.buffers[t.scalar(2)]
        dtype={0:'<f4',1:'<f2',2:'<i4',3:'u1',4:'<i8',6:'?',9:'i1'}[t.scalar(1,'b')]
        v=b.vector(0,'u1')
        return v.view(dtype).reshape(tuple(t.vector(0))) if len(v) else None
    def offset(self,i): return self.buffers[self.tensors[i].scalar(2)].indirect(0)+4
    def describe(self):
        names={0:'ADD',2:'CONCAT',3:'CONV_2D',6:'DEQUANTIZE',9:'FULLY_CONNECTED',14:'LOGISTIC',17:'MAX_POOL',18:'MUL',19:'RELU',22:'RESHAPE',28:'TANH',34:'PAD',39:'TRANSPOSE',45:'STRIDED_SLICE',65:'SLICE',67:'TRANSPOSE_CONV',97:'RESIZE_NEAREST',98:'LEAKY_RELU'}
        print('description',self.root.string(3),'opcodes',self.opcodes)
        print('INPUT',self.graph.vector(1),'OUTPUT',self.graph.vector(2))
        for i,t in enumerate(self.tensors):
            a=self.array(i)
            stats='' if a is None else f' range={a.min():.5g}..{a.max():.5g} mean={a.mean():.5g} std={a.std():.5g}'
            print(i,t.string(3),list(t.vector(0)),t.scalar(1,'b'),'buffer',t.scalar(2),stats)
        for i,o in enumerate(self.ops):
            code=self.opcodes[o.scalar(0,'I')]
            opts=o.table(4) if o.field(4) else None
            fields=[] if opts is None else [(j,opts.scalar(j),opts.scalar(j,'f')) for j in range(8) if opts.field(j)]
            print('OP',i,names.get(code,code),list(o.vector(1)),'->',list(o.vector(2)),'opts',fields)

if __name__=='__main__': Model().describe()
