"""Run the decoded float32 TFLite generator with PyTorch (no training)."""
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from inspect_model import Model

torch.set_num_threads(4)

class Generator:
    def __init__(self,path='model.tflite',edits=None):
        self.model=Model(path)
        self.layers=[]
        self.reverse_axes=()
        for op in self.model.ops:
            code=self.model.opcodes[op.scalar(0,'I')]
            if code==105:
                assert op is self.model.ops[-1], 'Only a final output reversal is supported'
                self.reverse_axes=tuple(int(x) for x in self.model.array(int(op.vector(1)[1])))
                continue
            if code not in (77,45,83,22,67,98,14):
                raise ValueError(f'Unsupported operation {code}')
            if code!=67: continue
            ids=op.vector(1); opts=op.table(4)
            wi,bi=int(ids[1]),int(ids[3])
            w=self.model.array(wi).copy(); b=self.model.array(bi).copy()
            if edits:
                w=edits.get(wi,w); b=edits.get(bi,b)
            self.layers.append((torch.from_numpy(w.transpose(3,0,1,2).copy()),torch.from_numpy(b),opts.scalar(1),opts.scalar(0,'b'),wi,bi))
    @torch.inference_mode()
    def __call__(self,z, trace=False):
        x=torch.as_tensor(z,dtype=torch.float32).reshape(-1,512,1,1)
        activations=[]
        for k,(w,b,s,pad,wi,bi) in enumerate(self.layers):
            h,wid=x.shape[2:]
            x=F.conv_transpose2d(x,w,None,stride=s)
            oh,ow=(h*s,wid*s) if pad==0 else tuple(x.shape[2:])
            top=max((x.shape[2]-oh)//2,0); left=max((x.shape[3]-ow)//2,0)
            x=x[:,:,top:top+oh,left:left+ow]+b[None,:,None,None]
            x=F.leaky_relu(x,negative_slope=0.3) if k<9 else torch.sigmoid(x)
            if trace: activations.append(x.clone())
        out=x.permute(0,2,3,1).numpy()
        if self.reverse_axes: out=np.flip(out,axis=self.reverse_axes).copy()
        return (out,activations) if trace else out

def grid(images,path,labels=None,cols=8,scale=2):
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        import subprocess, json
        tmp=Path(str(path)+'.grid.npy'); np.save(tmp,images)
        subprocess.run(['python3',str(Path(__file__).with_name('render_grid.py')),str(tmp),str(path),json.dumps(labels),str(cols),str(scale)],check=True)
        tmp.unlink()
        return
    n=len(images); size=128*scale; bar=22 if labels else 0
    canvas=Image.new('RGB',(cols*size,((n+cols-1)//cols)*(size+bar)),(30,30,30))
    draw=ImageDraw.Draw(canvas)
    for i,a in enumerate(images):
        im=Image.fromarray(np.round(np.clip(a,0,1)*255).astype('uint8')).resize((size,size))
        px=(i%cols)*size; py=(i//cols)*(size+bar)
        canvas.paste(im,(px,py))
        if labels: draw.text((px+4,py+size+3),labels[i],fill='white')
    canvas.save(path)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--model',default='model.tflite'); p.add_argument('--output',default='baseline.png'); p.add_argument('--seed',type=int,default=42); p.add_argument('--count',type=int,default=16); p.add_argument('--scale',type=float,default=1.0); a=p.parse_args()
    g=Generator(a.model); z=np.random.default_rng(a.seed).standard_normal((a.count,512)).astype('float32')*a.scale
    y=np.concatenate([g(z[i:i+4]) for i in range(0,len(z),4)])
    grid(y,a.output,labels=[str(i) for i in range(len(y))]); np.save(Path(a.output).with_suffix('.npy'),y)
    print(y.shape,float(y.min()),float(y.max()),flush=True)
