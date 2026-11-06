"""Function-preserving architecture prototype, ported analytically from v1.

No optimization or training. This is a PyTorch checkpoint, not a TFLite export.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,hashlib,datetime,shutil
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from inspect_model import Model
from generate import Generator,grid

torch.set_num_threads(4)
ROOT=Path(__file__).resolve().parent

class CroppedTranspose(nn.Module):
    """Retain the original asymmetric 3x3/stride-2 geometry exactly."""
    def __init__(self):
        super().__init__();self.conv=nn.ConvTranspose2d(1024,512,3,stride=2)
    def forward(self,x):
        h,w=x.shape[-2:];return self.conv(x)[:,:,:h*2,:w*2]

class PhaseUpsample(nn.Module):
    """Exact 2x2 stride-two transpose convolution as projection + pixel shuffle."""
    def __init__(self,ci,co):
        super().__init__();self.projection=nn.Conv2d(ci,4*co,1,bias=False);self.bias=nn.Parameter(torch.zeros(co))
    def forward(self,x):return F.pixel_shuffle(self.projection(x),2)+self.bias[None,:,None,None]

class SpatialRefiner(nn.Module):
    def __init__(self,channels=64):
        super().__init__();self.conv1=nn.Conv2d(channels,channels,3,padding=1);self.conv2=nn.Conv2d(channels,channels,3,padding=1)
        nn.init.dirac_(self.conv1.weight);nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.weight);nn.init.zeros_(self.conv2.bias)
    def forward(self,x):return x+self.conv2(F.leaky_relu(self.conv1(x),.3))

class RestructuredGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem=nn.Linear(512,1024*2*2)
        self.to4=CroppedTranspose()
        self.upsamplers=nn.ModuleList([PhaseUpsample(ci,co) for ci,co in [(512,256),(256,128),(128,64),(64,64),(64,64)]])
        self.refiner=SpatialRefiner(64)
        self.color=nn.ModuleList([nn.Conv2d(ci,co,1) for ci,co in [(64,32),(32,16),(16,3)]])
    def forward(self,z):
        x=F.leaky_relu(self.stem(z).reshape(-1,1024,2,2),.3)
        x=F.leaky_relu(self.to4(x),.3)
        for k,layer in enumerate(self.upsamplers):
            x=F.leaky_relu(layer(x),.3)
            if k==2:x=self.refiner(x)  # 32x32, before fine-scale upsampling
        for k,layer in enumerate(self.color):
            x=layer(x);x=F.leaky_relu(x,.3) if k<2 else torch.sigmoid(x)
        return x.permute(0,2,3,1)
    def freeze_inherited(self):
        for p in self.parameters():p.requires_grad_(False)
        for p in self.refiner.parameters():p.requires_grad_(True)

@torch.no_grad()
def transfer(source):
    m=Model(source);new=RestructuredGenerator()
    # NCHW flattening order is output channel, row, column.
    active=m.array(13)[:,1:3,1:3,:].copy().reshape(4096,512)
    new.stem.weight.copy_(torch.from_numpy(active))
    new.stem.bias.copy_(torch.from_numpy(np.repeat(m.array(14),4)))
    new.to4.conv.weight.copy_(torch.from_numpy(m.array(15).transpose(3,0,1,2).copy()))
    new.to4.conv.bias.copy_(torch.from_numpy(m.array(16).copy()))
    for layer,wi in zip(new.upsamplers,[17,19,21,23,25]):
        # PixelShuffle phase order: output channel, subpixel row, subpixel col.
        w=m.array(wi);co,kh,kw,ci=w.shape;assert (kh,kw)==(2,2)
        layer.projection.weight.copy_(torch.from_numpy(w.reshape(co*4,ci,1,1).copy()))
        layer.bias.copy_(torch.from_numpy(m.array(wi+1).copy()))
    for layer,wi in zip(new.color,[27,29,33]):
        layer.weight.copy_(torch.from_numpy(m.array(wi).transpose(0,3,1,2).copy()))
        layer.bias.copy_(torch.from_numpy(m.array(wi+1).copy()))
    new.freeze_inherited();return new

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    source=Path('model_improved.tflite');source_hash=sha(source)
    assert source_hash=='38c2106d8fded23210427ca1c27e30ec73acc00887597f2a26c82090fbbe31c9'
    out=ROOT/'checkpoints/001_compact_identity';out.mkdir(parents=True,exist_ok=False)
    (ROOT/'logs').mkdir(exist_ok=True);(ROOT/'figures').mkdir(exist_ok=True)
    start={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'event':'restructure_start','source_sha256':source_hash,'script_sha256':sha(__file__),'training':False}
    (ROOT/'logs/events.jsonl').write_text(json.dumps(start)+'\n')
    shutil.copy2(__file__,out/'restructure.py')
    new=transfer(source).eval();old=Generator(str(source));rng=np.random.default_rng(88430)
    cases=[('zero',np.zeros((1,512),dtype='float32')),('gaussian',rng.standard_normal((128,512)).astype('float32')),('uniform',rng.uniform(-1,1,(128,512)).astype('float32')),('large_gaussian',rng.standard_normal((16,512)).astype('float32')*3)]
    results=[];preview=[]
    with torch.inference_mode():
        for name,z in cases:
            a=[];b=[]
            for i in range(0,len(z),8):
                a.append(old(z[i:i+8]));b.append(new(torch.from_numpy(z[i:i+8])).numpy())
            a=np.concatenate(a);b=np.concatenate(b)
            result={'input':name,'count':len(z),'max_absolute_error':float(np.abs(a-b).max()),'rms_error':float(np.sqrt(np.mean((a-b)**2))),'finite':bool(np.isfinite(b).all()),'within_tolerance':bool(np.allclose(a,b,atol=2e-5,rtol=2e-5))}
            assert result['within_tolerance'] and result['finite'],result
            results.append(result)
            if name=='gaussian':
                preview=np.stack([a[:8],b[:8]],axis=1).reshape(-1,128,128,3)
    # A standalone negative-valued feature probe checks the new block itself.
    probe=torch.linspace(-3,3,64*7*9).reshape(1,64,7,9)
    with torch.no_grad():assert torch.equal(new.refiner(probe),probe)
    weights=out/'model_compact_identity.pt'
    torch.save(new.state_dict(),weights)
    # Round-trip state loading must preserve the exported model exactly.
    loaded=RestructuredGenerator().eval();loaded.load_state_dict(torch.load(weights,weights_only=True));loaded.freeze_inherited()
    with torch.inference_mode():
        check=torch.from_numpy(cases[1][1][:8]);assert torch.equal(new(check),loaded(check))
    params=sum(p.numel() for p in new.parameters());trainable=sum(p.numel() for p in new.parameters() if p.requires_grad)
    m=Model(source);parent=sum(a.size for i in range(len(m.tensors)) if (a:=m.array(i)) is not None and a.dtype==np.float32)
    record={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source':str(source),'source_sha256':source_hash,'checkpoint':str(weights.relative_to(ROOT)),'checkpoint_sha256':sha(weights),'source_bytes':source.stat().st_size,'checkpoint_bytes':weights.stat().st_size,'parent_float_parameters':int(parent),'prototype_parameters':params,'refiner_parameters':trainable,'inherited_parameters_frozen':True,'parameter_reduction_percent':100*(1-params/parent),'refiner_initialization':'first convolution center identity, final convolution and biases zero; no training','refiner_identity_probe_exact':True,'state_round_trip_exact':True,'tests':results,'tflite_exported':False,'native_litert_tested':False,'quality_or_diversity_improvement_claimed':False}
    assert sha(source)==source_hash
    (out/'manifest.json').write_text(json.dumps(record,indent=2))
    with (ROOT/'logs/events.jsonl').open('a') as f:f.write(json.dumps({'event':'checkpoint_validated',**record})+'\n')
    grid(preview,ROOT/'figures/v1_compact_identity.png',labels=[f'{i//2} '+('v1' if i%2==0 else 'compact + new block') for i in range(16)],cols=8)
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()
