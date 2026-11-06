"""Render samples from the serialized compact graph using the decoded backend.

For native acceptance testing use validate_native.py. This preview runner does
not claim to use a native TFLite/LiteRT interpreter.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,json
import numpy as np
from generate import grid
from round4.check_graph import GraphRunner

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default='round4/candidates/model_compact_refined.tflite');p.add_argument('--count',type=int,default=16);p.add_argument('--seed',type=int,default=88506);p.add_argument('--output',default='round4/figures/compact_preview.png');args=p.parse_args()
    assert args.count>0
    z=np.random.default_rng(args.seed).standard_normal((args.count,512)).astype('float32');g=GraphRunner(args.model)
    y=np.concatenate([g(z[i:i+8]) for i in range(0,len(z),8)])
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    grid(y,out,labels=[str(i) for i in range(len(y))])
    np.savez_compressed(out.with_suffix('.npz'),images=y,latents=z)
    out.with_suffix('.json').write_text(json.dumps({'model':args.model,'count':len(y),'seed':args.seed,'backend':'decoded GraphRunner with PyTorch arithmetic','native_litert':False},indent=2))
    print(out)

if __name__=='__main__':main()
