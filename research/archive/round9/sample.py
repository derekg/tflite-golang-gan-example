"""Round9 reproducible 256px samples using the decoded graph (not native LiteRT)."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse, json
import numpy as np
import torch
from round9.check_graph import GraphRunner
from generate import grid

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default=str(Path(__file__).parent/'candidates/model_spatial_experimental_2x.tflite'))
    p.add_argument('--seed',type=int,default=88720)
    p.add_argument('--count',type=int,default=8)
    p.add_argument('--output',required=True,help='New .npz file; matching .png also written')
    a=p.parse_args();out=Path(a.output)
    assert a.count>0 and out.suffix=='.npz'
    assert not out.exists() and not out.with_suffix('.png').exists()
    torch.set_num_threads(4)
    z=np.random.default_rng(a.seed).standard_normal((a.count,512)).astype('float32')
    runner=GraphRunner(a.model)
    images=np.concatenate([runner(v) for v in np.array_split(z,max(1,(a.count+7)//8))])
    np.savez_compressed(out,images=images,latents=z)
    grid(images,out.with_suffix('.png'),labels=[f'seed {a.seed}: {i}' for i in range(a.count)],cols=min(4,a.count),scale=1)
    print(json.dumps(dict(shape=list(images.shape),seed=a.seed,native_litert=False,output=str(out))))
