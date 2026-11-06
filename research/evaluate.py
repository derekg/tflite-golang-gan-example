"""Reproduce frozen original-versus-round11 measurements; no model changes."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from graph import GraphRunner
from metrics import stats
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--count',type=int,default=256)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.count<2:p.error('count must be at least 2')
    if a.output.exists():p.error('Choose a new output path; existing results are preserved')
    torch.set_num_threads(4)
    manifest=json.loads((ROOT/'web/models.json').read_text());runners=[]
    for name in ['original','round11']:
        spec=manifest[name];path=(ROOT/'web'/spec['url']).resolve()
        assert hashlib.sha256(path.read_bytes()).hexdigest()==spec['sha256']
        runners.append(GraphRunner(path))
    result={}
    for law,seed in [('gaussian',89202),('uniform',89203),('legacy_uniform_0_1',89304)]:
        rng=np.random.default_rng(seed)
        z=(rng.standard_normal((a.count,512)) if law=='gaussian' else rng.uniform(-1 if law=='uniform' else 0,1,(a.count,512))).astype('float32')
        outputs=[np.concatenate([g(z[i:i+8]) for i in range(0,a.count,8)]) for g in runners]
        old,new=outputs;result[law]=dict(count=a.count,seed=seed,baseline=stats(old,old),round11=stats(old,new))
        print(law,'complete',flush=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
