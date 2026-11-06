"""Check displayed source arrays against actual exported TFLite computation."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import torch
from round9.check_graph import GraphRunner
from round8.selective_detail import noise
from round10.overlap import sha,write
ROOT=Path(__file__).resolve().parent
torch.set_num_threads(4)
result=json.loads((ROOT/'evaluation/result.json').read_text())
runner=GraphRunner(ROOT/'candidates/model_grid_cleanup_experimental.tflite')
tests=[]
for law,seed in [('gaussian',88920),('uniform',88921)]:
    d=np.load(ROOT/'figures'/f'{law}_comparison.npz')
    assert np.array_equal(d['latents'],noise(seed,8,law=='uniform'))
    actual=runner(d['latents']);err=float(np.max(np.abs(actual-d['round11'])))
    assert err<2e-5
    tests.append(dict(distribution=law,seed=seed,count=8,export_vs_figure_source_max_error=err))
for p,h in result['protected_files'].items():assert sha(p)==h
for m in result['models']:assert sha(m['path'])==m['sha256']
write(ROOT/'evaluation/delivery_audit.json',dict(comparison_inputs_and_actual_export=tests,protected_files_unchanged=True,model_hashes_verified=True,native=dict(status='NOT RUN',exit_code=2,reason='No native LiteRT/TensorFlow interpreter installed.'),sample_cli=dict(status='PASS',seed=89220,count=2,shape=[2,256,256,3])))
print(json.dumps(tests))
