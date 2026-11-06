"""Native-runtime acceptance check to run when LiteRT or TensorFlow is installed.

Usage: python round4/validate_native.py
Exits 2 if the runtime is unavailable; this is not a passed native check.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import torch
from round4.refine_structure import ROOT,RefinedGenerator

def main():
    runtime=None
    try:
        from ai_edge_litert.interpreter import Interpreter
        runtime='ai_edge_litert'
    except ImportError:
        try:
            from tensorflow.lite import Interpreter
            runtime='tensorflow.lite'
        except ImportError:
            print('Native check NOT RUN: install ai-edge-litert or TensorFlow in a compatible environment.',file=sys.stderr);return 2
    delta=dict(np.load(ROOT/'checkpoints/012_frozen_selection/edits.npz'));smooth=float(delta.pop('smooth32'));module=RefinedGenerator(smooth,delta).eval()
    path=ROOT/'candidates/model_compact_refined.tflite';interpreter=Interpreter(model_path=str(path));interpreter.allocate_tensors()
    tests=[];rng=np.random.default_rng(88505)
    for n in [1,2,3]:
        inp=interpreter.get_input_details()[0];interpreter.resize_tensor_input(inp['index'],[n,512],strict=True);interpreter.allocate_tensors()
        inp=interpreter.get_input_details()[0];out=interpreter.get_output_details()[0]
        assert inp['dtype']==np.float32 and out['dtype']==np.float32
        z=rng.standard_normal((n,512)).astype('float32');interpreter.set_tensor(inp['index'],z);interpreter.invoke();actual=interpreter.get_tensor(out['index'])
        with torch.inference_mode():expected=module(torch.from_numpy(z)).numpy()
        assert actual.shape==(n,128,128,3) and np.isfinite(actual).all()
        assert np.allclose(actual,expected,atol=3e-5,rtol=3e-5)
        tests.append({'batch':n,'max_absolute_error':float(np.abs(actual-expected).max())})
    aliases=interpreter.get_signature_list();assert aliases['serving_default']['inputs']==['input_38'] and aliases['serving_default']['outputs']==['activation_27']
    record={'runtime':runtime,'native_litert_tested':True,'tests':tests,'signatures':aliases}
    (ROOT/'evaluation/native_validation.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2));return 0

if __name__=='__main__':sys.exit(main())
