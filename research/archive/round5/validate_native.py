"""Native acceptance check for the combined model, when a runtime is installed."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import torch
from round5.combine import ROOT,load_model

def main():
    try:
        from ai_edge_litert.interpreter import Interpreter
        runtime='ai_edge_litert'
    except ImportError:
        try:
            from tensorflow.lite import Interpreter
            runtime='tensorflow.lite'
        except ImportError:
            print('Native acceptance NOT RUN: no LiteRT or TensorFlow runtime is installed.',file=sys.stderr);return 2
    module=load_model(ROOT/'candidates/model_clean_diverse.pt');runner=Interpreter(model_path=str(ROOT/'candidates/model_clean_diverse.tflite'));runner.allocate_tensors();rng=np.random.default_rng(88644);tests=[]
    for n in [1,2,3]:
        inp=runner.get_input_details()[0];runner.resize_tensor_input(inp['index'],[n,512],strict=True);runner.allocate_tensors();inp=runner.get_input_details()[0];out=runner.get_output_details()[0]
        z=rng.standard_normal((n,512)).astype('float32');runner.set_tensor(inp['index'],z);runner.invoke();actual=runner.get_tensor(out['index'])
        with torch.inference_mode():expected=module(torch.from_numpy(z)).numpy()
        assert actual.shape==(n,128,128,3) and np.allclose(actual,expected,atol=3e-5,rtol=3e-5)
        tests.append({'batch':n,'max_error':float(np.abs(actual-expected).max())})
    record={'runtime':runtime,'native_tested':True,'tests':tests};(ROOT/'evaluation/native_validation.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2));return 0

if __name__=='__main__':sys.exit(main())
