"""Native acceptance gate: exit 2 means NOT RUN, not success.

Run on a machine with ai-edge-litert or TensorFlow installed. Compares actual
Interpreter outputs with decoded-graph outputs for both exports and batches 1/2/3.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import torch
from round6.check_graph import GraphRunner

def main():
    try:
        from ai_edge_litert.interpreter import Interpreter
        runtime='ai_edge_litert'
    except ImportError:
        try:
            import tensorflow as tf
            Interpreter=tf.lite.Interpreter;runtime='tensorflow.lite'
        except ImportError:
            print(json.dumps(dict(status='NOT RUN',reason='No native LiteRT/TensorFlow interpreter installed.')))
            return 2
    torch.set_num_threads(4);tests=[];rng=np.random.default_rng(88730)
    for model in sorted((Path(__file__).parent/'candidates').glob('*.tflite')):
        ref=GraphRunner(model)
        for count in [1,2,3]:
            runner=Interpreter(model_path=str(model),num_threads=1)
            index=runner.get_input_details()[0]['index'];runner.resize_tensor_input(index,[count,512],strict=True);runner.allocate_tensors()
            z=rng.standard_normal((count,512)).astype('float32');runner.set_tensor(index,z);runner.invoke()
            y=runner.get_tensor(runner.get_output_details()[0]['index']);expected=ref(z)
            assert y.shape==(count,256,256,3) and np.isfinite(y).all()
            assert np.allclose(y,expected,atol=2e-5,rtol=2e-5)
            tests.append(dict(model=model.name,batch=count,max_error=float(np.max(np.abs(y-expected)))))
    print(json.dumps(dict(status='PASS',runtime=runtime,tests=tests),indent=2));return 0

if __name__=='__main__':sys.exit(main())
