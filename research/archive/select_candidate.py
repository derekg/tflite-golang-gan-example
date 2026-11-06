import json
from pathlib import Path
import numpy as np
from generate import Generator,grid
from experiment import samples,metrics
from refine import correction,spectral

g=Generator(); means={int(k):v for k,v in np.load('results/channel_means.npz').items()}
z=np.random.default_rng(42).standard_normal((64,512)).astype('float32')
configs=[('original',0,0,1),('gain_only',.6,0,1.1),('s10g115',.6,.1,1.15),('s15g115',.6,.15,1.15),('s15g125',.6,.15,1.25),('s20g125',.6,.2,1.25)]
results={}; pics=[]; labels=[]
for name,dc,alpha,gain in configs:
    edits=correction(g,means,dc) if dc else {}
    if alpha or gain!=1: edits[13],_=spectral(g,alpha,gain)
    np.savez(f'results/{name}_edits.npz',**{str(k):v for k,v in edits.items()})
    y=samples(Generator(edits=edits),z)
    results[name]={'dc_strength':dc,'alpha':alpha,'gain':gain,**metrics(y)}
    print(name,json.dumps(results[name]),flush=True)
    pics.extend(y[:8]); labels.extend([f'{name} {i}' for i in range(8)])
grid(np.array(pics),'results/selection.png',labels=labels)
Path('results/selection_metrics.json').write_text(json.dumps(results,indent=2))
