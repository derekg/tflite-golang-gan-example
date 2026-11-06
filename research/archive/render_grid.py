import sys, json
import numpy as np
from PIL import Image, ImageDraw

images=np.load(sys.argv[1]); path=sys.argv[2]; labels=json.loads(sys.argv[3]); cols=int(sys.argv[4]); scale=int(sys.argv[5])
n=len(images); size=images.shape[2]*scale; height=images.shape[1]*scale; bar=22 if labels else 0
canvas=Image.new('RGB',(cols*size,((n+cols-1)//cols)*(height+bar)),(30,30,30)); draw=ImageDraw.Draw(canvas)
for i,a in enumerate(images):
    im=Image.fromarray(np.round(np.clip(a,0,1)*255).astype('uint8')).resize((size,height))
    px=(i%cols)*size; py=(i//cols)*(height+bar); canvas.paste(im,(px,py))
    if labels: draw.text((px+4,py+height+3),labels[i],fill='white')
canvas.save(path)
