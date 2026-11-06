"""Render saved, unnormalized samples with labels; no generated image editing."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent
def font(n):
    for path in ['/System/Library/Fonts/Supplemental/Arial.ttf','/System/Library/Fonts/Helvetica.ttc']:
        if Path(path).exists():return ImageFont.truetype(path,n)
    return ImageFont.load_default()
def render(key,seed,indices,columns,path):
    samples=np.load(ROOT/'figures'/f'{key}_comparison.npz');margin=24;gap=20;size=256;top=142;row=307
    im=Image.new('RGB',(margin*2+len(columns)*size+(len(columns)-1)*gap,top+len(indices)*row+44),'#111827');d=ImageDraw.Draw(im)
    title='Round8 vs new spatial filter' if len(columns)==2 else 'Original vs round8 vs round9'
    d.text((margin,18),title,font=font(26),fill='white')
    law='Normal(0,1)' if key=='gaussian' else 'Uniform[-1,1]'
    d.text((margin,54),f'Identical noise vectors | {law} | seed {seed}',font=font(17),fill='#d1d5db')
    d.text((margin,80),'Same 256px interpolation; no contrast adjustment or retouching.',font=font(15),fill='#d1d5db')
    labels={'original':'ORIGINAL model.tflite','round8':'ROUND8 preserved','round9':'ROUND9 spatial filter'}
    for col,k in enumerate(columns):d.text((margin+col*(size+gap),111),labels[k],font=font(18),fill=['#93c5fd','#6ee7b7','#fcd34d'][col])
    for r,i in enumerate(indices):
        for col,k in enumerate(columns):
            x=margin+col*(size+gap);y=top+r*row
            a=np.round(np.clip(samples[k][i],0,1)*255).astype('uint8');im.paste(Image.fromarray(a),(x,y))
            d.text((x,y+262),f'Sample #{i:02d} | same input',font=font(16),fill='white')
    d.text((margin,im.height-30),'First 8 samples in seed order; none screened out.',font=font(15),fill='#d1d5db');im.save(path)
for key,seed in [('gaussian',88920),('uniform',88921)]:
    for start in [0,4]:
        render(key,seed,list(range(start,start+4)),['round8','round9'],ROOT/'figures'/f'{key}_round8_vs_round9_{start:02d}_{start+3:02d}.png')
        render(key,seed,list(range(start,start+4)),['original','round8','round9'],ROOT/'figures'/f'{key}_three_way_{start:02d}_{start+3:02d}.png')
    render(key,seed,list(range(8)),['original','round8','round9'],ROOT/'figures'/f'{key}_all_8.png')
