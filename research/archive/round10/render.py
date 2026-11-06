"""Render genuine model output and fixed-scale edge maps. System Python/Pillow."""
from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent
def font(n):
    p='/System/Library/Fonts/Supplemental/Arial.ttf'
    return ImageFont.truetype(p,n) if Path(p).exists() else ImageFont.load_default()
def rgb(a,size=256):
    return Image.fromarray(np.round(np.clip(a,0,1)*255).astype('uint8')).resize((size,size),Image.Resampling.NEAREST)
def edge(a):
    dx=np.pad(np.abs(np.diff(a,axis=1)).mean(2),((0,0),(0,1)))
    dy=np.pad(np.abs(np.diff(a,axis=0)).mean(2),((0,1),(0,0)))
    return np.sqrt(dx*dx+dy*dy)
def panel(data,law,start,edge_view=False,holdout=False):
    keys=['round9','round10','edge9','edge10'] if edge_view else ['original','round9','round10']
    if holdout and not edge_view:keys=['round9','round10']
    labels={'original':'ORIGINAL','round9':'ROUND9 preserved','round10':'ROUND10 overlap','edge9':'ROUND9 edges','edge10':'ROUND10 edges'}
    width=40+len(keys)*276;im=Image.new('RGB',(width,140+4*300),'#101522');d=ImageDraw.Draw(im)
    d.text((20,18),'Grid cleanup: matched-input comparison',font=font(25),fill='white')
    seed=(89102 if law=='gaussian' else 89103) if holdout else (88920 if law=='gaussian' else 88921)
    d.text((20,55),f'{law} noise | seed {seed} | native 128px enlarged 2x',font=font(16),fill='#c8d0de')
    d.text((20,81),'Unedited RGB. Edge maps use one shared exposure.',font=font(15),fill='#c8d0de')
    edge_scale=max(float(np.percentile(np.stack([edge(a) for a in data['round9']]),99)),1e-6)
    for j,k in enumerate(keys):d.text((20+j*276,111),labels[k],font=font(18),fill='#ffd580' if '10' in k else '#9fd6ff')
    for row,i in enumerate(range(start,start+4)):
        for col,k in enumerate(keys):
            if k.startswith('edge'):
                e=np.clip(edge(data['round'+k[4:]][i])/edge_scale,0,1)**.65
                a=np.repeat(e[:,:,None],3,axis=2)
            else:a=data[k][i]
            x=20+col*276;y=140+row*300;im.paste(rgb(a),(x,y))
            d.text((x,y+265),f'#{i:02d} | identical latent',font=font(16),fill='white')
    name=f'{law}_{"holdout_" if holdout else ""}{"edges" if edge_view else "portraits"}_{start:02d}_{start+3:02d}.png'
    im.save(ROOT/'figures'/name)

def development():
    paths=sorted((ROOT/'checkpoints').glob('*/preview.npz'))
    cols=5;tile=192;gap=16;top=68;rowh=252
    for law in ['gaussian','uniform']:
        im=Image.new('RGB',(cols*(tile+gap)+gap,top+((len(paths)+cols-1)//cols)*rowh),'#101522');d=ImageDraw.Draw(im)
        d.text((16,16),f'Development contact sheet | {law} sample #0 | all 23 candidates',font=font(22),fill='white')
        for idx,p in enumerate(paths):
            m=json.loads((p.parent/'manifest.json').read_text());a=np.load(p)[law][0]
            x=gap+(idx%cols)*(tile+gap);y=top+(idx//cols)*rowh;im.paste(rgb(a,tile),(x,y))
            title=m['config']['name'];d.text((x,y+198),title,font=font(13),fill='white')
            d.text((x,y+218),f'grid ratio {m["score"]:.3f} | safe {m["structure_safe"]}',font=font(12),fill='#c8d0de')
        im.save(ROOT/'figures'/f'{law}_development.png')

if __name__=='__main__':
    development()
    for law in ['gaussian','uniform']:
        for holdout in [False,True]:
            p=ROOT/'figures'/f'{law}_{"holdout" if holdout else "comparison"}.npz'
            if not p.exists():continue
            data=np.load(p)
            for start in [0,4]:
                panel(data,law,start,False,holdout);panel(data,law,start,True,holdout)
