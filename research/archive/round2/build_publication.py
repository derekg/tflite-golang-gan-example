"""Render the repository's technical Markdown into a self-contained HTML article.

Intentionally small renderer for headings, paragraphs, tables, code, and figures
used by this document. No external fonts, scripts, libraries, or network assets.
"""
from pathlib import Path
import base64,html,re,json,zipfile,hashlib,datetime

ROOT=Path(__file__).resolve().parent

def inline(s):
    s=html.escape(s)
    s=re.sub(r'`([^`]+)`',r'<code>\1</code>',s)
    s=re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',s)
    return re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',s)

def render(md):
    lines=md.splitlines();out=[];i=0
    while i<len(lines):
        s=lines[i]
        if not s.strip():i+=1;continue
        if s.startswith('```'):
            i+=1;code=[]
            while i<len(lines) and not lines[i].startswith('```'):code.append(lines[i]);i+=1
            out.append('<pre><code>'+html.escape('\n'.join(code))+'</code></pre>');i+=1;continue
        m=re.match(r'^(#{1,6}) (.+)$',s)
        if m:
            level=len(m[1]);out.append(f'<h{level}>'+inline(m[2])+f'</h{level}>');i+=1;continue
        im=re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)',s)
        if im:
            path=ROOT/im[2];assert path.exists()
            uri='data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()
            out.append('<figure><img loading="lazy" src="'+uri+'" alt="'+html.escape(im[1],quote=True)+'"><figcaption>'+inline(im[1])+'</figcaption></figure>');i+=1;continue
        if s.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].startswith('|'):
                row=[x.strip() for x in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?',x) for x in row):rows.append(row)
                i+=1
            out.append('<div class="table"><table><thead><tr>'+''.join('<th>'+inline(x)+'</th>' for x in rows[0])+'</tr></thead><tbody>')
            for row in rows[1:]:out.append('<tr>'+''.join('<td>'+inline(x)+'</td>' for x in row)+'</tr>')
            out.append('</tbody></table></div>');continue
        para=[s];i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','```','|','![')):para.append(lines[i]);i+=1
        out.append('<p>'+inline(' '.join(para))+'</p>')
    return '\n'.join(out)

def main():
    md=(ROOT/'TECHNICAL_WRITEUP.md').read_text()
    body=render(md)
    style='''html{color:#20252b;background:#fff;font-family:Georgia,"Times New Roman",serif;line-height:1.65}body{max-width:1000px;margin:56px auto;padding:0 28px 70px}h1,h2,h3,th{font-family:system-ui,sans-serif;line-height:1.25}h1{font-size:2.6rem;letter-spacing:-.035em;margin:0 0 28px}h2{margin-top:52px;font-size:1.65rem}h3{margin-top:32px;font-size:1.2rem}p{font-size:1.1rem;margin:18px 0}a{color:#155bc2;text-underline-offset:3px}pre{background:#f4f5f7;padding:20px;overflow:auto;font-size:.86rem;line-height:1.6}code{font-family:ui-monospace,monospace;font-size:.88em;overflow-wrap:anywhere}pre code{font-size:inherit;overflow-wrap:normal}.table{overflow:auto;margin:26px 0}table{border-collapse:collapse;width:100%;font-family:system-ui,sans-serif;font-size:.95rem;line-height:1.5}td,th{text-align:left;vertical-align:top;padding:12px 14px;border-bottom:1px solid #ddd}th{background:#f4f5f7}figure{margin:30px 0}img{max-width:100%;height:auto;display:block}figcaption{font-family:system-ui,sans-serif;font-size:.9rem;color:#555;margin-top:10px}@media(max-width:600px){body{margin-top:28px;padding:0 16px 40px}h1{font-size:2rem}p{font-size:1rem}td,th{padding:8px}}@media print{body{max-width:none;margin:0}h2,h3{break-after:avoid}figure,pre,tr{break-inside:avoid}a{color:inherit}}'''
    doc='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Editing a TFLite portrait generator without retraining</title><style>'+style+'</style></head><body><main>'+body+'</main></body></html>'
    (ROOT/'TECHNICAL_WRITEUP.html').write_text(doc)
    assert doc.count('<figure>')==4 and doc.count('<table>')==4,(doc.count('<figure>'),doc.count('<table>'))
    # Reproducibility bundle: local original evidence remains in its checkpoints.
    # Models and large numerical patches stay separate from the posting assets.
    files=[ROOT/'TECHNICAL_WRITEUP.md',ROOT/'TECHNICAL_WRITEUP.html',ROOT/'WORK_LOG.md']
    files+=list((ROOT/'figures').glob('*.png'))+[p for p in (ROOT/'evaluation').glob('*.json') if p.name!='publication_manifest.json']
    files+=list((ROOT/'logs').glob('*'))+list((ROOT/'source').glob('*.py'))+list(ROOT.glob('*.py'))
    files+=list((ROOT/'checkpoints').glob('*/manifest.json'))+[ROOT/'checkpoints/016_frozen_selection/selection.json',ROOT/'checkpoints/016_frozen_selection/patch_manifest.json']
    files+=list((ROOT/'candidates').glob('*.json'))
    top=ROOT.parent
    scripts=[top/p for p in ['generate.py','render_grid.py','inspect_model.py','verify_forward.py','experiment.py','refine.py','select_candidate.py','finalize.py']]
    manifest={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':{},'model_files_included':False,'weight_patches_included':False}
    with zipfile.ZipFile(ROOT/'publication_bundle.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in dict.fromkeys(files+scripts):
            arc=str(p.relative_to(top));data=p.read_bytes();z.writestr(arc,data)
            manifest['files'][arc]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
        z.writestr('BUNDLE_README.txt','Publication assets, code, logs and metrics. Model files and large weight patches are not included. Obtain the exact model bytes identified by SHA-256 in round2/TECHNICAL_WRITEUP.md to reproduce numerical results. The HTML article embeds its figures and can be opened by itself. Checkpoint and code paths refer to the original workspace layout. Local full checkpoints remain available outside this bundle.\n')
        z.writestr('BUNDLE_MANIFEST.json',json.dumps(manifest,indent=2))
    (ROOT/'evaluation/publication_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'html_bytes':len(doc.encode()),'bundle_bytes':(ROOT/'publication_bundle.zip').stat().st_size,'bundle_files':len(manifest['files'])}))

if __name__=='__main__':main()
