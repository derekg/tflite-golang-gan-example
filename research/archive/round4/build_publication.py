"""Package this round's repository article, code, measurements and figures."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import datetime,hashlib,json,zipfile
import round2.build_publication as renderer

ROOT=Path(__file__).resolve().parent

def main():
    renderer.ROOT=ROOT
    body=renderer.render((ROOT/'TECHNICAL_WRITEUP.md').read_text())
    style='''body{max-width:1000px;margin:50px auto;padding:0 24px 70px;font:18px/1.65 Georgia,serif;color:#20252b}h1,h2,h3{font-family:system-ui,sans-serif;line-height:1.25}h1{font-size:2.5rem}h2{margin-top:48px;font-size:1.6rem}h3{margin-top:30px}a{color:#155bc2}pre{padding:18px;background:#f4f5f7;overflow:auto;font-size:.85rem;line-height:1.55}code{font-family:ui-monospace,monospace;font-size:.85em;overflow-wrap:anywhere}pre code{font-size:inherit;overflow-wrap:normal}.table{overflow:auto}table{border-collapse:collapse;width:100%;font:15px/1.5 system-ui,sans-serif;margin:24px 0}th,td{text-align:left;vertical-align:top;padding:12px;border-bottom:1px solid #ddd}th{background:#f4f5f7}figure{margin:30px 0}img{max-width:100%;height:auto;display:block}figcaption{font:14px/1.5 system-ui,sans-serif;color:#555;margin-top:10px}@media(max-width:600px){body{font-size:16px;padding:0 16px}h1{font-size:2rem}}'''
    doc='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Exporting and refining the restructured portrait generator</title><style>'+style+'</style></head><body><main>'+body+'</main></body></html>'
    assert doc.count('<figure>')==2 and doc.count('<table>')==4
    (ROOT/'TECHNICAL_WRITEUP.html').write_text(doc)
    files=[ROOT/'TECHNICAL_WRITEUP.md',ROOT/'TECHNICAL_WRITEUP.html',ROOT/'WORK_LOG.md',ROOT/'DATA_REQUIREMENTS.json']
    files+=list(ROOT.glob('*.py'))+list((ROOT/'source').glob('*.py'))+list((ROOT/'figures').glob('*before_after.png'))
    files+=[p for p in (ROOT/'evaluation').glob('*.json') if p.name!='publication_manifest.json']
    files+=list((ROOT/'logs').glob('*'))+list((ROOT/'checkpoints').glob('*/manifest.json'))+[ROOT/'checkpoints/012_frozen_selection/selection.json']
    files+=list((ROOT/'candidates').glob('*.graph.json'))
    work=ROOT.parent
    files+=[work/p for p in ['round3/restructure.py','round3/RESTRUCTURING.md','inspect_model.py','generate.py','render_grid.py','experiment.py','round2/run_experiments.py','round2/build_publication.py','refine.py']]
    manifest={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model_files_included':False,'files':{}}
    with zipfile.ZipFile(ROOT/'publication_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in dict.fromkeys(files):
            arc=str(p.relative_to(work));data=p.read_bytes();z.writestr(arc,data)
            manifest['files'][arc]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
        z.writestr('BUNDLE_MANIFEST.json',json.dumps(manifest,indent=2))
        z.writestr('BUNDLE_README.txt','This bundle contains the technical article, figures, code, logs and measurements. Models and large numerical checkpoints are not included. Reproduction requires the exact round-three parent checkpoint identified in the article, plus the original project environment and file layout. The HTML article embeds its images. No native LiteRT acceptance result is claimed.\n')
    (ROOT/'evaluation/publication_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'html_bytes':len(doc.encode()),'bundle_bytes':(ROOT/'publication_bundle.zip').stat().st_size,'bundle_files':len(manifest['files'])}))

if __name__=='__main__':main()
