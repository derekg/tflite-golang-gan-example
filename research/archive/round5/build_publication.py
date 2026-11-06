"""Package the noise-input cleanup and repeat-control experiment for review."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,zipfile,hashlib,datetime
import round2.build_publication as renderer

ROOT=Path(__file__).resolve().parent

def main():
    renderer.ROOT=ROOT;body=renderer.render((ROOT/'TECHNICAL_WRITEUP.md').read_text())
    css='''body{max-width:1000px;margin:50px auto;padding:0 24px 60px;font:18px/1.65 Georgia,serif;color:#20252b}h1,h2,h3{font-family:system-ui,sans-serif;line-height:1.25}h1{font-size:2.5rem}h2{font-size:1.6rem;margin-top:46px}a{color:#155bc2}pre{background:#f4f5f7;padding:18px;overflow:auto;font-size:.85rem}code{font-family:ui-monospace,monospace;overflow-wrap:anywhere;font-size:.85em}pre code{font-size:inherit;overflow-wrap:normal}.table{overflow:auto}table{border-collapse:collapse;width:100%;font:15px/1.5 system-ui,sans-serif;margin:24px 0}th,td{padding:12px;text-align:left;vertical-align:top;border-bottom:1px solid #ddd}th{background:#f4f5f7}img{max-width:100%;height:auto;display:block}figure{margin:28px 0}figcaption{font:14px/1.5 system-ui,sans-serif;color:#555;margin-top:10px}@media(max-width:600px){body{font-size:16px;padding:0 16px}h1{font-size:2rem}}'''
    doc='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Cleaner portraits and fewer recent repeats from random noise</title><style>'+css+'</style></head><body><main>'+body+'</main></body></html>'
    assert doc.count('<figure>')==3 and doc.count('<table>')==2
    (ROOT/'TECHNICAL_WRITEUP.html').write_text(doc)
    work=ROOT.parent
    files=[ROOT/'TECHNICAL_WRITEUP.md',ROOT/'TECHNICAL_WRITEUP.html',ROOT/'WORK_LOG.md']+list(ROOT.glob('*.py'))+list((ROOT/'source').glob('*.py'))+list((ROOT/'logs').glob('*'))
    files+=[p for p in (ROOT/'evaluation').glob('*.json') if p.name!='publication_manifest.json']
    files+=[ROOT/'figures'/n for n in ['gaussian_comparison.png','random_batch.png','repeat_aware_batch.png']]
    files+=list((ROOT/'checkpoints').glob('*/manifest.json'))+list((ROOT/'checkpoints').glob('*/selection.json'))
    files+=[p for p in (ROOT/'checkpoints/006_sampler_v1').glob('*') if p.suffix in ['.json','.py','.png']]
    files+=[work/p for p in ['round3/restructure.py','round4/refine_structure.py','round4/export_compact.py','round4/check_graph.py','round2/run_experiments.py','round2/build_publication.py','generate.py','render_grid.py','inspect_model.py','experiment.py','refine.py']]
    manifest={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model_files_included':False,'files':{}}
    with zipfile.ZipFile(ROOT/'publication_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in dict.fromkeys(files):
            arc=str(p.relative_to(work));data=p.read_bytes();z.writestr(arc,data);manifest['files'][arc]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
        z.writestr('BUNDLE_MANIFEST.json',json.dumps(manifest,indent=2))
        z.writestr('BUNDLE_README.txt','This review bundle contains the article, code, figures, logs and measurements, including the rejected first sampler. Models, numerical patches and runtime history caches are separate. The HTML embeds its figures. Reproduction requires the exact source model/checkpoint files and project layout recorded in the article and plans. Native LiteRT acceptance has not been performed.\n')
    (ROOT/'evaluation/publication_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({'html_bytes':len(doc.encode()),'bundle_files':len(manifest['files']),'bundle_bytes':(ROOT/'publication_bundle.zip').stat().st_size}))

if __name__=='__main__':main()
