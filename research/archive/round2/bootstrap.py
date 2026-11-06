"""Initialize a clean second-round experiment, or verify the preserved checkpoint."""
from pathlib import Path
import hashlib,json,shutil,datetime

ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
EXPECTED={'model.tflite':'0074d74ac51f7a67361c2f881faeeebb591f7297c3878f972538b4c1cb95e79b','model_improved.tflite':'38c2106d8fded23210427ca1c27e30ec73acc00887597f2a26c82090fbbe31c9'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    for name,expected in EXPECTED.items():assert sha(WORK/name)==expected,(name,'Unexpected model bytes')
    checkpoint=ROOT/'checkpoints/000_preserved_v1'
    if (checkpoint/'manifest.json').exists():
        assert sha(checkpoint/'model_improved.tflite')==EXPECTED['model_improved.tflite']
        print('Existing preserved checkpoint verified; no files overwritten.');return
    for name in ['checkpoints/000_preserved_v1','figures','logs','candidates','evaluation','source']:(ROOT/name).mkdir(parents=True,exist_ok=True)
    shutil.copy2(WORK/'model_improved.tflite',checkpoint/'model_improved.tflite')
    for p in list(WORK.glob('*.py'))+[WORK/'README.md']:
        if p.exists():shutil.copy2(p,checkpoint/p.name)
    if (WORK/'results').exists():shutil.copytree(WORK/'results',checkpoint/'results',dirs_exist_ok=False)
    event={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'event':'preserve_round1','files':{name:{'sha256':expected,'bytes':(WORK/name).stat().st_size} for name,expected in EXPECTED.items()},'checkpoint':str(checkpoint.relative_to(WORK))}
    (checkpoint/'manifest.json').write_text(json.dumps(event,indent=2))
    with (ROOT/'logs/events.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
    if not (ROOT/'WORK_LOG.md').exists():(ROOT/'WORK_LOG.md').write_text('# Second round experiment log\n\nInitialized a fresh preservation checkpoint.\n')
    print(json.dumps(event,indent=2))

if __name__=='__main__':main()
