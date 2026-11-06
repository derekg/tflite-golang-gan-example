import {gaussianNoise,slerp,smootherstep} from './math.mjs';
const $=id=>document.getElementById(id),canvases=[$('before'),$('after')],ctx=canvases.map(c=>c.getContext('2d'));
let mode='saved',sample=0,atlas,edgeAtlas,worker,ready=false,busy=false,loading=false,epoch=0,id=0;
let paused=matchMedia('(prefers-reduced-motion: reduce)').matches,hidden=document.hidden,frozen=null;
let from,to,current,pair,phase='travel',startTime=0,duration=20000,journey=1,pending;
function rand(){const a=new Uint32Array(1);crypto.getRandomValues(a);return(a[0]+.5)/4294967296;}
function noise(){return $('law').value==='gaussian'?gaussianNoise():Float32Array.from({length:512},()=>$('law').value==='legacy'?rand():2*rand()-1);}
export function blend(a,b,t,law){return law==='gaussian'?slerp(a,b,t):Float32Array.from(a,(v,i)=>v+(b[i]-v)*t);}
function label(){
 if(mode==='saved')$('sample-label').textContent=`${sample<8?'Gaussian · seed 88920':'Uniform −1–1 · seed 88921'} · sample ${String(sample%8).padStart(2,'0')} / 07`;
 else $('sample-label').textContent=`Live · ${$('law').selectedOptions[0].textContent} · journey ${journey}`;
}
function drawSaved(){if(!atlas||!edgeAtlas)return;const image=$('edges').checked?edgeAtlas:atlas;ctx.forEach((c,i)=>c.drawImage(image,i*128,sample*128,128,128,0,0,128,128));label();}
export function edges(a){const e=new Float32Array(128*128);for(let y=0;y<128;y++)for(let x=0;x<128;x++){
 let dx=0,dy=0;const at=(y*128+x)*3;for(let c=0;c<3;c++){if(x<127)dx+=Math.abs(a[at+c+3]-a[at+c])/3;if(y<127)dy+=Math.abs(a[at+c+384]-a[at+c])/3;}e[y*128+x]=Math.hypot(dx,dy);
 }return e;}
function drawPair(){
 if(!pair)return;let maps,scale;
 if($('edges').checked){maps=pair.map(edges);const sorted=Array.from(maps[0]).sort((a,b)=>a-b);scale=Math.max(sorted[Math.floor(.99*(sorted.length-1))],1e-6);}
 pair.forEach((a,k)=>{const p=new Uint8ClampedArray(128*128*4);for(let i=0;i<128*128;i++){for(let c=0;c<3;c++)p[i*4+c]=Math.round(255*(maps?Math.min(1,maps[k][i]/scale)**.65:a[i*3+c]));p[i*4+3]=255;}ctx[k].putImageData(new ImageData(p,128,128),0,0);});label();
}
function freeze(){const stop=paused||hidden||mode!=='live';if(stop&&frozen===null)frozen=performance.now();if(!stop&&frozen!==null){startTime+=performance.now()-frozen;frozen=null;}$('pause').textContent=paused?'Resume':'Pause';$('pause').setAttribute('aria-pressed',String(paused));}
function setMode(value){mode=value;epoch++;$('saved-controls').hidden=value!=='saved';$('live-controls').hidden=value!=='live';for(const v of ['saved','live']){$(v).classList.toggle('active',v===value);$(v).setAttribute('aria-pressed',String(v===value));}freeze();if(mode==='saved'){drawSaved();$('status').textContent='Sixteen saved pairs. No cherry-picking, no independent flips.';}else{if(pair)drawPair();$('status').textContent=ready?'Both models ready. Every frame uses the same noise.':'Load the models to generate synchronized pairs locally.';}}
$('saved').onclick=()=>setMode('saved');$('live').onclick=()=>setMode('live');
$('sample').oninput=e=>{sample=Number(e.target.value);drawSaved();};
for(const [name,delta]of [['prev',-1],['next',1]])$(name).onclick=()=>{sample=(sample+delta+16)%16;$('sample').value=sample;drawSaved();};
$('edges').onchange=()=>mode==='saved'?drawSaved():pair?drawPair():drawSaved();
$('pause').onclick=()=>{paused=!paused;epoch++;freeze();};
function destination(reset=false){epoch++;from=reset||!current?noise():current.slice();to=noise();current=from.slice();phase='travel';startTime=frozen??performance.now();journey++;$('progress').value=0;label();}
$('destination').onclick=()=>destination();$('law').onchange=()=>destination(true);
$('pace').oninput=e=>{const now=frozen??performance.now();if(phase==='travel')startTime=now-(now-startTime)/duration*Number(e.target.value)*1000;duration=Number(e.target.value)*1000;$('pace-label').textContent=`${e.target.value} seconds`;};
document.addEventListener('visibilitychange',()=>{hidden=document.hidden;epoch++;freeze();});
function fail(message){$('error').hidden=false;$('error').textContent=message;$('status').textContent='Live mode stopped. Saved comparisons still work.';worker?.terminate();ready=false;busy=false;loading=false;$('start').disabled=false;$('start').textContent='Retry loading';$('pause').disabled=true;$('destination').disabled=true;}
async function fetchModel(spec,name){
 const response=await fetch(spec.url,{cache:'force-cache'});if(!response.ok)throw Error(`${name} download failed (${response.status}). Serve the repository root, including model/.`);
 const reader=response.body?.getReader();let buffer;
 if(reader){const chunks=[];let n=0;while(true){const{done,value}=await reader.read();if(done)break;chunks.push(value);n+=value.length;$('status').textContent=`Loading ${name}: ${(n/1e6).toFixed(1)} / ${(spec.bytes/1e6).toFixed(1)} MB`;}const all=new Uint8Array(n);let cursor=0;for(const c of chunks){all.set(c,cursor);cursor+=c.length;}buffer=all.buffer;}else buffer=await response.arrayBuffer();
 if(buffer.byteLength!==spec.bytes)throw Error(`${name} file size mismatch`);
 if(globalThis.crypto?.subtle){const digest=await crypto.subtle.digest('SHA-256',buffer);const hash=Array.from(new Uint8Array(digest),v=>v.toString(16).padStart(2,'0')).join('');if(hash!==spec.sha256)throw Error(`${name} checksum mismatch`);}
 return buffer;
}
$('start').onclick=async()=>{
 if(loading||ready)return;loading=true;$('start').disabled=true;$('error').hidden=true;
 try{
  const response=await fetch('models.json');if(!response.ok)throw Error('Model manifest unavailable');const manifest=await response.json();
  const original=await fetchModel(manifest.original,'original'),improved=await fetchModel(manifest.round11,'round11');
  worker=new Worker(new URL('./worker.mjs',import.meta.url),{type:'module'});
  worker.onerror=e=>fail(e.message||'Worker could not start');
  worker.onmessage=({data:m})=>{
   if(m.type==='error'){fail(m.message);return;}
   if(m.type==='ready'){ready=true;loading=false;$('start').textContent='Models loaded';$('pause').disabled=false;$('destination').disabled=false;from=noise();to=noise();current=from.slice();startTime=performance.now();frozen=null;freeze();send(from.slice(),0);return;}
   if(m.type==='frame'){busy=false;if(m.epoch!==epoch){if(!pair&&mode==='live')send(current.slice(),0);return;}pair=m.outputs;current=pending.slice();if(mode==='live')drawPair();$('progress').value=m.t;$('status').textContent=`Live · ${m.backend.join(' / ')} · ${(m.ms/1000).toFixed(2)}s per paired frame${paused?' · paused':''}`;if(m.t===1){phase='rest';startTime=performance.now();}if(m.id===1){startTime=performance.now();if(frozen!==null)frozen=startTime;}}
  };
  $('status').textContent='Checking inference engines. The first pair may take a moment…';worker.postMessage({type:'init',models:[original,improved]},[original,improved]);
 }catch(error){fail(error.message);}
};
function send(z,t){if(busy||!ready)return;busy=true;pending=z;worker.postMessage({type:'frame',id:++id,epoch,t,noise:z});}
function tick(now){requestAnimationFrame(tick);if(!ready||busy||paused||hidden||mode!=='live')return;if(phase==='rest'){if(now-startTime<1200)return;destination();}const t=Math.max(0,Math.min(1,(now-startTime)/duration));send(blend(from,to,smootherstep(t),$('law').value),t);}
requestAnimationFrame(tick);
$('save').onclick=()=>{const c=document.createElement('canvas');c.width=552;c.height=312;const d=c.getContext('2d');d.fillStyle='#f4f0e7';d.fillRect(0,0,552,312);d.fillStyle='#242923';d.font='14px sans-serif';d.fillText('Original · 2021',12,24);d.fillText('Round11 · edited',284,24);d.imageSmoothingEnabled=false;d.drawImage(canvases[0],12,36,256,256);d.drawImage(canvases[1],284,36,256,256);c.toBlob(blob=>{if(!blob)return;const a=document.createElement('a'),url=URL.createObjectURL(blob);a.href=url;a.download=`portrait-comparison-${sample}.png`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});};
function loadImage(src){return new Promise((resolve,reject)=>{const i=new Image();i.onload=()=>resolve(i);i.onerror=()=>reject(Error('Saved comparison images could not load'));i.src=src;});}
Promise.all([loadImage('samples.png'),loadImage('edges.png')]).then(([a,b])=>{atlas=a;edgeAtlas=b;drawSaved();}).catch(e=>{ $('error').hidden=false;$('error').textContent=e.message;});
