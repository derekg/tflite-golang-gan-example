// Simulated DOM/worker tests; explicitly not browser rendering or GPU tests.
import fs from 'node:fs';import vm from 'node:vm';import assert from 'node:assert/strict';import {webcrypto} from 'node:crypto';
import {gaussianNoise,slerp,smootherstep} from './math.mjs';
globalThis.crypto??=webcrypto;
const nodes=new Map(),events={},sent=[],draws=[],checks=[];let now=0,raf,worker,fetches=0;
function element(id){if(nodes.has(id))return nodes.get(id);const e={id,value:id==='law'?'gaussian':0,hidden:false,disabled:false,checked:false,textContent:'',selectedOptions:[{textContent:'Gaussian'}],classList:{toggle(){}},setAttribute(){},getContext(){return{drawImage(...v){draws.push([id,...v]);},putImageData(){draws.push([id,'pixels']);},fillRect(){},fillText(){}};},toBlob(fn){fn(new Blob());},click(){events.download=true;}};nodes.set(id,e);return e;}
class FakeWorker{constructor(){worker=this;}postMessage(m){sent.push(m);}terminate(){this.terminated=true;}emit(m){this.onmessage({data:m});}}
const sandbox={console,Float32Array,Uint8ClampedArray,Uint8Array,Uint32Array,Array,Math,Number,Promise,Error,URL,Blob,crypto:webcrypto,gaussianNoise,slerp,smootherstep,performance:{now:()=>now},requestAnimationFrame:f=>raf=f,matchMedia:()=>({matches:false}),setTimeout:()=>{},ImageData:class{},Image:class{set src(v){this.onload();}},Worker:FakeWorker,document:{hidden:false,getElementById:element,addEventListener:(n,f)=>events[n]=f,createElement:element},fetch:async url=>{fetches++;return url==='models.json'?{ok:true,json:async()=>({original:{url:'a',bytes:0},round11:{url:'b',bytes:0}})}:{ok:true,arrayBuffer:async()=>new ArrayBuffer(0)};}};
// Suppress checksum only for zero-byte mock downloads, not production code.
sandbox.crypto={getRandomValues:webcrypto.getRandomValues.bind(webcrypto)};
vm.createContext(sandbox);let code=fs.readFileSync(new URL('./app.mjs',import.meta.url),'utf8').replace(/^import .*\n/,'').replaceAll('export function','function').replaceAll('import.meta.url',JSON.stringify(import.meta.url));
vm.runInContext(code,sandbox);await new Promise(r=>setImmediate(r));
function check(name,fn){fn();checks.push(name);}const click=id=>element(id).onclick();
function frame(){const m=sent.at(-1);assert.equal(m.type,'frame');worker.emit({...m,type:'frame',outputs:[new Float32Array(49152),new Float32Array(49152)],ms:1000,backend:['CPU','CPU']});return m;}
check('Saved mode makes no model requests',()=>assert.equal(fetches,0));
click('prev');check('Gallery wraps backward',()=>assert.equal(element('sample').value,15));click('next');
check('Gallery wraps forward',()=>assert.equal(element('sample').value,0));
click('live');await click('start');worker.emit({type:'ready'});frame();
check('One message contains the shared noise vector',()=>assert.equal(sent.at(-1).noise.length,512));
now=5000;raf(now);const first=frame();check('Walk advances',()=>assert.equal(first.t,.25));
click('pause');const count=sent.length;now=15000;raf(now);check('Pause stops requests',()=>assert.equal(sent.length,count));
click('pause');raf(now);check('Resume preserves progress',()=>assert.equal(frame().t,.25));
element('pace').oninput({target:{value:'40'}});raf(now);check('Changing duration preserves progress',()=>assert.equal(frame().t,.25));
now=16000;raf(now);const before=draws.length;click('destination');frame();check('Stale frame is discarded',()=>assert.equal(draws.length,before));
raf(now);check('New destination starts from last displayed noise',()=>{const m=sent.at(-1);assert.equal(m.t,0);assert.ok(m.noise.every(Number.isFinite));});frame();
element('law').value='legacy';element('law').onchange();raf(now);check('Legacy endpoints stay within zero to one',()=>assert.ok(sent.at(-1).noise.every(v=>v>=0&&v<=1)));frame();
sandbox.document.hidden=true;events.visibilitychange();const n=sent.length;now=26000;raf(now);check('Hidden tab stops requests',()=>assert.equal(sent.length,n));sandbox.document.hidden=false;events.visibilitychange();raf(now);check('Visibility resume preserves progress',()=>assert.equal(frame().t,0));
now=66000;raf(now);check('Endpoint is rendered exactly',()=>assert.equal(frame().t,1));const end=sent.length;now=66500;raf(now);check('Endpoint hold avoids extra inference',()=>assert.equal(sent.length,end));now=67300;raf(now);check('Next journey starts automatically',()=>assert.equal(frame().t,0));
click('saved');const stopped=sent.length;now+=5000;raf(now);check('Saved mode suspends live inference',()=>assert.equal(sent.length,stopped));
click('save');check('Saving creates a paired image',()=>assert.equal(events.download,true));
worker.emit({type:'error',message:'Deliberate test error'});check('Error exposes retry and preserves gallery',()=>{assert.equal(element('start').disabled,false);assert.equal(element('error').hidden,false);click('next');});
const result={status:'PASS',checks,simulated_dom:true,browser_rendering_tested:false,webgl_tested:false};fs.writeFileSync(new URL('../evidence/ui-controls-validation.json',import.meta.url),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result));
