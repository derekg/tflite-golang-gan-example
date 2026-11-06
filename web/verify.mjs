// Run with Node from any directory. No browser/GPU acceptance claim.
import fs from 'node:fs';import path from 'node:path';import assert from 'node:assert/strict';import {fileURLToPath} from 'node:url';import {createHash} from 'node:crypto';
import {TFLiteModel,CPURunner} from './engine.mjs';import {slerp} from './math.mjs';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const bytes=p=>{const b=fs.readFileSync(p);return b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength);};
const manifest=JSON.parse(fs.readFileSync(path.join(root,'web/models.json'))),runners=[],buffers=[];
for(const key of ['original','round11']){const m=manifest[key],b=bytes(path.resolve(root,'web',m.url));assert.equal(createHash('sha256').update(new Uint8Array(b)).digest('hex'),m.sha256);assert.equal(b.byteLength,m.bytes);buffers.push(b);runners.push(new CPURunner(new TFLiteModel(b)));}
const z=new Float32Array(bytes(path.join(root,'evidence/inputs.f32'))),expected=new Float32Array(bytes(path.join(root,'evidence/expected.f32'))),pixels=128*128*3,records=[];
for(let i=0;i<5;i++){const noise=z.slice(i*512,(i+1)*512),copy=noise.slice();for(let k=0;k<2;k++){const start=performance.now(),actual=runners[k].run(noise);let error=0;for(let j=0;j<pixels;j++)error=Math.max(error,Math.abs(actual[j]-expected[(i*2+k)*pixels+j]));assert.ok(error<2e-5);assert.ok(actual.every(Number.isFinite));records.push({case:i,model:k===0?'original':'round11',max_error:error,ms:performance.now()-start});}assert.deepEqual(noise,copy);}
let messages=[];globalThis.self={postMessage:m=>messages.push(m)};await import('./worker.mjs');
await self.onmessage({data:{type:'init',models:buffers}});assert.equal(messages.at(-1).type,'ready');
await self.onmessage({data:{type:'frame',noise:z.slice(512,1024),id:7,epoch:4,t:.5}});
const m=messages.at(-1);assert.equal(m.type,'frame');assert.equal(m.id,7);assert.equal(m.epoch,4);assert.equal(m.t,.5);assert.equal(m.outputs.length,2);
for(let k=0;k<2;k++){let e=0;for(let j=0;j<pixels;j++)e=Math.max(e,Math.abs(m.outputs[k][j]-expected[(2+k)*pixels+j]));assert.ok(e<2e-5);}
const mid=slerp(z.slice(512,1024),z.slice(1024,1536),.5);assert.ok(mid.every((v,i)=>Math.abs(v-z[1536+i])<2e-6));
const record={status:'PASS',records,paired_worker_protocol:true,identical_input_for_both_models:true,browser_rendering_tested:false,webgl_tested:false,native_litert_tested:false};
fs.writeFileSync(path.join(root,'evidence/browser-engine-validation.json'),JSON.stringify(record,null,2)+'\n');console.log(JSON.stringify({status:'PASS',max_error:Math.max(...records.map(r=>r.max_error)),records:records.length}));
