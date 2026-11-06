import {TFLiteModel,CPURunner,WebGLRunner} from './engine.mjs';
let runners=[],busy=false;
self.onmessage=async({data:m})=>{
 try{
  if(m.type==='init'){
   runners=m.models.map(buffer=>{const model=new TFLiteModel(buffer);return{cpu:new CPURunner(model),gpu:null};});
   for(const r of runners){
    try{
     r.gpu=new WebGLRunner(r.cpu.model);
     for(const z of [new Float32Array(512),Float32Array.from({length:512},(_,i)=>Math.sin(i*1.371)+.3*Math.cos(i*.173))]){
      const a=r.cpu.run(z),b=r.gpu.run(z);let error=0;
      for(let i=0;i<a.length;i++){if(!Number.isFinite(b[i]))throw Error('Nonfinite GPU output');error=Math.max(error,Math.abs(a[i]-b[i]));}
      if(error>.0002)throw Error('GPU numerical check failed');
     }
    }catch(error){r.gpu?.dispose();r.gpu=null;}
   }
   self.postMessage({type:'ready',backend:runners.map(r=>r.gpu?'WebGL2':'CPU')});
  }else if(m.type==='frame'){
   if(busy||runners.length!==2)throw Error('Comparison worker is not ready');
   busy=true;const start=performance.now();
   const outputs=runners.map(r=>{
    if(r.gpu){try{return r.gpu.run(m.noise);}catch(error){r.gpu.dispose();r.gpu=null;}}
    return r.cpu.run(m.noise);
   });
   for(const a of outputs)if(a.length!==128*128*3||!a.every(Number.isFinite))throw Error('Invalid portrait output');
   self.postMessage({type:'frame',id:m.id,epoch:m.epoch,t:m.t,outputs,ms:performance.now()-start,backend:runners.map(r=>r.gpu?'WebGL2':'CPU')},outputs.map(x=>x.buffer));busy=false;
  }
 }catch(error){busy=false;self.postMessage({type:'error',message:error.message});}
};
