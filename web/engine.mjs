// Purpose-built reader/evaluator for the float32 portrait TFLite graphs.
// It executes the file's operators; it is not Google's native LiteRT runtime.
export class Table {
  constructor(view, position) { this.v = view; this.p = position; }
  field(n) {
    const t = this.p - this.v.getInt32(this.p, true), slot = 4 + 2 * n;
    if (slot >= this.v.getUint16(t, true)) return 0;
    const d = this.v.getUint16(t + slot, true); return d ? this.p + d : 0;
  }
  scalar(n, kind = 'i', fallback = 0) {
    const p = this.field(n); if (!p) return fallback;
    if (kind === 'b') return this.v.getInt8(p);
    if (kind === 'B') return this.v.getUint8(p);
    if (kind === 'f') return this.v.getFloat32(p, true);
    return this.v.getInt32(p, true);
  }
  indirect(n) { const p = this.field(n); return p ? p + this.v.getUint32(p, true) : 0; }
  table(n) { return new Table(this.v, this.indirect(n)); }
  vector(n, Type = Int32Array) {
    const p = this.indirect(n); if (!p) return new Type(0);
    const count = this.v.getUint32(p, true), offset = p + 4;
    if (offset % Type.BYTES_PER_ELEMENT === 0) return new Type(this.v.buffer, this.v.byteOffset + offset, count);
    return new Type(this.v.buffer.slice(this.v.byteOffset + offset, this.v.byteOffset + offset + count * Type.BYTES_PER_ELEMENT));
  }
  tables(n) {
    const p = this.indirect(n); if (!p) return [];
    return Array.from({ length: this.v.getUint32(p, true) }, (_, i) => {
      const q = p + 4 + i * 4; return new Table(this.v, q + this.v.getUint32(q, true));
    });
  }
  string(n) { return new TextDecoder().decode(this.vector(n, Uint8Array)); }
}

export class TFLiteModel {
  constructor(buffer) {
    const view = new DataView(buffer);
    if (new TextDecoder().decode(new Uint8Array(buffer, 4, 4)) !== 'TFL3') throw Error('Not a TFLite model.');
    const root = new Table(view, view.getUint32(0, true));
    if (root.scalar(0) !== 3) throw Error('Unsupported model version.');
    const graphs = root.tables(2); if (graphs.length !== 1) throw Error('Expected one model graph.');
    const graph = graphs[0], buffers = root.tables(4);
    const codes = root.tables(1).map(t => t.scalar(3, 'i', t.scalar(0, 'b')));
    this.tensors = graph.tables(0).map(t => {
      const shape = Array.from(t.vector(0)), type = t.scalar(1, 'b'), raw = buffers[t.scalar(2)].vector(0, Uint8Array);
      if (![0, 2].includes(type)) throw Error(`Unsupported tensor type ${type}.`);
      let data = null;
      if (raw.length) {
        const Type = type === 0 ? Float32Array : Int32Array;
        data = raw.byteOffset % 4 ? new Type(raw.slice().buffer) : new Type(raw.buffer, raw.byteOffset, raw.byteLength / 4);
      }
      return { name: t.string(3), shape, type, data };
    });
    this.input = graph.vector(1)[0]; this.output = graph.vector(2)[0];
    this.ops = graph.tables(3).map(t => ({ code: codes[t.scalar(0)], inputs: Array.from(t.vector(1)), output: t.vector(2)[0], options: t.field(4) ? t.table(4) : null }));
    const supported = new Set([9, 22, 98, 77, 45, 83, 67, 3, 5, 0, 14, 100, 4]);
    for (const op of this.ops) {
      if (!supported.has(op.code)) throw Error(`Unsupported operator ${op.code}.`);
      const o = op.options;
      if (op.code === 3 && (o.scalar(1) !== 1 || o.scalar(2) !== 1 || o.scalar(3, 'b') || o.scalar(4, 'i', 1) !== 1 || o.scalar(5, 'i', 1) !== 1)) throw Error('Unsupported convolution options.');
      if (op.code === 9 && o.scalar(0, 'b')) throw Error('Unsupported dense activation.');
      if (op.code === 0 && o.scalar(0, 'b')) throw Error('Unsupported addition activation.');
      if (op.code === 67 && (o.scalar(1) !== o.scalar(2) || o.scalar(0, 'b') !== 0)) throw Error('Unsupported transpose convolution.');
      if (op.code === 4 && (o.scalar(0, 'b') !== 1 || o.scalar(1) !== 1 || o.scalar(2) !== 1 || o.scalar(3) !== 1 || o.scalar(4, 'b') || o.scalar(5, 'i', 1) !== 1 || o.scalar(6, 'i', 1) !== 1)) throw Error('Unsupported depthwise options.');
      if (op.code === 45 && o.scalar(4) !== 1) throw Error('Unsupported slice options.');
      if (op.code === 83 && o.scalar(1) !== 0) throw Error('Unsupported pack axis.');
      if (op.code === 100) {
        const p = this.tensors[op.inputs[1]].data;
        if (o.scalar(0, 'b') !== 1 || String(p) !== '0,0,1,1,1,1,0,0') throw Error('Unsupported mirror padding.');
      }
    }
    if (String(this.tensors[this.input].shape) !== '1,512' || !['1,128,128,3','1,256,256,3'].includes(String(this.tensors[this.output].shape))) throw Error('Unexpected model input/output shapes.');
    this.uses = new Int32Array(this.tensors.length);
    for (const op of this.ops) for (const id of op.inputs) this.uses[id]++;
  }
}

export function shape4(shape) {
  if (shape.length === 4) return shape;
  return [...Array(4 - shape.length).fill(1), ...shape];
}

export class CPURunner {
  constructor(model) {
    this.model = model; this.sparse = new Map();
    // The inherited spatial residual contains identity/mostly-zero filters.
    // Skip exact zero multiplies, preserving coefficients and accumulation order.
    for (const op of model.ops) if (op.code === 3) {
      const id = op.inputs[1], t = model.tensors[id], [co, kh, kw, ci] = t.shape;
      if (kh === 1 && kw === 1) continue;
      let count = 0; for (const v of t.data) if (v !== 0) count++;
      if (count > t.data.length / 5) continue;
      const rows = Array.from({ length: co }, () => []);
      for (let c = 0; c < co; c++) for (let dy = 0; dy < kh; dy++) for (let dx = 0; dx < kw; dx++) for (let j = 0; j < ci; j++) {
        const v = t.data[((c * kh + dy) * kw + dx) * ci + j];
        if (v !== 0) rows[c].push({ dy, dx, j, v });
      }
      this.sparse.set(id, rows);
    }
  }
  run(z) {
    if (z.length !== 512) throw Error('Expected 512 noise values.');
    const m = this.model, values = m.tensors.map(t => t.data), uses = m.uses.slice();
    values[m.input] = z;
    for (const op of m.ops) {
      const args = op.inputs.map(id => values[id]), o = op.options, out = m.tensors[op.output];
      let y;
      if (op.code === 77) y = Int32Array.from(m.tensors[op.inputs[0]].shape);
      else if (op.code === 45) y = Int32Array.of(args[0][args[1][0]]);
      else if (op.code === 83) y = Int32Array.from(args.map(x => x[0]));
      else if (op.code === 22) y = args[0];
      else if (op.code === 9) {
        const [x, w, b] = args, ci = x.length, co = b.length; y = new Float32Array(co);
        for (let c = 0; c < co; c++) {
          let sum = 0, offset = c * ci;
          for (let j = 0; j < ci; j++) sum += x[j] * w[offset + j];
          y[c] = sum + b[c];
        }
      } else if (op.code === 98 || op.code === 14) {
        const x = args[0]; y = new Float32Array(x.length);
        const alpha = op.code === 98 ? o.scalar(0, 'f') : 0;
        for (let i = 0; i < x.length; i++) y[i] = op.code === 14 ? 1 / (1 + Math.exp(-x[i])) : x[i] >= 0 ? x[i] : x[i] * alpha;
      } else if (op.code === 0) {
        const [a, b] = args; y = new Float32Array(a.length);
        if (a.length === b.length) for (let i = 0; i < a.length; i++) y[i] = a[i] + b[i];
        else for (let i = 0; i < a.length; i++) y[i] = a[i] + b[i % b.length];
      } else if (op.code === 3 || op.code === 67) {
        const transpose = op.code === 67;
        const x = args[transpose ? 2 : 0], w = args[1], b = args[transpose ? 3 : 2];
        const [, ih, iw, ci] = m.tensors[op.inputs[transpose ? 2 : 0]].shape;
        const [, oh, ow, co] = out.shape;
        const [, kh, kw] = m.tensors[op.inputs[1]].shape;
        const stride = transpose ? o.scalar(1) : 1;
        const ph = transpose ? Math.floor(Math.max((ih - 1) * stride + kh - oh, 0) / 2) : o.scalar(0, 'b') === 0 ? Math.floor(kh / 2) : 0;
        const pw = transpose ? Math.floor(Math.max((iw - 1) * stride + kw - ow, 0) / 2) : o.scalar(0, 'b') === 0 ? Math.floor(kw / 2) : 0;
        y = new Float32Array(oh * ow * co);
        const sparse = !transpose && this.sparse.get(op.inputs[1]);
        if (sparse) {
          for (let yy = 0; yy < oh; yy++) for (let xx = 0; xx < ow; xx++) for (let c = 0; c < co; c++) {
            let sum = 0;
            for (const entry of sparse[c]) {
              const iy = yy + entry.dy - ph, ix = xx + entry.dx - pw;
              if (iy >= 0 && iy < ih && ix >= 0 && ix < iw) sum += x[(iy * iw + ix) * ci + entry.j] * entry.v;
            }
            y[(yy * ow + xx) * co + c] = sum + b[c];
          }
        } else for (let yy = 0; yy < oh; yy++) for (let xx = 0; xx < ow; xx++) for (let c = 0; c < co; c++) {
          let sum = 0;
          for (let dy = 0; dy < kh; dy++) {
            const iy0 = yy + ph - (transpose ? dy : -dy), iy = transpose ? iy0 / stride : yy + dy - ph;
            if (!Number.isInteger(iy) || iy < 0 || iy >= ih) continue;
            for (let dx = 0; dx < kw; dx++) {
              const ix0 = xx + pw - dx, ix = transpose ? ix0 / stride : xx + dx - pw;
              if (!Number.isInteger(ix) || ix < 0 || ix >= iw) continue;
              const xi = (iy * iw + ix) * ci, wi = ((c * kh + dy) * kw + dx) * ci;
              for (let j = 0; j < ci; j++) sum += x[xi + j] * w[wi + j];
            }
          }
          y[(yy * ow + xx) * co + c] = sum + b[c];
        }
      } else if (op.code === 5) {
        const x = args[0], [, ih, iw, ci] = m.tensors[op.inputs[0]].shape;
        const [, oh, ow, co] = out.shape, size = o.scalar(0); y = new Float32Array(oh * ow * co);
        for (let yy = 0; yy < oh; yy++) for (let xx = 0; xx < ow; xx++) {
          const phase = ((yy % size) * size + xx % size) * co;
          const src = (Math.floor(yy / size) * iw + Math.floor(xx / size)) * ci + phase;
          y.set(x.subarray(src, src + co), (yy * ow + xx) * co);
        }
      } else if (op.code === 100) {
        const x = args[0], [, ih, iw, c] = m.tensors[op.inputs[0]].shape, [, oh, ow] = out.shape;
        y = new Float32Array(oh * ow * c);
        for (let yy = 0; yy < oh; yy++) for (let xx = 0; xx < ow; xx++) {
          const src = (Math.max(0, Math.min(ih - 1, yy - 1)) * iw + Math.max(0, Math.min(iw - 1, xx - 1))) * c;
          y.set(x.subarray(src, src + c), (yy * ow + xx) * c);
        }
      } else if (op.code === 4) {
        const [x, w, b] = args, [, ih, iw, c] = m.tensors[op.inputs[0]].shape;
        const [, oh, ow] = out.shape, [, kh, kw] = m.tensors[op.inputs[1]].shape;
        y = new Float32Array(oh * ow * c);
        for (let yy = 0; yy < oh; yy++) for (let xx = 0; xx < ow; xx++) for (let ch = 0; ch < c; ch++) {
          let sum = 0;
          for (let dy = 0; dy < kh; dy++) for (let dx = 0; dx < kw; dx++) sum += x[((yy + dy) * iw + xx + dx) * c + ch] * w[(dy * kw + dx) * c + ch];
          y[(yy * ow + xx) * c + ch] = sum + b[ch];
        }
      } else throw Error(`Unsupported CPU operator ${op.code}.`);
      if (y.length !== out.shape.reduce((a, b) => a * b, 1)) throw Error(`Unexpected tensor size after operator ${op.code}.`);
      values[op.output] = y;
      for (const id of op.inputs) if (--uses[id] === 0 && !m.tensors[id].data) values[id] = null;
    }
    return values[m.output];
  }
}

const VERTEX = `#version 300 es
void main(){vec2 p=vec2(float((gl_VertexID<<1)&2),float(gl_VertexID&2));gl_Position=vec4(p*2.0-1.0,0.0,1.0);}`;
const FRAGMENT = `#version 300 es
precision highp float;precision highp int;
uniform highp sampler2D X;uniform highp sampler2D W;uniform highp sampler2D B;
uniform ivec4 IS;uniform ivec4 OS;uniform ivec4 BS;uniform int WT;uniform float A;
out vec4 result;
vec4 weightAt(int i){return texelFetch(W,ivec2(i%WT,i/WT),0);}
vec4 inputAt(ivec2 p,int g){return texelFetch(X,ivec2(p.x*((IS.w+3)/4)+g,p.y),0);}
float inputChannel(ivec2 p,int c){vec4 v=inputAt(p,c/4);return v[c%4];}
vec4 biasAt(int g){return texelFetch(B,ivec2(g,0),0);}
`;

export class WebGLRunner {
  constructor(model) {
    this.model = model;
    const canvas = new OffscreenCanvas(1, 1), gl = canvas.getContext('webgl2', { antialias: false, depth: false, stencil: false, preserveDrawingBuffer: false });
    if (!gl || !gl.getExtension('EXT_color_buffer_float')) throw Error('Float WebGL2 unavailable.');
    this.gl = gl; this.maxTexture = gl.getParameter(gl.MAX_TEXTURE_SIZE); this.framebuffer = gl.createFramebuffer();
    gl.bindVertexArray(gl.createVertexArray()); gl.disable(gl.DITHER); gl.disable(gl.BLEND);
    this.programs = new Map(); this.constants = new Map(); this.weights = new Map();
    this.vertex = this.shader(gl.VERTEX_SHADER, VERTEX);
    this.canvas = canvas;
  }
  shader(type, source) {
    const gl = this.gl, shader = gl.createShader(type); gl.shaderSource(shader, source); gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) { const message = gl.getShaderInfoLog(shader); gl.deleteShader(shader); throw Error(message); }
    return shader;
  }
  texture(width, height, data = null) {
    const gl = this.gl;
    if (width > this.maxTexture || height > this.maxTexture) throw Error('Model exceeds GPU texture limits.');
    const texture = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, data);
    return { texture, width, height };
  }
  upload(data, shape) {
    const [, h, w, c] = shape4(shape), groups = Math.ceil(c / 4), array = new Float32Array(h * w * groups * 4);
    if (c % 4 === 0) array.set(data);
    else for (let i = 0; i < h * w; i++) array.set(data.subarray(i * c, (i + 1) * c), i * groups * 4);
    return this.texture(w * groups, h, array);
  }
  constant(id) {
    if (!this.constants.has(id)) { const t = this.model.tensors[id]; this.constants.set(id, this.upload(t.data, t.shape)); }
    return this.constants.get(id);
  }
  weight(id, ci) {
    const key = `${id}:${ci}`;
    if (!this.weights.has(key)) {
      const data = this.model.tensors[id].data, groups = Math.ceil(ci / 4), rows = data.length / ci;
      const texels = rows * groups, width = Math.min(1024, this.maxTexture, texels), height = Math.ceil(texels / width);
      const packed = new Float32Array(width * height * 4);
      if (ci % 4 === 0) packed.set(data);
      else for (let i = 0; i < rows; i++) packed.set(data.subarray(i * ci, (i + 1) * ci), i * groups * 4);
      this.weights.set(key, this.texture(width, height, packed));
    }
    return this.weights.get(key);
  }
  draw(body, input, inShape, outShape, weight = null, bias = null, alpha = 0, biasShape = [1, 1, 1, 1]) {
    const gl = this.gl;
    if (!this.programs.has(body)) {
      const fragment = this.shader(gl.FRAGMENT_SHADER, FRAGMENT + body), p = gl.createProgram();
      gl.attachShader(p, this.vertex); gl.attachShader(p, fragment); gl.linkProgram(p); gl.deleteShader(fragment);
      if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw Error(gl.getProgramInfoLog(p));
      const uniforms = Object.fromEntries(['X', 'W', 'B', 'IS', 'OS', 'BS', 'WT', 'A'].map(n => [n, gl.getUniformLocation(p, n)]));
      this.programs.set(body, { p, uniforms });
    }
    const { p, uniforms: u } = this.programs.get(body), os = shape4(outShape), is = shape4(inShape);
    const target = this.texture(os[2] * Math.ceil(os[3] / 4), os[1]);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer); gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, target.texture, 0);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) throw Error('GPU framebuffer unavailable.');
    gl.viewport(0, 0, target.width, target.height); gl.useProgram(p);
    for (const [unit, name, tex] of [[0, 'X', input], [1, 'W', weight || input], [2, 'B', bias || input]]) {
      gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, tex.texture); gl.uniform1i(u[name], unit);
    }
    gl.uniform4iv(u.IS, is); gl.uniform4iv(u.OS, os); gl.uniform4iv(u.BS, shape4(biasShape)); gl.uniform1i(u.WT, weight?.width || 1); gl.uniform1f(u.A, alpha);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    return target;
  }
  run(z) {
    const m = this.model, gl = this.gl, values = new Map(), uses = m.uses.slice(), temporary = new Set();
    const allocate = value => { temporary.add(value.texture); return value; };
    values.set(m.input, allocate(this.upload(z, [1, 512])));
    const get = id => values.get(id) || this.constant(id);
    try {
      for (const op of m.ops) {
        const o = op.options, outShape = m.tensors[op.output].shape;
        const ids = op.inputs, inShape = m.tensors[ids[0]].shape;
        let y;
        if (op.code === 77) y = Int32Array.from(inShape);
        else if (op.code === 45) y = Int32Array.of((values.get(ids[0]) || m.tensors[ids[0]].data)[m.tensors[ids[1]].data[0]]);
        else if (op.code === 83) y = Int32Array.from(ids.map(id => (values.get(id) || m.tensors[id].data)[0]));
        else {
          let body = '', input = get(ids[0]), weight = null, bias = null, alpha = 0, actualIn = inShape, bs = [1, 1, 1, 1];
          if (op.code === 22) body = `void main(){ivec2 p=ivec2(gl_FragCoord.xy);int width=OS.z*((OS.w+3)/4);int oldWidth=IS.z*((IS.w+3)/4);int i=p.y*width+p.x;result=texelFetch(X,ivec2(i%oldWidth,i/oldWidth),0);}`;
          else if (op.code === 98 || op.code === 14) {
            alpha = op.code === 98 ? o.scalar(0, 'f') : 0;
            body = op.code === 98 ? 'void main(){vec4 v=texelFetch(X,ivec2(gl_FragCoord.xy),0);result=max(v,0.0)+A*min(v,0.0);}' : 'void main(){vec4 v=texelFetch(X,ivec2(gl_FragCoord.xy),0);result=1.0/(1.0+exp(-v));}';
          } else if (op.code === 0) {
            bias = get(ids[1]); bs = m.tensors[ids[1]].shape;
            body = `void main(){ivec2 p=ivec2(gl_FragCoord.xy);int g=p.x%((OS.w+3)/4);int x=p.x/((OS.w+3)/4);ivec2 b=ivec2((BS.z==1?0:x)*((BS.w+3)/4)+g,BS.y==1?0:p.y);result=texelFetch(X,p,0)+texelFetch(B,b,0);}`;
          } else if (op.code === 100) body = `void main(){ivec2 p=ivec2(gl_FragCoord.xy);int cg=(OS.w+3)/4;ivec2 v=ivec2(clamp(p.x/cg-1,0,IS.z-1),clamp(p.y-1,0,IS.y-1));result=inputAt(v,p.x%cg);}`;
          else if (op.code === 5) {
            const block = o.scalar(0);
            body = `void main(){ivec2 p=ivec2(gl_FragCoord.xy);int cg=(OS.w+3)/4;int x=p.x/cg;int c=(p.x%cg)*4;int phase=((p.y%${block})*${block}+x%${block})*OS.w;ivec2 v=ivec2(x/${block},p.y/${block});result=vec4(0.0);for(int lane=0;lane<4;lane++){if(c+lane<OS.w)result[lane]=inputChannel(v,phase+c+lane);}}`;
          } else if (op.code === 9) {
            const ci = inShape[inShape.length - 1], groups = Math.ceil(ci / 4);
            weight = this.weight(ids[1], ci); bias = this.constant(ids[2]);
            body = `void main(){int c=int(gl_FragCoord.x)*4;vec4 sum=vec4(0.0);for(int j=0;j<${groups};j++){vec4 v=texelFetch(X,ivec2(j,0),0);sum+=vec4(dot(v,weightAt(c*${groups}+j)),dot(v,weightAt((c+1)*${groups}+j)),dot(v,weightAt((c+2)*${groups}+j)),dot(v,weightAt((c+3)*${groups}+j)));}result=sum+biasAt(c/4);}`;
          } else if (op.code === 3 || op.code === 67) {
            const transpose = op.code === 67;
            if (transpose) { input = get(ids[2]); actualIn = m.tensors[ids[2]].shape; }
            const [, ih, iw, ci] = actualIn, [, oh, ow, co] = outShape;
            const [, kh, kw] = m.tensors[ids[1]].shape, groups = Math.ceil(ci / 4);
            const stride = transpose ? o.scalar(1) : 1;
            const ph = transpose ? Math.floor(Math.max((ih - 1) * stride + kh - oh, 0) / 2) : o.scalar(0, 'b') === 0 ? Math.floor(kh / 2) : 0;
            const pw = transpose ? Math.floor(Math.max((iw - 1) * stride + kw - ow, 0) / 2) : o.scalar(0, 'b') === 0 ? Math.floor(kw / 2) : 0;
            weight = this.weight(ids[1], ci); bias = this.constant(ids[transpose ? 3 : 2]);
            const coords = transpose ? `ivec2 q=xy+ivec2(${pw},${ph})-ivec2(dx,dy);if(q.x%${stride}!=0||q.y%${stride}!=0)continue;q/=${stride};` : `ivec2 q=xy+ivec2(dx-${pw},dy-${ph});`;
            body = `void main(){ivec2 p=ivec2(gl_FragCoord.xy);int cg=(OS.w+3)/4;ivec2 xy=ivec2(p.x/cg,p.y);int c=(p.x%cg)*4;vec4 sum=vec4(0.0);for(int dy=0;dy<${kh};dy++){for(int dx=0;dx<${kw};dx++){${coords}if(q.x<0||q.y<0||q.x>=IS.z||q.y>=IS.y)continue;for(int j=0;j<${groups};j++){vec4 v=inputAt(q,j);int i=(dy*${kw}+dx)*${groups}+j;vec4 dots=vec4(0.0);for(int lane=0;lane<4;lane++){if(c+lane<${co})dots[lane]=dot(v,weightAt((c+lane)*${kh * kw * groups}+i));}sum+=dots;}}}result=sum+biasAt(c/4);}`;
          } else if (op.code === 4) {
            const c = actualIn[3], [, kh, kw] = m.tensors[ids[1]].shape, groups = Math.ceil(c / 4);
            weight = this.weight(ids[1], c); bias = this.constant(ids[2]);
            body = `void main(){ivec2 p=ivec2(gl_FragCoord.xy);int g=p.x%${groups};ivec2 xy=ivec2(p.x/${groups},p.y);vec4 sum=vec4(0.0);for(int dy=0;dy<${kh};dy++)for(int dx=0;dx<${kw};dx++){sum+=inputAt(xy+ivec2(dx,dy),g)*weightAt((dy*${kw}+dx)*${groups}+g);}result=sum+biasAt(g);}`;
          } else throw Error(`Unsupported GPU operator ${op.code}.`);
          y = allocate(this.draw(body, input, actualIn, outShape, weight, bias, alpha, bs));
        }
        values.set(op.output, y);
        for (const id of ids) if (--uses[id] === 0 && !m.tensors[id].data) {
          const old = values.get(id); if (old?.texture) { gl.deleteTexture(old.texture); temporary.delete(old.texture); } values.delete(id);
        }
      }
      const final = values.get(m.output), [, h, w, c] = m.tensors[m.output].shape;
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer); gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, final.texture, 0);
      const packed = new Float32Array(final.width * final.height * 4); gl.readPixels(0, 0, final.width, final.height, gl.RGBA, gl.FLOAT, packed);
      if (gl.isContextLost() || gl.getError() !== gl.NO_ERROR) throw Error('GPU execution failed.');
      const result = new Float32Array(h * w * c), stride = Math.ceil(c / 4) * 4;
      for (let i = 0; i < h * w; i++) result.set(packed.subarray(i * stride, i * stride + c), i * c);
      return result;
    } finally { for (const texture of temporary) gl.deleteTexture(texture); }
  }
  dispose() { this.gl.getExtension('WEBGL_lose_context')?.loseContext(); }
}
