// Gaussian endpoints; spherical direction interpolation with interpolated radii.
// This avoids the mid-journey variance collapse of a straight Gaussian lerp.
export function gaussianNoise(length = 512, random = () => {
  const value = new Uint32Array(1); crypto.getRandomValues(value);
  return (value[0] + 0.5) / 4294967296;
}) {
  const result = new Float32Array(length);
  for (let i = 0; i < length; i += 2) {
    const r = Math.sqrt(-2 * Math.log(Math.max(Number.MIN_VALUE, random()))), angle = Math.PI * 2 * random();
    result[i] = r * Math.cos(angle); if (i + 1 < length) result[i + 1] = r * Math.sin(angle);
  }
  return result;
}
export function smootherstep(t) { t = Math.max(0, Math.min(1, t)); return t * t * t * (t * (t * 6 - 15) + 10); }
export function slerp(a, b, t) {
  if (a.length !== b.length) throw Error('Noise dimensions differ.');
  if (t <= 0) return a.slice(); if (t >= 1) return b.slice();
  let aa = 0, bb = 0, ab = 0;
  for (let i = 0; i < a.length; i++) { aa += a[i] * a[i]; bb += b[i] * b[i]; ab += a[i] * b[i]; }
  if (aa === bb && ab === aa) return a.slice();
  const na = Math.sqrt(aa), nb = Math.sqrt(bb), out = new Float32Array(a.length);
  if (na < 1e-8 || nb < 1e-8) { for (let i = 0; i < a.length; i++) out[i] = a[i] * (1 - t) + b[i] * t; return out; }
  const cosine = Math.max(-1, Math.min(1, ab / (na * nb))), radius = na * (1 - t) + nb * t;
  if (cosine > 0.9995) {
    let norm = 0;
    for (let i = 0; i < a.length; i++) { out[i] = (1 - t) * a[i] / na + t * b[i] / nb; norm += out[i] * out[i]; }
    norm = Math.sqrt(norm); for (let i = 0; i < a.length; i++) out[i] *= radius / norm;
    return out;
  }
  if (cosine < -0.9995) {
    // Deterministic perpendicular basis avoids the antipodal singularity.
    let norm = 0;
    const perpendicular = new Float64Array(a.length);
    for (let i = 0; i < a.length; i++) { perpendicular[i] = b[i] / nb - cosine * a[i] / na; norm += perpendicular[i] ** 2; }
    let angle = Math.acos(cosine);
    if (norm < 1e-12) {
      let axis = 0; for (let i = 1; i < a.length; i++) if (Math.abs(a[i]) < Math.abs(a[axis])) axis = i;
      const projection = a[axis] / na; norm = 0; angle = Math.PI;
      for (let i = 0; i < a.length; i++) { perpendicular[i] = (i === axis ? 1 : 0) - projection * a[i] / na; norm += perpendicular[i] ** 2; }
    }
    norm = Math.sqrt(norm);
    for (let i = 0; i < a.length; i++) out[i] = radius * (Math.cos(angle * t) * a[i] / na + Math.sin(angle * t) * perpendicular[i] / norm);
    return out;
  }
  const theta = Math.acos(cosine), denominator = Math.sin(theta);
  const wa = Math.sin((1 - t) * theta) / denominator / na, wb = Math.sin(t * theta) / denominator / nb;
  for (let i = 0; i < a.length; i++) out[i] = radius * (wa * a[i] + wb * b[i]);
  return out;
}
