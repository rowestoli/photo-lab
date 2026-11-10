import { crop, downsample, makeImg, mulberry32, shift, type Img } from './image';

/* ---------- 1. Align a handheld bracket (Ward's median threshold bitmaps) ---------- */

export function toGray(img: Img): Img {
  const out = makeImg(img.w, img.h, 1);
  for (let i = 0; i < img.w * img.h; i++) {
    out.data[i] = 0.299 * img.data[i * 3] + 0.587 * img.data[i * 3 + 1] + 0.114 * img.data[i * 3 + 2];
  }
  return out;
}

function median(values: Float32Array): number {
  const hist = new Uint32Array(256);
  for (const v of values) hist[Math.min(255, Math.max(0, Math.round(v * 255)))]++;
  let acc = 0;
  for (let i = 0; i < 256; i++) { acc += hist[i]; if (acc >= values.length / 2) return i / 255; }
  return 0.5;
}

/** Bitmap of pixels above the median, plus a mask that ignores pixels too close to it to be stable. */
export function bitmaps(gray: Img, tolerance = 4 / 255): { bits: Uint8Array; keep: Uint8Array } {
  const m = median(gray.data);
  const n = gray.w * gray.h;
  const bits = new Uint8Array(n), keep = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const v = gray.data[i];
    bits[i] = v > m ? 1 : 0;
    keep[i] = Math.abs(v - m) > tolerance ? 1 : 0;
  }
  return { bits, keep };
}

export function mtbError(a: ReturnType<typeof bitmaps>, b: ReturnType<typeof bitmaps>, w: number, h: number, dy: number, dx: number): number {
  let err = 0;
  const y0 = Math.max(0, dy), y1 = Math.min(h, h + dy), x0 = Math.max(0, dx), x1 = Math.min(w, w + dx);
  for (let y = y0; y < y1; y++) {
    const ra = y * w, rb = (y - dy) * w - dx;
    for (let x = x0; x < x1; x++) err += (a.bits[ra + x] ^ b.bits[rb + x]) & a.keep[ra + x] & b.keep[rb + x];
  }
  // Normalize by overlap so large shifts are not rewarded for covering less of the frame.
  return err / ((y1 - y0) * (x1 - x0));
}

/** Translation (dy, dx) that moves `img` onto `ref`. Exposure-invariant, so it works across a bracket. */
export function mtbAlign(ref: Img, img: Img, maxLevels = 6): { dy: number; dx: number } {
  // Stop while the coarsest level is still at least ~16 px on its short side.
  const levels = Math.max(1, Math.min(maxLevels, Math.floor(Math.log2(Math.min(ref.w, ref.h) / 16)) + 1));
  const ra = [toGray(ref)], rb = [toGray(img)];
  for (let i = 1; i < levels; i++) { ra.push(downsample(ra[i - 1])); rb.push(downsample(rb[i - 1])); }
  let dy = 0, dx = 0;
  for (let l = levels - 1; l >= 0; l--) {
    dy *= 2; dx *= 2;
    const a = bitmaps(ra[l]), b = bitmaps(rb[l]);
    let best = Infinity, by = dy, bx = dx;
    // Center first, so ties (flat regions carry no signal) keep the current estimate.
    for (const [oy, ox] of [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [-1, 1], [1, -1], [1, 1]]) {
      const e = mtbError(a, b, ra[l].w, ra[l].h, dy + oy, dx + ox);
      if (e < best) { best = e; by = dy + oy; bx = dx + ox; }
    }
    dy = by; dx = bx;
  }
  return { dy, dx };
}

/* ---------- 2. Recover the camera response (Debevec & Malik 1997) ---------- */

export const hat = (z: number) => Math.min(z, 255 - z) + 1;

/** Solve a symmetric positive definite system in place with Cholesky. */
export function choleskySolve(A: Float64Array, b: Float64Array, n: number): Float64Array {
  for (let j = 0; j < n; j++) {
    let s = A[j * n + j];
    for (let k = 0; k < j; k++) s -= A[j * n + k] * A[j * n + k];
    const d = Math.sqrt(Math.max(s, 1e-12));
    A[j * n + j] = d;
    for (let i = j + 1; i < n; i++) {
      let t = A[i * n + j];
      for (let k = 0; k < j; k++) t -= A[i * n + k] * A[j * n + k];
      A[i * n + j] = t / d;
    }
  }
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let t = b[i];
    for (let k = 0; k < i; k++) t -= A[i * n + k] * y[k];
    y[i] = t / A[i * n + i];
  }
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let t = y[i];
    for (let k = i + 1; k < n; k++) t -= A[k * n + i] * x[k];
    x[i] = t / A[i * n + i];
  }
  return x;
}

/**
 * g(z) = ln(exposure) for pixel value z. `Z[i][j]` is the value of sample i in exposure j.
 * Builds the normal equations of the weighted least-squares problem directly.
 */
export function solveResponse(Z: Uint8Array[], logT: number[], lambda: number): Float64Array {
  const N = Z.length, P = logT.length, n = 256 + N;
  const A = new Float64Array(n * n), b = new Float64Array(n);
  // Adds w * (sum coef_k * x_k) = w * rhs as one row.
  const addRow = (idx: number[], coef: number[], rhs: number, w: number) => {
    const w2 = w * w;
    for (let p = 0; p < idx.length; p++) {
      b[idx[p]] += w2 * coef[p] * rhs;
      for (let q = 0; q < idx.length; q++) A[idx[p] * n + idx[q]] += w2 * coef[p] * coef[q];
    }
  };
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < P; j++) {
      const z = Z[i][j];
      addRow([z, 256 + i], [1, -1], logT[j], hat(z));
    }
  }
  addRow([128], [1], 0, 1); // fixes the curve's free offset: g(128) = 0
  for (let z = 1; z < 255; z++) addRow([z - 1, z, z + 1], [1, -2, 1], 0, lambda * hat(z));
  for (let i = 0; i < n; i++) A[i * n + i] += 1e-9;
  return choleskySolve(A, b, n).slice(0, 256);
}

/** Pick sample pixels that span the tonal range of the middle exposure, inside the valid area. */
function pickSamples(mid: Img, count: number, seed = 1290): number[] {
  const rand = mulberry32(seed);
  const bins: number[][] = Array.from({ length: 16 }, () => []);
  const m = Math.floor(Math.min(mid.w, mid.h) * 0.03);
  for (let tries = 0; tries < count * 60; tries++) {
    const x = m + Math.floor(rand() * (mid.w - 2 * m)), y = m + Math.floor(rand() * (mid.h - 2 * m));
    const i = y * mid.w + x;
    const v = (mid.data[i * 3] + mid.data[i * 3 + 1] + mid.data[i * 3 + 2]) / 3;
    const bin = bins[Math.min(15, Math.floor(v * 16))];
    if (bin.length < Math.ceil(count / 16)) bin.push(i);
  }
  return bins.flat();
}

export interface HdrResult {
  radiance: Img; // linear relative radiance, 3 channels
  curves: Float64Array[]; // g for R, G, B
  shifts: { dy: number; dx: number }[];
  samples: number;
  ms: number;
}

export function buildHdr(frames: Img[], seconds: number[], opts: { lambda?: number; samples?: number } = {}): HdrResult {
  const t0 = performance.now();
  const ref = Math.floor(frames.length / 2);
  const shifts = frames.map((f, i) => (i === ref ? { dy: 0, dx: 0 } : mtbAlign(frames[ref], f)));
  const moved = frames.map((f, i) => shift(f, shifts[i].dy, shifts[i].dx));
  // Keep only the area every shifted frame covers.
  const y0 = Math.max(0, ...shifts.map((s) => s.dy)), y1 = Math.min(...shifts.map((s) => frames[0].h + Math.min(0, s.dy)));
  const x0 = Math.max(0, ...shifts.map((s) => s.dx)), x1 = Math.min(...shifts.map((s) => frames[0].w + Math.min(0, s.dx)));
  const imgs = moved.map((f) => crop(f, x0, y0, x1 - x0, y1 - y0));
  const { w, h } = imgs[0];

  const logT = seconds.map(Math.log);
  const idx = pickSamples(imgs[ref], opts.samples ?? 200);
  const curves: Float64Array[] = [];
  for (let c = 0; c < 3; c++) {
    const Z = idx.map((i) => Uint8Array.from(imgs.map((im) => Math.round(im.data[i * 3 + c] * 255))));
    curves.push(solveResponse(Z, logT, opts.lambda ?? 50));
  }

  const radiance = makeImg(w, h, 3);
  for (let i = 0; i < w * h; i++) {
    for (let c = 0; c < 3; c++) {
      let num = 0, den = 0;
      for (let j = 0; j < imgs.length; j++) {
        const z = Math.round(imgs[j].data[i * 3 + c] * 255);
        const wt = hat(z);
        num += wt * (curves[c][z] - logT[j]);
        den += wt;
      }
      radiance.data[i * 3 + c] = Math.exp(num / den);
    }
  }
  return { radiance, curves, shifts, samples: idx.length, ms: performance.now() - t0 };
}

/* ---------- 3. Tone mapping ---------- */

export const luminance = (d: Float32Array, i: number) => 0.2126 * d[i * 3] + 0.7152 * d[i * 3 + 1] + 0.0722 * d[i * 3 + 2];

function percentile(values: Float32Array, p: number): number {
  const sorted = Float32Array.from(values).sort();
  return sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))];
}

/**
 * Edge-preserving blur via a bilateral grid (Chen, Paris & Durand 2007):
 * splat into (x, y, value) cells, blur the 3-D grid, read back with trilinear interpolation.
 */
export function bilateralGrid(f: Float32Array, w: number, h: number, sigmaS: number, sigmaR: number): Float32Array {
  let lo = Infinity, hi = -Infinity;
  for (const v of f) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const pad = 2;
  const gx = Math.ceil((w - 1) / sigmaS) + 1 + 2 * pad, gy = Math.ceil((h - 1) / sigmaS) + 1 + 2 * pad;
  const gz = Math.ceil((hi - lo) / sigmaR) + 1 + 2 * pad;
  const size = gx * gy * gz;
  let val = new Float32Array(size), wt = new Float32Array(size);
  const cell = (x: number, y: number, z: number) => (z * gy + y) * gx + x;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = f[y * w + x];
      const k = cell(Math.round(x / sigmaS) + pad, Math.round(y / sigmaS) + pad, Math.round((v - lo) / sigmaR) + pad);
      val[k] += v; wt[k] += 1;
    }
  }
  const strides = [1, gx, gx * gy], dims = [gx, gy, gz];
  for (let axis = 0; axis < 3; axis++) {
    const s = strides[axis], dim = dims[axis];
    const nv = new Float32Array(size), nw = new Float32Array(size);
    for (let k = 0; k < size; k++) {
      const pos = Math.floor(k / s) % dim;
      let a = 6 * val[k], b = 6 * wt[k];
      if (pos > 0) { a += 4 * val[k - s]; b += 4 * wt[k - s]; }
      if (pos > 1) { a += val[k - 2 * s]; b += wt[k - 2 * s]; }
      if (pos < dim - 1) { a += 4 * val[k + s]; b += 4 * wt[k + s]; }
      if (pos < dim - 2) { a += val[k + 2 * s]; b += wt[k + 2 * s]; }
      nv[k] = a / 16; nw[k] = b / 16;
    }
    val = nv; wt = nw;
  }
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = f[y * w + x];
      const fx = x / sigmaS + pad, fy = y / sigmaS + pad, fz = (v - lo) / sigmaR + pad;
      const ix = Math.floor(fx), iy = Math.floor(fy), iz = Math.floor(fz);
      const tx = fx - ix, ty = fy - iy, tz = fz - iz;
      let a = 0, b = 0;
      for (let dz = 0; dz < 2; dz++) for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
        const c = (dx ? tx : 1 - tx) * (dy ? ty : 1 - ty) * (dz ? tz : 1 - tz);
        const k = cell(Math.min(gx - 1, ix + dx), Math.min(gy - 1, iy + dy), Math.min(gz - 1, iz + dz));
        a += c * val[k]; b += c * wt[k];
      }
      out[y * w + x] = b > 1e-8 ? a / b : v;
    }
  }
  return out;
}

export interface ToneParams {
  operator: 'global' | 'local';
  exposure: number; // stops
  saturation: number; // 0..1.5
  detail: number; // local operator: detail boost, 1 = unchanged
  contrast: number; // local operator: output contrast in stops
}

export const defaultTone: ToneParams = { operator: 'local', exposure: 0, saturation: 0.75, detail: 1.25, contrast: 5 };

/** Cache of the expensive part of the local operator, keyed by the radiance object. */
const baseCache = new WeakMap<Img, { L: Float32Array; logL: Float32Array; base: Float32Array; avgLog: number; white: number }>();

function prepare(radiance: Img) {
  let hit = baseCache.get(radiance);
  if (hit) return hit;
  const n = radiance.w * radiance.h;
  const L = new Float32Array(n), logL = new Float32Array(n);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    L[i] = Math.max(luminance(radiance.data, i), 1e-8);
    logL[i] = Math.log10(L[i]);
    sum += logL[i];
  }
  const base = bilateralGrid(logL, radiance.w, radiance.h, Math.max(radiance.w, radiance.h) * 0.02, 0.4);
  hit = { L, logL, base, avgLog: sum / n, white: percentile(L, 0.995) };
  baseCache.set(radiance, hit);
  return hit;
}

const GAMMA = Float32Array.from({ length: 4096 }, (_, i) => Math.pow(i / 4095, 1 / 2.2));

export function toneMap(radiance: Img, p: ToneParams): Img {
  const { L, logL, base, avgLog, white: whiteL } = prepare(radiance);
  const n = L.length;
  const out = makeImg(radiance.w, radiance.h, 3);
  const Ld = new Float32Array(n);
  const gain = Math.pow(2, p.exposure);
  if (p.operator === 'global') {
    // Reinhard et al. 2002: scale to a mid-grey key, then compress with a soft white point.
    const scale = 0.18 / Math.pow(10, avgLog);
    const white = whiteL * scale * gain;
    for (let i = 0; i < n; i++) {
      const Lm = L[i] * scale * gain;
      Ld[i] = (Lm * (1 + Lm / (white * white))) / (1 + Lm);
    }
  } else {
    // Durand & Dorsey 2002: compress only the large-scale base layer, keep (or boost) the detail.
    let bmax = -Infinity, bmin = Infinity;
    for (let i = 0; i < n; i++) { if (base[i] > bmax) bmax = base[i]; if (base[i] < bmin) bmin = base[i]; }
    const k = (p.contrast * Math.log10(2)) / Math.max(bmax - bmin, 1e-6);
    for (let i = 0; i < n; i++) {
      const detail = logL[i] - base[i];
      Ld[i] = Math.pow(10, (base[i] - bmax) * k + detail * p.detail) * gain;
    }
  }
  const d = radiance.data, o = out.data, s = p.saturation;
  for (let i = 0; i < n; i++) {
    const inv = 1 / L[i], l = Ld[i];
    for (let c = 0; c < 3; c++) {
      const v = Math.exp(s * Math.log(Math.max(d[i * 3 + c] * inv, 1e-6))) * l;
      o[i * 3 + c] = v >= 1 ? 1 : GAMMA[(v * 4095) | 0];
    }
  }
  return out;
}

/** False-color view of log luminance, so the range the camera captured is visible. */
const RAMP = [[13, 8, 135], [84, 2, 163], [139, 10, 165], [185, 50, 137], [219, 92, 104], [244, 136, 73], [254, 188, 43], [240, 249, 33]];
export function radianceView(radiance: Img): { img: Img; lo: number; hi: number } {
  const { logL } = prepare(radiance);
  const lo = percentile(logL, 0.005), hi = percentile(logL, 0.9995);
  const out = makeImg(radiance.w, radiance.h, 3);
  for (let i = 0; i < logL.length; i++) {
    const t = Math.min(1, Math.max(0, (logL[i] - lo) / (hi - lo))) * (RAMP.length - 1);
    const a = Math.floor(Math.min(t, RAMP.length - 2)), f = t - a;
    for (let c = 0; c < 3; c++) out.data[i * 3 + c] = (RAMP[a][c] * (1 - f) + RAMP[a + 1][c] * f) / 255;
  }
  return { img: out, lo, hi };
}

/** Captured dynamic range in stops, robust to a few hot or dead pixels. */
export function dynamicRangeStops(radiance: Img): number {
  const { logL } = prepare(radiance);
  return (percentile(logL, 0.999) - percentile(logL, 0.001)) / Math.log10(2);
}
