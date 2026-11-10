import { makeImg, type Img } from './image';

export type BlendMode = 'paste' | 'poisson' | 'mixed';

export interface BlendResult {
  out: Img;
  unknowns: number;
  iterations: number;
  ms: number;
}

/**
 * Paste `src` (where `mask` is set) into `dst` with its top-left corner at (ox, oy).
 *
 * poisson: inside the mask, solve  Δf = Δsrc  with f = dst on the boundary (Pérez et al. 2003).
 * mixed:   same, but each edge keeps whichever gradient is stronger, source or destination,
 *          so background texture shows through thin or transparent parts of the object.
 *
 * The 5-point Laplacian system is symmetric positive definite, so it is solved
 * matrix-free with conjugate gradients, warm-started from `init` when given.
 */
export function blend(
  src: Img, mask: Uint8Array, dst: Img, ox: number, oy: number, mode: BlendMode,
  opts: { tol?: number; maxIter?: number; init?: Img } = {},
): BlendResult {
  const t0 = performance.now();
  const out = makeImg(dst.w, dst.h, 3);
  out.data.set(dst.data);
  const { w: W, h: H } = dst;

  // Unknown pixels: masked source pixels that land strictly inside the destination.
  const index = new Int32Array(W * H).fill(-1);
  const coords: number[] = [];
  for (let y = 0; y < src.h; y++) {
    for (let x = 0; x < src.w; x++) {
      if (!mask[y * src.w + x]) continue;
      const tx = x + ox, ty = y + oy;
      if (tx < 1 || ty < 1 || tx >= W - 1 || ty >= H - 1) continue;
      index[ty * W + tx] = coords.length;
      coords.push(ty * W + tx);
    }
  }
  const n = coords.length;
  if (mode === 'paste' || n === 0) {
    for (const p of coords) {
      const sx = (p % W) - ox, sy = Math.floor(p / W) - oy;
      for (let c = 0; c < 3; c++) out.data[p * 3 + c] = src.data[(sy * src.w + sx) * 3 + c];
    }
    return { out, unknowns: n, iterations: 0, ms: performance.now() - t0 };
  }

  // Neighbor table: index of each neighbor unknown, or -1 if it is fixed boundary.
  const nbr = new Int32Array(n * 4);
  const offs = [-1, 1, -W, W];
  for (let i = 0; i < n; i++) for (let k = 0; k < 4; k++) nbr[i * 4 + k] = index[coords[i] + offs[k]];
  const srcAt = (p: number, c: number) => {
    const sx = Math.min(src.w - 1, Math.max(0, (p % W) - ox));
    const sy = Math.min(src.h - 1, Math.max(0, Math.floor(p / W) - oy));
    return src.data[(sy * src.w + sx) * 3 + c];
  };

  const applyA = (x: Float64Array, y: Float64Array) => {
    for (let i = 0; i < n; i++) {
      let s = 4 * x[i];
      for (let k = 0; k < 4; k++) { const j = nbr[i * 4 + k]; if (j >= 0) s -= x[j]; }
      y[i] = s;
    }
  };

  let iterations = 0;
  const b = new Float64Array(n), x = new Float64Array(n), r = new Float64Array(n), d = new Float64Array(n), q = new Float64Array(n);
  for (let c = 0; c < 3; c++) {
    for (let i = 0; i < n; i++) {
      const p = coords[i];
      let rhs = 0;
      for (let k = 0; k < 4; k++) {
        const pq = p + offs[k];
        let g = srcAt(p, c) - srcAt(pq, c);
        if (mode === 'mixed') {
          const gd = dst.data[p * 3 + c] - dst.data[pq * 3 + c];
          if (Math.abs(gd) > Math.abs(g)) g = gd;
        }
        rhs += g;
        if (nbr[i * 4 + k] < 0) rhs += dst.data[pq * 3 + c];
      }
      b[i] = rhs;
      x[i] = opts.init ? opts.init.data[p * 3 + c] : srcAt(p, c);
    }
    // Conjugate gradients.
    applyA(x, q);
    let rr = 0, bb = 0;
    for (let i = 0; i < n; i++) { r[i] = b[i] - q[i]; d[i] = r[i]; rr += r[i] * r[i]; bb += b[i] * b[i]; }
    const stop = (opts.tol ?? 1e-5) ** 2 * Math.max(bb, 1e-12);
    const maxIter = opts.maxIter ?? 3000;
    let it = 0;
    while (rr > stop && it < maxIter) {
      applyA(d, q);
      let dq = 0;
      for (let i = 0; i < n; i++) dq += d[i] * q[i];
      const alpha = rr / dq;
      let rr2 = 0;
      for (let i = 0; i < n; i++) { x[i] += alpha * d[i]; r[i] -= alpha * q[i]; rr2 += r[i] * r[i]; }
      const beta = rr2 / rr;
      for (let i = 0; i < n; i++) d[i] = r[i] + beta * d[i];
      rr = rr2;
      it++;
    }
    iterations = Math.max(iterations, it);
    for (let i = 0; i < n; i++) out.data[coords[i] * 3 + c] = Math.min(1, Math.max(0, x[i]));
  }
  return { out, unknowns: n, iterations, ms: performance.now() - t0 };
}
