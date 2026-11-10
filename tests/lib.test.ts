import { describe, expect, it } from 'vitest';
import { makeImg, mulberry32, shift, type Img } from '../src/lib/image';
import { alignPyramid, ncc } from '../src/lib/align';
import { bilateralGrid, buildHdr, choleskySolve, mtbAlign, solveResponse } from '../src/lib/hdr';
import { blend } from '../src/lib/poisson';

/** Smooth random texture so alignment has something to lock onto. */
function texture(w: number, h: number, c = 1, seed = 7): Img {
  const img = makeImg(w, h, c);
  const rand = mulberry32(seed);
  const blobs = Array.from({ length: 40 }, () => [rand() * w, rand() * h, 4 + rand() * 20, rand()]);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let v = 0.2;
    for (const [bx, by, r, a] of blobs) v += 0.3 * a * Math.exp(-((x - bx) ** 2 + (y - by) ** 2) / (2 * r * r));
    for (let k = 0; k < c; k++) img.data[(y * w + x) * c + k] = Math.min(1, v * (0.6 + 0.2 * k));
  }
  return img;
}

/** Hard-edged rectangles: MTB needs edges, since it ignores pixels near the median. */
function blocks(w: number, h: number, seed = 4): Img {
  const img = makeImg(w, h, 3);
  img.data.fill(0.25);
  const rand = mulberry32(seed);
  for (let n = 0; n < 60; n++) {
    const x0 = Math.floor(rand() * w), y0 = Math.floor(rand() * h), bw = 4 + Math.floor(rand() * 30), bh = 4 + Math.floor(rand() * 30);
    const v = 0.05 + rand() * 0.9;
    for (let y = y0; y < Math.min(h, y0 + bh); y++) for (let x = x0; x < Math.min(w, x0 + bw); x++) {
      for (let k = 0; k < 3; k++) img.data[(y * w + x) * 3 + k] = v * (0.8 + 0.1 * k);
    }
  }
  return img;
}

describe('alignment', () => {
  it('scores an exact match as 1', () => {
    const t = texture(80, 60);
    expect(ncc(t, t, 0, 0)).toBeCloseTo(1, 6);
  });

  it('recovers a known translation through the pyramid', () => {
    const ref = texture(320, 280);
    for (const [dy, dx] of [[9, -13], [-21, 6], [0, 0]]) {
      const moved = shift(ref, -dy, -dx); // undoing (dy, dx) aligns it again
      for (const metric of ['edges', 'brightness'] as const) {
        const r = alignPyramid(ref, moved, { metric });
        expect([r.dy, r.dx]).toEqual([dy, dx]);
      }
    }
  });
});

describe('HDR', () => {
  it('solves a small SPD system', () => {
    const A = new Float64Array([4, 2, 0, 2, 5, 1, 0, 1, 3]);
    const x = choleskySolve(A, new Float64Array([2, 3, 1]), 3);
    expect(x[0] * 4 + x[1] * 2).toBeCloseTo(2, 9);
    expect(x[0] * 2 + x[1] * 5 + x[2]).toBeCloseTo(3, 9);
  });

  it('MTB alignment ignores exposure differences', () => {
    const ref = blocks(256, 192);
    const dark = makeImg(ref.w, ref.h, 3);
    dark.data.set(ref.data.map((v) => v * 0.3));
    const moved = shift(dark, 5, -7);
    expect(mtbAlign(ref, moved)).toEqual({ dy: -5, dx: 7 });
  });

  it('recovers a known gamma response curve up to its offset', () => {
    const rand = mulberry32(3);
    const N = 120, seconds = [1, 1 / 4, 1 / 16, 1 / 64];
    const radiance = Array.from({ length: N }, () => Math.exp(rand() * 7 - 3));
    const response = (e: number) => Math.round(255 * Math.min(1, Math.pow(e, 1 / 2.2) / 4));
    const Z = radiance.map((E) => Uint8Array.from(seconds.map((t) => response(E * t))));
    const g = solveResponse(Z, seconds.map(Math.log), 20);
    // True g(z) = 2.2 ln(4z/255); compare slopes between well-exposed values.
    const truth = (z: number) => 2.2 * Math.log((4 * z) / 255);
    for (const [a, b] of [[40, 120], [80, 200]]) {
      expect(g[b] - g[a]).toBeCloseTo(truth(b) - truth(a), 0);
    }
  });

  it('merges a synthetic bracket into radiance proportional to the scene', () => {
    const scene = blocks(96, 72, 11);
    const seconds = [1, 1 / 4, 1 / 16];
    const frames = seconds.map((t) => {
      const f = makeImg(scene.w, scene.h, 3);
      f.data.set(scene.data.map((v) => Math.min(1, Math.pow(v * 3 * t, 1 / 2.2))));
      return f;
    });
    const { radiance, shifts } = buildHdr(frames, seconds, { samples: 160 });
    expect(shifts.every((s) => s.dy === 0 && s.dx === 0)).toBe(true);
    // Ratios between two pixels should match the scene's ratio.
    const i = 10 * 96 + 10, j = 40 * 96 + 50;
    const ratio = radiance.data[i * 3 + 1] / radiance.data[j * 3 + 1];
    const truth = scene.data[i * 3 + 1] / scene.data[j * 3 + 1];
    expect(Math.abs(Math.log(ratio / truth))).toBeLessThan(0.15);
  });

  it('bilateral grid leaves a constant image unchanged and keeps a step edge', () => {
    const w = 64, h = 32;
    const flat = new Float32Array(w * h).fill(0.7);
    for (const v of bilateralGrid(flat, w, h, 4, 0.1)) expect(v).toBeCloseTo(0.7, 5);
    const step = new Float32Array(w * h).map((_, i) => ((i % w) < w / 2 ? 0 : 1));
    const out = bilateralGrid(step, w, h, 4, 0.1);
    expect(out[10 * w + w / 2 - 1]).toBeLessThan(0.05);
    expect(out[10 * w + w / 2]).toBeGreaterThan(0.95);
  });
});

describe('Poisson blending', () => {
  const dst = texture(60, 50, 3, 5);
  const mask = new Uint8Array(20 * 20).fill(1);

  it('reproduces the destination when the source is the destination', () => {
    const src = makeImg(20, 20, 3);
    for (let y = 0; y < 20; y++) for (let x = 0; x < 20; x++) for (let c = 0; c < 3; c++) {
      src.data[(y * 20 + x) * 3 + c] = dst.data[((y + 15) * 60 + x + 20) * 3 + c];
    }
    const { out } = blend(src, mask, dst, 20, 15, 'poisson', { tol: 1e-9 });
    for (let i = 0; i < out.data.length; i++) expect(out.data[i]).toBeCloseTo(dst.data[i], 4);
  });

  it('is invariant to a constant brightness offset in the source', () => {
    const src = texture(20, 20, 3, 9);
    const brighter = makeImg(20, 20, 3);
    brighter.data.set(src.data.map((v) => v * 0.5 + 0.3));
    const half = makeImg(20, 20, 3);
    half.data.set(src.data.map((v) => v * 0.5));
    const a = blend(brighter, mask, dst, 20, 15, 'poisson', { tol: 1e-9 }).out;
    const b = blend(half, mask, dst, 20, 15, 'poisson', { tol: 1e-9 }).out;
    for (let i = 0; i < a.data.length; i++) expect(a.data[i]).toBeCloseTo(b.data[i], 4);
  });

  it('paste copies the source and ignores pixels outside the frame', () => {
    const src = texture(20, 20, 3, 9);
    const r = blend(src, mask, dst, 50, 40, 'paste');
    expect(r.unknowns).toBe(9 * 9); // only the part strictly inside the 60x50 frame
    expect(r.out.data[(45 * 60 + 55) * 3]).toBeCloseTo(src.data[(5 * 20 + 5) * 3], 6);
  });
});
