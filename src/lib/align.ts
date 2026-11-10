import { crop, downsample, gradientMagnitude, makeImg, shift, type Img } from './image';

export type Metric = 'edges' | 'brightness';

export interface LevelTrace {
  level: number;
  width: number;
  dy: number;
  dx: number;
  score: number;
  tried: number;
}

export interface AlignResult {
  dy: number;
  dx: number;
  score: number;
  trace: LevelTrace[];
}

/**
 * Normalized cross-correlation between `ref` and `tgt` shifted by (dy, dx),
 * scored only over the interior (a `margin` fraction is ignored on every side,
 * because plate borders are scratched and never line up).
 */
export function ncc(ref: Img, tgt: Img, dy: number, dx: number, margin = 0.1): number {
  const { w, h } = ref;
  const mx = Math.floor(w * margin), my = Math.floor(h * margin);
  const x0 = Math.max(mx, mx + dx), x1 = Math.min(w - mx, w - mx + dx);
  const y0 = Math.max(my, my + dy), y1 = Math.min(h - my, h - my + dy);
  if (x1 - x0 < 2 || y1 - y0 < 2) return -1;
  let sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0, n = 0;
  const a = ref.data, b = tgt.data;
  for (let y = y0; y < y1; y++) {
    const ra = y * w, rb = (y - dy) * w - dx;
    for (let x = x0; x < x1; x++) {
      const va = a[ra + x], vb = b[rb + x];
      sa += va; sb += vb; saa += va * va; sbb += vb * vb; sab += va * vb; n++;
    }
  }
  const cov = sab - (sa * sb) / n;
  const den = Math.sqrt((saa - (sa * sa) / n) * (sbb - (sb * sb) / n));
  return den > 1e-12 ? cov / den : -1;
}

/**
 * Coarse-to-fine search: exhaustive ±radius at the coarsest level, then a ±2
 * refinement at each finer level after doubling the estimate.
 */
export function alignPyramid(ref: Img, tgt: Img, opts: { metric?: Metric; radius?: number; minSize?: number } = {}): AlignResult {
  const metric = opts.metric ?? 'edges';
  const prep = (im: Img) => (metric === 'edges' ? gradientMagnitude(im) : im);
  const refs = [prep(ref)], tgts = [prep(tgt)];
  const minSize = opts.minSize ?? 48;
  while (Math.min(refs[refs.length - 1].w, refs[refs.length - 1].h) / 2 >= minSize) {
    refs.push(downsample(refs[refs.length - 1]));
    tgts.push(downsample(tgts[tgts.length - 1]));
  }
  const top = refs.length - 1;
  const coarseRadius = opts.radius ?? Math.max(4, Math.ceil(refs[top].w * 0.08));
  let dy = 0, dx = 0, best = -Infinity;
  const trace: LevelTrace[] = [];
  for (let level = top; level >= 0; level--) {
    const r = level === top ? coarseRadius : 2;
    if (level !== top) { dy *= 2; dx *= 2; }
    let by = dy, bx = dx, tried = 0;
    best = -Infinity;
    for (let y = dy - r; y <= dy + r; y++) {
      for (let x = dx - r; x <= dx + r; x++) {
        const s = ncc(refs[level], tgts[level], y, x);
        tried++;
        if (s > best) { best = s; by = y; bx = x; }
      }
    }
    dy = by; dx = bx;
    trace.push({ level, width: refs[level].w, dy, dx, score: best, tried });
  }
  return { dy, dx, score: best, trace };
}

/** Split a scanned glass plate (blue, green, red stacked top to bottom). */
export function splitPlate(plate: Img): [Img, Img, Img] {
  const h = Math.floor(plate.h / 3);
  return [crop(plate, 0, 0, plate.w, h), crop(plate, 0, h, plate.w, h), crop(plate, 0, 2 * h, plate.w, h)];
}

/** Rows/columns that are mostly black or white border are trimmed (same rule as the course version). */
export function detectContent(img: Img, low = 0.1, high = 0.9, minContent = 0.1): { top: number; bottom: number; left: number; right: number } {
  const { w, h, data } = img;
  let lo = Infinity, hi = -Infinity;
  for (const v of data) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const norm = (v: number) => (v - lo) / (hi - lo + 1e-8);
  const rowOk = (y: number) => {
    let mid = 0;
    for (let x = 0; x < w; x++) { const v = norm(data[y * w + x]); if (v >= low && v <= high) mid++; }
    return mid / w > minContent;
  };
  const colOk = (x: number) => {
    let mid = 0;
    for (let y = 0; y < h; y++) { const v = norm(data[y * w + x]); if (v >= low && v <= high) mid++; }
    return mid / h > minContent;
  };
  let top = 0; while (top < h - 1 && !rowOk(top)) top++;
  let bottom = h - 1; while (bottom > top && !rowOk(bottom)) bottom--;
  let left = 0; while (left < w - 1 && !colOk(left)) left++;
  let right = w - 1; while (right > left && !colOk(right)) right--;
  return { top, bottom, left, right };
}

export interface PlateResult {
  naive: Img;
  aligned: Img;
  green: AlignResult;
  red: AlignResult;
  ms: number;
}

/** Full pipeline: split, align green and red to blue, stack, crop to the area all three channels cover. */
export function alignPlate(plate: Img, metric: Metric = 'edges'): PlateResult {
  const t0 = performance.now();
  const [b, g, r] = splitPlate(plate);
  const green = alignPyramid(b, g, { metric });
  const red = alignPyramid(b, r, { metric });
  const gs = shift(g, green.dy, green.dx), rs = shift(r, red.dy, red.dx);

  const stack = (rc: Img, gc: Img, bc: Img) => {
    const out = makeImg(bc.w, bc.h, 3);
    for (let i = 0; i < bc.w * bc.h; i++) {
      out.data[i * 3] = rc.data[i]; out.data[i * 3 + 1] = gc.data[i]; out.data[i * 3 + 2] = bc.data[i];
    }
    return out;
  };

  // Overlap of the three channels after shifting, then trim the plate's own border.
  const y0 = Math.max(0, green.dy, red.dy), y1 = Math.min(b.h, b.h + green.dy, b.h + red.dy);
  const x0 = Math.max(0, green.dx, red.dx), x1 = Math.min(b.w, b.w + green.dx, b.w + red.dx);
  // Each channel has its own scratched border: keep what all three agree is content, minus a small inset.
  const boxes = [b, gs, rs].map((c) => detectContent(crop(c, x0, y0, x1 - x0, y1 - y0)));
  const inset = Math.round(Math.min(x1 - x0, y1 - y0) * 0.02);
  const left = Math.max(...boxes.map((q) => q.left)) + inset, right = Math.min(...boxes.map((q) => q.right)) - inset;
  const top = Math.max(...boxes.map((q) => q.top)) + inset, bottom = Math.min(...boxes.map((q) => q.bottom)) - inset;
  const cx = x0 + left, cy = y0 + top;
  const cw = right - left + 1, ch = bottom - top + 1;

  const naive0 = crop(stack(r, g, b), cx, cy, cw, ch);
  const aligned0 = crop(stack(rs, gs, b), cx, cy, cw, ch);
  const t = trimFringes(aligned0);
  const naive = crop(naive0, t.left, t.top, t.w, t.h);
  const aligned = crop(aligned0, t.left, t.top, t.w, t.h);
  return { naive, aligned, green, red, ms: performance.now() - t0 };
}

/**
 * After alignment the channels agree inside the picture but not on leftover plate edge,
 * where one channel is still scratched border. Trim edge rows and columns whose
 * channel disagreement is far above the image's typical level (at most 8% per side).
 */
export function trimFringes(img: Img): { left: number; top: number; w: number; h: number } {
  const { w, h, data } = img;
  const dis = (i: number) => {
    const r = data[i * 3], g = data[i * 3 + 1], b = data[i * 3 + 2];
    return Math.abs(r - g) + Math.abs(g - b) + Math.abs(r - b);
  };
  const lum = (i: number) => (data[i * 3] + data[i * 3 + 1] + data[i * 3 + 2]) / 3;
  const line = (n: number, len: number, at: (k: number, j: number) => number) =>
    Array.from({ length: n }, (_, k) => {
      let d = 0, l = 0;
      for (let j = 0; j < len; j++) { const i = at(k, j); d += dis(i); l += lum(i); }
      return { d: d / len, l: l / len };
    });
  const rows = line(h, w, (y, x) => y * w + x), cols = line(w, h, (x, y) => y * w + x);
  const typical = (v: { d: number }[]) => v.map((q) => q.d).sort((a, b) => a - b)[Math.floor(v.length / 2)];
  // A line is still border if the channels disagree strongly or it is nearly black.
  const bad = (q: { d: number; l: number }, limit: number) => q.d > limit || q.l < 0.06;
  const edge = (v: { d: number; l: number }[], fromEnd: boolean) => {
    const limit = typical(v) * 2.2, max = Math.floor(v.length * 0.08);
    let trim = 0;
    for (let k = 0; k < max; k++) if (bad(v[fromEnd ? v.length - 1 - k : k], limit)) trim = k + 1;
    return trim;
  };
  const top = edge(rows, false), bottom = h - 1 - edge(rows, true);
  const left = edge(cols, false), right = w - 1 - edge(cols, true);
  return { left, top, w: right - left + 1, h: bottom - top + 1 };
}
