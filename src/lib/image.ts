/** Float image, values 0..1, channels interleaved. */
export interface Img {
  w: number;
  h: number;
  c: number;
  data: Float32Array;
}

export function makeImg(w: number, h: number, c: number): Img {
  return { w, h, c, data: new Float32Array(w * h * c) };
}

/** RGBA bytes (canvas ImageData) to a float image with `c` channels (1 = luminance-free gray from R). */
export function fromRGBA(rgba: Uint8ClampedArray, w: number, h: number, c: 1 | 3): Img {
  const img = makeImg(w, h, c);
  for (let i = 0, n = w * h; i < n; i++) {
    for (let k = 0; k < c; k++) img.data[i * c + k] = rgba[i * 4 + k] / 255;
  }
  return img;
}

/** Float image to RGBA bytes. Gray images are replicated to RGB. */
export function toRGBA(img: Img): Uint8ClampedArray<ArrayBuffer> {
  const out = new Uint8ClampedArray(img.w * img.h * 4);
  for (let i = 0, n = img.w * img.h; i < n; i++) {
    for (let k = 0; k < 3; k++) out[i * 4 + k] = img.data[i * img.c + (img.c === 1 ? 0 : k)] * 255 + 0.5;
    out[i * 4 + 3] = 255;
  }
  return out;
}

export function crop(img: Img, x0: number, y0: number, w: number, h: number): Img {
  const out = makeImg(w, h, img.c);
  for (let y = 0; y < h; y++) {
    const src = ((y0 + y) * img.w + x0) * img.c;
    out.data.set(img.data.subarray(src, src + w * img.c), y * w * img.c);
  }
  return out;
}

/** Translate without wrapping; uncovered pixels become 0. Positive dx moves content right. */
export function shift(img: Img, dy: number, dx: number): Img {
  const out = makeImg(img.w, img.h, img.c);
  for (let y = 0; y < img.h; y++) {
    const sy = y - dy;
    if (sy < 0 || sy >= img.h) continue;
    for (let x = 0; x < img.w; x++) {
      const sx = x - dx;
      if (sx < 0 || sx >= img.w) continue;
      for (let k = 0; k < img.c; k++) out.data[(y * img.w + x) * img.c + k] = img.data[(sy * img.w + sx) * img.c + k];
    }
  }
  return out;
}

/** Separable [1 2 1]/4 blur then 2x decimation (one pyramid level). Single channel. */
export function downsample(img: Img): Img {
  const { w, h } = img;
  const tmp = new Float32Array(w * h);
  const d = img.data;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const l = d[y * w + Math.max(0, x - 1)], r = d[y * w + Math.min(w - 1, x + 1)];
      tmp[y * w + x] = (l + 2 * d[y * w + x] + r) / 4;
    }
  }
  const ow = Math.max(1, w >> 1), oh = Math.max(1, h >> 1);
  const out = makeImg(ow, oh, 1);
  for (let y = 0; y < oh; y++) {
    const sy = y * 2;
    const u = Math.max(0, sy - 1), dn = Math.min(h - 1, sy + 1);
    for (let x = 0; x < ow; x++) {
      const sx = x * 2;
      out.data[y * ow + x] = (tmp[u * w + sx] + 2 * tmp[sy * w + sx] + tmp[dn * w + sx]) / 4;
    }
  }
  return out;
}

/** Sobel gradient magnitude of a single-channel image. */
export function gradientMagnitude(img: Img): Img {
  const { w, h, data: d } = img;
  const out = makeImg(w, h, 1);
  const at = (x: number, y: number) => d[Math.min(h - 1, Math.max(0, y)) * w + Math.min(w - 1, Math.max(0, x))];
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const gx = at(x + 1, y - 1) + 2 * at(x + 1, y) + at(x + 1, y + 1) - at(x - 1, y - 1) - 2 * at(x - 1, y) - at(x - 1, y + 1);
      const gy = at(x - 1, y + 1) + 2 * at(x, y + 1) + at(x + 1, y + 1) - at(x - 1, y - 1) - 2 * at(x, y - 1) - at(x + 1, y - 1);
      out.data[y * w + x] = Math.hypot(gx, gy);
    }
  }
  return out;
}

/** Small seeded PRNG so sampling is reproducible. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
