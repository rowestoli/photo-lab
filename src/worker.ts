/// <reference lib="webworker" />
import { alignPlate, type Metric } from './lib/align';
import { buildHdr, defaultTone, dynamicRangeStops, radianceView, toneMap, type ToneParams } from './lib/hdr';
import { crop, fromRGBA, toRGBA, type Img } from './lib/image';
import { blend, type BlendMode } from './lib/poisson';

export type Request =
  | { kind: 'plate'; rgba: Uint8ClampedArray; w: number; h: number; metric: Metric }
  | { kind: 'hdr'; frames: Uint8ClampedArray[]; w: number; h: number; seconds: number[] }
  | { kind: 'tone'; params: ToneParams }
  | { kind: 'blend-scene'; src: Uint8ClampedArray; sw: number; sh: number; mask: Uint8ClampedArray; dst: Uint8ClampedArray; dw: number; dh: number }
  | { kind: 'blend'; x: number; y: number; mode: BlendMode };

let radiance: Img | null = null;
let scene: { src: Img; mask: Uint8Array; dst: Img } | null = null;

const ctx = self as unknown as DedicatedWorkerGlobalScope;

ctx.onmessage = (e: MessageEvent<{ id: number; req: Request }>) => {
  const { id, req } = e.data;
  const reply = (body: object, transfer: Transferable[] = []) => ctx.postMessage({ id, ...body }, transfer);
  try {
    if (req.kind === 'plate') {
      const r = alignPlate(fromRGBA(req.rgba, req.w, req.h, 1), req.metric);
      const naive = toRGBA(r.naive), aligned = toRGBA(r.aligned);
      reply({ naive, aligned, w: r.aligned.w, h: r.aligned.h, green: r.green, red: r.red, ms: r.ms }, [naive.buffer, aligned.buffer]);
    } else if (req.kind === 'hdr') {
      const frames = req.frames.map((f) => fromRGBA(f, req.w, req.h, 3));
      const r = buildHdr(frames, req.seconds);
      radiance = r.radiance;
      const t0 = performance.now();
      const view = radianceView(radiance);
      const tone = toRGBA(toneMap(radiance, defaultTone));
      // The same crop of each input frame, for side-by-side comparison.
      const y0 = Math.max(0, ...r.shifts.map((s) => s.dy)), x0 = Math.max(0, ...r.shifts.map((s) => s.dx));
      const aligned = frames.map((f, i) => {
        const s = r.shifts[i];
        return toRGBA(crop(f, x0 - s.dx, y0 - s.dy, radiance!.w, radiance!.h));
      });
      const radView = toRGBA(view.img);
      reply({
        w: radiance.w, h: radiance.h, shifts: r.shifts, curves: r.curves.map((c) => Array.from(c)),
        samples: r.samples, ms: r.ms + performance.now() - t0, stops: dynamicRangeStops(radiance),
        lo: view.lo, hi: view.hi, radView, tone, aligned,
      }, [radView.buffer, tone.buffer, ...aligned.map((a) => a.buffer)]);
    } else if (req.kind === 'tone') {
      if (!radiance) throw new Error('No HDR image yet');
      const t0 = performance.now();
      const tone = toRGBA(toneMap(radiance, req.params));
      reply({ tone, ms: performance.now() - t0 }, [tone.buffer]);
    } else if (req.kind === 'blend-scene') {
      const src = fromRGBA(req.src, req.sw, req.sh, 3), dst = fromRGBA(req.dst, req.dw, req.dh, 3);
      const mask = new Uint8Array(req.sw * req.sh);
      for (let i = 0; i < mask.length; i++) mask[i] = req.mask[i * 4] > 127 ? 1 : 0;
      scene = { src, mask, dst };
      reply({ ok: true });
    } else if (req.kind === 'blend') {
      if (!scene) throw new Error('No scene loaded');
      const r = blend(scene.src, scene.mask, scene.dst, req.x, req.y, req.mode);
      const out = toRGBA(r.out);
      reply({ out, unknowns: r.unknowns, iterations: r.iterations, ms: r.ms }, [out.buffer]);
    }
  } catch (err) {
    reply({ error: String(err) });
  }
};
