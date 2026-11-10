import './style.css';
import type { Request } from './worker';
import type { BlendMode } from './lib/poisson';
import type { ToneParams } from './lib/hdr';
import type { AlignResult, Metric } from './lib/align';

interface Manifest {
  plates: string[];
  bracket: { file: string; seconds: number; label: string }[];
  objects: { id: string; home: string; x: number; y: number }[];
  scenes: string[];
}

const $ = <T extends Element = HTMLElement>(sel: string, root: ParentNode = document) => root.querySelector(sel) as T;
const fmt = (n: number, d = 0) => n.toLocaleString('en-US', { maximumFractionDigits: d, minimumFractionDigits: d });

/* ---------- worker RPC: one request in flight per channel, newest wins ---------- */

const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
let nextId = 0;
const pending = new Map<number, (v: any) => void>();
worker.onmessage = (e) => {
  const cb = pending.get(e.data.id);
  pending.delete(e.data.id);
  if (e.data.error) console.error(e.data.error);
  cb?.(e.data);
};
function call<T = any>(req: Request, transfer: Transferable[] = []): Promise<T> {
  const id = nextId++;
  return new Promise((res) => { pending.set(id, res); worker.postMessage({ id, req }, transfer); });
}

/** Coalesces rapid requests (slider drags): runs one at a time, skipping to the latest. */
function latest<A>(run: (a: A) => Promise<void>) {
  let busy = false, queued: { a: A } | null = null;
  const go = async (a: A) => {
    busy = true;
    await run(a);
    busy = false;
    if (queued) { const q = queued; queued = null; go(q.a); }
  };
  return (a: A) => { if (busy) queued = { a }; else go(a); };
}

/* ---------- image helpers ---------- */

const imageCache = new Map<string, Promise<HTMLImageElement>>();
function loadImage(src: string): Promise<HTMLImageElement> {
  if (!imageCache.has(src)) {
    imageCache.set(src, new Promise((res, rej) => {
      const im = new Image();
      im.onload = () => res(im);
      im.onerror = rej;
      im.src = src;
    }));
  }
  return imageCache.get(src)!;
}

async function pixels(src: string): Promise<{ data: Uint8ClampedArray; w: number; h: number }> {
  const im = await loadImage(src);
  const c = document.createElement('canvas');
  c.width = im.naturalWidth; c.height = im.naturalHeight;
  const g = c.getContext('2d', { willReadFrequently: true })!;
  g.drawImage(im, 0, 0);
  return { data: g.getImageData(0, 0, c.width, c.height).data, w: c.width, h: c.height };
}

function paint(canvas: HTMLCanvasElement, rgba: Uint8ClampedArray, w: number, h: number) {
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  canvas.getContext('2d')!.putImageData(new ImageData(new Uint8ClampedArray(rgba), w, h), 0, 0);
}

function setStats(dl: HTMLElement, rows: [string, string][]) {
  dl.innerHTML = rows.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join('');
}

/** Segmented control: buttons with data-v and aria-pressed. */
function seg(el: HTMLElement, onChange: (v: string) => void) {
  el.addEventListener('click', (e) => {
    const b = (e.target as HTMLElement).closest('button');
    if (!b || b.getAttribute('aria-pressed') === 'true') return;
    el.querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
    onChange(b.dataset.v!);
  });
}

function compare(fig: HTMLElement) {
  const range = $<HTMLInputElement>('.compare__range', fig);
  range.addEventListener('input', () => fig.style.setProperty('--pos', `${range.value}%`));
  return { a: $<HTMLCanvasElement>('.compare__a', fig), b: $<HTMLCanvasElement>('.compare__b', fig) };
}

/* ---------- tabs ---------- */

const tabs = [...document.querySelectorAll<HTMLButtonElement>('[role=tab]')];
const tabBar = $('.tabs__bar');
const panelInit = new Map<string, () => void>();
function placeBar(tab: HTMLElement) {
  tabBar.style.setProperty('--x', `${tab.offsetLeft}px`);
  tabBar.style.setProperty('--w', `${tab.offsetWidth}px`);
}
function select(tab: HTMLButtonElement, push = true) {
  for (const t of tabs) {
    const on = t === tab;
    t.setAttribute('aria-selected', String(on));
    t.tabIndex = on ? 0 : -1;
    document.getElementById(t.getAttribute('aria-controls')!)!.hidden = !on;
  }
  placeBar(tab);
  const id = tab.getAttribute('aria-controls')!;
  panelInit.get(id)?.();
  panelInit.delete(id);
  if (push) history.replaceState(null, '', `#${id}`);
}
tabs.forEach((t, i) => {
  t.addEventListener('click', () => select(t));
  t.addEventListener('keydown', (e) => {
    const d = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0;
    if (!d) return;
    const n = tabs[(i + d + tabs.length) % tabs.length];
    n.focus(); select(n);
  });
});
addEventListener('resize', () => placeBar(tabs.find((t) => t.getAttribute('aria-selected') === 'true')!));

/* ---------- color plates ---------- */

function describeShift(r: AlignResult) {
  const v = r.dy === 0 ? '' : `${Math.abs(r.dy)} ${r.dy > 0 ? 'down' : 'up'}`;
  const h = r.dx === 0 ? '' : `${Math.abs(r.dx)} ${r.dx > 0 ? 'right' : 'left'}`;
  return [v, h].filter(Boolean).join(', ') || 'none';
}

function initPlates(m: Manifest) {
  const pick = $('#plate-pick');
  const fig = $('#plate-compare');
  const { a, b } = compare(fig);
  const busy = $('.busy', fig);
  let plate = m.plates[0];
  let metric: Metric = 'edges';

  pick.innerHTML = m.plates.map((p, i) =>
    `<button role="option" aria-selected="${i === 0}" data-p="${p}" aria-label="Plate ${p}"><img src="/data/plates/${p}.jpg" alt="" loading="lazy" /></button>`).join('');

  const run = latest(async () => {
    busy.hidden = false;
    const px = await pixels(`/data/plates/${plate}.jpg`);
    const r = await call<any>({ kind: 'plate', rgba: px.data, w: px.w, h: px.h, metric });
    busy.hidden = true;
    paint(a, r.naive, r.w, r.h);
    paint(b, r.aligned, r.w, r.h);
    fig.style.setProperty('--ar', String(r.w / r.h));
    const tried = r.green.trace.reduce((s: number, t: any) => s + t.tried, 0) + r.red.trace.reduce((s: number, t: any) => s + t.tried, 0);
    setStats($('#plate-stats'), [
      ['Green moved', describeShift(r.green)],
      ['Red moved', describeShift(r.red)],
      ['Offsets scored', fmt(tried)],
      ['Time', `${fmt(r.ms)} ms`],
    ]);
    $('#plate-trace').innerHTML = `<thead><tr><th>Width</th><th>Green (x, y)</th><th>Red (x, y)</th></tr></thead><tbody>${r.green.trace.map((t: any, i: number) => {
      const rt = r.red.trace[i];
      return `<tr><td>${t.width} px</td><td>${t.dx}, ${t.dy}</td><td>${rt.dx}, ${rt.dy}</td></tr>`;
    }).join('')}</tbody>`;
    const fail = metric === 'brightness' && plate === '00153v';
    $('#plate-hint').textContent = fail
      ? 'Brightness matching fails on this one: the emir\'s robe is bright through the blue filter and dark through the red, so the raw channels disagree. Switch to edges.'
      : metric === 'edges'
        ? 'Compares gradient magnitude, so it works even when a color is bright in one channel and dark in another. Try brightness on the emir (third plate).'
        : 'Compares raw pixel values. Fast, and fine for most plates.';
  });

  pick.addEventListener('click', (e) => {
    const btn = (e.target as HTMLElement).closest('button');
    if (!btn) return;
    plate = btn.dataset.p!;
    pick.querySelectorAll('button').forEach((x) => x.setAttribute('aria-selected', String(x === btn)));
    run(undefined);
  });
  seg($('#plate-metric'), (v) => { metric = v as Metric; run(undefined); });
  run(undefined);
}

/* ---------- HDR ---------- */

const SLIDERS: { key: keyof ToneParams; label: string; min: number; max: number; step: number; unit: string; local?: boolean }[] = [
  { key: 'exposure', label: 'Exposure', min: -3, max: 3, step: 0.1, unit: ' stops' },
  { key: 'saturation', label: 'Saturation', min: 0, max: 1.5, step: 0.05, unit: '' },
  { key: 'contrast', label: 'Output contrast', min: 2, max: 9, step: 0.25, unit: ' stops', local: true },
  { key: 'detail', label: 'Detail', min: 0.5, max: 2.5, step: 0.05, unit: '×', local: true },
];

function drawCurves(svg: SVGSVGElement, curves: number[][]) {
  const W = 360, H = 220, L = 44, B = 26, T = 8, R = 8;
  const all = curves.flat();
  const lo = Math.min(...all), hi = Math.max(...all);
  const x = (g: number) => L + ((g - lo) / (hi - lo)) * (W - L - R);
  const y = (z: number) => H - B - (z / 255) * (H - B - T);
  const colors = ['#e2735f', '#8fcba2', '#7aa7e0'];
  const ticks = [0, 128, 255].map((z) => `<text x="${L - 6}" y="${y(z) + 4}" text-anchor="end">${z}</text><line x1="${L}" x2="${W - R}" y1="${y(z)}" y2="${y(z)}" class="grid"/>`).join('');
  const paths = curves.map((g, c) =>
    `<path d="${g.map((v, z) => `${z ? 'L' : 'M'}${x(v).toFixed(1)},${y(z).toFixed(1)}`).join('')}" stroke="${colors[c]}" fill="none" stroke-width="2" style="--d:${c * 120}ms"/>`).join('');
  svg.innerHTML = `${ticks}${paths}
    <text x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle">Light reaching the sensor (log scale) →</text>
    <text x="8" y="${(H - B) / 2}" text-anchor="middle" transform="rotate(-90 8 ${(H - B) / 2})">Pixel value</text>`;
}

function initHdr(m: Manifest) {
  const fig = $('#hdr-compare');
  const { a, b } = compare(fig);
  const busy = $('.busy', fig);
  const strip = $('#bracket');
  const params: ToneParams = { operator: 'local', exposure: 0, saturation: 0.75, detail: 1.25, contrast: 5 };
  let view = 'compare';
  let single = 2; // which exposure is shown on the comparison side
  let result: any = null;
  let tone: Uint8ClampedArray | null = null;

  strip.innerHTML = m.bracket.map((f, i) =>
    `<button aria-pressed="${i === single}" data-i="${i}"><img src="/data/hdr/${f.file}" alt="Exposure at ${f.label}" /><span>${f.label}</span><small></small></button>`).join('');

  const show = () => {
    if (!result) return;
    const { w, h } = result;
    fig.style.setProperty('--ar', String(w / h));
    fig.classList.toggle('compare--solo', view !== 'compare');
    $('#hdr-legend').hidden = view !== 'radiance';
    if (view === 'radiance') paint(b, result.radView, w, h);
    else if (tone) paint(b, tone, w, h);
    paint(a, result.aligned[single], w, h);
    $('#hdr-tag-a').textContent = `Single exposure, ${m.bracket[single].label}`;
  };

  const retone = latest(async () => {
    const r = await call<any>({ kind: 'tone', params: { ...params } });
    tone = r.tone;
    show();
    $('#hdr-stats').querySelector<HTMLElement>('[data-k=tone]')!.textContent = `${fmt(r.ms)} ms`;
  });

  // Sliders
  $('#hdr-sliders').innerHTML = SLIDERS.map((s) => `
    <label class="slider" data-local="${!!s.local}">
      <span>${s.label} <output>${params[s.key]}${s.unit}</output></span>
      <input type="range" min="${s.min}" max="${s.max}" step="${s.step}" value="${params[s.key]}" data-k="${s.key}" />
    </label>`).join('');
  $('#hdr-sliders').addEventListener('input', (e) => {
    const inp = e.target as HTMLInputElement;
    const s = SLIDERS.find((x) => x.key === inp.dataset.k)!;
    (params as any)[s.key] = Number(inp.value);
    inp.previousElementSibling!.querySelector('output')!.textContent = `${inp.value}${s.unit}`;
    if (view === 'radiance') { view = 'tone'; $('#hdr-view').querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', String(x.dataset.v === 'tone'))); }
    retone(undefined);
  });

  const opHint = () => {
    $('#hdr-op-hint').textContent = params.operator === 'local'
      ? 'Durand and Dorsey: an edge-preserving filter splits the image into a base layer and fine detail. Only the base is compressed, so texture survives.'
      : 'Reinhard: one curve for every pixel. Simple and natural, but detail flattens in the brightest and darkest areas.';
    document.querySelectorAll<HTMLElement>('#hdr-sliders [data-local=true]').forEach((el) => { el.hidden = params.operator !== 'local'; });
  };
  opHint();
  seg($('#hdr-op'), (v) => { params.operator = v as ToneParams['operator']; opHint(); retone(undefined); });
  seg($('#hdr-view'), (v) => { view = v; show(); });
  strip.addEventListener('click', (e) => {
    const btn = (e.target as HTMLElement).closest('button');
    if (!btn) return;
    single = Number(btn.dataset.i);
    strip.querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', String(x === btn)));
    if (view !== 'compare') { view = 'compare'; $('#hdr-view').querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', String(x.dataset.v === 'compare'))); }
    show();
  });

  (async () => {
    const frames = await Promise.all(m.bracket.map((f) => pixels(`/data/hdr/${f.file}`)));
    const r = await call<any>({ kind: 'hdr', frames: frames.map((f) => f.data), w: frames[0].w, h: frames[0].h, seconds: m.bracket.map((f) => f.seconds) });
    result = r;
    tone = r.tone;
    busy.hidden = true;
    strip.querySelectorAll('small').forEach((s, i) => {
      const d = Math.hypot(r.shifts[i].dx, r.shifts[i].dy);
      s.textContent = d === 0 ? 'reference' : `moved ${fmt(d)} px`;
    });
    setStats($('#hdr-stats'), [
      ['Range captured', `${fmt(r.stops, 1)} stops`],
      ['Sample pixels', String(r.samples)],
      ['Merge', `${fmt(r.ms)} ms`],
      ['Tone map', ''],
    ]);
    $('#hdr-stats').lastElementChild!.querySelector('dd')!.dataset.k = 'tone';
    drawCurves($('#hdr-curve') as unknown as SVGSVGElement, r.curves);
    show();
    retone(undefined);
  })();
}

/* ---------- blending ---------- */

const BLEND_HINTS: Record<BlendMode, string> = {
  paste: 'Copies the pixels as they are. The edge of the cutout and the mismatched lighting give it away.',
  poisson: 'Keeps the object\'s gradients and takes its edge colors from the background, so it picks up the scene\'s light.',
  mixed: 'At each pixel keeps whichever gradient is stronger, object or background, so background texture shows through thin or see-through parts.',
};

function initBlend(m: Manifest) {
  const canvas = $<HTMLCanvasElement>('#blend-canvas');
  const objEl = $('#blend-object'), sceneEl = $('#blend-scene');
  let obj = m.objects[2], scene = obj.home;
  let mode: BlendMode = 'poisson';
  let pos = { x: obj.x, y: obj.y };
  let size = { sw: 0, sh: 0, dw: 0, dh: 0 };

  const chips = (el: HTMLElement, items: string[], active: string) => {
    el.innerHTML = items.map((v) => `<button aria-pressed="${v === active}" data-v="${v}">${v[0].toUpperCase()}${v.slice(1)}</button>`).join('');
  };
  chips(objEl, m.objects.map((o) => o.id), obj.id);
  chips(sceneEl, m.scenes, scene);

  const solve = latest(async () => {
    const r = await call<any>({ kind: 'blend', x: Math.round(pos.x), y: Math.round(pos.y), mode });
    paint(canvas, r.out, size.dw, size.dh);
    setStats($('#blend-stats'), [
      ['Unknown pixels', fmt(r.unknowns)],
      ['Solver iterations', mode === 'paste' ? 'none' : fmt(r.iterations)],
      ['Solve time', `${fmt(r.ms)} ms`],
    ]);
  });

  const load = async (resetPos: boolean) => {
    const [src, mask, dst] = await Promise.all([
      pixels(`/data/blend/${obj.id}.jpg`), pixels(`/data/blend/${obj.id}-mask.png`), pixels(`/data/blend/${scene}.jpg`),
    ]);
    size = { sw: src.w, sh: src.h, dw: dst.w, dh: dst.h };
    if (resetPos) {
      pos = scene === obj.home ? { x: obj.x, y: obj.y } : { x: (dst.w - src.w) / 2, y: (dst.h - src.h) / 2 };
    }
    $('#blend-stage').style.setProperty('--ar', String(dst.w / dst.h));
    await call({ kind: 'blend-scene', src: src.data, sw: src.w, sh: src.h, mask: mask.data, dst: dst.data, dw: dst.w, dh: dst.h });
    solve(undefined);
  };

  objEl.addEventListener('click', (e) => {
    const b = (e.target as HTMLElement).closest('button'); if (!b) return;
    obj = m.objects.find((o) => o.id === b.dataset.v)!;
    scene = obj.home;
    chips(objEl, m.objects.map((o) => o.id), obj.id);
    chips(sceneEl, m.scenes, scene);
    load(true);
  });
  sceneEl.addEventListener('click', (e) => {
    const b = (e.target as HTMLElement).closest('button'); if (!b) return;
    scene = b.dataset.v!;
    chips(sceneEl, m.scenes, scene);
    load(true);
  });
  seg($('#blend-mode'), (v) => { mode = v as BlendMode; $('#blend-hint').textContent = BLEND_HINTS[mode]; solve(undefined); });
  $('#blend-hint').textContent = BLEND_HINTS[mode];

  // Drag to move; positions are in destination pixels.
  let drag: { px: number; py: number; x: number; y: number } | null = null;
  const scale = () => size.dw / canvas.getBoundingClientRect().width;
  canvas.addEventListener('pointerdown', (e) => {
    drag = { px: e.clientX, py: e.clientY, x: pos.x, y: pos.y };
    canvas.setPointerCapture(e.pointerId);
    canvas.classList.add('dragging');
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!drag) return;
    const s = scale();
    pos = {
      x: Math.max(-size.sw * 0.6, Math.min(size.dw - size.sw * 0.4, drag.x + (e.clientX - drag.px) * s)),
      y: Math.max(-size.sh * 0.6, Math.min(size.dh - size.sh * 0.4, drag.y + (e.clientY - drag.py) * s)),
    };
    solve(undefined);
  });
  const end = () => { drag = null; canvas.classList.remove('dragging'); };
  canvas.addEventListener('pointerup', end);
  canvas.addEventListener('pointercancel', end);
  canvas.addEventListener('keydown', (e) => {
    const step = e.shiftKey ? 10 : 2;
    const d = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] }[e.key];
    if (!d) return;
    e.preventDefault();
    pos = { x: pos.x + d[0], y: pos.y + d[1] };
    solve(undefined);
  });

  load(true);
}

/* ---------- gallery ---------- */

function initGallery() {
  seg($('#damp-toggle'), (v) => {
    const img = $<HTMLImageElement>('#damp-img');
    img.src = `/data/gallery/ducky-${v}.jpg`;
    img.alt = `Image pyramids and the final flow field for a test scene, ${v === 'damp' ? 'with' : 'without'} damping`;
  });
}

/* ---------- boot ---------- */

(async () => {
  const m: Manifest = await (await fetch('/data/manifest.json')).json();
  panelInit.set('plates', () => initPlates(m));
  panelInit.set('hdr', () => initHdr(m));
  panelInit.set('blend', () => initBlend(m));
  panelInit.set('gallery', initGallery);
  const start = tabs.find((t) => `#${t.getAttribute('aria-controls')}` === location.hash) ?? tabs[0];
  select(start, false);
})();
