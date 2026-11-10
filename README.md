# Photo lab

Computational photography in the browser. Live at **https://rowestoli-photo-lab.netlify.app**.

I first built these in Python for CSCI 1290 (Computational Photography) at Brown. This version is rewritten in
TypeScript with no image libraries, so every result is computed live, in a Web Worker on the visitor's machine.

| Technique | What it does | Live? |
| --- | --- | --- |
| Color plates | Aligns the blue, green and red negatives of Prokudin-Gorskii glass plates with a coarse-to-fine normalized cross-correlation search | Yes |
| HDR | Aligns a handheld 6-shot phone bracket (median threshold bitmaps), recovers the response curve (Debevec & Malik), merges a radiance map and tone-maps it (Reinhard global or Durand local with a bilateral grid) | Yes |
| Blending | Poisson and mixed-gradient blending, solved with matrix-free conjugate gradients as you drag the object | Yes |
| Panoramas, optical flow | SIFT + RANSAC homography panoramas, Lucas-Kanade slow motion | Results from the course pipeline |

## Changes from the course version

- **Alignment scores edges, not brightness.** Raw-intensity NCC puts the red channel of plate 00153v about 40 px off,
  because the emir's robe is bright in one channel and dark in another. Gradient magnitude aligns all ten plates.
  The page lets you switch between the two to see it.
- **Scores ignore the plate border** and never wrap pixels around the frame (the original used `np.roll`).
- **HDR works on a handheld bracket.** The original assumed a tripod. Frames here moved up to 26 px, so they are
  aligned first with Ward's median threshold bitmaps, which do not care about exposure.
- **Sample pixels are spread across the tonal range** instead of drawn uniformly, so the response curve is
  constrained in shadows and highlights too.
- **The local tone mapper uses a bilateral grid**, which is fast enough to re-run on every slider move.

## Run it

```bash
npm install
npm run dev        # http://localhost:5173
npm test           # synthetic-data tests for every algorithm
npm run data -- /path/to/csci1290-repo   # rebuild public/data from the course files
```

## Code

- `src/lib/align.ts`: plate splitting, NCC, image pyramid search, border cropping
- `src/lib/hdr.ts`: MTB alignment, response curve solve (Cholesky on the normal equations), merge, tone mapping
- `src/lib/poisson.ts`: Poisson and mixed-gradient blending with conjugate gradients
- `src/worker.ts`: runs all of the above off the main thread
- `tests/lib.test.ts`: recovers known shifts, a known gamma curve and known radiance ratios; checks blending invariants

## Credits

Glass plates: Prokudin-Gorskii Collection, Library of Congress (public domain). Blending and panorama images are
from the CSCI 1290 course materials. The HDR bracket is my own.
