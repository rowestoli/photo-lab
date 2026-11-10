"""Copy and downsize the course data into public/data for the web app.

Usage: python3 scripts/prepare_data.py /path/to/csci1290-repo
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SRC = Path(sys.argv[1])
OUT = Path(__file__).resolve().parent.parent / 'public' / 'data'
if OUT.exists():
    shutil.rmtree(OUT)

# 1. Prokudin-Gorskii glass plates (Library of Congress, public domain)
plates_dir = OUT / 'plates'
plates_dir.mkdir(parents=True)
plates = []
for f in sorted((SRC / 'proj1_rgbalign' / 'data').glob('*.jpg')):
    shutil.copy(f, plates_dir / f.name)
    plates.append(f.stem)

# 2. Exposure bracket shot for lab 1 (handheld phone, so frames need aligning)
hdr_dir = OUT / 'hdr'
hdr_dir.mkdir()
bracket = []
for denom in [25, 60, 250, 1600, 5000, 10000]:
    im = Image.open(SRC / 'lab1_exposure' / 'results' / f'1:{denom}.jpg').convert('RGB')
    im = im.resize((1024, 768), Image.LANCZOS)
    name = f'1-{denom}.jpg'
    im.save(hdr_dir / name, quality=90)
    bracket.append({'file': name, 'seconds': 1 / denom, 'label': f'1/{denom} s'})

# 3. Poisson blending pairs (course test images)
blend_dir = OUT / 'blend'
blend_dir.mkdir()
pairs = {'01': ('bear', 'pool'), '04': ('octopus', 'carrier'), '05': ('jet', 'valley'), '06': ('rainbow', 'island')}
offsets = np.loadtxt(SRC / 'proj3_poissonblending' / 'data' / 'offsets.csv', delimiter=',', skiprows=1)
objects, scenes = [], []
for idx, (obj, scene) in pairs.items():
    d = SRC / 'proj3_poissonblending' / 'data'
    Image.open(d / f'source_{idx}.jpg').convert('RGB').save(blend_dir / f'{obj}.jpg', quality=92)
    mask = np.asarray(Image.open(d / f'mask_{idx}.jpg').convert('L')) > 127
    Image.fromarray((mask * 255).astype(np.uint8)).save(blend_dir / f'{obj}-mask.png', optimize=True)
    Image.open(d / f'target_{idx}.jpg').convert('RGB').save(blend_dir / f'{scene}.jpg', quality=92)
    x, y = offsets[int(idx) - 1]
    objects.append({'id': obj, 'home': scene, 'x': int(x), 'y': int(y)})
    scenes.append(scene)

# 4. Gallery: panoramas and optical-flow results
gallery = OUT / 'gallery'
gallery.mkdir()
pano = SRC / 'proj4_panorama' / 'results'
Image.open(pano / 'hero_image.png').convert('RGB').save(gallery / 'yosemite.jpg', quality=88)
im = Image.open(pano / 'panorama_5_cylindrical.png').convert('RGB')
im.save(gallery / 'yosemite-cylindrical.jpg', quality=88)
flow = SRC / 'proj5_opticalflow' / 'results'
for name in ['ducky_no_damp', 'ducky_damp']:
    Image.open(flow / f'{name}.png').convert('RGB').save(gallery / f'{name.replace("_", "-")}.jpg', quality=88)
subprocess.run([
    'ffmpeg', '-v', 'error', '-y', '-ss', '0', '-t', '14', '-i', str(flow / 'JellyFish.mp4'),
    '-vf', 'scale=960:-2', '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '28',
    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(gallery / 'jellyfish.mp4'),
], check=True)
subprocess.run([
    'ffmpeg', '-v', 'error', '-y', '-ss', '2', '-i', str(flow / 'JellyFish.mp4'), '-frames:v', '1',
    '-vf', 'scale=960:-2', '-q:v', '4', str(gallery / 'jellyfish-poster.jpg'),
], check=True)

(OUT / 'manifest.json').write_text(json.dumps({
    'plates': plates, 'bracket': bracket, 'objects': objects, 'scenes': scenes,
}, indent=1))
print('wrote', sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file()) // 1024, 'KB')
