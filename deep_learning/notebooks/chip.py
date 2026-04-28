"""
FFSL Building Footprint - WMTS Chipper
Standalone script for SLURM batch execution.

Generates COCO-format training chips from the UGRC Discover WMTS service
using reviewed building polygon labels.
All labels are used directly.

Usage:
    sbatch chip.slurm

Author: Magnus Tveit
Project: Utah HRWUI Building Footprint Extraction
"""

import os
import json
import random
import subprocess
import warnings
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings('ignore')

import geopandas as gpd
from shapely.geometry import box, Point
from shapely.ops import unary_union
import requests
from PIL import Image
import numpy as np
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE       = '/uufs/chpc.utah.edu/common/home/u0972368/FFSL_HRWUI'
LABELS_SHP = os.path.join(BASE, 'OLD_STUFF/Data/Training_Labels_WGS84/Training_Labels_WGS84.shp')
OUT_DIR    = os.path.join(BASE, 'OLD_STUFF/Data/Chips')
IMG_DIR    = os.path.join(OUT_DIR, 'images')

CHIP_PX    = 512
RESOLUTION = 0.149291       # Hexagon 15cm in meters/pixel (Web Mercator)
CHIP_M     = CHIP_PX * RESOLUTION   # ~76.5 meters per chip side
NEG_RATIO  = 1.0            # negative chips per positive chip
N_WORKERS  = 8              # parallel WMTS request workers
RANDOM_SEED = 42

WMTS_URL   = 'https://discover.agrc.utah.gov/login/path/ENTER_YOUR_UGRC_WMTS_KEY_HERE/wmts/1.0.0/WMTSCapabilities.xml'
WMTS_LAYER = 'utah'
# =============================================================================

os.makedirs(IMG_DIR, exist_ok=True)
random.seed(RANDOM_SEED)
half = CHIP_M / 2

print('Starting FFSL Chipper')
print(f'Output:    {OUT_DIR}')
print(f'Workers:   {N_WORKERS}')
print(f'Chip size: {CHIP_PX}px = {CHIP_M:.1f}m')

# Load and reproject labels
TARGET_CRS = 'EPSG:3857'
labels = gpd.read_file(LABELS_SHP).to_crs(TARGET_CRS)
print(f'Labels loaded: {len(labels)}')

# Use all labels directly — no boundary clipping needed for chipping
labels_clipped = labels.copy()

# Lightweight convex hull around all labels — used only for placing negative chips
clip_boundary = labels_clipped.union_all().convex_hull
print(f'Labels to chip: {len(labels_clipped)}')
print(f'Convex hull area: {clip_boundary.area / 1e6:.2f} km2')


def fetch_wmts_chip(bbox_3857, chip_px=512):
    """Fetch a chip from WMTS via WMS GetMap. Returns PIL Image or None."""
    wms_url = WMTS_URL.replace('/wmts/1.0.0/WMTSCapabilities.xml', '/wms')
    minx, miny, maxx, maxy = bbox_3857
    params = {
        'SERVICE': 'WMS', 'VERSION': '1.1.1', 'REQUEST': 'GetMap',
        'LAYERS': WMTS_LAYER, 'STYLES': '', 'SRS': 'EPSG:3857',
        'BBOX': f'{minx},{miny},{maxx},{maxy}',
        'WIDTH': chip_px, 'HEIGHT': chip_px,
        'FORMAT': 'image/png', 'TRANSPARENT': 'FALSE'
    }
    try:
        r = requests.get(wms_url, params=params, timeout=30)
        if r.status_code == 200:
            return Image.open(io.BytesIO(r.content)).convert('RGB')
        return None
    except Exception:
        return None


def polygon_to_coco(geom, chip_bbox):
    """Convert shapely polygon to COCO segmentation + pixel bbox."""
    minx, miny, maxx, maxy = chip_bbox
    chip_w = maxx - minx
    chip_h = maxy - miny

    if geom.geom_type == 'Polygon':
        coords = list(geom.exterior.coords)
    elif geom.geom_type == 'MultiPolygon':
        coords = list(max(geom.geoms, key=lambda g: g.area).exterior.coords)
    else:
        return None, None

    px = []
    for x, y in coords:
        px.extend([
            round((x - minx) / chip_w * CHIP_PX, 2),
            round((maxy - y) / chip_h * CHIP_PX, 2)
        ])

    xs, ys = px[0::2], px[1::2]
    bbox_px = [
        round(min(xs), 2), round(min(ys), 2),
        round(max(xs) - min(xs), 2),
        round(max(ys) - min(ys), 2)
    ]
    return [px], bbox_px


# COCO structure
coco = {
    'info': {
        'description': 'FFSL Utah HRWUI Building Footprints',
        'version': '1.0',
        'contributor': 'Magnus Tveit'
    },
    'categories': [{'id': 1, 'name': 'building', 'supercategory': 'structure'}],
    'images': [],
    'annotations': []
}

image_id      = 0
annotation_id = 0
skipped       = 0


def process_building(args):
    """Fetch one chip centered on a building and return COCO entries."""
    idx, row, image_id_local = args
    geom = row.geometry
    if geom is None or geom.is_empty:
        return None

    cx, cy    = geom.centroid.x, geom.centroid.y
    chip_bbox = (cx - half, cy - half, cx + half, cy + half)
    chip_box  = box(*chip_bbox)

    img = fetch_wmts_chip(chip_bbox)
    if img is None:
        return None

    fname = f'{image_id_local:09d}.png'
    img.save(os.path.join(IMG_DIR, fname))

    img_info = {
        'id': image_id_local,
        'file_name': fname,
        'width': CHIP_PX,
        'height': CHIP_PX
    }

    # Find all buildings that overlap this chip
    anns = []
    overlapping = labels_clipped[labels_clipped.geometry.intersects(chip_box)]
    for _, bldg in overlapping.iterrows():
        clipped_geom = bldg.geometry.intersection(chip_box)
        if clipped_geom.is_empty:
            continue
        # Only include buildings that are mostly within the chip (>50% area)
        if clipped_geom.area / bldg.geometry.area < 0.5:
            continue
        seg, bbox_px = polygon_to_coco(clipped_geom, chip_bbox)
        if seg is None:
            continue
        anns.append({
            'image_id': image_id_local,
            'category_id': 1,
            'segmentation': seg,
            'bbox': bbox_px,
            'area': round(clipped_geom.area, 2),
            'iscrowd': 0
        })

    return img_info, anns


# ── Positive chips ─────────────────────────────────────────────────────────
tasks = [
    (idx, row, i + 1)
    for i, (idx, row) in enumerate(labels_clipped.iterrows())
]

print(f'\nGenerating positive chips for {len(tasks)} buildings...')

with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
    futures = {executor.submit(process_building, t): t for t in tasks}
    for future in tqdm(as_completed(futures), total=len(tasks), desc='Positive'):
        result = future.result()
        if result is None:
            skipped += 1
            continue
        img_info, anns = result
        image_id += 1
        img_info['id'] = image_id
        for ann in anns:
            ann['image_id'] = image_id
            annotation_id += 1
            ann['id'] = annotation_id
        coco['images'].append(img_info)
        coco['annotations'].extend(anns)

n_positive = image_id
print(f'Positive: {n_positive} chips, {annotation_id} annotations, {skipped} skipped')


# ── Negative chips ─────────────────────────────────────────────────────────
n_negatives  = int(n_positive * NEG_RATIO)
neg_count    = 0
labels_union = unary_union(labels_clipped.geometry.buffer(CHIP_M))
bx_min, by_min, bx_max, by_max = clip_boundary.bounds

print(f'\nGenerating {n_negatives} negative chips...')


def fetch_negative(image_id_local):
    """Find a random empty location and fetch a chip."""
    for _ in range(50):
        rx = random.uniform(bx_min, bx_max)
        ry = random.uniform(by_min, by_max)
        pt = Point(rx, ry)
        if not clip_boundary.contains(pt):
            continue
        if labels_union.contains(pt):
            continue
        chip_bbox = (rx - half, ry - half, rx + half, ry + half)
        img = fetch_wmts_chip(chip_bbox)
        if img is None:
            continue
        fname = f'{image_id_local:09d}.png'
        img.save(os.path.join(IMG_DIR, fname))
        return {
            'id': image_id_local,
            'file_name': fname,
            'width': CHIP_PX,
            'height': CHIP_PX
        }
    return None


neg_tasks = list(range(image_id + 1, image_id + 1 + n_negatives))

with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
    futures = {executor.submit(fetch_negative, iid): iid for iid in neg_tasks}
    for future in tqdm(as_completed(futures), total=len(neg_tasks), desc='Negative'):
        result = future.result()
        if result:
            image_id += 1
            neg_count += 1
            coco['images'].append(result)

# ── Save annotations ───────────────────────────────────────────────────────
ann_path = os.path.join(OUT_DIR, 'annotations.json')
with open(ann_path, 'w') as f:
    json.dump(coco, f)

print(f'\n=== Chipping Complete ===')
print(f'Total images:      {len(coco["images"])}')
print(f'Total annotations: {len(coco["annotations"])}')
print(f'Positive chips:    {n_positive}')
print(f'Negative chips:    {neg_count}')
print(f'Skipped:           {skipped}')
print(f'Annotations file:  {ann_path}')

result = subprocess.run(['du', '-sh', OUT_DIR], capture_output=True, text=True)
print(f'Folder size:       {result.stdout.split()[0]}')
