"""
FFSL Building Footprint — Full Inference
Standalone script for SLURM batch execution.

Tiles the prediction area, fetches chips from WMTS, runs MaskRCNN inference,
converts masks to polygons, applies post-processing filters, and exports
a shapefile with confidence scores.

Usage:
    1. Set WMTS_URL below
    2. Set SCOPE ('Random', 'Medium', or 'Extent')
    3. sbatch infer.slurm

Author: Magnus Tveit
Project: Utah HRWUI Building Footprint Extraction
"""

import sys
print('Imports starting', flush=True)

import os
import io
import json
import zipfile
import warnings
warnings.filterwarnings('ignore')

print('Loading torch...', flush=True)
import torch
print('Torch loaded', flush=True)

import torchvision.transforms.functional as TF
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import box, Polygon
from shapely.prepared import prep
from skimage import measure
import requests
from PIL import Image
from tqdm import tqdm

print('All imports done', flush=True)

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE        = '/uufs/chpc.utah.edu/common/home/u0972368/FFSL_HRWUI'
BEST_MODEL  = os.path.join(BASE, 'deep_learning/models/best_model.pth')
DLPK_PATH   = os.path.join(BASE, 'deep_learning/models/usa_building_footprints.dlpk')
PTH_PATH    = os.path.join(BASE, 'deep_learning/models/usa_building_footprints.pth')
TEST_SHP    = os.path.join(BASE, 'OLD_STUFF/Data/TestPoly/TestPoly.shp')
COUNTY_SHP  = os.path.join(BASE, 'OLD_STUFF/Data/Counties/Counties.shp')
EXTENT_RAS  = os.path.join(BASE, 'OLD_STUFF/Data/Extent_57M/extent_57m')
OUT_DIR     = os.path.join(BASE, 'data/predictions')

COUNTY_KEY = {
    1: 'Beaver',      2: 'Box_Elder',    3: 'Cache',
    4: 'Carbon',      5: 'Daggett',      6: 'Davis',
    7: 'Duchesne',    8: 'Emery',        9: 'Garfield',
    10: 'Grand',      11: 'Iron',        12: 'Juab',
    13: 'Kane',       14: 'Millard',     15: 'Morgan',
    16: 'Piute',      17: 'Rich',        18: 'Salt_Lake',
    19: 'San_Juan',   20: 'Sanpete',     21: 'Sevier',
    22: 'Summit',     23: 'Tooele',      24: 'Uintah',
    25: 'Utah',       26: 'Wasatch',     27: 'Washington',
    28: 'Wayne',      29: 'Weber'
}

# ── Scope ─────────────────────────────────────────────────────────────────────
# For testing:    set SCOPE = 'Random' or 'Medium', run python infer.py
# For production: set SCOPE = 'County', pass county number via SLURM argument
#                 e.g. python infer.py 18  →  runs Salt Lake County
SCOPE = 'Random'

# Read county number from SLURM argument if provided
if len(sys.argv) > 1:
    COUNTY_NBR = int(sys.argv[1])
    SCOPE      = 'County'
else:
    COUNTY_NBR = None

# WMTS
WMTS_URL    = 'https://discover.agrc.utah.gov/login/path/ENTER_YOUR_UGRC_WMTS_KEY_HERE/wmts/1.0.0/WMTSCapabilities.xml'
WMTS_LAYER  = 'utah'

# ── Imagery source ────────────────────────────────────────────────────────────
# 'wmts'  = fetch live from UGRC web service (requires valid WMTS_URL key)
# 'local' = read from downloaded GeoTIFF files on disk
IMAGERY_SOURCE    = 'wmts'
LOCAL_IMAGERY_DIR = '/path/to/downloaded/imagery/'   # only used if IMAGERY_SOURCE = 'local'

# Chip parameters
CHIP_PX     = 512
RESOLUTION  = 0.149291
CHIP_M      = CHIP_PX * RESOLUTION   # ~76.4m
STRIDE_M    = CHIP_M * 0.75          # 25% overlap — ~57.3m

# Detection thresholds
SCORE_THRESH = 0.5
NMS_IOU      = 0.15
MIN_AREA_M2  = 23.23                  # 250 sq ft minimum

# Checkpoint
SAVE_EVERY   = 50
# =============================================================================

# ── Output naming ─────────────────────────────────────────────────────────────
if SCOPE == 'County':
    scope_label = COUNTY_KEY.get(COUNTY_NBR, f'county_{COUNTY_NBR}')
else:
    scope_label = SCOPE.lower()

RUN_DIR    = os.path.join(OUT_DIR, f'predictions_{scope_label}')
OUT_SHP    = os.path.join(RUN_DIR, f'predictions_{scope_label}.shp')
CKPT_INFER = os.path.join(RUN_DIR, f'infer_checkpoint_{scope_label}.json')

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RUN_DIR, exist_ok=True)

print('FFSL Inference Starting', flush=True)
print(f'Scope:       {SCOPE}', flush=True)
if SCOPE == 'County':
    print(f'County:      {COUNTY_NBR} — {scope_label}', flush=True)
print(f'Output:      {OUT_SHP}', flush=True)
print(f'Score thresh:{SCORE_THRESH}', flush=True)
print(f'Min area:    {MIN_AREA_M2} m2 (250 sq ft)', flush=True)

# ── Load boundaries ───────────────────────────────────────────────────────────
TARGET_CRS = 'EPSG:3857'

print('\nLoading boundaries...', flush=True)

if SCOPE == 'County':
    county_gdf     = gpd.read_file(COUNTY_SHP).to_crs(TARGET_CRS)
    extent_gdf     = gpd.read_file(
        os.path.join(BASE, 'OLD_STUFF/Data/SES7_8_400MBuff_Dissolve/SES7_8_400MBuff_Dissolve.shp')
    ).to_crs(TARGET_CRS)
    extent_polygon = extent_gdf.geometry.iloc[0]
    county_row = county_gdf[county_gdf['COUNTYNBR'] == str(COUNTY_NBR).zfill(2)]
    if len(county_row) == 0:
        print(f'ERROR: COUNTYNBR {COUNTY_NBR} not found in shapefile', flush=True)
        exit(1)
    county_polygon = county_row.geometry.iloc[0]
    pred_boundary  = county_polygon.intersection(extent_polygon)
    print(f'County {COUNTY_NBR} ({scope_label}) intersected with SES7_8 extent', flush=True)

elif SCOPE in ('Random', 'Medium'):
    extent_gdf     = gpd.read_file(
        os.path.join(BASE, 'OLD_STUFF/Data/SES7_8_400MBuff_Dissolve/SES7_8_400MBuff_Dissolve.shp')
    ).to_crs(TARGET_CRS)
    extent_polygon = extent_gdf.geometry.iloc[0]
    testpoly       = gpd.read_file(TEST_SHP).to_crs(TARGET_CRS)
    pred_boundary  = testpoly[testpoly['Name'] == SCOPE].union_all()
    print(f'Using test polygon: {SCOPE}', flush=True)

else:
    print(f'ERROR: Unknown SCOPE {SCOPE}', flush=True)
    exit(1)

area_km2 = pred_boundary.area / 1e6
minx, miny, maxx, maxy = pred_boundary.bounds
print(f'Boundary area:   {area_km2:.2f} km2', flush=True)

# ── Build chip list ───────────────────────────────────────────────────────────
print('\nBuilding chip list...', flush=True)

if SCOPE == 'County':
    # Load raster, subset to county bbox, clip to county+extent intersection
    with rasterio.open(EXTENT_RAS) as src:
        transform = src.transform
        mask      = src.read(1)

    origin_x = transform.c
    origin_y = transform.f
    cell_w   = transform.a
    cell_h   = abs(transform.e)

    row_min = max(0, int((origin_y - maxy) / cell_h))
    row_max = min(mask.shape[0], int((origin_y - miny) / cell_h) + 1)
    col_min = max(0, int((minx - origin_x) / cell_w))
    col_max = min(mask.shape[1], int((maxx - origin_x) / cell_w) + 1)

    mask_sub    = mask[row_min:row_max, col_min:col_max]
    rows, cols  = np.where(mask_sub == 1)
    xs = origin_x + (cols + col_min) * cell_w
    ys = origin_y - (rows + row_min) * cell_h

    pred_prep = prep(pred_boundary)
    chip_list = []
    for x, y in zip(xs, ys):
        chip_box = box(float(x), float(y), float(x) + CHIP_M, float(y) + CHIP_M)
        if pred_prep.intersects(chip_box):
            chip_list.append((float(x), float(y)))

elif SCOPE in ('Random', 'Medium'):
    # Small test polygon — strip filter inline
    x_starts  = np.arange(minx, maxx, STRIDE_M)
    y_starts  = np.arange(miny, maxy, STRIDE_M)
    pred_prep = prep(pred_boundary)
    chip_list = []
    for x0 in x_starts:
        strip = box(x0, miny, x0 + CHIP_M, maxy)
        if not pred_prep.intersects(strip):
            continue
        for y0 in y_starts:
            chip_box = box(x0, y0, x0 + CHIP_M, y0 + CHIP_M)
            if pred_prep.intersects(chip_box):
                chip_list.append((x0, y0))

total_chips = len(chip_list)
print(f'Chips to process: {total_chips:,}', flush=True)

# ── Load model ────────────────────────────────────────────────────────────────
print('\nLoading model...', flush=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}', flush=True)
if torch.cuda.is_available():
    print(f'GPU:    {torch.cuda.get_device_name(0)}', flush=True)

if not os.path.exists(PTH_PATH):
    print('Extracting .pth from .dlpk...', flush=True)
    with zipfile.ZipFile(DLPK_PATH, 'r') as z:
        z.extract('usa_building_footprints.pth', os.path.join(BASE, 'deep_learning/models/'))

model = maskrcnn_resnet50_fpn(weights=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, 2)

state = torch.load(BEST_MODEL, map_location='cpu')
model.load_state_dict(state)
model = model.to(device)
model.eval()
print(f'Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters', flush=True)

# ── WMTS fetcher ──────────────────────────────────────────────────────────────
def fetch_wmts_chip(bbox_3857, chip_px=512):
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

# ── Local imagery fetcher ─────────────────────────────────────────────────────
def fetch_local_chip(bbox_3857, chip_px=512):
    """
    Reads a chip from locally downloaded GeoTIFF files.
    Searches LOCAL_IMAGERY_DIR for any .tif that overlaps bbox_3857.
    Returns a PIL Image (RGB) or None if no coverage found.
    """
    minx, miny, maxx, maxy = bbox_3857
    chip_box = box(minx, miny, maxx, maxy)
    try:
        for fname in os.listdir(LOCAL_IMAGERY_DIR):
            if not fname.lower().endswith(('.tif', '.tiff')):
                continue
            fpath = os.path.join(LOCAL_IMAGERY_DIR, fname)
            with rasterio.open(fpath) as src:
                left, bottom, right, top = src.bounds
                tile_box = box(left, bottom, right, top)
                if not chip_box.intersects(tile_box):
                    continue
                window = rasterio.windows.from_bounds(
                    minx, miny, maxx, maxy,
                    transform=src.transform
                )
                data = src.read([1, 2, 3], window=window,
                                out_shape=(3, chip_px, chip_px),
                                resampling=rasterio.enums.Resampling.bilinear)
                img_array = np.transpose(data, (1, 2, 0)).astype(np.uint8)
                return Image.fromarray(img_array, 'RGB')
        return None
    except Exception:
        return None


# ── Inference loop ────────────────────────────────────────────────────────────

# Load checkpoint if exists (resumes after preemption)
if os.path.exists(CKPT_INFER):
    print(f'Resuming from checkpoint: {CKPT_INFER}', flush=True)
    with open(CKPT_INFER) as f:
        ckpt = json.load(f)
    all_detections = [
        {
            'geometry':   Polygon(d['coords']),
            'confidence': d['confidence'],
            'chip_id':    d['chip_id']
        }
        for d in ckpt['detections']
    ]
    completed_chips = set(tuple(c) for c in ckpt['completed_chips'])
    chip_id         = ckpt['chip_id']
    skipped         = ckpt['skipped']
    print(f'Resumed: {len(completed_chips)} chips done, {len(all_detections)} detections so far', flush=True)
else:
    all_detections  = []
    completed_chips = set()
    chip_id         = 0
    skipped         = 0

print(f'\nRunning inference on {total_chips:,} chips...', flush=True)

with tqdm(total=total_chips, desc='Inference') as pbar:
    for x0, y0 in chip_list:
        chip_bbox = (x0, y0, x0 + CHIP_M, y0 + CHIP_M)
        chip_box  = box(*chip_bbox)
        chip_id  += 1

        # Skip already completed chips (resume after preemption)
        if (x0, y0) in completed_chips:
            pbar.update(1)
            continue

        img = fetch_wmts_chip(chip_bbox) if IMAGERY_SOURCE == 'wmts' else fetch_local_chip(chip_bbox)
        if img is None:
            skipped += 1
            pbar.update(1)
            continue

        img_tensor = TF.to_tensor(img).to(device)
        with torch.no_grad():
            output = model([img_tensor])[0]

        scores = output['scores'].cpu().numpy()
        masks  = output['masks'].cpu().numpy()

        cx0, cy0, cx1, cy1 = chip_bbox
        chip_w = cx1 - cx0
        chip_h = cy1 - cy0

        for score, mask in zip(scores, masks):
            if score < SCORE_THRESH:
                continue

            mask_bin = (mask[0] > 0.3).astype(np.uint8)
            if mask_bin.sum() < 10:
                continue

            # skimage contours with explicit Y-axis flip
            contours = measure.find_contours(mask_bin, 0.5)
            for contour in contours:
                if len(contour) < 4:
                    continue
                coords = []
                for row, col in contour:
                    geo_x = cx0 + (col / CHIP_PX) * chip_w
                    geo_y = cy1 - (row / CHIP_PX) * chip_h  # Y flip
                    coords.append((geo_x, geo_y))
                if len(coords) < 4:
                    continue
                geom = Polygon(coords)
                if not geom.is_valid:
                    geom = geom.buffer(0)
                if geom.is_empty or geom.area < 1:
                    continue
                if geom.geom_type != 'Polygon':
                    continue
                all_detections.append({
                    'geometry':   geom,
                    'confidence': round(float(score), 4),
                    'chip_id':    chip_id
                })

        completed_chips.add((x0, y0))

        # Save checkpoint every SAVE_EVERY chips
        if len(completed_chips) % SAVE_EVERY == 0:
            CKPT_TMP = CKPT_INFER + '.tmp'
            with open(CKPT_TMP, 'w') as f:
                json.dump({
                    'detections': [
                        {
                            'coords': list(d['geometry'].exterior.coords) if d['geometry'].geom_type == 'Polygon' else list(d['geometry'].convex_hull.exterior.coords),
                            'confidence': d['confidence'],
                            'chip_id':    d['chip_id']
                        }
                        for d in all_detections
                    ],
                    'completed_chips': [list(c) for c in completed_chips],
                    'chip_id':  chip_id,
                    'skipped':  skipped
                }, f)
            os.replace(CKPT_TMP, CKPT_INFER)
            print(f'\nCheckpoint saved: {len(completed_chips)}/{total_chips} chips', flush=True)

        pbar.update(1)

print(f'\nInference complete', flush=True)
print(f'Chips processed: {chip_id}', flush=True)
print(f'Chips skipped:   {skipped}', flush=True)
print(f'Raw detections:  {len(all_detections)}', flush=True)

# ── Post-processing ───────────────────────────────────────────────────────────
if not all_detections:
    print('No detections found — check SCORE_THRESH or model quality')
else:
    gdf_raw = gpd.GeoDataFrame(all_detections, geometry='geometry', crs=TARGET_CRS)
    print(f'\nPost-processing {len(gdf_raw)} detections...')

    # Step 1 — Aspect ratio filter
    def get_aspect(geom):
        minx, miny, maxx, maxy = geom.bounds
        w = maxx - minx
        h = maxy - miny
        if h == 0 or w == 0:
            return 999
        return max(w, h) / min(w, h)

    gdf_raw['aspect'] = gdf_raw.geometry.apply(get_aspect)
    before = len(gdf_raw)
    gdf_raw = gdf_raw[gdf_raw['aspect'] < 4.0].reset_index(drop=True)
    print(f'After aspect filter:     {len(gdf_raw)} (removed {before - len(gdf_raw)})')

    # Step 2 — Area filter pre-dedup
    gdf_raw['area_m2'] = gdf_raw.geometry.area
    before = len(gdf_raw)
    gdf_raw = gdf_raw[gdf_raw['area_m2'] > MIN_AREA_M2].reset_index(drop=True)
    print(f'After area filter:       {len(gdf_raw)} (removed {before - len(gdf_raw)})')

    # Step 3 — Simplify
    gdf_raw['geometry'] = gdf_raw.geometry.simplify(2.0, preserve_topology=True)

    # Step 4 — NMS deduplication
    gdf_raw = gdf_raw.sort_values('confidence', ascending=False).reset_index(drop=True)
    keep = [True] * len(gdf_raw)
    for i in range(len(gdf_raw)):
        if not keep[i]:
            continue
        geom_i = gdf_raw.iloc[i].geometry
        for j in range(i+1, len(gdf_raw)):
            if not keep[j]:
                continue
            geom_j = gdf_raw.iloc[j].geometry
            if not geom_i.intersects(geom_j):
                continue
            intersection = geom_i.intersection(geom_j).area
            union        = geom_i.union(geom_j).area
            iou          = intersection / union if union > 0 else 0
            if iou > NMS_IOU:
                keep[j] = False

    gdf_dedup = gdf_raw[keep].reset_index(drop=True)
    print(f'After NMS dedup:         {len(gdf_dedup)}')

    # Step 5 — Dissolve touching polygons, preserve mean confidence
    gdf_dedup['geometry'] = gdf_dedup.geometry.buffer(0.1)
    dissolved = gdf_dedup.dissolve(by=None)
    dissolved = dissolved.explode(index_parts=False).reset_index(drop=True)
    dissolved['geometry'] = dissolved.geometry.buffer(-0.1)
    dissolved['geometry'] = dissolved.geometry.simplify(2.0, preserve_topology=True)

    # Re-assign confidence as mean of contributing detections
    joined   = gpd.sjoin(dissolved[['geometry']], gdf_dedup[['geometry', 'confidence']], how='left', predicate='intersects')
    conf_avg = joined.groupby(joined.index)['confidence'].mean()
    dissolved['confidence'] = dissolved.index.map(conf_avg)
    dissolved['chip_id']    = gdf_dedup['chip_id'].iloc[0]
    print(f'After dissolve:          {len(dissolved)}')

    # Step 6 — Area filter post-dissolve
    dissolved['area_m2'] = dissolved.geometry.area
    before = len(dissolved)
    dissolved = dissolved[dissolved['area_m2'] > MIN_AREA_M2].reset_index(drop=True)
    print(f'After post-dissolve area:{len(dissolved)} (removed {before - len(dissolved)})')

    # Step 7 — Extent filter: keep only buildings FULLY within SES extent
    gdf_filtered = dissolved[
        dissolved.geometry.apply(lambda g: extent_polygon.contains(g))
    ].reset_index(drop=True)

    print(f'Fully within extent:     {len(gdf_filtered)}')
    print(f'Removed at boundary:     {len(dissolved) - len(gdf_filtered)}')

    if len(gdf_filtered) > 0:
        print(f'\nConfidence stats:')
        print(f'  Min:    {gdf_filtered.confidence.min():.3f}')
        print(f'  Max:    {gdf_filtered.confidence.max():.3f}')
        print(f'  Mean:   {gdf_filtered.confidence.mean():.3f}')
        print(f'  > 0.9:  {(gdf_filtered.confidence > 0.9).sum()}')
        print(f'  0.7-0.9:{((gdf_filtered.confidence >= 0.7) & (gdf_filtered.confidence < 0.9)).sum()}')
        print(f'  0.5-0.7:{((gdf_filtered.confidence >= 0.5) & (gdf_filtered.confidence < 0.7)).sum()}')

        # Export shapefile in NAD83
        gdf_export = gdf_filtered[['geometry', 'confidence', 'chip_id']].copy()
        gdf_export = gdf_export.set_crs('EPSG:3857', allow_override=True)
        gdf_export = gdf_export.to_crs('EPSG:4269')
        gdf_export.to_file(OUT_SHP)

        # Clean up checkpoint now that we're done
        if os.path.exists(CKPT_INFER):
            os.remove(CKPT_INFER)
            print('Checkpoint cleared')

        print(f'\n=== Inference Complete ===')
        print(f'Detections:  {len(gdf_export)}')
        print(f'CRS:         EPSG:4269 (NAD83)')
        print(f'Output:      {OUT_SHP}')
        print(f'Fields:      geometry, confidence (0-1), chip_id')
        print(f'QA tip:      Sort by confidence ASC — review low confidence first')
    else:
        print('No detections passed all filters')
