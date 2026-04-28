# Deep Learning Pipeline

This folder contains the full MaskRCNN pipeline for automated building footprint extraction across Utah's High-Risk Wildland Urban Interface (HRWUI).

---

## Pipeline Overview

| Script | Purpose |
|---|---|
| `chip.py` | Generates 512×512px training chips from WMTS/local imagery |
| `train.py` | Fine-tunes MaskRCNN on reviewed building polygons |
| `infer.py` | Runs county-by-county inference on CHPC |
| `infer.slurm` | SLURM submission script for RAI GPU nodes |

---

## Environment Setup (CHPC)

The pipeline runs inside the CHPC `deeplearning/2025.4` Singularity container. See `ENVIRONMENT_deep_learning.md` for full details.

```bash
module load deeplearning/2025.4
python deep_learning/notebooks/infer.py 29   # runs Weber County
```

Three packages are installed to the user home directory with `pip install --user`:
- `shapely`
- `geopandas`
- `rasterio`

---

## Data Requirements

Place the following in `data/` before running:

### Inputs (required for inference)
| File | Location | Description |
|---|---|---|
| `Counties/Counties.shp` | `data/inputs` | Utah county boundaries |
| `Extent_57M/extent_57m` | `data/inputs` | Binary raster mask at 57m cell size |
| `SES7_8_400MBuff_Dissolve.shp` | `data/inputs` | HRWUI study boundary with 400m buffer |
| `best_model.pth` | `deep_learning/models/` | Fine-tuned model weights (Google Drive) |

### Imagery
Inference requires either:
- **WMTS** - a valid UGRC Discover key (set `IMAGERY_SOURCE = 'wmts'` in `infer.py`)
- **Local GeoTIFFs** - downloaded imagery directory (set `IMAGERY_SOURCE = 'local'` and `LOCAL_IMAGERY_DIR` in `infer.py`)

> **Note:** The original UGRC WMTS key was suspended after generating high server load during statewide inference. The recommended path forward is to download study area imagery (~17 TB) locally from UGRC and set `IMAGERY_SOURCE = 'local'`.

---

## Running Inference

### 1. Set the county
Edit `infer.slurm` and set `COUNTY_NBR` to the desired county number:

```bash
COUNTY_NBR=29   # Weber County
```

County key:
```
1: Beaver      2: Box_Elder   3: Cache      4: Carbon     5: Daggett
6: Davis       7: Duchesne    8: Emery      9: Garfield  10: Grand
11: Iron       12: Juab       13: Kane      14: Millard   15: Morgan
16: Piute      17: Rich       18: Salt_Lake 19: San_Juan  20: Sanpete
21: Sevier     22: Summit     23: Tooele    24: Uintah    25: Utah
26: Wasatch    27: Washington 28: Wayne     29: Weber
```

### 2. Submit the job
```bash
sbatch deep_learning/notebooks/infer.slurm
```

### 3. Monitor
```bash
tail -f deep_learning/logs/infer_JOBID.out
```

---

## Model Details

| Parameter | Value |
|---|---|
| Architecture | MaskRCNN - ResNet50 FPN backbone |
| Base weights | ESRI USA Building Footprints (pretrained) |
| Parameters | 43,922,395 |
| Training data | 17,191 polygons → ~34,000 chips (512×512px) |
| Best val. loss | 0.3407 (epoch 18, early stop epoch 23) |
| Score threshold | 0.5 |
| NMS IoU threshold | 0.15 |
| Min detection area | 23.23 m² (250 sq ft) |
| Inference hardware | CHPC RAI nodes - NVIDIA H200 GPUs |

Training loss history is saved in `models/loss_curves.json`.

---

## Post-Processing Steps

Detections go through the following filters in order:
1. Aspect ratio filter (remove detections with ratio > 4.0)
2. Area filter (remove detections < 23.23 m²)
3. Geometry simplification
4. NMS deduplication (IoU > 0.15, keep highest confidence)
5. Dissolve touching polygons
6. Post-dissolve area filter
7. Extent clip to HRWUI boundary

Output: EPSG:4269 (NAD83) shapefile with `confidence` field (0-1).

---

## Completed Inference

| County | Detections | Notes |
|---|---|---|
| Davis | 358 | Completed in ~26 hours |
| Weber | 2,350 | Completed in ~50 hours |