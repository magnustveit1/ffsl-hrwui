# Identifying Structures Within the High-Risk Wildland–Urban Interface (WUI)

**Author:** Magnus Tveit, University of Utah GIS Capstone  
**Community Partner:** Tommy Thompson, Utah Division of Forestry, Fire & State Lands (FFSL)  
**Academic Supervisor:** Phoebe McNeally, University of Utah  
**Status:** In progress — Fall 2026 completion  

---

## Project Overview

Utah's wildfire risk is mapped using a Structural Exposure Score (SES), where scores of 7 or 8 represent the highest-risk areas — covering approximately 138,000 km², or roughly 63% of Utah's land area. The **High-Risk Wildland–Urban Interface (HRWUI)** is a subset of this zone, defined as areas where two or more structures exist within 250 meters of each other within the SES 7/8 boundary.

Under **HB48**, the Utah Division of Forestry, Fire & State Lands (FFSL) is responsible for identifying all structures that meet this criteria and maintaining a complete, accurate inventory of qualifying buildings. This dataset did not previously exist in verified form.

This project develops a two-track approach to build that inventory:
1. **Manual review** of existing Overture Maps Foundation (OMF) building polygon data in ArcGIS Pro
2. **Deep learning pipeline** to identify structures missed by current datasets using 15cm Hexagon aerial imagery

---

## Repository Structure

```
ffsl-hrwui/
│
├── README.md                          ← this file
│
├── data/
│   ├── inputs/                        ← validation input shapefiles (CHPC only)
│   ├── predictions/                   ← completed DL inference outputs
│   │   ├── predictions_Davis/         ← 358 detections
│   │   └── predictions_Weber/         ← 2,350 detections
│   └── outputs/                       ← figures and CSV outputs
│
├── deep_learning/
│   ├── ENVIRONMENT_deep_learning.md   ← CHPC environment setup
│   ├── notebooks/                     ← pipeline scripts
│   │   ├── chip.py                    ← training chip generation
│   │   ├── train.py                   ← MaskRCNN fine-tuning
│   │   ├── infer.py                   ← county-by-county inference
│   │   └── infer.slurm               ← SLURM submission script
│   └── models/                        ← model weights (CHPC only, see below)
│
├── SQL/
│   ├── ffsl_parcel_join.sql           ← three-way join script
│   └── parcel_join_strategy.csv       ← county-specific parcel ID cleaning rules
│
└── validation/
    ├── hrwui_validation.ipynb         ← validation notebook
    └── environment.yml                ← conda environment for validation
```

---

## Data

All large data files live on CHPC at:
`/uufs/chpc.utah.edu/common/home/u0972368/FFSL_HRWUI/`

They are **not stored in this repository**. See each subfolder's README for what data is needed and where to obtain it.

### Model Files (too large for GitHub)

Model weights are hosted on Google Drive. Download and place at `deep_learning/models/`:

| File | Size | Description |
|---|---|---|
| `best_model.pth` | 176 MB | Fine-tuned MaskRCNN weights — use for inference |
| `usa_building_footprints.pth` | 176 MB | ESRI base model weights |
| `usa_building_footprints.dlpk` | 165 MB | ESRI base model package |

> Google Drive link: *(add link here)*

---

## Quick Start

### Deep Learning Inference
See `deep_learning/README.md` for full setup and inference instructions.

### SQL Parcel Join
See `SQL/README.md` for instructions on running the parcel join for any Utah county.

### Validation
See `validation/README.md` for instructions on running the validation notebook.

---

## Results Summary (Weber County)

| Metric | Value |
|---|---|
| Ground truth structures | 572 (manually verified) |
| DL detections (full county) | 2,350 |
| DL detections (SES 7/8 only) | 1,466 |
| True Positives | 534 |
| False Positives | 932 |
| False Negatives | 64 |
| Recall | 89.3% |
| Precision | 36.4% |
| F1 Score | 0.517 |
| Detections outside existing HRWUI | 468 |

High recall is the right outcome for this use case — missing a real structure is more costly than reviewing a false detection under HB48 fee administration.

---

## Acknowledgements

- Tommy Thompson & Utah Division of Forestry, Fire & State Lands (FFSL)
- Phoebe McNeally, University of Utah
- CHPC (Martin Cuma) for HPC resource allocation
- UGRC for Hexagon 15cm aerial imagery
- Overture Maps Foundation for building polygon data
