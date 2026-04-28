# Validation

Contains the validation notebook and environment for evaluating deep learning inference results against manually verified ground truth structures.

---

## Overview

Validation was performed for Weber County, comparing DL predictions against 572 manually verified structures (Johnson dataset) within the SES 7/8 boundary.

**Method:** A detection is counted as a True Positive if the DL polygon spatially overlaps with a verified structure polygon. Inference was run across the full study area including the 400m buffer, but validation is performed within the SES 7/8 boundary only. All spatial operations in EPSG:5070 (meters).

---

## Files

| File | Description |
|---|---|
| `hrwui_validation.ipynb` | Main validation notebook |
| `environment.yml` | Conda environment for running the notebook |

---

## Environment Setup

```bash
conda env create -f validation/environment.yml
conda activate ffsl_validation
python -m ipykernel install --user --name ffsl_validation --display-name "FFSL Validation"
```

Then open `hrwui_validation.ipynb` in VS Code or JupyterLab and select the **FFSL Validation** kernel.

---

## Data Requirements

Place the following in `data/inputs/` before running:

| File | Description | Source |
|---|---|---|
| `Johnson_Weber/Johnson_Weber.shp` | 572 manually verified structures (ground truth) | Project supervisor |
| `FFSL26_Weber/FFSL26_Weber.shp` | FFSL 2026 official structure centroids (506 structures) | FFSL |

Place the following in `data/predictions/`:

| File | Description |
|---|---|
| `predictions_Weber/predictions_Weber.shp` | Full Weber County DL predictions (2,350 detections) |

---

## Notebook Structure

| Section | Description |
|---|---|
| Imports & Directories | All libraries and file paths defined once at the top |
| Statistics for Poster | Calculates TP, FP, FN, Precision, Recall, F1 |
| Figure 3 | Validation results table by confidence bin |
| Figure 4 | TP and FP detections by confidence score (line chart) |
| Figure 5 | Detection outcome legend (2D color grid) |

All figures are saved to `data/outputs/` at 300-600 DPI.

---

## Results (Weber County)

| Metric | Value |
|---|---|
| Ground truth structures | 572 |
| True Positives (TP) | 534 |
| False Positives (FP) | 932 |
| False Negatives (FN) | 64 |
| Precision | 36.4% |
| Recall | 89.3% |
| F1 Score | 0.517 |
| Optimal threshold (best F1) | 0.90 |

High recall is the appropriate outcome for HB48 fee administration - missing a real structure is more costly than reviewing a false detection.

---

## Future Work

- Update notebook to clip predictions to SES 7/8 boundary internally (rather than using pre-clipped file) - requires adding SES 7/8 boundary without buffer as an input
- Extend validation to additional counties as inference completes
