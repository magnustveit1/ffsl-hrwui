# FFSL Environment Setup and Reproducibility

**Last Updated:** March 2026  
**Author:** Magnus Tveit

---

## Overview

The pipeline runs inside the CHPC `deeplearning/2025.4` Singularity container,
which provides a frozen, reproducible PyTorch environment that works on all CHPC
GPU nodes without any conda environment management.

Three additional packages not included in the container are installed to the user's
home directory with `pip install --user` and persist across sessions automatically.

---

## On CHPC (Primary Environment)

### Activate environment
```bash
module load deeplearning/2025.4
python your_script.py
```

That's it. No conda activate, no LD_PRELOAD, no CUDA variables needed.

### Container details

| Component | Version |
|-----------|---------|
| Module | deeplearning/2025.4 |
| Container type | Singularity (read-only image) |
| Python | 3.11.11 |
| PyTorch | 2.7.0+cu126 |
| torchvision | 0.22.0+cu126 |
| CUDA | 12.6.3 |
| numpy | 2.1.3 |
| pandas | 2.2.3 |
| scikit-image | included |
| PIL (Pillow) | included |
| tqdm | included |
| requests | 2.32.3 |
| shapely | 2.1.2 (user-installed) |
| geopandas | 1.1.3 (user-installed) |
| rasterio | 1.4.4 (user-installed) |

### Supported GPU compute capabilities
PyTorch 2.7.0 supports: sm_50, sm_60, sm_70, sm_75, sm_80, sm_86, sm_90

This covers every GPU on CHPC granite-gpu-guest including RTX6000 (sm_86),
A800 (sm_80), H100 (sm_90), H200 (sm_90), and all others.

### User-installed packages (pip --user)

These are stored at `~/.local/lib/python3.11/site-packages/` and load
automatically when the deeplearning module is active.

To reinstall if lost:
```bash
module load deeplearning/2025.4
pip install --user geopandas==1.1.3 rasterio==1.4.4 shapely==2.1.2
```

Full user package list (from pip list --user --format=freeze):
```
affine==2.4.0
click-plugins==1.1.1.2
cligj==0.7.2
geopandas==1.1.3
pyogrio==0.12.1
pyproj==3.7.2
rasterio==1.4.4
shapely==2.1.2
```

---

## Outside CHPC (Local Machine / Other Systems)

The pipeline does not depend on any CHPC-specific infrastructure except the
file paths and WMTS URL. To run on another system:

### Requirements

- Python 3.11+
- CUDA-capable GPU recommended (CPU works but is very very slow)

### Installation

```bash
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126
pip install geopandas==1.1.3 rasterio==1.4.4 shapely==2.1.2
pip install requests tqdm scikit-image Pillow numpy pandas
```

### Path changes needed

Update the `BASE` variable at the top of each script:
```python
BASE = '/your/local/path/to/FFSL'
```

---

## Validating the Environment

Run the test script to confirm all components work:
```bash
module load deeplearning/2025.4
python /uufs/chpc.utah.edu/common/home/u0972368/FFSL/Notebooks/test_env.py
```

Expected output — all 7 tests should show OK:
```
=== Test 1: Imports ===         chip.py imports: OK / train.py imports: OK
=== Test 2: WMTS Fetch ===      WMTS fetch: OK
=== Test 3: Label Loading ===   Labels loaded: 17191 polygons
=== Test 4: COCO Annotation === COCO annotation: OK
=== Test 5: Model Build ===     Model build + load: OK
=== Test 6: Mini Training ===   Mini training loop: OK
=== Test 7: Best Model ===      Best model loading: OK
```

---

## Why Not conda?

The original `arcgis_dl` conda environment (27GB) was replaced with the CHPC
deeplearning module for three reasons:

1. The conda environment caused CUDA initialization hangs on certain nodes
2. The deeplearning module works on all GPU types including H100/H200
3. The module is version-frozen and maintained by CHPC — no dependency drift

The `arcgis_dl` environment has been deleted to recover ~27GB of quota space.
The `arcgis` package itself (for ArcGIS Online auth) is no longer needed since
the pipeline uses the WMTS URL directly without portal authentication.

---

## SLURM Template

```bash
#!/bin/bash
#SBATCH --job-name=ffsl_infer1
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=64G
#SBATCH --account=johnsonca
#SBATCH --partition=granite-gpu-guest
#SBATCH --qos=granite-gpu-guest
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/uufs/chpc.utah.edu/common/home/u0972368/FFSL/logs/infer_%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=u0972368@utah.edu

COUNTY_NBR=6   # Change this number to run a different county

module load deeplearning/2025.4
python /uufs/chpc.utah.edu/common/home/u0972368/FFSL/Notebooks/infer.py $COUNTY_NBR
```
