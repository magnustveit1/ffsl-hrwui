"""
FFSL Building Footprint — Model Training
Standalone script for SLURM batch execution.

Fine-tunes the ESRI pre-trained MaskRCNN building footprint model on
reviewed Utah HRWUI building polygons using COCO-format chips.

Usage:
    sbatch train.slurm

Author: Magnus Tveit
Project: Utah HRWUI Building Footprint Extraction
"""

import os
import json
import random
import zipfile
import warnings
import time
warnings.filterwarnings('ignore')

import torch
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE        = '/uufs/chpc.utah.edu/common/home/u0972368/FFSL_HRWUI'
CHIPS_DIR   = os.path.join(BASE, 'OLD_STUFF/Data/Chips')
ANN_PATH    = os.path.join(CHIPS_DIR, 'annotations.json')
IMG_DIR     = os.path.join(CHIPS_DIR, 'images')
DLPK_PATH   = os.path.join(BASE, 'deep_learning/models/usa_building_footprints.dlpk')
PTH_PATH    = os.path.join(BASE, 'deep_learning/models/usa_building_footprints.pth')
CKPT_DIR    = os.path.join(BASE, 'deep_learning/models/checkpoints')
BEST_MODEL  = os.path.join(BASE, 'deep_learning/models/best_model.pth')

MAX_SAMPLES = None      # None = full dataset
VAL_SPLIT   = 0.15      # 85% train / 15% val
BATCH_SIZE  = 4         # reduce to 2 if out-of-memory
MAX_EPOCHS  = 30
PATIENCE    = 5         # early stopping patience
LR          = 0.0001
NUM_WORKERS = 4
RANDOM_SEED = 42
# =============================================================================

os.makedirs(CKPT_DIR, exist_ok=True)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device:      {device}')
if torch.cuda.is_available():
    print(f'GPU:         {torch.cuda.get_device_name(0)}')
    print(f'VRAM:        {round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)} GB')
print(f'PyTorch:     {torch.__version__}')
print(f'torchvision: {torchvision.__version__}')


# =============================================================================
# DATASET
# =============================================================================
class BuildingDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, ann_path, image_ids=None, augment=False):
        self.img_dir = img_dir
        self.augment = augment

        with open(ann_path) as f:
            coco = json.load(f)

        self.images = {img['id']: img for img in coco['images']}
        self.ann_by_image = {}
        for ann in coco['annotations']:
            iid = ann['image_id']
            self.ann_by_image.setdefault(iid, []).append(ann)

        self.image_ids = image_ids if image_ids is not None else list(self.images.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        iid      = self.image_ids[idx]
        img_info = self.images[iid]
        img_path = os.path.join(self.img_dir, img_info['file_name'])

        img  = Image.open(img_path).convert('RGB')
        W, H = img.size
        anns = self.ann_by_image.get(iid, [])

        boxes, masks, labels = [], [], []
        for ann in anns:
            x, y, w, h = ann['bbox']
            if w < 2 or h < 2:
                continue
            boxes.append([x, y, x+w, y+h])
            labels.append(1)
            mask = Image.new('L', (W, H), 0)
            draw = ImageDraw.Draw(mask)
            for seg in ann['segmentation']:
                pts = list(zip(seg[0::2], seg[1::2]))
                if len(pts) >= 3:
                    draw.polygon(pts, fill=1)
            masks.append(np.array(mask, dtype=np.uint8))

        img_tensor = TF.to_tensor(img)

        if len(boxes) == 0:
            target = {
                'boxes':  torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,),    dtype=torch.int64),
                'masks':  torch.zeros((0, H, W), dtype=torch.uint8),
            }
        else:
            target = {
                'boxes':  torch.tensor(boxes,           dtype=torch.float32),
                'labels': torch.tensor(labels,          dtype=torch.int64),
                'masks':  torch.tensor(np.array(masks), dtype=torch.uint8),
            }

        if self.augment:
            img_tensor, target = self._augment(img_tensor, target)

        return img_tensor, target

    def _augment(self, img, target):
        if random.random() > 0.5:
            img = TF.hflip(img)
            if len(target['boxes']):
                _, H, W = img.shape
                boxes = target['boxes'].clone()
                boxes[:, [0, 2]] = W - boxes[:, [2, 0]]
                target['boxes'] = boxes
                target['masks'] = torch.flip(target['masks'], [2])
        if random.random() > 0.5:
            img = TF.vflip(img)
            if len(target['boxes']):
                _, H, W = img.shape
                boxes = target['boxes'].clone()
                boxes[:, [1, 3]] = H - boxes[:, [3, 1]]
                target['boxes'] = boxes
                target['masks'] = torch.flip(target['masks'], [1])
        if random.random() > 0.5:
            img = T.ColorJitter(
                brightness=0.3, contrast=0.3,
                saturation=0.2, hue=0.1
            )(img)
        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


# =============================================================================
# DATA LOADING
# =============================================================================
print('\nLoading dataset...')
with open(ANN_PATH) as f:
    coco_meta = json.load(f)

all_ids = [img['id'] for img in coco_meta['images']]
random.shuffle(all_ids)

if MAX_SAMPLES:
    all_ids = all_ids[:MAX_SAMPLES]

n_val     = int(len(all_ids) * VAL_SPLIT)
n_train   = len(all_ids) - n_val
train_ids = all_ids[:n_train]
val_ids   = all_ids[n_train:]

print(f'Total chips:  {len(all_ids)}')
print(f'Train chips:  {len(train_ids)}')
print(f'Val chips:    {len(val_ids)}')

train_dataset = BuildingDataset(IMG_DIR, ANN_PATH, image_ids=train_ids, augment=True)
val_dataset   = BuildingDataset(IMG_DIR, ANN_PATH, image_ids=val_ids,   augment=False)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE,
    shuffle=True, collate_fn=collate_fn,
    num_workers=NUM_WORKERS
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE,
    shuffle=False, collate_fn=collate_fn,
    num_workers=NUM_WORKERS
)

print(f'Train batches: {len(train_loader)}')
print(f'Val batches:   {len(val_loader)}')


# =============================================================================
# MODEL
# =============================================================================
print('\nLoading model...')

if not os.path.exists(PTH_PATH):
    print('Extracting .pth from .dlpk...')
    with zipfile.ZipFile(DLPK_PATH, 'r') as z:
        z.extract('usa_building_footprints.pth',
                  os.path.join(BASE, 'deep_learning/models/'))

model = maskrcnn_resnet50_fpn(weights=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, 2)

state = torch.load(PTH_PATH, map_location='cpu')
missing, unexpected = model.load_state_dict(state, strict=False)
model = model.to(device)

print(f'Model on:       {device}')
print(f'Parameters:     {sum(p.numel() for p in model.parameters()):,}')
print(f'Missing keys:   {len(missing)}')
print(f'Unexpected keys:{len(unexpected)}')


# =============================================================================
# TRAINING
# =============================================================================
optimizer = torch.optim.SGD(
    [p for p in model.parameters() if p.requires_grad],
    lr=LR, momentum=0.9, weight_decay=0.0005
)
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=8, gamma=0.5
)

# Check for existing checkpoint to resume from
start_epoch    = 0
best_val_loss  = float('inf')
patience_count = 0
train_losses   = []
val_losses     = []

if os.path.exists(CKPT_DIR):
    ckpts = sorted([
        f for f in os.listdir(CKPT_DIR) if f.endswith('.pth')
    ])
    if ckpts:
        latest_ckpt = os.path.join(CKPT_DIR, ckpts[-1])
        print(f'\nResuming from checkpoint: {latest_ckpt}')
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        start_epoch    = ckpt['epoch'] + 1
        best_val_loss  = ckpt.get('best_val_loss', float('inf'))
        train_losses   = ckpt.get('train_losses', [])
        val_losses     = ckpt.get('val_losses', [])
        patience_count = ckpt.get('patience_count', 0)
        print(f'Resuming from epoch {start_epoch}')

print(f'\nStarting training from epoch {start_epoch+1}/{MAX_EPOCHS}')
print(f'LR: {LR}, Batch: {BATCH_SIZE}, Patience: {PATIENCE}')
print()

for epoch in range(start_epoch, MAX_EPOCHS):
    epoch_start = time.time()

    # Training
    model.train()
    train_loss = 0.0
    for images, targets in tqdm(train_loader,
                                desc=f'Epoch {epoch+1}/{MAX_EPOCHS} [train]',
                                leave=False):
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses    = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        train_loss += losses.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    lr_scheduler.step()

    # Validation
    model.train()
    val_loss = 0.0
    with torch.no_grad():
        for images, targets in tqdm(val_loader,
                                    desc=f'Epoch {epoch+1}/{MAX_EPOCHS} [val]',
                                    leave=False):
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses    = sum(loss for loss in loss_dict.values())
            val_loss += losses.item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    epoch_time = time.time() - epoch_start
    print(f'Epoch {epoch+1:2d}/{MAX_EPOCHS} | '
          f'Train: {train_loss:.4f} | '
          f'Val: {val_loss:.4f} | '
          f'Time: {epoch_time/60:.1f}min')

    # Save checkpoint
    ckpt_path = os.path.join(CKPT_DIR, f'epoch_{epoch+1:03d}.pth')
    torch.save({
        'epoch':           epoch,
        'model_state':     model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'train_loss':      train_loss,
        'val_loss':        val_loss,
        'best_val_loss':   best_val_loss,
        'train_losses':    train_losses,
        'val_losses':      val_losses,
        'patience_count':  patience_count,
    }, ckpt_path)

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss  = val_loss
        patience_count = 0
        torch.save(model.state_dict(), BEST_MODEL)
        print(f'  -> New best model (val_loss={val_loss:.4f})')
    else:
        patience_count += 1
        print(f'  -> No improvement ({patience_count}/{PATIENCE})')

    # Early stopping
    if patience_count >= PATIENCE:
        print(f'\nEarly stopping at epoch {epoch+1}')
        break

# Save final loss curves
curves_path = os.path.join(BASE, 'deep_learning/models/loss_curves.json')
with open(curves_path, 'w') as f:
    json.dump({'train': train_losses, 'val': val_losses}, f)

print(f'\n=== Training Complete ===')
print(f'Best val loss:  {best_val_loss:.4f}')
print(f'Best model:     {BEST_MODEL}')
print(f'Loss curves:    {curves_path}')
print(f'Checkpoints:    {CKPT_DIR}')
