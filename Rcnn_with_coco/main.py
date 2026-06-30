#!/usr/bin/env python3
"""
Helmet Detection — Faster R-CNN (ResNet50-FPN)
TH Deggendorf · MSS-M-2 · SS26

Usage:
  python main.py train                        # No-COCO, ImageNet backbone, 10 epochs
  python main.py train --coco                 # COCO pretrained, head replaced
  python main.py train --epochs 30            # More epochs
  python main.py train --no-imagenet          # Completely from scratch
  python main.py train --resume checkpoints/checkpoint_epoch010.pth

  python main.py evaluate                     # Test set, best_model.pth
  python main.py evaluate --split val
  python main.py evaluate --checkpoint checkpoints_coco/best_model.pth

  python main.py compare                      # 2-model chart (no-COCO vs YOLOv8)
  python main.py compare \\
      --rcnn-coco-map50-val 0.8832 \\
      --rcnn-coco-map50-test 0.8127 \\
      --rcnn-coco-map 0.4919 \\
      --rcnn-coco-helmet-ap50 0.5225 \\
      --rcnn-coco-head-ap50 0.4614

  python main.py inference --image path/to/img.jpg
  python main.py inference --test-samples 5
"""

# ===========================================================================
# Imports
# ===========================================================================

import argparse
import json
import os
import random
import time
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as TF


# ===========================================================================
# Config
# ===========================================================================

ARCHIVE_ROOT    = "/Users/harundolcan/Downloads/archive (1)"
IMAGES_DIR      = os.path.join(ARCHIVE_ROOT, "images")
ANNOTATIONS_DIR = os.path.join(ARCHIVE_ROOT, "annotations")

YOLO_ROOT        = "/Users/harundolcan/Downloads/ppe detection.yolov8"
USE_YOLO_DATASET = True

# 0 = background (reserved by Faster R-CNN)
# 1 = helmet  (helmet ON the head)
# 2 = head    (bare head / No Helmet)
NUM_CLASSES    = 3
CLASS_NAMES    = ["background", "helmet", "head"]
CLASS_MAP      = {"helmet": 1, "head": 2}
YOLO_CLASS_MAP = {0: 1, 1: 2}   # YOLO 0=Helmet, 1=No_Helmet

TRAIN_RATIO  = 0.80
VAL_RATIO    = 0.10
RANDOM_SEED  = 42

BATCH_SIZE    = 4
NUM_EPOCHS    = 10
LEARNING_RATE = 0.005
MOMENTUM      = 0.9
WEIGHT_DECAY  = 0.0005
LR_STEP_SIZE  = 10
LR_GAMMA      = 0.1
EVAL_EVERY_N_EPOCHS = 2

USE_IMAGENET_BACKBONE = True

CHECKPOINT_DIR = "./checkpoints"
SPLITS_FILE    = "./data_splits.json"
RESULTS_FILE   = "./results.json"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps")
    return torch.device("cpu")


# ===========================================================================
# Dataset
# ===========================================================================

class HelmetDataset(Dataset):
    """Archive dataset: Pascal VOC XML annotations, 416x416 PNG images."""

    def __init__(self, file_list, augment=False):
        self.file_list = file_list
        self.augment   = augment

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        base     = self.file_list[idx]
        img_path = os.path.join(IMAGES_DIR,      base + ".png")
        ann_path = os.path.join(ANNOTATIONS_DIR, base + ".xml")

        image         = Image.open(img_path).convert("RGB")
        boxes, labels = _parse_voc_xml(ann_path)

        if len(boxes) == 0:
            return self.__getitem__((idx + 1) % len(self))

        boxes  = torch.as_tensor(boxes,  dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        if self.augment and random.random() > 0.5:
            image, boxes = _hflip(image, boxes)

        image = TF.to_tensor(image)
        target = {
            "boxes":    boxes,
            "labels":   labels,
            "image_id": torch.tensor([idx]),
            "area":     (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd":  torch.zeros(len(labels), dtype=torch.int64),
        }
        return image, target


class YoloHelmetDataset(Dataset):
    """
    YOLOv8-format dataset (Roboflow export).
    YOLO label: <class_id> <cx> <cy> <w> <h>  (normalized 0-1)
    Class mapping: 0 → helmet (1), 1 → head/no-helmet (2)
    """

    def __init__(self, images_dir, labels_dir, augment=False, id_offset=0):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.augment    = augment
        self.id_offset  = id_offset
        self.samples    = []
        for fname in sorted(os.listdir(images_dir)):
            if not (fname.lower().endswith(".jpg") or fname.lower().endswith(".png")):
                continue
            stem     = os.path.splitext(fname)[0]
            lbl_path = os.path.join(labels_dir, stem + ".txt")
            if os.path.exists(lbl_path):
                self.samples.append((os.path.join(images_dir, fname), lbl_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]
        image  = Image.open(img_path).convert("RGB")
        w, h   = image.size
        boxes, labels = _parse_yolo_label(lbl_path, w, h)

        if len(boxes) == 0:
            return self.__getitem__((idx + 1) % len(self))

        boxes  = torch.as_tensor(boxes,  dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        if self.augment and random.random() > 0.5:
            image, boxes = _hflip(image, boxes)

        image = TF.to_tensor(image)
        target = {
            "boxes":    boxes,
            "labels":   labels,
            "image_id": torch.tensor([self.id_offset + idx]),
            "area":     (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd":  torch.zeros(len(labels), dtype=torch.int64),
        }
        return image, target


def _parse_voc_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes, labels = [], []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip().lower()
        if name not in CLASS_MAP:
            continue
        b    = obj.find("bndbox")
        xmin = float(b.find("xmin").text)
        ymin = float(b.find("ymin").text)
        xmax = float(b.find("xmax").text)
        ymax = float(b.find("ymax").text)
        if xmax > xmin and ymax > ymin:
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(CLASS_MAP[name])
    return boxes, labels


def _parse_yolo_label(label_path, img_w, img_h):
    boxes, labels = [], []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            if cls_id not in YOLO_CLASS_MAP:
                continue
            cx, cy, bw, bh = map(float, parts[1:5])
            xmin = max(0.0, (cx - bw / 2) * img_w)
            ymin = max(0.0, (cy - bh / 2) * img_h)
            xmax = min(img_w, (cx + bw / 2) * img_w)
            ymax = min(img_h, (cy + bh / 2) * img_h)
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(YOLO_CLASS_MAP[cls_id])
    return boxes, labels


def _hflip(image, boxes):
    w     = image.width
    image = TF.hflip(image)
    boxes = boxes.clone()
    boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
    return image, boxes


def get_splits():
    """Load or create 80/10/10 archive splits (persisted to disk)."""
    if os.path.exists(SPLITS_FILE):
        with open(SPLITS_FILE) as f:
            splits = json.load(f)
        return splits["train"], splits["val"], splits["test"]

    all_files = sorted(
        f.replace(".png", "")
        for f in os.listdir(IMAGES_DIR)
        if f.endswith(".png")
    )
    rng      = random.Random(RANDOM_SEED)
    shuffled = all_files.copy()
    rng.shuffle(shuffled)
    n         = len(shuffled)
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))
    train = shuffled[:train_end]
    val   = shuffled[train_end:val_end]
    test  = shuffled[val_end:]
    with open(SPLITS_FILE, "w") as f:
        json.dump({"train": train, "val": val, "test": test}, f, indent=2)
    print(f"Splits saved to {SPLITS_FILE}")
    return train, val, test


def get_datasets(augment_train=True, use_yolo=None):
    """Returns (train_dataset, val_dataset, test_dataset)."""
    if use_yolo is None:
        use_yolo = USE_YOLO_DATASET

    train_files, val_files, test_files = get_splits()

    archive_train = HelmetDataset(train_files, augment=augment_train)
    archive_val   = HelmetDataset(val_files,   augment=False)
    archive_test  = HelmetDataset(test_files,  augment=False)

    if use_yolo:
        yolo_train = YoloHelmetDataset(
            images_dir=os.path.join(YOLO_ROOT, "train", "images"),
            labels_dir=os.path.join(YOLO_ROOT, "train", "labels"),
            augment=augment_train, id_offset=100_000,
        )
        yolo_val = YoloHelmetDataset(
            images_dir=os.path.join(YOLO_ROOT, "valid", "images"),
            labels_dir=os.path.join(YOLO_ROOT, "valid", "labels"),
            augment=False, id_offset=200_000,
        )
        yolo_test = YoloHelmetDataset(
            images_dir=os.path.join(YOLO_ROOT, "test", "images"),
            labels_dir=os.path.join(YOLO_ROOT, "test", "labels"),
            augment=False, id_offset=300_000,
        )
        train_ds = ConcatDataset([archive_train, yolo_train])
        val_ds   = ConcatDataset([archive_val,   yolo_val])
        test_ds  = ConcatDataset([archive_test,  yolo_test])
        print(
            f"[Data] Combined  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}\n"
            f"         Archive  {len(archive_train)} / {len(archive_val)} / {len(archive_test)}\n"
            f"         YOLO     {len(yolo_train)} / {len(yolo_val)} / {len(yolo_test)}"
        )
    else:
        train_ds = archive_train
        val_ds   = archive_val
        test_ds  = archive_test
        print(f"[Data] Archive only  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    return train_ds, val_ds, test_ds


def collate_fn(batch):
    return tuple(zip(*batch))


# ===========================================================================
# Model
# ===========================================================================

def build_model(num_classes=NUM_CLASSES, use_imagenet_backbone=USE_IMAGENET_BACKBONE,
                use_coco=False):
    """
    Faster R-CNN with ResNet50-FPN backbone.

    use_coco=True  → COCO_V1 pretrained weights, detection head replaced for num_classes
    use_coco=False → weights=None (no COCO), optional ImageNet backbone (prof. requirement)
    """
    if use_coco:
        model = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
            min_size=416,
            max_size=416,
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        print(f"[Model] Faster R-CNN | {num_classes} classes | COCO pretrained | head replaced")
    else:
        backbone_weights = ResNet50_Weights.IMAGENET1K_V1 if use_imagenet_backbone else None
        model = fasterrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=backbone_weights,
            num_classes=num_classes,
            min_size=416,
            max_size=416,
        )
        mode = "ImageNet backbone" if use_imagenet_backbone else "from scratch"
        print(f"[Model] Faster R-CNN | {num_classes} classes | {mode} | NO COCO")
    return model


# ===========================================================================
# Training
# ===========================================================================

def train_one_epoch(model, optimizer, loader, device, epoch, total_epochs,
                    warmup_scheduler=None):
    model.train()
    total_loss  = 0.0
    valid_steps = 0
    t0          = time.time()

    for step, (images, targets) in enumerate(loader, 1):
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss      = sum(loss_dict.values())

        if not torch.isfinite(loss):
            print(f"  [skip] non-finite loss at step {step}")
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if warmup_scheduler is not None:
            warmup_scheduler.step()

        total_loss  += loss.item()
        valid_steps += 1

        if step % 100 == 0 or step == len(loader):
            elapsed = time.time() - t0
            details = {k: f"{v.item():.4f}" for k, v in loss_dict.items()}
            lr_now  = optimizer.param_groups[0]["lr"]
            print(
                f"  Ep [{epoch}/{total_epochs}] Step [{step}/{len(loader)}] "
                f"Loss: {loss.item():.4f}  LR: {lr_now:.6f}  ({elapsed:.0f}s)\n"
                f"    {details}"
            )

    return total_loss / max(valid_steps, 1)


def compute_map(model, loader, device):
    """Compute mAP on a DataLoader, returns dict with map_50, per_class, etc."""
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)

    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            preds = [
                {"boxes": o["boxes"].cpu(), "scores": o["scores"].cpu(),
                 "labels": o["labels"].cpu()}
                for o in outputs
            ]
            gts = [
                {"boxes": t["boxes"].cpu(), "labels": t["labels"].cpu()}
                for t in targets
            ]
            metric.update(preds, gts)

    raw    = metric.compute()
    result = {k: float(v) for k, v in raw.items() if hasattr(v, "numel") and v.numel() == 1}

    per_class = {}
    if "map_per_class" in raw:
        for i, ap in enumerate(raw["map_per_class"]):
            per_class[CLASS_NAMES[i + 1]] = float(ap)
    result["per_class"] = per_class
    return result


def cmd_train(args):
    global CHECKPOINT_DIR, RESULTS_FILE

    use_yolo     = not args.no_yolo
    use_imagenet = not args.no_imagenet
    use_coco     = args.coco

    if use_coco:
        CHECKPOINT_DIR = "./checkpoints_coco"
        RESULTS_FILE   = "./results_coco.json"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    device = torch.device(args.device) if args.device else get_device()
    print(f"[Device] {device}")

    train_ds, val_ds, _ = get_datasets(augment_train=True, use_yolo=use_yolo)
    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=1,
                              shuffle=False, num_workers=0, collate_fn=collate_fn)

    model = build_model(use_imagenet_backbone=use_imagenet, use_coco=use_coco)
    model.to(device)

    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr,
                                momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    WARMUP_STEPS      = 500
    warmup_scheduler  = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_STEPS)
    main_scheduler    = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    start_epoch = 1
    best_map50  = -1.0
    best_loss   = float("inf")
    history     = []
    global_step = 0

    if args.resume:
        ckpt        = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_map50  = ckpt.get("map_50", -1.0)
        best_loss   = ckpt.get("loss", float("inf"))
        history     = ckpt.get("history", [])
        global_step = ckpt.get("global_step", WARMUP_STEPS + 1)
        print(f"[Resume] epoch={ckpt['epoch']}  map50={best_map50:.4f}  loss={best_loss:.4f}")

    steps_per_epoch = len(train_loader)

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'='*65}")
        active_warmup = warmup_scheduler if global_step < WARMUP_STEPS else None
        avg_loss      = train_one_epoch(model, optimizer, train_loader, device,
                                        epoch, args.epochs, warmup_scheduler=active_warmup)
        global_step  += steps_per_epoch

        if global_step >= WARMUP_STEPS:
            main_scheduler.step()

        lr_now     = optimizer.param_groups[0]["lr"]
        epoch_info = {"epoch": epoch, "loss": avg_loss, "lr": lr_now,
                      "global_step": global_step}

        map50 = None
        if args.eval_freq > 0 and epoch % args.eval_freq == 0:
            print(f"  Computing val mAP (epoch {epoch})...")
            t_eval     = time.time()
            map_result = compute_map(model, val_loader, device)
            map50      = map_result.get("map_50", 0.0)
            epoch_info.update({
                "map":       map_result.get("map", 0.0),
                "map_50":    map50,
                "map_75":    map_result.get("map_75", 0.0),
                "per_class": map_result.get("per_class", {}),
            })
            print(f"  Val mAP@0.50={map50:.4f}  "
                  f"mAP@0.50:0.95={map_result.get('map', 0):.4f}  "
                  f"({time.time()-t_eval:.0f}s)")
            for cls, ap in map_result.get("per_class", {}).items():
                print(f"    {cls}: {ap:.4f}")

        print(
            f"Epoch [{epoch}/{args.epochs}]  Loss={avg_loss:.4f}  LR={lr_now:.6f}"
            + (f"  mAP@0.5={map50:.4f}" if map50 is not None else "")
        )
        history.append(epoch_info)

        ckpt_data = {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss":                 avg_loss,
            "map_50":               map50 if map50 is not None else -1.0,
            "use_imagenet":         use_imagenet,
            "use_coco":             use_coco,
            "use_yolo":             use_yolo,
            "global_step":          global_step,
            "history":              history,
        }

        if epoch % 5 == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch{epoch:03d}.pth")
            torch.save(ckpt_data, ckpt_path)

        is_best = False
        if map50 is not None and map50 > best_map50:
            best_map50 = map50
            is_best    = True
        elif map50 is None and avg_loss < best_loss:
            best_loss = avg_loss
            is_best   = True

        if is_best:
            torch.save(ckpt_data, os.path.join(CHECKPOINT_DIR, "best_model.pth"))
            marker = f"mAP@0.5={best_map50:.4f}" if map50 is not None else f"loss={avg_loss:.4f}"
            print(f"  -> Best model saved  ({marker})")

    history_path = os.path.join(CHECKPOINT_DIR, "train_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete. History saved to {history_path}")
    if best_map50 >= 0:
        print(f"Best mAP@0.5 on val: {best_map50:.4f}")


# ===========================================================================
# Evaluation
# ===========================================================================

def cmd_evaluate(args):
    device   = torch.device(args.device) if args.device else get_device()
    use_yolo = not args.no_yolo
    print(f"[Device] {device}")

    _, val_ds, test_ds = get_datasets(augment_train=False, use_yolo=use_yolo)
    dataset = val_ds if args.split == "val" else test_ds
    print(f"[Eval]  split={args.split}  n={len(dataset)}")

    loader = DataLoader(dataset, batch_size=1,
                        shuffle=False, num_workers=0, collate_fn=collate_fn)

    ckpt      = torch.load(args.checkpoint, map_location=device)
    use_coco  = ckpt.get("use_coco", False)
    model     = build_model(use_coco=use_coco)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.roi_heads.score_thresh = args.score_threshold
    print(f"[Model] epoch={ckpt.get('epoch', '?')}  {args.checkpoint}")

    print("\nRunning evaluation...")
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            preds = [{"boxes": o["boxes"].cpu(), "scores": o["scores"].cpu(),
                      "labels": o["labels"].cpu()} for o in outputs]
            gts   = [{"boxes": t["boxes"].cpu(), "labels": t["labels"].cpu()}
                     for t in targets]
            metric.update(preds, gts)

    results      = metric.compute()
    summary      = {k: float(v) for k, v in results.items()
                    if hasattr(v, "numel") and v.numel() == 1}
    per_class_ap = {}
    if "map_per_class" in results:
        for i, ap in enumerate(results["map_per_class"]):
            per_class_ap[CLASS_NAMES[i + 1]] = float(ap)

    print("\n" + "=" * 50)
    print("  mAP Results")
    print("=" * 50)
    print(f"  mAP@0.50:0.95  : {summary.get('map',        0):.4f}")
    print(f"  mAP@0.50       : {summary.get('map_50',     0):.4f}")
    print(f"  mAP@0.75       : {summary.get('map_75',     0):.4f}")
    print(f"  mAP small      : {summary.get('map_small',  0):.4f}")
    print(f"  mAP medium     : {summary.get('map_medium', 0):.4f}")
    print(f"  mAP large      : {summary.get('map_large',  0):.4f}")
    print("-" * 50)
    print("  Per-class AP@0.50:0.95:")
    for cls, ap in per_class_ap.items():
        print(f"    {cls:<12}: {ap:.4f}")
    print("=" * 50)

    model_label = "Faster R-CNN (COCO pretrained)" if use_coco else "Faster R-CNN (no COCO)"
    results_file = args.results_file or RESULTS_FILE
    output = {
        "model":        model_label,
        "split":        args.split,
        "checkpoint":   args.checkpoint,
        "metrics":      summary,
        "per_class_ap": per_class_ap,
    }
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_file}")


# ===========================================================================
# Comparison Chart
# ===========================================================================

YOLO_RESULTS = {
    "name":        "YOLOv8s\n(COCO pretrained)",
    "map50_val":   0.949,
    "map50_test":  None,
    "map5095":     None,
    "helmet_ap50": 0.961,
    "head_ap50":   0.937,
    "fps":         103,
    "color":       "#3498db",
}

RCNN_NOCOCO_RESULTS = {
    "name":        "Faster R-CNN\n(no COCO)",
    "map50_val":   0.8612,
    "map50_test":  0.7733,
    "map5095":     0.4409,
    "helmet_ap50": 0.4725,
    "head_ap50":   0.4093,
    "fps":         None,
    "color":       "#e74c3c",
}


def make_chart(models, out="comparison.png"):
    n      = len(models)
    names  = [m["name"] for m in models]
    colors = [m["color"] for m in models]
    x      = np.arange(n)
    w      = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("Model Comparison — Helmet Detection", fontsize=14, fontweight="bold")

    # Left: mAP@0.50 val vs test
    ax = axes[0]
    for i, m in enumerate(models):
        val_v  = m["map50_val"]  or 0
        test_v = m["map50_test"] or 0
        b1 = ax.bar(i - w/2, val_v,  w, color=m["color"], alpha=0.9,
                    label="Val"  if i == 0 else "")
        b2 = ax.bar(i + w/2, test_v, w, color=m["color"], alpha=0.55,
                    label="Test" if i == 0 else "",
                    hatch="//", edgecolor="white")
        for bar, v, orig in [(b1, val_v, m["map50_val"]), (b2, test_v, m["map50_test"])]:
            lbl = f"{v:.3f}" if orig is not None else "—"
            ax.text(bar[0].get_x() + bar[0].get_width() / 2,
                    bar[0].get_height() + 0.005,
                    lbl, ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("mAP@0.50")
    ax.set_title("mAP@0.50  (solid=Val, hatched=Test)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Middle: mAP@0.50:0.95
    ax2   = axes[1]
    vals  = [m["map5095"] for m in models]
    bars2 = ax2.bar(x, [v or 0 for v in vals], 0.55, color=colors, alpha=0.85,
                    edgecolor="white")
    for bar, v in zip(bars2, vals):
        lbl = f"{v:.3f}" if v is not None else "N/A"
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.005,
                 lbl, ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=9)
    ax2.set_ylim(0, 1.08)
    ax2.set_ylabel("mAP@0.50:0.95")
    ax2.set_title("mAP@0.50:0.95 (Test)")
    ax2.grid(axis="y", alpha=0.3)

    # Right: per-class mAP@0.50
    ax3     = axes[2]
    xc      = np.arange(2)
    wc      = 0.25
    offsets = np.linspace(-(n - 1) * wc / 2, (n - 1) * wc / 2, n)
    for i, m in enumerate(models):
        hv    = m.get("helmet_ap50")
        dv    = m.get("head_ap50")
        vc    = [hv or 0, dv or 0]
        bars3 = ax3.bar(xc + offsets[i], vc, wc,
                        label=m["name"].replace("\n", " "),
                        color=m["color"], alpha=0.85, edgecolor="white")
        for bar, v, orig in zip(bars3, vc, [hv, dv]):
            lbl = f"{v:.3f}" if orig is not None else "N/A"
            ax3.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.005,
                     lbl, ha="center", va="bottom", fontsize=8)
    ax3.set_xticks(xc)
    ax3.set_xticklabels(["Helmet", "Head (No Helmet)"], fontsize=10)
    ax3.set_ylim(0, 1.08)
    ax3.set_ylabel("mAP@0.50")
    ax3.set_title("Per-Class mAP@0.50")
    ax3.legend(fontsize=7, loc="lower right")
    ax3.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")

    print("\n" + "=" * 72)
    print(f"  {'Model':<28} {'Val mAP@0.50':>13} {'Test mAP@0.50':>14} {'FPS':>6}")
    print("=" * 72)
    for m in models:
        fps_s  = f"{int(m['fps'])}" if m["fps"] else "—"
        val_s  = f"{m['map50_val']:.4f}"  if m["map50_val"]  is not None else "N/A"
        test_s = f"{m['map50_test']:.4f}" if m["map50_test"] is not None else "N/A"
        print(f"  {m['name'].replace(chr(10), ' '):<28} {val_s:>13} {test_s:>14} {fps_s:>6}")
    print("=" * 72)


def cmd_compare(args):
    models = [YOLO_RESULTS, RCNN_NOCOCO_RESULTS]

    if args.rcnn_coco_map50_val is not None or args.rcnn_coco_map50_test is not None:
        rcnn_coco = {
            "name":        "Faster R-CNN\n(COCO pretrained)",
            "map50_val":   args.rcnn_coco_map50_val,
            "map50_test":  args.rcnn_coco_map50_test,
            "map5095":     args.rcnn_coco_map,
            "helmet_ap50": args.rcnn_coco_helmet_ap50,
            "head_ap50":   args.rcnn_coco_head_ap50,
            "fps":         args.rcnn_coco_fps,
            "color":       "#2ecc71",
        }
        models.insert(1, rcnn_coco)

    make_chart(models, out=args.output)


# ===========================================================================
# Inference
# ===========================================================================

CLASS_COLORS = {1: "lime", 2: "red"}


def predict(model, image_path, device, threshold=0.5):
    image  = Image.open(image_path).convert("RGB")
    tensor = TF.to_tensor(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(tensor)[0]
    keep = output["scores"] >= threshold
    return image, {
        "boxes":  output["boxes"][keep].cpu(),
        "labels": output["labels"][keep].cpu(),
        "scores": output["scores"][keep].cpu(),
    }


def visualize(image, prediction, title="", save_path=None):
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(image)
    for box, label, score in zip(
        prediction["boxes"], prediction["labels"], prediction["scores"]
    ):
        x1, y1, x2, y2 = box.tolist()
        cls_name = CLASS_NAMES[label.item()]
        color    = CLASS_COLORS.get(label.item(), "yellow")
        rect     = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x1, y1 - 4, f"{cls_name} {score:.2f}",
            color=color, fontsize=9, fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.4, pad=1, edgecolor="none"),
        )
    helmet_count = (prediction["labels"] == 1).sum().item()
    head_count   = (prediction["labels"] == 2).sum().item()
    ax.set_title(f"{title}\nHelmet: {helmet_count}  No-Helmet (head): {head_count}", fontsize=11)
    ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def cmd_inference(args):
    device = get_device()
    ckpt   = torch.load(args.checkpoint, map_location=device)
    model  = build_model(use_coco=ckpt.get("use_coco", False))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    print(f"[Model] Loaded epoch {ckpt.get('epoch', '?')} from {args.checkpoint}")

    images_to_process = []
    if args.image:
        images_to_process.append(args.image)
    if args.test_samples > 0:
        _, _, test_files = get_splits()
        chosen = random.sample(test_files, min(args.test_samples, len(test_files)))
        images_to_process += [os.path.join(IMAGES_DIR, f + ".png") for f in chosen]

    if not images_to_process:
        print("No images specified. Use --image or --test-samples.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    for img_path in images_to_process:
        print(f"Processing: {img_path}")
        image, pred = predict(model, img_path, device, threshold=args.threshold)
        base        = os.path.splitext(os.path.basename(img_path))[0]
        save_path   = os.path.join(args.output_dir, base + "_pred.png")
        visualize(image, pred, title=base, save_path=save_path)


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Helmet Detection — Faster R-CNN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # ---- train ----
    p_train = sub.add_parser("train", help="Train the model")
    p_train.add_argument("--epochs",      type=int,   default=NUM_EPOCHS)
    p_train.add_argument("--batch",       type=int,   default=BATCH_SIZE)
    p_train.add_argument("--lr",          type=float, default=LEARNING_RATE)
    p_train.add_argument("--eval-freq",   type=int,   default=EVAL_EVERY_N_EPOCHS,
                         help="Compute val mAP every N epochs (0=skip)")
    p_train.add_argument("--coco",        action="store_true",
                         help="Fine-tune COCO pretrained model")
    p_train.add_argument("--no-imagenet", action="store_true",
                         help="Train completely from scratch")
    p_train.add_argument("--no-yolo",     action="store_true",
                         help="Use archive dataset only")
    p_train.add_argument("--device",      type=str, default=None)
    p_train.add_argument("--resume",      type=str, default=None)

    # ---- evaluate ----
    p_eval = sub.add_parser("evaluate", help="Evaluate on val or test split")
    p_eval.add_argument("--checkpoint",      type=str,
                        default=os.path.join(CHECKPOINT_DIR, "best_model.pth"))
    p_eval.add_argument("--split",           choices=["val", "test"], default="test")
    p_eval.add_argument("--score-threshold", type=float, default=0.5)
    p_eval.add_argument("--no-yolo",         action="store_true")
    p_eval.add_argument("--device",          type=str, default=None)
    p_eval.add_argument("--results-file",    type=str, default=None,
                        help="Override output JSON path")

    # ---- compare ----
    p_cmp = sub.add_parser("compare", help="Generate comparison chart")
    p_cmp.add_argument("--rcnn-coco-map50-val",   type=float, default=None)
    p_cmp.add_argument("--rcnn-coco-map50-test",  type=float, default=None)
    p_cmp.add_argument("--rcnn-coco-map",         type=float, default=None)
    p_cmp.add_argument("--rcnn-coco-helmet-ap50", type=float, default=None)
    p_cmp.add_argument("--rcnn-coco-head-ap50",   type=float, default=None)
    p_cmp.add_argument("--rcnn-coco-fps",         type=float, default=None)
    p_cmp.add_argument("--output",                type=str,   default="comparison.png")

    # ---- inference ----
    p_inf = sub.add_parser("inference", help="Run inference on images")
    p_inf.add_argument("--image",        type=str, default=None)
    p_inf.add_argument("--checkpoint",   type=str,
                       default=os.path.join(CHECKPOINT_DIR, "best_model.pth"))
    p_inf.add_argument("--threshold",    type=float, default=0.5)
    p_inf.add_argument("--test-samples", type=int,   default=0)
    p_inf.add_argument("--output-dir",   type=str,   default="./inference_output")

    args = parser.parse_args()

    if args.mode == "train":
        cmd_train(args)
    elif args.mode == "evaluate":
        cmd_evaluate(args)
    elif args.mode == "compare":
        cmd_compare(args)
    elif args.mode == "inference":
        cmd_inference(args)


if __name__ == "__main__":
    main()
