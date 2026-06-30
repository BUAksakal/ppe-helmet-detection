import os
import json
import random
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset, ConcatDataset
from PIL import Image
import torchvision.transforms.functional as TF

import config


class HelmetDataset(Dataset):
    """Archive dataset: Pascal VOC XML annotations, 416x416 PNG images."""

    def __init__(self, file_list, augment=False):
        self.file_list = file_list
        self.augment   = augment

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        base     = self.file_list[idx]
        img_path = os.path.join(config.IMAGES_DIR,      base + ".png")
        ann_path = os.path.join(config.ANNOTATIONS_DIR, base + ".xml")

        image          = Image.open(img_path).convert("RGB")
        boxes, labels  = _parse_voc_xml(ann_path)

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
    YOLOv8-format dataset from the ppe-detection.yolov8 Roboflow export.
    YOLO label file: <class_id> <cx> <cy> <w> <h>  (all normalized 0-1)
    Class mapping:  0 → helmet (1), 1 → head/no-helmet (2)
    """

    def __init__(self, images_dir, labels_dir, augment=False, id_offset=0):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.augment    = augment
        self.id_offset  = id_offset

        self.samples = []
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


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _parse_voc_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes, labels = [], []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip().lower()
        if name not in config.CLASS_MAP:
            continue
        b    = obj.find("bndbox")
        xmin = float(b.find("xmin").text)
        ymin = float(b.find("ymin").text)
        xmax = float(b.find("xmax").text)
        ymax = float(b.find("ymax").text)
        if xmax > xmin and ymax > ymin:
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(config.CLASS_MAP[name])
    return boxes, labels


def _parse_yolo_label(label_path, img_w, img_h):
    """Convert normalized YOLO coords → pixel [xmin, ymin, xmax, ymax]."""
    boxes, labels = [], []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            if cls_id not in config.YOLO_CLASS_MAP:
                continue
            cx, cy, bw, bh = map(float, parts[1:5])
            xmin = max(0.0, (cx - bw / 2) * img_w)
            ymin = max(0.0, (cy - bh / 2) * img_h)
            xmax = min(img_w, (cx + bw / 2) * img_w)
            ymax = min(img_h, (cy + bh / 2) * img_h)
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(config.YOLO_CLASS_MAP[cls_id])
    return boxes, labels


def _hflip(image, boxes):
    w     = image.width
    image = TF.hflip(image)
    boxes = boxes.clone()
    boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
    return image, boxes


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def get_splits():
    """Load or create archive-only 80/10/10 splits (persisted to disk)."""
    if os.path.exists(config.SPLITS_FILE):
        with open(config.SPLITS_FILE) as f:
            splits = json.load(f)
        return splits["train"], splits["val"], splits["test"]

    all_files = sorted(
        f.replace(".png", "")
        for f in os.listdir(config.IMAGES_DIR)
        if f.endswith(".png")
    )

    rng      = random.Random(config.RANDOM_SEED)
    shuffled = all_files.copy()
    rng.shuffle(shuffled)

    n         = len(shuffled)
    train_end = int(n * config.TRAIN_RATIO)
    val_end   = int(n * (config.TRAIN_RATIO + config.VAL_RATIO))

    train = shuffled[:train_end]
    val   = shuffled[train_end:val_end]
    test  = shuffled[val_end:]

    with open(config.SPLITS_FILE, "w") as f:
        json.dump({"train": train, "val": val, "test": test}, f, indent=2)

    print(f"Splits saved to {config.SPLITS_FILE}")
    return train, val, test


def get_datasets(augment_train=True, use_yolo=None):
    """
    Returns (train_dataset, val_dataset, test_dataset).

    If use_yolo is None it reads config.USE_YOLO_DATASET.
    When enabled the YOLOv8 splits are merged into the corresponding
    archive splits so validation/test sets are also larger.
    """
    if use_yolo is None:
        use_yolo = config.USE_YOLO_DATASET

    train_files, val_files, test_files = get_splits()

    archive_train = HelmetDataset(train_files, augment=augment_train)
    archive_val   = HelmetDataset(val_files,   augment=False)
    archive_test  = HelmetDataset(test_files,  augment=False)

    if use_yolo:
        yolo_train = YoloHelmetDataset(
            images_dir=os.path.join(config.YOLO_ROOT, "train", "images"),
            labels_dir=os.path.join(config.YOLO_ROOT, "train", "labels"),
            augment=augment_train, id_offset=100_000,
        )
        yolo_val = YoloHelmetDataset(
            images_dir=os.path.join(config.YOLO_ROOT, "valid", "images"),
            labels_dir=os.path.join(config.YOLO_ROOT, "valid", "labels"),
            augment=False, id_offset=200_000,
        )
        yolo_test = YoloHelmetDataset(
            images_dir=os.path.join(config.YOLO_ROOT, "test",  "images"),
            labels_dir=os.path.join(config.YOLO_ROOT, "test",  "labels"),
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
        print(
            f"[Data] Archive only  "
            f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}"
        )

    return train_ds, val_ds, test_ds


def collate_fn(batch):
    return tuple(zip(*batch))
