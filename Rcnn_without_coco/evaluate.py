"""
Evaluate Faster R-CNN on val or test split and compute mAP.

Usage:
    python evaluate.py
    python evaluate.py --split val
    python evaluate.py --checkpoint checkpoints/best_model.pth
    python evaluate.py --no-yolo   # evaluate on archive test set only
"""
import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import config
from dataset import get_datasets, collate_fn
from model import build_model


def evaluate(model, loader, device):
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)

    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            preds = [
                {
                    "boxes":  o["boxes"].cpu(),
                    "scores": o["scores"].cpu(),
                    "labels": o["labels"].cpu(),
                }
                for o in outputs
            ]
            gts = [
                {
                    "boxes":  t["boxes"].cpu(),
                    "labels": t["labels"].cpu(),
                }
                for t in targets
            ]
            metric.update(preds, gts)

    return metric.compute()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",      type=str,
                        default=os.path.join(config.CHECKPOINT_DIR, "best_model.pth"))
    parser.add_argument("--split",           choices=["val", "test"], default="test")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--no-yolo",         action="store_true")
    parser.add_argument("--device",          type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else config.get_device()
    print(f"[Device] {device}")

    use_yolo = not args.no_yolo
    _, val_ds, test_ds = get_datasets(augment_train=False, use_yolo=use_yolo)
    dataset = val_ds if args.split == "val" else test_ds
    print(f"[Eval]  split={args.split}  n={len(dataset)}")

    # num_workers=0: avoids macOS multiprocessing hangs with MPS
    loader = DataLoader(
        dataset, batch_size=1,
        shuffle=False, num_workers=0, collate_fn=collate_fn,
    )

    model = build_model()
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    print(f"[Model] epoch={ckpt.get('epoch','?')}  {args.checkpoint}")

    model.roi_heads.score_thresh = args.score_threshold

    print("\nRunning evaluation...")
    results = evaluate(model, loader, device)

    summary      = {k: float(v) for k, v in results.items() if hasattr(v, "numel") and v.numel() == 1}
    per_class_ap = {}
    if "map_per_class" in results:
        for i, ap in enumerate(results["map_per_class"]):
            cls_name = config.CLASS_NAMES[i + 1]
            per_class_ap[cls_name] = float(ap)

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

    output = {
        "model":        "Faster R-CNN (no COCO)",
        "split":        args.split,
        "checkpoint":   args.checkpoint,
        "metrics":      summary,
        "per_class_ap": per_class_ap,
    }
    with open(config.RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {config.RESULTS_FILE}")


if __name__ == "__main__":
    main()
