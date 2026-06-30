"""
Train Faster R-CNN (no COCO) for helmet detection.

Usage:
    python train.py                          # 30 epochs, combined dataset
    python train.py --epochs 50 --eval-freq 5
    python train.py --no-imagenet            # completely from scratch
    python train.py --no-yolo                # archive dataset only
    python train.py --resume checkpoints/checkpoint_epoch010.pth
"""
import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import config
from dataset import get_datasets, collate_fn
from model import build_model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, optimizer, loader, device, epoch, total_epochs,
                    warmup_scheduler=None):
    model.train()
    total_loss = 0.0
    valid_steps = 0
    t0 = time.time()

    for step, (images, targets) in enumerate(loader, 1):
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss      = sum(loss_dict.values())

        # Skip NaN/Inf batches instead of crashing
        if not torch.isfinite(loss):
            print(f"  [skip] non-finite loss at step {step}, skipping batch")
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Warmup: step LR scheduler every iteration during warmup
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


# ---------------------------------------------------------------------------
# Validation mAP
# ---------------------------------------------------------------------------

def compute_map(model, loader, device):
    """Return dict with map, map_50, map_75 and per-class AP."""
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)

    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            preds = [
                {"boxes": o["boxes"].cpu(), "scores": o["scores"].cpu(), "labels": o["labels"].cpu()}
                for o in outputs
            ]
            gts = [
                {"boxes": t["boxes"].cpu(), "labels": t["labels"].cpu()}
                for t in targets
            ]
            metric.update(preds, gts)

    raw = metric.compute()
    result = {k: float(v) for k, v in raw.items() if hasattr(v, "numel") and v.numel() == 1}

    per_class = {}
    if "map_per_class" in raw:
        for i, ap in enumerate(raw["map_per_class"]):
            cls_name = config.CLASS_NAMES[i + 1]
            per_class[cls_name] = float(ap)
    result["per_class"] = per_class

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",      type=int,   default=config.NUM_EPOCHS)
    parser.add_argument("--batch",       type=int,   default=config.BATCH_SIZE)
    parser.add_argument("--lr",          type=float, default=config.LEARNING_RATE)
    parser.add_argument("--eval-freq",   type=int,   default=config.EVAL_EVERY_N_EPOCHS,
                        help="Compute val mAP every N epochs (0 = skip)")
    parser.add_argument("--no-imagenet", action="store_true",
                        help="Train completely from scratch (no ImageNet backbone)")
    parser.add_argument("--coco",        action="store_true",
                        help="Fine-tune COCO pretrained model (replaces detection head)")
    parser.add_argument("--no-yolo",     action="store_true",
                        help="Use archive dataset only (skip YOLOv8 data)")
    parser.add_argument("--device",      type=str,   default=None,
                        help="Force device: cpu | mps | cuda (default: auto)")
    parser.add_argument("--resume",      type=str,   default=None)
    args = parser.parse_args()

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    device = torch.device(args.device) if args.device else config.get_device()
    print(f"[Device] {device}")

    use_yolo      = not args.no_yolo
    use_imagenet  = not args.no_imagenet
    use_coco      = args.coco

    # COCO run → ayrı checkpoint klasörü (no-COCO sonuçlarını ezmemek için)
    if use_coco:
        config.CHECKPOINT_DIR = "./checkpoints_coco"
        config.RESULTS_FILE   = "./results_coco.json"
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    train_ds, val_ds, _ = get_datasets(augment_train=True, use_yolo=use_yolo)

    # num_workers=0: avoids multiprocessing hangs with MPS on macOS
    train_loader = DataLoader(
        train_ds, batch_size=args.batch,
        shuffle=True, num_workers=0, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1,
        shuffle=False, num_workers=0, collate_fn=collate_fn,
    )

    model = build_model(use_imagenet_backbone=use_imagenet, use_coco=use_coco)
    model.to(device)

    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr,
        momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY,
    )

    # Linear warmup: LR ramps from lr/100 → lr over the first WARMUP_STEPS steps.
    # This prevents gradient explosion from the randomly initialized detection head.
    WARMUP_STEPS = 500
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,   # start at 1% of target LR
        end_factor=1.0,
        total_iters=WARMUP_STEPS,
    )

    # After warmup, step-decay LR every LR_STEP_SIZE epochs
    main_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.LR_STEP_SIZE, gamma=config.LR_GAMMA,
    )

    start_epoch     = 1
    best_map50      = -1.0
    best_loss       = float("inf")
    history         = []
    global_step     = 0   # tracks total steps for warmup cutoff

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch  = ckpt["epoch"] + 1
        best_map50   = ckpt.get("map_50", -1.0)
        best_loss    = ckpt.get("loss", float("inf"))
        history      = ckpt.get("history", [])
        global_step  = ckpt.get("global_step", WARMUP_STEPS + 1)
        print(f"[Resume] epoch={ckpt['epoch']}  map50={best_map50:.4f}  loss={best_loss:.4f}")

    steps_per_epoch = len(train_loader)

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'='*65}")

        # Pass warmup_scheduler only while still in warmup window
        active_warmup = warmup_scheduler if global_step < WARMUP_STEPS else None

        avg_loss = train_one_epoch(
            model, optimizer, train_loader, device, epoch, args.epochs,
            warmup_scheduler=active_warmup,
        )
        global_step += steps_per_epoch

        # Step-decay only after warmup is complete
        if global_step >= WARMUP_STEPS:
            main_scheduler.step()

        lr_now     = optimizer.param_groups[0]["lr"]
        epoch_info = {"epoch": epoch, "loss": avg_loss, "lr": lr_now, "global_step": global_step}

        # --- Optional per-epoch mAP ---
        map50 = None
        if args.eval_freq > 0 and epoch % args.eval_freq == 0:
            print(f"  Computing val mAP (epoch {epoch})...")
            t_eval = time.time()
            map_result = compute_map(model, val_loader, device)
            map50 = map_result.get("map_50", 0.0)
            epoch_info["map"]         = map_result.get("map",    0.0)
            epoch_info["map_50"]      = map50
            epoch_info["map_75"]      = map_result.get("map_75", 0.0)
            epoch_info["per_class"]   = map_result.get("per_class", {})
            print(
                f"  Val mAP@0.50={map50:.4f}  "
                f"mAP@0.50:0.95={map_result.get('map',0):.4f}  "
                f"({time.time()-t_eval:.0f}s)"
            )
            if map_result.get("per_class"):
                for cls, ap in map_result["per_class"].items():
                    print(f"    {cls}: {ap:.4f}")

        print(
            f"Epoch [{epoch}/{args.epochs}]  Loss={avg_loss:.4f}  "
            f"LR={lr_now:.6f}"
            + (f"  mAP@0.5={map50:.4f}" if map50 is not None else "")
        )
        history.append(epoch_info)

        # --- Checkpointing ---
        ckpt_data = {
            "epoch":               epoch,
            "model_state_dict":    model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss":                avg_loss,
            "map_50":              map50 if map50 is not None else -1.0,
            "use_imagenet":        use_imagenet,
            "use_yolo":            use_yolo,
            "global_step":         global_step,
            "history":             history,
        }

        # Save every-5-epoch checkpoint
        if epoch % 5 == 0:
            ckpt_path = os.path.join(
                config.CHECKPOINT_DIR, f"checkpoint_epoch{epoch:03d}.pth"
            )
            torch.save(ckpt_data, ckpt_path)

        # Save best model: prefer best mAP@0.5, fall back to best loss
        is_best = False
        if map50 is not None and map50 > best_map50:
            best_map50 = map50
            is_best    = True
        elif map50 is None and avg_loss < best_loss:
            best_loss = avg_loss
            is_best   = True

        if is_best:
            torch.save(ckpt_data, os.path.join(config.CHECKPOINT_DIR, "best_model.pth"))
            marker = f"mAP@0.5={best_map50:.4f}" if map50 is not None else f"loss={avg_loss:.4f}"
            print(f"  -> Best model saved  ({marker})")

    # Save training history
    history_path = os.path.join(config.CHECKPOINT_DIR, "train_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete. History saved to {history_path}")
    print(f"Best mAP@0.5 on val: {best_map50:.4f}" if best_map50 >= 0 else "")


if __name__ == "__main__":
    main()
