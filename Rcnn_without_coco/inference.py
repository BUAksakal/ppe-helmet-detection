"""
Run inference on individual images and visualize bounding boxes.
Usage:
    python inference.py --image path/to/image.png
    python inference.py --image path/to/image.png --threshold 0.6
    python inference.py --test-samples 5        # Random samples from test split
"""
import argparse
import os
import random

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torchvision.transforms.functional as TF

import config
from dataset import get_splits
from model import build_model

# Colors per class (background excluded)
CLASS_COLORS = {1: "lime", 2: "red"}   # 1=helmet, 2=head


def predict(model, image_path, device, threshold=0.5):
    image = Image.open(image_path).convert("RGB")
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
        cls_name = config.CLASS_NAMES[label.item()]
        color = CLASS_COLORS.get(label.item(), "yellow")

        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor="none"
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",        type=str,   default=None)
    parser.add_argument("--checkpoint",   type=str,
                        default=os.path.join(config.CHECKPOINT_DIR, "best_model.pth"))
    parser.add_argument("--threshold",    type=float, default=0.5)
    parser.add_argument("--test-samples", type=int,   default=0,
                        help="Randomly pick N images from the test split")
    parser.add_argument("--output-dir",   type=str,   default="./inference_output")
    args = parser.parse_args()

    device = config.get_device()
    model  = build_model()
    ckpt   = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    print(f"[Model] Loaded epoch {ckpt.get('epoch', '?')} from {args.checkpoint}")

    images_to_process = []

    if args.image:
        images_to_process.append(args.image)

    if args.test_samples > 0:
        _, _, test_files = get_splits()
        chosen = random.sample(test_files, min(args.test_samples, len(test_files)))
        for f in chosen:
            images_to_process.append(os.path.join(config.IMAGES_DIR, f + ".png"))

    if not images_to_process:
        print("No images specified. Use --image or --test-samples.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    for img_path in images_to_process:
        print(f"Processing: {img_path}")
        image, pred = predict(model, img_path, device, threshold=args.threshold)
        base = os.path.splitext(os.path.basename(img_path))[0]
        save_path = os.path.join(args.output_dir, base + "_pred.png")
        visualize(image, pred, title=base, save_path=save_path)


if __name__ == "__main__":
    main()
