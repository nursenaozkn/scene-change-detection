"""
Inference and change visualization for the scene change detection models.

Given a pair of images (a "before" image A and an "after" image B), the
trained model predicts a binary change mask. The script then:
  - localizes the changed regions,
  - interprets them (changed area %, number of distinct change regions,
    and IoU / F1 against the ground-truth mask when it is available),
  - saves a side-by-side visualization with the changes highlighted in red.

By default it runs on two sample pairs from the validation set, satisfying
the task requirement of demonstrating detected changes on selected images.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # headless backend so it works inside Docker
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.models.simple_model import SimpleChangeNet
from src.models.siamese_model import SiameseChangeNet
from src.models.siamese_unet_model import SiameseUNet

MODELS = {
    "simple": SimpleChangeNet,
    "siamese": SiameseChangeNet,
    "siamese_unet": SiameseUNet,
}


def load_image(path):
    """Read an image from disk and return it as an RGB uint8 array."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_tensor(img_rgb):
    """Convert an RGB uint8 array to a normalized [1, 3, H, W] tensor."""
    t = img_rgb.astype("float32") / 255.0
    return torch.from_numpy(t).permute(2, 0, 1).unsqueeze(0)


def predict_mask(model, img_a, img_b, device, threshold):
    """Run the model on one image pair and return (probability, binary mask)."""
    a = to_tensor(img_a).to(device)
    b = to_tensor(img_b).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(a, b))[0, 0].cpu().numpy()
    binary = (prob >= threshold).astype("uint8")
    return prob, binary


def interpret(binary):
    """Summarize a binary change mask into human-readable statistics."""
    changed = int(binary.sum())
    pct = 100.0 * changed / binary.size
    num_labels, _ = cv2.connectedComponents(binary)
    regions = max(num_labels - 1, 0)  # subtract the background label
    return changed, pct, regions


def metrics_vs_gt(pred, gt):
    """Compute IoU and F1 of the prediction against a ground-truth mask."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = intersection / union if union > 0 else 1.0
    denom = pred.sum() + gt.sum()
    f1 = (2 * intersection / denom) if denom > 0 else 1.0
    return iou, f1


def overlay_mask(img_rgb, binary, color=(255, 0, 0), alpha=0.5):
    """Blend the changed pixels of `binary` onto `img_rgb` in the given color."""
    overlay = img_rgb.copy()
    mask = binary.astype(bool)
    overlay[mask] = (alpha * np.array(color) + (1 - alpha) * overlay[mask]).astype("uint8")
    return overlay


def visualize(name, img_a, img_b, prob, binary, gt, out_path):
    """Save a side-by-side figure of the inputs, prediction and overlay."""
    panels = [("Image A (before)", img_a), ("Image B (after)", img_b)]
    if gt is not None:
        panels.append(("Ground truth", gt * 255))
    panels.append(("Predicted change", binary * 255))
    panels.append(("Change overlay on B", overlay_mask(img_b, binary)))

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4.5))
    for ax, (title, image) in zip(axes, panels):
        cmap = "gray" if image.ndim == 2 else None
        ax.imshow(image, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"Scene Change Detection - {name}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def collect_sample_pairs(data_root, split, num_samples):
    """Pick sample pairs from the dataset, preferring ones that contain change."""
    split_dir = Path(data_root) / split
    a_dir, b_dir, label_dir = split_dir / "A", split_dir / "B", split_dir / "label"
    if not a_dir.is_dir():
        raise FileNotFoundError(f"Dataset split not found: {split_dir}")

    files = sorted(a_dir.glob("*"), key=lambda f: int(f.stem) if f.stem.isdigit() else 0)
    with_change, without_change = [], []
    for f in files:
        label_path = label_dir / f.name
        gt = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE) if label_path.exists() else None
        target = with_change if (gt is not None and gt.sum() > 0) else without_change
        target.append((f.name, a_dir / f.name, b_dir / f.name, label_path))
        if len(with_change) >= num_samples:
            break

    chosen = (with_change + without_change)[:num_samples]
    if not chosen:
        raise FileNotFoundError(f"No images found in {a_dir}")
    return chosen


def parse_args():
    parser = argparse.ArgumentParser(description="Run change detection inference and visualize the result.")
    parser.add_argument("--model", choices=list(MODELS), default="siamese_unet",
                        help="Model architecture (default: siamese_unet).")
    parser.add_argument("--checkpoint",
                        default=str(PROJECT_ROOT / "outputs" / "checkpoints" / "best_siamese_unet_model.pth"),
                        help="Path to the trained model weights.")
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data" / "raw" / "building-change"),
                        help="Dataset root (used when no custom image pair is given).")
    parser.add_argument("--split", default="val", help="Dataset split to sample from (default: val).")
    parser.add_argument("--num-samples", type=int, default=2,
                        help="Number of sample pairs to run when no custom pair is given.")
    parser.add_argument("--image-a", help="Path to a custom 'before' image.")
    parser.add_argument("--image-b", help="Path to a custom 'after' image.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold for a pixel to count as changed.")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "predictions"),
                        help="Directory where visualizations are saved.")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MODELS[args.model]().to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded {args.model} model from {args.checkpoint} (device: {device})")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build the list of pairs to run: either one custom pair, or samples from the dataset.
    if args.image_a and args.image_b:
        pairs = [("custom", Path(args.image_a), Path(args.image_b), None)]
    elif args.image_a or args.image_b:
        raise SystemExit("Provide both --image-a and --image-b, or neither.")
    else:
        pairs = collect_sample_pairs(args.data_root, args.split, args.num_samples)

    for name, a_path, b_path, label_path in pairs:
        stem = Path(name).stem
        img_a = load_image(a_path)
        img_b = load_image(b_path)

        gt = None
        if label_path is not None and Path(label_path).exists():
            gt = (cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE) > 0).astype("uint8")

        prob, binary = predict_mask(model, img_a, img_b, device, args.threshold)
        changed, pct, regions = interpret(binary)

        print(f"\n[{stem}] detected change")
        print(f"  changed pixels : {changed} ({pct:.2f}% of the image)")
        print(f"  change regions : {regions}")
        if gt is not None:
            iou, f1 = metrics_vs_gt(binary, gt)
            print(f"  IoU vs GT      : {iou:.3f}")
            print(f"  F1  vs GT      : {f1:.3f}")

        out_path = output_dir / f"prediction_{stem}.png"
        visualize(stem, img_a, img_b, prob, binary, gt, out_path)
        print(f"  visualization  : {out_path}")

    print(f"\nDone. {len(pairs)} visualization(s) saved to {output_dir}")


if __name__ == "__main__":
    main()
