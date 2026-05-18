"""
Training entry point for the scene change detection models.

Trains one of the available architectures (simple / siamese / siamese_unet),
logs the train/validation loss to a CSV file and to TensorBoard, and saves
the best checkpoint (lowest validation loss).
"""
import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# TensorBoard is optional: the loss is always written to a CSV file, and
# additionally streamed to TensorBoard when the package is importable.
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except Exception:
    TENSORBOARD_AVAILABLE = False

# Make the project importable regardless of the working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.datasets.scd_dataset import SCDDataset
from src.models.simple_model import SimpleChangeNet
from src.models.siamese_model import SiameseChangeNet
from src.models.siamese_unet_model import SiameseUNet
from src.utils.losses import combined_loss

MODELS = {
    "simple": SimpleChangeNet,
    "siamese": SiameseChangeNet,
    "siamese_unet": SiameseUNet,
}


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0

    for batch in tqdm(loader, desc="Training", leave=False):
        image_a = batch["image_a"].to(device)
        image_b = batch["image_b"].to(device)
        mask = batch["mask"].to(device)

        optimizer.zero_grad()

        output = model(image_a, image_b)
        loss = combined_loss(output, mask)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate_one_epoch(model, loader, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            image_a = batch["image_a"].to(device)
            image_b = batch["image_b"].to(device)
            mask = batch["mask"].to(device)

            output = model(image_a, image_b)
            loss = combined_loss(output, mask)

            running_loss += loss.item()

    return running_loss / len(loader)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a scene change detection model.")
    parser.add_argument("--model", choices=list(MODELS), default="siamese_unet",
                        help="Model architecture to train (default: siamese_unet).")
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data" / "raw" / "building-change"),
                        help="Dataset root containing train/ and val/ folders.")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs"),
                        help="Directory for checkpoints and logs.")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs" / args.model
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model: {args.model} | Device: {device}")

    train_dataset = SCDDataset(root_dir=args.data_root, split="train")
    val_dataset = SCDDataset(root_dir=args.data_root, split="val")
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    model = MODELS[args.model]().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Loss is always logged to a CSV file, and to TensorBoard when available.
    writer = SummaryWriter(log_dir=str(log_dir)) if TENSORBOARD_AVAILABLE else None
    if writer is None:
        print("TensorBoard unavailable - logging losses to CSV only.")
    metrics_path = log_dir / "metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss"])

    checkpoint_path = checkpoint_dir / f"best_{args.model}_model.pth"
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch [{epoch}/{args.epochs}]")

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate_one_epoch(model, val_loader, device)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")

        if writer is not None:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.4f}", f"{val_loss:.4f}"])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Best model saved to: {checkpoint_path}")

    if writer is not None:
        writer.close()
    print(f"\nTraining finished. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Loss log:   {metrics_path}")


if __name__ == "__main__":
    main()
