import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(r"c:\Users\Abra\scene-change-detection")

from src.datasets.scd_dataset import SCDDataset
from src.models.simple_model import SimpleChangeNet
from src.utils.losses import combined_loss


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


def main():
    root_dir = r"c:\Users\Abra\scene-change-detection\data\raw\building-change"
    checkpoint_dir = Path(r"c:\Users\Abra\scene-change-detection\outputs\checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    batch_size = 4
    lr = 1e-3
    num_epochs = 6

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_dataset = SCDDataset(root_dir=root_dir, split="train")
    val_dataset = SCDDataset(root_dir=root_dir, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    model = SimpleChangeNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate_one_epoch(model, val_loader, device)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = checkpoint_dir / "best_model.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Best model saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()