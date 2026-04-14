"""
Amaç:
Dataset’ten tek tek sample değil, batch batch veri almak
training sırasında kullanacağımız yapıyı kurmak
shape’lerin doğru olduğundan emin olmak
DataLoader bunu yapar:
veriyi batch’ler
shuffle yapabilir
training’i hızlandırır
"""
import matplotlib.pyplot as plt
import sys
sys.path.append(r"c:\Users\Abra\scene-change-detection")

from torch.utils.data import DataLoader
from src.datasets.scd_dataset import SCDDataset

# Train dataset
train_dataset = SCDDataset(
    root_dir=r"c:\Users\Abra\scene-change-detection\data\raw\building-change",
    split="train"
)

# Val dataset
val_dataset = SCDDataset(
    root_dir=r"c:\Users\Abra\scene-change-detection\data\raw\building-change",
    split="val"
)

# DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0
)
"""
shuffle=True neden var?
Train loader’da: shuffle=True çünkü model her epoch’ta veriyi farklı sırada görsün isteriz.
Val loader’da: shuffle=False çünkü validation sabit ve tutarlı olsun isteriz.
"""
print("Train dataset size:", len(train_dataset))
print("Val dataset size:", len(val_dataset))

# Take one batch from train loader
batch = next(iter(train_loader))

print("\nBatch keys:", batch.keys())
print("Image A batch shape:", batch["image_a"].shape)
print("Image B batch shape:", batch["image_b"].shape)
print("Mask batch shape:", batch["mask"].shape)
print("File names:", batch["file_name"])
print("Mask unique values in batch:", batch["mask"].unique())

img_a = batch["image_a"][0].permute(1, 2, 0).numpy()
img_b = batch["image_b"][0].permute(1, 2, 0).numpy()
mask = batch["mask"][0].squeeze(0).numpy()

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img_a)
plt.title("Batch Image A")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(img_b)
plt.title("Batch Image B")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(mask, cmap="gray")
plt.title("Batch Mask")
plt.axis("off")

plt.show()