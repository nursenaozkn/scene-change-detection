import sys
sys.path.append(r"c:\Users\Abra\scene-change-detection")

from src.datasets.scd_dataset import SCDDataset

dataset = SCDDataset(r"c:\Users\Abra\scene-change-detection\data\raw\building-change", split="train")

print("Dataset length:", len(dataset))

for i in range(len(dataset)):
    sample = dataset[i]
    unique_vals = sample["mask"].unique()

    if len(unique_vals) > 1 or unique_vals.item() == 1.0:
        print("\nFound non-empty mask sample")
        print("Index:", i)
        print("File name:", sample["file_name"])
        print("Image A shape:", sample["image_a"].shape)
        print("Image B shape:", sample["image_b"].shape)
        print("Mask shape:", sample["mask"].shape)
        print("Mask unique values:", unique_vals)
        break