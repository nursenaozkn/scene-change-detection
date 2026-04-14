"""
A görüntüsünü oku
B görüntüsünü oku
label maskesini oku
maskeyi 0/255 → 0/1 çevir
hepsini tensor yap
modelin kullanacağı formatta döndür
"""
from pathlib import Path
import cv2
import torch
from torch.utils.data import Dataset


class SCDDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        """
        Args:
            root_dir (str): dataset root path
                example: data/raw/building-change
            split (str): "train" or "val"
        """
        self.root_dir = Path(root_dir)
        self.split = split

        self.a_dir = self.root_dir / split / "A"
        self.b_dir = self.root_dir / split / "B"
        self.label_dir = self.root_dir / split / "label"

        self.files = sorted(
            [f.name for f in self.a_dir.glob("*") if f.is_file()],
            key=lambda x: int(Path(x).stem)
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]

        a_path = self.a_dir / file_name
        b_path = self.b_dir / file_name
        label_path = self.label_dir / file_name

        # Read images
        img_a = cv2.imread(str(a_path), cv2.IMREAD_COLOR)
        img_b = cv2.imread(str(b_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)

        if img_a is None:
            raise FileNotFoundError(f"Could not read image A: {a_path}")
        if img_b is None:
            raise FileNotFoundError(f"Could not read image B: {b_path}")
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {label_path}")

        # BGR -> RGB
        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)

        # Normalize images to [0, 1]
        img_a = img_a.astype("float32") / 255.0
        img_b = img_b.astype("float32") / 255.0

        # Normalize mask: 0/255 -> 0/1
        mask = mask.astype("float32") / 255.0

        # HWC -> CHW
        img_a = torch.from_numpy(img_a).permute(2, 0, 1)
        img_b = torch.from_numpy(img_b).permute(2, 0, 1)

        # Add channel dimension to mask: [H, W] -> [1, H, W]
        mask = torch.from_numpy(mask).unsqueeze(0)

        return {
            "image_a": img_a,
            "image_b": img_b,
            "mask": mask,
            "file_name": file_name
        }