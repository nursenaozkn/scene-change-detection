import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Klasör yolları
base_path = Path("c:/Users/Abra/scene-change-detection/data/raw/building-change/train")

A_path = base_path / "A"
B_path = base_path / "B"
L_path = base_path / "label"

# Dosya listesi al (A klasöründen)
files = sorted(list(A_path.glob("*")))[:5]

for file in files:
    name = file.name

    img1_path = A_path / name
    img2_path = B_path / name
    mask_path = L_path / name

    # Oku
    img1 = cv2.cvtColor(cv2.imread(str(img1_path)), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread(str(img2_path)), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    # Plot
    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.imshow(img1)
    plt.title(f"A")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(img2)
    plt.title("B")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(mask, cmap="gray")
    plt.title("Mask")
    plt.axis("off")

    plt.show()