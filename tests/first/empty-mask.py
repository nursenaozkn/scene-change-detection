import cv2
from pathlib import Path
import numpy as np

label_path = Path("c:/Users/Abra/scene-change-detection/data/raw/building-change/train/label")

files = list(label_path.glob("*"))

empty = 0
non_empty = 0

for f in files:
    mask = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)

    if np.sum(mask) == 0:
        empty += 1
    else:
        non_empty += 1

print("Empty (no change):", empty)
print("Non-empty (change):", non_empty)