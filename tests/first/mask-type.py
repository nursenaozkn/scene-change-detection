import cv2
import numpy as np

mask_path = r"c:\Users\Abra\scene-change-detection\data\raw\building-change\train\label\10.png"

mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

print("Shape:", mask.shape)
print("Dtype:", mask.dtype)
print("Unique values:", np.unique(mask))
print("Min:", mask.min(), "Max:", mask.max())

"""
Shape: (512, 512) Dtype: uint8 Unique values: [ 0 255] Min: 0 Max: 255
Loss (özellikle BCEWithLogitsLoss) şunu bekler:
0 → no change  
1 → change
Ama mask:
0 → no change  
255 → change
Bu haliyle verirsen;model saçma öğrenir,loss yanlış hesaplanır.
"""

