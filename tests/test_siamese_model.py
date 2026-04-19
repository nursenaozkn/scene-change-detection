import sys
sys.path.append(r"c:\Users\Abra\scene-change-detection")

import torch
from src.models.siamese_model import SiameseChangeNet

model = SiameseChangeNet()

image_a = torch.randn(4, 3, 512, 512)
image_b = torch.randn(4, 3, 512, 512)

output = model(image_a, image_b)

print("Output shape:", output.shape)