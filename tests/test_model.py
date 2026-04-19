"""
Model kodu teknik olarak çalışıyor mu?”
Gerçek görüntüye geçmeden önce bu daha güvenli.
"""
import sys
sys.path.append(r"c:\Users\Abra\scene-change-detection")

import torch
from src.models.simple_model import SimpleChangeNet

model = SimpleChangeNet()

image_a = torch.randn(4, 3, 512, 512)
"""
4 → batch size, yani aynı anda 4 örnek
3 → RGB kanal sayısı
512, 512 → görüntü boyutu
Yani image_a şunu temsil ediyor: 4 tane sahte RGB görüntü
"""
image_b = torch.randn(4, 3, 512, 512)

output = model(image_a, image_b)

print("Output shape:", output.shape)