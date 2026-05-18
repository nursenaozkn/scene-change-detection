"""Shape test for the Siamese U-Net change detection model.

This is the final model: a shared encoder extracts multi-scale features
from A and B, their absolute differences feed skip connections, and a
U-Net decoder reconstructs a full-resolution change mask.
"""
import torch

from src.models.siamese_unet_model import SiameseUNet


def test_siamese_unet_output_shape():
    model = SiameseUNet()

    # Spatial size must be divisible by 8 (three 2x downsampling stages).
    image_a = torch.randn(2, 3, 256, 256)
    image_b = torch.randn(2, 3, 256, 256)

    output = model(image_a, image_b)

    assert output.shape == (2, 1, 256, 256)
