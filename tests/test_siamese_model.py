"""Shape test for the Siamese change detection model.

The shared encoder processes A and B with the same weights; this test
confirms the difference-and-decode path yields a single-channel mask at
the input resolution.
"""
import torch

from src.models.siamese_model import SiameseChangeNet


def test_siamese_model_output_shape():
    model = SiameseChangeNet()

    image_a = torch.randn(2, 3, 256, 256)
    image_b = torch.randn(2, 3, 256, 256)

    output = model(image_a, image_b)

    assert output.shape == (2, 1, 256, 256)
