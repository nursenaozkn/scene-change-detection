"""Shape test for the baseline 6-channel model.

Verifies the model runs end-to-end and produces a single-channel change
mask at the same spatial resolution as the input, before training on real data.
"""
import torch

from src.models.simple_model import SimpleChangeNet


def test_simple_model_output_shape():
    model = SimpleChangeNet()

    # Two fake RGB image pairs (batch=2, 3 channels, 256x256).
    image_a = torch.randn(2, 3, 256, 256)
    image_b = torch.randn(2, 3, 256, 256)

    output = model(image_a, image_b)

    # Output is a single-channel mask matching the input resolution.
    assert output.shape == (2, 1, 256, 256)
