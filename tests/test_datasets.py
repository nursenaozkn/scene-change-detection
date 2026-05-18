"""Tests for the SCD dataset.

These tests require the real dataset on disk and are skipped automatically
when it is not available (e.g. in a clean checkout, since data/ is gitignored).
"""
from pathlib import Path

import pytest

from src.datasets.scd_dataset import SCDDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "building-change"

requires_data = pytest.mark.skipif(
    not (DATA_ROOT / "train" / "A").is_dir(),
    reason="building-change dataset not available",
)


@requires_data
def test_dataset_is_not_empty():
    dataset = SCDDataset(str(DATA_ROOT), split="train")
    assert len(dataset) > 0


@requires_data
def test_dataset_sample_shapes_and_mask_values():
    dataset = SCDDataset(str(DATA_ROOT), split="train")
    sample = dataset[0]

    # Images are CHW with 3 channels; mask has a single channel.
    assert sample["image_a"].shape[0] == 3
    assert sample["image_b"].shape[0] == 3
    assert sample["mask"].shape[0] == 1
    assert sample["image_a"].shape[1:] == sample["mask"].shape[1:]

    # Mask must be normalized from {0, 255} to {0, 1}.
    unique_values = set(sample["mask"].unique().tolist())
    assert unique_values.issubset({0.0, 1.0})

    # Image pixels are normalized to the [0, 1] range.
    assert 0.0 <= sample["image_a"].min() and sample["image_a"].max() <= 1.0
