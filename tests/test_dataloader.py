"""Tests for batching the SCD dataset with a PyTorch DataLoader.

Confirms that the dataset integrates with DataLoader and yields correctly
shaped batches. Skipped when the dataset is not available.
"""
from pathlib import Path

import pytest
from torch.utils.data import DataLoader

from src.datasets.scd_dataset import SCDDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "building-change"

requires_data = pytest.mark.skipif(
    not (DATA_ROOT / "train" / "A").is_dir(),
    reason="building-change dataset not available",
)


@requires_data
def test_dataloader_batch_shapes():
    dataset = SCDDataset(str(DATA_ROOT), split="train")
    batch_size = 4
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    batch = next(iter(loader))

    assert batch["image_a"].shape[0] == batch_size
    assert batch["image_a"].shape[1] == 3
    assert batch["image_b"].shape == batch["image_a"].shape
    assert batch["mask"].shape[0] == batch_size
    assert batch["mask"].shape[1] == 1
    assert len(batch["file_name"]) == batch_size
