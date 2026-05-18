# Scene Change Detection (SCD)

Detect and localize the changes between two images of the same scene
(a "before" image **A** and an "after" image **B**) and produce a binary
change mask. The project is implemented in **Python + PyTorch** and the
post-training inference workflow runs inside **Docker**.

## Task Requirements

| Requirement | Status | Where |
|---|---|---|
| Data preprocessing + custom data loader | Done | `src/datasets/scd_dataset.py` |
| Model architecture | Done | `src/models/` (3 models) |
| Training + inference from the trained model | Done | `src/training/train.py`, `src/inference/predict.py` |
| Loss logging during training | Done | CSV (`metrics.csv`) + TensorBoard |
| Change visualization on selected images | Done | `src/inference/predict.py` |
| Dockerized post-training workflow | Done | `Dockerfile`, `.dockerignore` |
| Tests, requirements, documentation | Done | `tests/`, `requirements.txt`, this file |

## Project Structure

```
scene-change-detection/
├── Dockerfile              # Container for the inference workflow
├── .dockerignore
├── requirements.txt        # Python dependencies
├── conftest.py             # Makes the project importable for pytest
├── README.md
├── data/
│   └── raw/building-change/{train,val}/{A,B,label}/   # dataset (not in git)
├── outputs/
│   ├── checkpoints/        # trained weights (not in git)
│   ├── logs/               # TensorBoard events + metrics.csv (not in git)
│   └── predictions/        # saved visualizations (not in git)
├── src/
│   ├── datasets/scd_dataset.py        # custom Dataset
│   ├── models/
│   │   ├── simple_model.py            # baseline (6-channel input)
│   │   ├── siamese_model.py           # shared-encoder baseline
│   │   └── siamese_unet_model.py      # final model
│   ├── training/train.py              # training loop + loss logging
│   ├── inference/predict.py           # inference + change visualization
│   └── utils/losses.py                # BCE + Dice loss
└── tests/                  # pytest test suite
```

`data/` and `outputs/` are git-ignored; they are mounted into the container
at runtime (see [Docker](#docker)).

## Dataset

The model is trained on a building change-detection dataset organized as:

```
data/raw/building-change/
├── train/
│   ├── A/        # before images
│   ├── B/        # after images
│   └── label/    # binary change masks
└── val/
    ├── A/
    ├── B/
    └── label/
```

- Image size: `512 x 512`, RGB.
- Each `A`/`B`/`label` triplet shares the same file name.
- Mask values on disk: `0` (no change) and `255` (change).
- Train split: 2026 samples with change, 642 with no change (class imbalance).

## Data Pipeline

`SCDDataset` (`src/datasets/scd_dataset.py`) is a custom `torch.utils.data.Dataset` that:

- reads the `A`, `B` and `label` images,
- converts images from BGR to RGB and normalizes pixels to `[0, 1]`,
- converts masks from `{0, 255}` to `{0, 1}`,
- returns CHW tensors: `image_a [3,H,W]`, `image_b [3,H,W]`, `mask [1,H,W]`.

It is consumed by a standard `DataLoader` (batching, shuffling for training).

## Models

Three architectures were implemented; complexity was increased step by step.

### 1. `SimpleChangeNet` — baseline
A and B are concatenated along the channel axis (`[6,H,W]`) and passed
through a small encoder–decoder CNN. Useful to validate the full pipeline.

### 2. `SiameseChangeNet` — shared-encoder baseline
A and B are encoded separately by a **shared** encoder; the absolute
difference of the feature maps is decoded into a change mask. This compares
the two images explicitly instead of relying on the network to learn the
comparison from concatenated channels.

### 3. `SiameseUNet` — final model
A U-Net with a Siamese encoder:

- a **shared encoder** extracts multi-scale features from A and B
  (3 downsampling stages + a bottleneck, with `BatchNorm`),
- the **absolute difference** of A/B features is computed at every scale,
- a **U-Net decoder** upsamples the bottleneck difference and fuses the
  per-scale differences through **skip connections**, restoring full
  spatial resolution.

The skip connections preserve fine spatial detail, which clearly improves
the sharpness and accuracy of the predicted mask. **`SiameseUNet` is the
final model** (~1.93M parameters); its weights are
`outputs/checkpoints/best_siamese_unet_model.pth`.

## Loss Function

Because the change regions are sparse and the classes are imbalanced, the
training loss combines two terms (`src/utils/losses.py`):

- **`BCEWithLogitsLoss`** — pixel-wise classification.
- **Dice loss** — region overlap; robust to class imbalance.

`combined_loss = BCE + Dice`.

## Training

```bash
python src/training/train.py --model siamese_unet
```

- Optimizer: Adam, learning rate `1e-3`.
- Batch size: 4. Default: 6 epochs.
- The best checkpoint (lowest validation loss) is saved to
  `outputs/checkpoints/best_<model>_model.pth`.
- **Loss logging:** the per-epoch train/validation loss is written to
  `outputs/logs/<model>/metrics.csv` and, when the package is available,
  streamed to **TensorBoard** in the same directory.

Useful options: `--model {simple,siamese,siamese_unet}`, `--epochs`,
`--batch-size`, `--lr`, `--data-root`, `--output-dir`.

View the TensorBoard curves:

```bash
tensorboard --logdir outputs/logs
```

### Baseline experiments

The two earlier baselines were trained for 6 epochs under identical
settings. Their recorded validation loss:

| Model | Best val loss (epoch 5) |
|---|---|
| `SimpleChangeNet` | 0.7924 |
| `SiameseChangeNet` | 0.7481 |

Both baselines started to overfit after epoch 5. The Siamese design
outperformed the concatenation baseline, which motivated the final
`SiameseUNet` with skip connections. On the sample validation pairs,
`SiameseUNet` reaches an IoU/F1 far above the plain Siamese model
(see the numbers printed by the inference script below).

## Inference and Change Visualization

`src/inference/predict.py` loads a trained model, runs it on image pairs,
and produces an interpretable visualization of the detected changes.

Default run — picks 2 sample pairs from the validation set:

```bash
python src/inference/predict.py
```

Run on a specific image pair:

```bash
python src/inference/predict.py --image-a path/to/A.png --image-b path/to/B.png
```

For each pair the script:

- predicts a change probability map and thresholds it into a binary mask,
- **interprets** the result: number of changed pixels and their percentage,
  number of distinct change regions (connected components), and — when a
  ground-truth mask is available — the **IoU** and **F1** score,
- saves a side-by-side figure to `outputs/predictions/prediction_<name>.png`
  showing *Image A*, *Image B*, the ground truth (if any), the predicted
  change mask, and the changes highlighted **in red** over image B.

Options: `--model`, `--checkpoint`, `--threshold`, `--num-samples`,
`--split`, `--data-root`, `--output-dir`.

## Docker

The post-training workflow (inference + visualization) is containerized.

Build the image (run from the project root, so the trained checkpoints are
included in the build context):

```bash
docker build -t scd .
```

Run inference — mount the dataset and the outputs directory so the
visualizations are written back to the host:

```bash
docker run --rm \
  -v ${PWD}/data:/app/data \
  -v ${PWD}/outputs:/app/outputs \
  scd
```

Results appear in `outputs/predictions/`. To train inside the container:

```bash
docker run --rm \
  -v ${PWD}/data:/app/data \
  -v ${PWD}/outputs:/app/outputs \
  scd python src/training/train.py --model siamese_unet
```

(On Windows PowerShell `${PWD}` works as shown; on cmd.exe use `%cd%`.)

## Tests

The test suite uses `pytest`:

```bash
pytest
```

- Model tests check that each architecture produces a correctly shaped
  change mask — these run without the dataset.
- Dataset / DataLoader tests verify preprocessing, tensor shapes and
  batching; they are **skipped automatically** when the dataset is not
  present (e.g. in a fresh checkout).

`tests/first/` contains the early exploratory scripts used to inspect the
dataset; they are kept for reference and are not part of the pytest suite.

## Setup

Requires Python 3.9+.

```bash
pip install -r requirements.txt
```

Then place the dataset under `data/raw/building-change/` as shown above and
follow the [Training](#training) or [Inference](#inference-and-change-visualization)
sections — or use [Docker](#docker).

## Possible Improvements

- Train `SiameseUNet` for more epochs with early stopping on validation loss.
- Add data augmentation (flips, rotations) to reduce overfitting.
- Report IoU / F1 over the full validation set, not only sample pairs.
- Train on a GPU for faster experimentation.
