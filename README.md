# Scene Change Detection

## Problem
This project focuses on scene change detection using PyTorch. The goal is to detect and localize changed regions between two input images and, if possible, identify change categories.

## Task Requirements
- Data preprocessing
- Custom dataloader
- Model design
- Training and inference
- Loss logging
- Visualization on 2 sample images or videos
- Dockerized post-training workflow
- Documentation and setup files

## Dataset Structure

data/raw/building-change/
├── train/
│   ├── A/
│   ├── B/
│   └── label/
└── val/
    ├── A/
    ├── B/
    └── label/

Each sample consists of:
- A: before image
- B: after image
- label: binary change mask

Train split statistics:
- Empty masks (no change): 642
- Non-empty masks (change): 2026

The dataset structure was verified manually and visually.

### Data Characteristics

- Image size: 512x512
- Mask values: 0 (no change), 255 (change)
- Converted to: 0 and 1 during preprocessing

## Data Pipeline

### Preprocessing

- Images converted from BGR to RGB
- Pixel values normalized to [0, 1]
- Masks converted from 0/255 to 0/1

### Custom Dataset

A PyTorch Dataset class was implemented to:
- Load paired images (A and B)
- Load corresponding masks
- Convert all inputs to tensors
- Return data in the format:
  - image_a: [3, H, W]
  - image_b: [3, H, W]
  - mask: [1, H, W]

### DataLoader

- Batch size: 4 (CPU training)
- Shuffle enabled for training
- num_workers used for parallel data loading

### Visualization

Sample inputs and masks were visualized to verify:
- correct pairing of A and B
- correct mask alignment
- proper normalization

## Models

Two different model architectures were implemented and compared.

### 1. Baseline Model (6-Channel Input)

In this approach:
- Image A and Image B are concatenated along the channel dimension
- Input shape becomes [6, H, W]

Architecture:
- Simple encoder-decoder CNN
- Convolution + ReLU + MaxPool in encoder
- Transposed convolution in decoder
- Output: single-channel change mask

### 2. Siamese Model

In this approach:
- Image A and Image B are processed separately using a shared encoder
- The same weights are used for both inputs

Steps:
1. Extract features from A and B using a shared encoder
2. Compute absolute difference between feature maps
3. Pass the difference through a decoder to produce change mask

This design is more suitable for change detection as it explicitly compares features from both images.

## Loss Function

Due to class imbalance and sparse change regions, a combination of Binary Cross Entropy and Dice Loss was used to ensure both pixel-wise accuracy and region-level overlap.

### Binary Cross Entropy (BCEWithLogitsLoss)
- Handles pixel-wise classification

### Dice Loss
- Improves performance on imbalanced data
- Focuses on overlap between predicted and ground truth masks

## Training Setup

- Optimizer: Adam
- Learning rate: 1e-3
- Batch size: 4
- Device: CPU (optionally GPU in Colab)
- Epochs tested: up to 6

## Training Results

### Baseline Model (6 epochs)

| Epoch | Train Loss | Val Loss |
|------|----------|---------|
| 1 | 1.1336 | 1.0637 |
| 2 | 1.0398 | 0.9596 |
| 3 | 0.9241 | 0.9304 |
| 4 | 0.8385 | 0.7970 |
| 5 | 0.7903 | **0.7924** |
| 6 | 0.7577 | 0.8543 |

Best validation loss: 0.7924 (Epoch 5)

### Siamese Model (6 epochs)

| Epoch | Train Loss | Val Loss |
|------|----------|---------|
| 1 | 1.1390 | 1.0385 |
| 2 | 0.9609 | 0.9542 |
| 3 | 0.8255 | 0.8579 |
| 4 | 0.7464 | 0.7873 |
| 5 | 0.7984 | **0.7481** |
| 6 | 0.6683 | 0.8119 |

Best validation loss: 0.7481 (Epoch 5)

## Model Comparison

Both models were trained under the same conditions (same dataset, loss, optimizer, and number of epochs).

Results:

- Baseline best validation loss: 0.7924
- Siamese best validation loss: 0.7481

Conclusion:

The Siamese model achieved better performance and was selected as the final model.

Both models showed increased validation loss after epoch 5, indicating the start of overfitting. Therefore, the checkpoint from epoch 5 was used as the final model.

## Final Model

Selected model:
- Siamese Change Detection Network

Best checkpoint:

outputs/checkpoints/best_model.pth

## Key Learnings

- A simple baseline model is useful to validate the full pipeline before using more complex architectures
- Siamese networks are more suitable for change detection tasks
- Increasing the number of epochs improves performance up to a point
- Validation loss is critical for detecting overfitting
- The best model is not the last epoch, but the one with the lowest validation loss

## Next Steps

- Visualize model predictions
- Evaluate with additional metrics (IoU, F1-score)
- Improve architecture (e.g., U-Net, skip connections)
- Train on GPU for faster experimentation

## How to Run

1. Prepare dataset in the correct folder structure
2. Run training:

python src/training/train.py

3. Best model will be saved in:

outputs/checkpoints/

