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

## Day 1 Progress
- Repository initialized
- Project structure created
- Dataset research started
- Environment setup completed

## Loss
Due to class imbalance and sparse change regions, a combination of Binary Cross Entropy and Dice Loss was used to ensure both pixel-wise accuracy and region-level overlap.
