import torch
import torch.nn as nn


bce_loss_fn = nn.BCEWithLogitsLoss()


def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)

    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = (pred * target).sum()
    dice_score = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    return 1.0 - dice_score


def combined_loss(pred, target):
    bce = bce_loss_fn(pred, target)
    dice = dice_loss(pred, target)
    return bce + dice