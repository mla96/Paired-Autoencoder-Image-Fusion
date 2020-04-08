#!/usr/bin/env python3
"""
This file contains custom loss functions for model training.

Contents
---
    weighted_l1_loss() : incorporates sample weights for an imbalanced dataset for L1 loss
    weighted_mse_loss() : Incorporates sample weights for an imbalanced dataset for MSE loss
"""


from tensorboard_functions import *
import torch


def weighted_l1_loss(output, target, sample_weight):
    loss = 0
    errors = np.abs(output - target)
    for w, error in zip(sample_weight, errors):
        loss += torch.mean(w * error)
    return loss


def weighted_mse_loss(output, target, sample_weight):
    loss = 0
    errors = (output - target) ** 2
    for w, error in zip(sample_weight, errors):
        loss += torch.mean(w * error)
    return loss
