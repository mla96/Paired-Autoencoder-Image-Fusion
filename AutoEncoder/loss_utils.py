#!/usr/bin/env python3
"""
This file contains custom loss functions for model training with PyTorch.

Contents
---
    weighted_l1_loss() : incorporates sample weights for an imbalanced dataset for L1 loss
    weighted_mse_loss() : incorporates sample weights for an imbalanced dataset for MSE loss
    mutual_information_calc() : calculates mutual information (in progress, currently unused)
    mutual_information_loss() : calculates mutual information loss for MINE and steps backward (in progress, currently unused)
"""


from tensorboard_utils import *
import torch
from ssim import SSIM, MS_SSIM


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


def mutual_information_calc(joint, marginal, mine_net):
    t = mine_net(joint)
    et = torch.exp(mine_net(marginal))
    mi_lb = torch.mean(t) - torch.log(torch.mean(et))
    return mi_lb, t, et


def mutual_information_loss(batch, mine_net, mine_net_optim, ma_et, ma_rate=0.01):
    # batch is a tuple of (joint, marginal)
    joint, marginal = batch
    joint = torch.autograd.Variable(torch.FloatTensor(joint)).cuda()
    marginal = torch.autograd.Variable(torch.FloatTensor(marginal)).cuda()
    mi_lb, t, et = mutual_information_calc(joint, marginal, mine_net)
    ma_et = (1 - ma_rate) * ma_et + ma_rate * torch.mean(et)

    # unbiasing use moving average
    loss = -(torch.mean(t) - (1 / ma_et.mean()).detach() * torch.mean(et))
    # use biased estimator
    #     loss = - mi_lb

    mine_net_optim.zero_grad()
    torch.autograd.backward(loss)
    mine_net_optim.step()
    return mi_lb, ma_et


class SSIM_Loss(SSIM):
    def forward(self, img1, img2):
        return 1 - super(SSIM_Loss, self).forward(img1, img2)


class MS_SSIM_Loss(MS_SSIM):
    def forward(self, img1, img2):
        return 1 - super(MS_SSIM_Loss, self).forward(img1, img2)
