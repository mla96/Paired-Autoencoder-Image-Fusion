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


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ssim import SSIM, MS_SSIM
from mefssim import MEFSSIM, MEF_MSSSIM


def get_loss_type(type, channel=3):
    loss_types = {'MSE': nn.MSELoss(),
                  'L1': nn.L1Loss(),
                  'SSIM': SSIM_Loss(data_range=1),
                  'MS-SSIM': MS_SSIM_Loss(data_range=1, nonnegative_ssim=True, win_size=11, channel=channel,
                                          K=(0.01, 0.03)),
                  'MS-SSIM-L1': MS_SSIM_L1_Loss(l1_weight=0.1, data_range=1, nonnegative_ssim=True, win_size=11,
                                                channel=channel, K=(0.01, 0.03)),
                  'MEFSSIM': MEFSSIM_Loss(),
                  'MEF-MSSSIM': MEFSSIM_Loss(),
                  'MEF-MSSSIM-L1': MEF_MSSSIM_L1_Loss(l1_weight=0.1)}
    return loss_types.get(type)


def weighted_loss(output, target, criterion, sample_weight):
    loss = 0
    for w, img1, img2 in zip(sample_weight, output, target):
        img1, img2 = torch.unsqueeze(img1, dim=0), torch.unsqueeze(img2, dim=0)
        loss += w * criterion(img1, img2)
    return loss


def cosine_similarity_loss(output, target):
    # Resize into 1 dimensional array to get a single value
    output = output.flatten(1, -1)
    target = target.flatten(1, -1)
    return torch.tensor(1).cuda() - F.cosine_similarity(output, target, dim=-1).mean()


class SSIM_Loss(SSIM):
    def forward(self, img1, img2):
        return 1 - super(SSIM_Loss, self).forward(img1, img2)


class MS_SSIM_Loss(MS_SSIM):
    def forward(self, img1, img2):
        return 1 - super(MS_SSIM_Loss, self).forward(img1, img2)


class MS_SSIM_L1_Loss(MS_SSIM):
    def __init__(self, l1_weight=0.1, **kwargs):
        super(MS_SSIM_L1_Loss, self).__init__(**kwargs)
        self.l1_weight = l1_weight

    def forward(self, img1, img2):
        ms_ssim_loss = 1 - super(MS_SSIM_L1_Loss, self).forward(img1, img2)
        l1_loss = F.l1_loss(img1, img2)
        return ms_ssim_loss + self.l1_weight * l1_loss


class MEFSSIM_Loss(MEFSSIM):
    def forward(self, img1, img2):
        return 1 - super(MEFSSIM_Loss, self).forward(img1, img2)


class MEF_MSSSIM_Loss(MEF_MSSSIM):
    def forward(self, img1, img2):
        return 1 - super(MEF_MSSSIM_Loss, self).forward(img1, img2)


class MEF_MSSSIM_L1_Loss(MEF_MSSSIM):
    def __init__(self, l1_weight=0.1, **kwargs):
        super(MEF_MSSSIM_L1_Loss, self).__init__(**kwargs)
        self.l1_weight = l1_weight

    def forward(self, img1, img2):
        mef_ssim_loss = 1 - super(MEF_MSSSIM_L1_Loss, self).forward(img1, img2)
        l1_loss = F.l1_loss(img1, img2)
        return mef_ssim_loss + self.l1_weight * l1_loss


def mutual_information_calc(t, et):
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
