#!/usr/bin/env python3
"""
This file contains classes and functions to construct model architecture blocks with forward propagation in PyTorch.

Contents
---
    ConvBlock :
    DoubleConvBlock :
    DownBlock :
    DownResBlock :
    UpBlock :
    UpResBlock :
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, activation='relu'):
        super().__init__()
        activations = nn.ModuleDict({
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh()
        })

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            activations[activation]
        )

    def forward(self, x):
        return self.conv_block(x)


class DoubleConvBlock(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.double_conv_block = nn.Sequential(
            ConvBlock(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            ConvBlock(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        )

    def forward(self, x):
        return self.double_conv_block(x)


class DownBlock(nn.Module):
    # Max pooling, double convolution
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.down_block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvBlock(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        )

    def forward(self, x):
        return self.down_block(x)


class DownResBlock(nn.Module):
    # For concatenating tensors, enabling skip connections within encoder (Down)
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.down_block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvBlock(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        )

    def forward(self, x):
        return self.down_block(x)


class UpBlock(nn.Module):
    # trainable=False for bilinear upsampling default
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, trainable=False):
        super().__init__()

        if trainable:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.up_conv_block = nn.Sequential(
            self.up,
            DoubleConvBlock(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        )

    def forward(self, x, dummy=None):
        return self.up_conv_block(x)


class UpResBlock(nn.Module):
    # For concatenating tensors, enabling skip connections from encoder (Down)
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, trainable=False):
        super().__init__()

        if trainable:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.double_conv_block = DoubleConvBlock(2 * in_channels, out_channels, kernel_size=kernel_size,
                                                 padding=padding)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diff_height = x2.size()[2] - x1.size()[2]
        diff_width = x2.size()[3] - x1.size()[3]
        # Pad tensors if there is a dimension mismatch due to convolving without padding
        x1 = F.pad(x1, [diff_width // 2, diff_width - diff_width // 2,
                        diff_height // 2, diff_height - diff_height // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.double_conv_block(x)


# class OutConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(OutConv, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#
#     def forward(self, x):
#         return self.conv(x)
