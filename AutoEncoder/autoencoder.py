#!/usr/bin/env python3
"""
This file contains classes and functions to construct complete model architectures with forward propagation in PyTorch.

Contents
---
    AutoEncoder :
    AutoEncoder_ResEncoder : Res34 encoder

"""


from fastai.vision import *
from torchvision import models

from autoencoder_blocks import *


class AutoEncoder(nn.Module):

    def __init__(self, n_channels, n_encoder_filters, n_decoder_filters, trainable=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_encoder_filters = n_encoder_filters
        self.n_decoder_filters = n_decoder_filters.insert(0, n_encoder_filters[-1])
        self.trainable = trainable

        self.double_conv_block = DoubleConvBlock(n_channels, n_encoder_filters[0])

        # [32, 64, 128, 256]
        down_blocks = [DownBlock(in_channels, out_channels)
                       for in_channels, out_channels in zip(n_encoder_filters, n_encoder_filters[1:])]
        self.down_blocks = nn.Sequential(*down_blocks)

        # [128, 64, 32]
        up_blocks = [UpBlock(in_channels, out_channels, trainable=trainable)
                     for in_channels, out_channels in zip(n_decoder_filters, n_decoder_filters[1:])]
        up_blocks[-1] = UpBlock(n_decoder_filters[-2], n_decoder_filters[-1], trainable=trainable, is_batch_norm=False)
        self.up_blocks = nn.Sequential(*up_blocks)

        # Uses tanh output layer to ensure -1 to 1
        # Potential parameters are kernel_size=3 and padding=1
        self.out_conv = ConvBlock(n_decoder_filters[-1], n_channels, kernel_size=1, padding=0, activation='tanh')

    def forward(self, x):
        x = self.double_conv_block(x)
        x = self.down_blocks(x)
        x = self.up_blocks(x)
        return self.out_conv(x)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0.01)


class AutoEncoder_ResEncoder(nn.Module):
    def __init__(self, n_channels, n_decoder_filters, trainable=False):
        super().__init__()

        resnet = models.resnet34()
        resnet_layers = list(resnet.children())
        self.resnet_block = nn.Sequential(*resnet_layers[:8])  # Stops right before linear layer

        self.n_channels = n_channels
        self.n_decoder_filters = n_decoder_filters.insert(0, 512)  # Insert here the number of filters the resnet ends on
        self.trainable = trainable

        # [256, 128, 64, 32]
        up_blocks = [UpBlock(in_channels, out_channels, trainable=trainable)
                     for in_channels, out_channels in zip(n_decoder_filters, n_decoder_filters[1:])]
        up_blocks[-1] = UpBlock(n_decoder_filters[-2], n_decoder_filters[-1], trainable=trainable, is_batch_norm=False)
        self.up_blocks = nn.Sequential(*up_blocks)

        # Uses tanh output layer to ensure -1 to 1
        # Potential parameters are kernel_size=3 and padding=1
        self.out_conv = ConvBlock(n_decoder_filters[-1], n_channels, kernel_size=1, padding=0, activation='tanh', is_batch_norm=False)

    def forward(self, x):
        x = self.resnet_block(x)
        x = self.up_blocks(x)
        return self.out_conv(x)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
