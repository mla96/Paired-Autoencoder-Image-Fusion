from fastai.imports import *
from fastai.vision import *
from fastai.data_block import *
from fastai.basic_train import *
import torch
import torch.nn as nn

import pandas as pd
import os

import torch.nn.functional as F

from autoencoder_aux_functions import *


# tfms = get_transforms(do_flip=False)
# data = ImageDataBunch.from_folder(data_path, 'train', bs=1, ds_tfms=tfms, size=1612)
# data.show_batch(rows=3, figsize=(7, 6))


class AutoEncoder(nn.Module):

    def __init__(self, n_channels, n_encoder_filters, n_decoder_filters, trainable=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_encoder_filters = n_encoder_filters
        self.n_decoder_filters = n_decoder_filters
        self.trainable = trainable

        self.double_conv_block = DoubleConvBlock(n_channels, n_encoder_filters[0])
        # [32, 64, 128, 256, 256]
        down_blocks = [DownBlock(in_channels, out_channels)
                       for in_channels, out_channels in zip(n_encoder_filters, n_encoder_filters[1:])]
        self.down_block1, self.down_block2, self.down_block3, self.down_block4 = down_blocks[:]
        # [256, 128, 64, 32, 32]
        # The first number is the output from the down blocks, and should be doubled if concatenation for skip connections is happening
        up_blocks = [UpBlock(in_channels, out_channels, trainable=trainable)
                     for in_channels, out_channels in zip(n_decoder_filters, n_decoder_filters[1:])]
        self.up_block1, self.up_block2, self.up_block3, self.up_block4 = up_blocks[:]
        self.out_conv = nn.Conv2d(n_decoder_filters[-1], n_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.double_conv_block(x)
        x2 = self.down_block1(x1)
        x3 = self.down_block2(x2)
        x4 = self.down_block3(x3)
        x = self.down_block4(x4)
        x = self.up_block1(x, x4)  # x4, etc. are dummy variables for up_blocks; no concatenation currently happening
        x = self.up_block2(x, x3)
        x = self.up_block3(x, x2)
        x = self.up_block4(x, x1)
        logits = self.out_conv(x)
        return torch.sigmoid(logits)  # Sigmoidal output layer to ensure 0-1

        # Get rid of skip connections
        # x = self.up1(x5, x4)
        # x = self.up2(x, x3)
        # x = self.up3(x, x2)
        # x = self.up4(x, x1)


    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0.01)

    # def init_weight(self):
    #     def init_func(m):  # define the initialization function
    #         classname = m.__class__.__name__
    #         if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
    #             torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    #
    #      return self.segop.apply(init_func)


# print(torch.cuda.is_available())
# print(torch.backends.cudnn.enabled)

