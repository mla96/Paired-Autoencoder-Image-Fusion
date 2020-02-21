from fastai.imports import *
from fastai.vision import *
from fastai.data_block import *
from fastai.basic_train import *
import torch
import torch.nn as nn

import pandas as pd
import os


import torch.nn.functional as F

from autoencoder_aux import *


# data_path = "../STARE_Data/Test_Crop"
#
# tfms = get_transforms(do_flip=False)
# data = ImageDataBunch.from_folder(data_path, 'train', bs=1, ds_tfms=tfms, size=1612)
# data.show_batch(rows=3, figsize=(7, 6))


class AutoEncoder(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super(AutoEncoder, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)  # Consider reduce filter numbers in half to prevent overfitting
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 256)
        self.up1 = Up(512, 128, bilinear)
        self.up2 = Up(256, 64, bilinear)
        self.up3 = Up(128, 32, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # Get rid of skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        logits = torch.sigmoid(logits)  # Add sigmoidal output layer to ensure 0-1
        return logits

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
#
#
#
# model = vision.models.resnet34().cuda()
#
# learn = Learner(data, model, loss_func=F.mse_loss)
# learn = create_cnn(data, model, loss_func=F.mse_loss)
#
# # learn.fit_one_cycle(10, 0.01)

