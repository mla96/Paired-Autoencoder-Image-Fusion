#!/usr/bin/env python3
"""
This file contains functions for augmenting PyTorch model training with TensorBoard.

Contents
---
    plot_tensors_tensorboard() : at some step during model training, plot input and output tensors as matplotlib figure,
    then save as image and write to TensorBoard SummaryWriter
"""


import os
import numpy as np
from matplotlib import pyplot as plt
plt.switch_backend('agg')


def plot_tensors_tensorboard(input, output, step, epoch, loss, writer, output_path):
    if not os.path.exists(output_path):  # Create plot path directory if it doesn't exist
        os.mkdir(output_path)

    fig = plt.figure()
    plt.title("Step " + str(step) + "; Epoch: {}".format(epoch + 1) + "\n Loss: {:.5f}".format(loss), y=0.9)
    plt.axis('off')
    fig.add_subplot(1, 2, 1)
    plt.imshow((input * 255).astype(np.uint8))
    plt.axis('off')
    fig.add_subplot(1, 2, 2)
    plt.imshow((output * 255).astype(np.uint8))
    plt.axis('off')
    plt.savefig(os.path.join(output_path, "trainfig_step" + str(step)))
    writer.add_figure('train_fig', fig, step)
    print('\n Figure added \n')
