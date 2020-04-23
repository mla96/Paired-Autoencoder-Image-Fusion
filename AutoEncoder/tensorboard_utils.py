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


def add_image_tensorboard(model, image, has_feature_output=False, **kwargs):
    """
    Parameters
    ---
    model: PyTorch model
    image: input tensor
    has_feature_output: bool
    **kwargs: to be fed directly into plot_tensors_tensorboard
        1. step
        2. epoch
        3. loss
        4. writer
        5. output_path
    """
    model.eval()  # Change to evaluation mode to test the model
    if has_feature_output:
        _, output = model(image)
    else:
        output = model(image)

    randint = np.random.randint(0, image.size()[0])
    input_im = image[randint].detach().cpu().numpy()
    output_im = output[randint].detach().cpu().numpy()
    input_im = denormalize_and_rescale(np.transpose(input_im, (1, 2, 0)))
    output_im = denormalize_and_rescale(np.transpose(output_im, (1, 2, 0)))
    plot_tensors_tensorboard(input_im, output_im, **kwargs)

    model.train()  # Change back to train for backpropagation


def plot_tensors_tensorboard(input, output, step, epoch, loss, writer, output_path):
    if not os.path.exists(output_path):  # Create plot path directory if it doesn't exist
        os.mkdir(output_path)

    fig = plt.figure()
    plt.title(f"Step {step}; Epoch: {epoch + 1} \n Loss: {loss}", y=0.9)
    plt.axis('off')
    fig.add_subplot(1, 2, 1)
    plt.imshow(input.astype(np.uint8))
    plt.axis('off')
    fig.add_subplot(1, 2, 2)
    plt.imshow(output.astype(np.uint8))
    plt.axis('off')
    plt.savefig(os.path.join(output_path, "trainfig_step" + str(step)))
    writer.add_figure('train_fig', fig, step)
    print('\n Figure added \n')


def denormalize(image):
    return 0.5 * image + 0.5


def denormalize_and_rescale(image):
    return 255 * denormalize(image)
