#!/usr/bin/env python3
"""
This file contains functions for paired AutoEncoder model training and testing with PyTorch.

Contents
---
    apply_criterion() : calculates latent features and loss given criterion between output and target
    calc_fundus_flio_loss() : calculates custom loss weighted by fundus and FLIO losses
    train() : trains model
    test() : validates model
    tensors_to_images() : converts tensors to RGB PIL images and saves them
"""


import numpy as np
import os
import PIL.Image
import torch

from autoencoder_traintest_functions import StopCondition
from loss_utils import cosine_similarity_loss, SSIM, MS_SSIM, MEF_MSSSIM_Loss
from tensorboard_utils import add_image_tensorboard, denormalize, denormalize_and_rescale


def apply_criterion(model, image, target, criterion, save_outputs=None):
    """
    :param model: PyTorch model
    :param image: input image data
    :param target: given input, this is the desired output; clone of image, as reconstruction is the goal
    :param criterion: loss type
    :param save_outputs: save model output to variable
    :return: latent features and loss calculation given criterion between output and target
    """
    latent_features, output = model.encoder(image), model(image)
    if isinstance(save_outputs, list):
        save_outputs.append(output)
    if isinstance(criterion, SSIM) or isinstance(criterion, MS_SSIM):
        output, target = denormalize(output), denormalize(target)
    return latent_features, criterion(output, target)


def calc_fundus_flio_loss(fundus_loss, fundus_latent_features, flio_loss, flio_latent_features, criterion_latent):
    """
    :param fundus_loss, flio_loss: loss criterion for each model
    :param fundus_latent_features, flio_latent_features: features of latent space for each model
    :param criterion_latent: loss criterion for shared constraints and model learning based on paired latent spaces
    :return: custom loss weighted by fundus and FLIO losses
    """
    return 0.01 * fundus_loss + 0.01 * flio_loss + criterion_latent(fundus_latent_features, flio_latent_features)


def train(model, model2, trainloader, epoch_num, criterion, criterion2, criterion_latent, optimizer, scheduler, device,
          writer, output_path, plot_steps=1000, stop_condition=4000):
    """
    :param model, model2: paired PyTorch models
    :param trainloader: PyTorch DataLoader
    :param epoch_num: number of training epochs
    :param criterion, criterion2: loss type
    :param criterion_latent: loss criterion for shared constraints and model learning based on paired latent spaces
    :param optimizer: optimization algorithm for stochastic gradient descent
    :param scheduler: learning rate scheduler
    :param device: PyTorch device ('cpu' or 'cuda')
    :param writer: write to TensorBoard
    :param output_path: directory to save figures created during training
    :param plot_steps: number of steps to plot training progress in TensorBoard
    :param stop_condition: number of steps without improvement for early stopping
    """
    model.train()
    model2.train()
    if stop_condition:
        stop_condition = StopCondition(stop_condition)
    epoch_log = open("log.txt", "a")

    for epoch in range(epoch_num):
        running_loss = 0
        for i, (fundus_image, flio_image, fundus_target, flio_target, image_files) in enumerate(trainloader):
            optimizer.zero_grad()
            fundus_image, fundus_target = fundus_image.to(device), fundus_target.to(device)
            fundus_latent_features, fundus_loss = apply_criterion(model, fundus_image, fundus_target, criterion)

            flio_image, flio_target = flio_image.to(device), flio_target.to(device)
            flio_latent_features, flio_loss = apply_criterion(model2, flio_image, flio_target, criterion2)

            total_loss = calc_fundus_flio_loss(fundus_loss, fundus_latent_features, flio_loss, flio_latent_features,
                                               criterion_latent)

            total_loss.backward()
            optimizer.step()

            step = i + epoch * len(trainloader)
            batch_loss = total_loss.item()
            running_loss += batch_loss
            # if i % 100 == 0:
            if step % plot_steps == 0:  # Generate training progress reconstruction figures every # steps
                add_image_tensorboard(model, fundus_image, step=step, epoch=epoch, loss=batch_loss, writer=writer,
                                      output_path=output_path)

            if i % len(trainloader) == len(trainloader) - 1:
                print('[Epoch: {}, i: {}] loss: {:.5f}'.format(epoch + 1, i + 1, running_loss / len(trainloader)))
                # TensorBoard
                writer.add_scalar('train_scalar', running_loss / len(trainloader), step)

                if stop_condition:
                    stop_condition.evaluate_stop(running_loss)
                    if stop_condition.stop:
                        print('Early stop at [Epoch: {}, i: {}] loss: {:.5f}'.format(epoch + 1, i + 1, running_loss / len(trainloader)))

                running_loss = 0

        # torch.save(model.state_dict(), 'checkpoint.pth')
        epoch_log.write('Epoch: ' + str(epoch))
        scheduler.step(total_loss)
    epoch_log.close()


def test(model, model2, testloader, criterion, criterion2, criterion_latent, device):
    """
    :param model, model2: paired PyTorch models
    :param testloader: PyTorch DataLoader
    :param criterion, criterion2: loss type
    :param criterion_latent: loss criterion for shared constraints and model learning based on paired latent spaces
    :param device: PyTorch device ('cpu' or 'cuda')
    """
    model.eval()
    fundus_outputs = []
    flio_outputs = []
    losses = []
    fundus_filenames = []
    flio_filenames = []
    with torch.no_grad():
        for fundus_image, flio_image, fundus_target, flio_target, image_files in testloader:
            fundus_image, fundus_target = fundus_image.to(device), fundus_target.to(device)
            fundus_latent_features, fundus_loss = apply_criterion(model, fundus_image, fundus_target, criterion,
                                                                  save_outputs=fundus_outputs)

            flio_image, flio_target = flio_image.to(device), flio_target.to(device)
            flio_latent_features, flio_loss = apply_criterion(model2, flio_image, flio_target, criterion2,
                                                              save_outputs=flio_outputs)

            total_loss = calc_fundus_flio_loss(fundus_loss, fundus_latent_features, flio_loss, flio_latent_features,
                                               criterion_latent)
            losses.append(total_loss)
            fundus_filenames.append(image_files[0])
            flio_filenames.append(image_files[1])

    return fundus_outputs, flio_outputs, losses, fundus_filenames, flio_filenames


# Convert output to RGB images
def tensors_to_images(tensors, filenames, valid_data_path):
    quality_val = 90
    for tensor, file_name in zip(tensors, filenames):
        volume = tensor[0]  # Batch size will always be 1
        volume = volume.cpu().numpy().transpose((1, 2, 0))
        volume = denormalize_and_rescale(volume)  # Denormalizes to [0, 1] and scales to [0, 255]
        volume = np.squeeze(volume)
        image = PIL.Image.fromarray(np.uint8(volume))
        file_name = file_name[0].split('/')[-3] + '_' + os.path.basename(file_name[0]).split('.')[0]
        image.save(os.path.join(valid_data_path, file_name + '_valid.jpg'), 'JPEG',
                   quality=quality_val)

    # quality_val = 90
    # transform = transforms.ToPILImage()
    # for i in range(len(tensors)):
    #     for volume in tensors[i]:
    #         volume = volume.cpu().numpy().transpose((1, 2, 0))
    #         for j in range(volume.shape[2]):
    #             channel = volume[:, :, j]
    #             minimum = np.min(channel)
    #             maximum = np.max(channel)
    #             volume[:, :, j] = 255 * (channel - minimum) / (maximum - minimum)
    #         image = transform(np.uint8(volume))
    #         file_name = filenames[i][0].split('.')
    #         image.save(os.path.join(valid_data_path, file_name[0] + '_valid.jpg'), 'JPEG',
    #                    quality=quality_val)
