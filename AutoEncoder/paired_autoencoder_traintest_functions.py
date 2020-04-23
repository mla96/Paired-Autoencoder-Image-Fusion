#!/usr/bin/env python3
"""
This file contains functions for paired AutoEncoder model training and testing with PyTorch.

Contents
---
    train() : trains model
    test() : validates model
    tensors_to_images() : converts tensors to RGB PIL images and saves them
"""


import numpy as np
import os
import PIL.Image
import torch

from autoencoder_traintest_functions import StopCondition
from loss_utils import cosine_similarity_loss, SSIM, MS_SSIM
from tensorboard_utils import add_image_tensorboard, denormalize, denormalize_and_rescale


def apply_criterion(model, image, target, criterion, save_outputs=None):
    latent_features, output = model(image)
    if isinstance(save_outputs, list):
        save_outputs.append(output)
    if isinstance(criterion, SSIM) or isinstance(criterion, MS_SSIM):
        output, target = denormalize(output), denormalize(target)
    return latent_features, criterion(output, target)


def train(model, model2, trainloader, epoch_num, criterion, criterion2, optimizer, scheduler, device, writer,
          output_path, plot_steps=1000, stop_condition=4000):
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

            total_loss = fundus_loss + 3 * flio_loss + 3 * cosine_similarity_loss(fundus_latent_features, flio_latent_features)

            total_loss.backward()
            optimizer.step()

            step = i + epoch * len(trainloader)
            batch_loss = total_loss.item()
            running_loss += batch_loss
            # if i % 100 == 0:
            print(i)
            if step % plot_steps == 0:  # Generate training progress reconstruction figures every # steps
                add_image_tensorboard(model, fundus_image, step=step, epoch=epoch, has_feature_output=True,
                                      loss=batch_loss, writer=writer, output_path=output_path)

            if i % len(trainloader) == len(trainloader) - 1:
                print('[Epoch: {}, i: {}] loss: {:.5f}'.format(epoch + 1, i + 1, running_loss / len(trainloader)))
                # Tensorboard
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


def test(model, testloader, criterion, device):
    model.eval()
    outputs = []
    losses = []
    filenames = []
    with torch.no_grad():
        for fundus_image, flio_image, fundus_target, flio_target, image_files in testloader:
            fundus_image, fundus_target = fundus_image.to(device), fundus_target.to(device)
            _, fundus_loss = apply_criterion(model, fundus_image, fundus_target, criterion, save_outputs=outputs)
            losses.append(fundus_loss)
            filenames.append(image_files[0])

    return outputs, losses, filenames


# Convert output to RGB images
def tensors_to_images(tensors, filenames, valid_data_path):
    quality_val = 90
    for tensor, file_name in zip(tensors, filenames):
        volume = tensor[0]  # Batch size will always be 1
        volume = volume.cpu().numpy().transpose((1, 2, 0))
        volume = denormalize_and_rescale(volume)  # Denormalizes to [0, 1] and scales to [0, 255]
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
