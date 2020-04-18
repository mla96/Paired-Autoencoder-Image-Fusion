#!/usr/bin/env python3
"""
This file contains functions for AutoEncoder model training and testing with PyTorch.

Contents
---
    StopCondition :
        evaluate_stop() : evaluates whether early stopping of training should occur at some step
    train() : trains model
    test() : validates model
    tensors_to_images() : converts tensors to RGB PIL images and saves them
"""

from matplotlib import pyplot as plt
plt.switch_backend('agg')
import PIL.Image

from loss_utils import *
from tensorboard_utils import add_image_tensorboard, denormalize


# Early stopping
class StopCondition:
    def __init__(self, patience):
        self.patience = patience  # How many steps to evaluate model improvement
        self.min_loss = float('inf')
        self.stop = False
        self.patience_step = 0

    def evaluate_stop(self, loss):
        if loss < self.min_loss:
            self.min_loss = loss
            self.patience_step = 0
        else:
            self.patience_step += 1
            if self.patience_step >= self.patience:
                self.stop = True


def train(model, trainloader, epoch_num, criterion, optimizer, scheduler, device, writer, output_path,
          plot_steps=1000, stop_condition=4000, sample_weights=None):
    model.train()
    if stop_condition:
        stop_condition = StopCondition(stop_condition)
    epoch_log = open("log.txt", "a")

    for epoch in range(epoch_num):
        running_loss = 0
        # file_labelweight is a tuple containing the file_path and a weight label
        for i, (image, target, file_labelweight) in enumerate(trainloader):
            optimizer.zero_grad()
            image, target = image.to(device), target.to(device)
            output = model(image)

            # If file_labelweight is a tuple and sample_weights is defined, then use a weighted_mse_loss
            if sample_weights:
                _, label = file_labelweight
                sample_weight = torch.tensor([sample_weights[l] for l in label]).cuda()
                loss = weighted_mse_loss(output, target, sample_weight)
            else:
                if isinstance(criterion, SSIM_Loss) or isinstance(criterion, MS_SSIM_Loss):
                    output, target = denormalize(output), denormalize(target)
                loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            step = i + epoch * len(trainloader)
            running_loss += loss.item()
            if i % 100 == 0:
                print(i)
            if step % plot_steps == 0:  # Generate training progress reconstruction figures every # steps
                add_image_tensorboard(model, image, step=step, epoch=epoch, avg_loss=running_loss / len(trainloader),
                                      writer=writer, output_path=output_path)

            if i % len(trainloader) == len(trainloader) - 1:
                print('[Epoch: {}, i: {}] loss: {:.5f}'.format(epoch + 1, i + 1, running_loss / len(trainloader)))
                # Tensorboard
                writer.add_scalar('train_scalar', running_loss / len(trainloader), step)

                if stop_condition:
                    stop_condition.evaluate_stop(running_loss)
                    if stop_condition.stop:
                        print('Early stop at [Epoch: {}, i: {}] loss: {:.5f}'.format(epoch + 1, i + 1, running_loss / len(trainloader)))

                running_loss = 0

        torch.save(model.state_dict(), 'checkpoint.pth')
        epoch_log.write('Epoch: ' + str(epoch))
        scheduler.step(loss)
    epoch_log.close()


def test(model, testloader, criterion, device):
    model.eval()
    outputs = []
    losses = []
    filenames = []
    with torch.no_grad():
        for image, target, file_labelweight in testloader:
            image, target = image.to(device), target.to(device)
            output = model(image)
            outputs.append(output)

            if isinstance(criterion, SSIM_Loss) or isinstance(criterion, MS_SSIM_Loss):
                output, target = denormalize(output), denormalize(target)
            losses.append(criterion(output, target))

            if isinstance(file_labelweight, list):
                filenames.append(file_labelweight[0])  # appends file_name
            else:
                filenames.append(file_labelweight)

    return outputs, losses, filenames


def tensors_to_images(tensors, filenames, valid_data_path):
    quality_val = 90
    for tensor, file_name in zip(tensors, filenames):
        volume = tensor[0]  # Batch size will always be 1
        volume = volume.cpu().numpy().transpose((1, 2, 0))
        volume = denormalize_and_rescale(volume)  # Denormalizes to [0, 1] and scales to [0, 255]
        image = PIL.Image.fromarray(np.uint8(volume))
        file_name = file_name[0].split('.')
        image.save(os.path.join(valid_data_path, file_name[0] + '_valid.jpg'), 'JPEG',
                   quality=quality_val)
    # transform = transforms.ToPILImage()
    # for i in range(len(tensors)):
    #     for volume in tensors[i]:
    #         volume = volume.cpu().numpy().transpose((1, 2, 0))
    #         minimum = np.min(volume)  # Normalize images by volume for correct color rendering
    #         maximum = np.max(volume)
    #         for j in range(volume.shape[2]):
    #             channel = volume[:, :, j]
    #             volume[:, :, j] = 255 * (channel - minimum) / (maximum - minimum)
    #         image = transform(np.uint8(volume))
    #         file_name = filenames[i][0].split('.')
    #         image.save(os.path.join(valid_data_path, file_name[0] + '_valid.jpg'), 'JPEG',
    #                    quality=quality_val)
