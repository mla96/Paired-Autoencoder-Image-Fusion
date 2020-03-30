import os

import numpy as np
import torch
from torchvision.transforms import transforms
from tensorboardX import SummaryWriter
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


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


def custom_mse_loss(output, target, sample_weight):
    loss = 0
    errors = (output - target) ** 2
    for w, error in zip(sample_weight, errors):
        loss += torch.mean(w * error)
    return loss


def train(model, trainloader, epoch_num, criterion, optimizer, scheduler, device, path,
          plot_steps=1000, stop_condition=4000, sample_weights=None):
    model.train()
    writer = SummaryWriter(path)
    if stop_condition:
        stop_condition = StopCondition(stop_condition)
    epoch_log = open("log.txt", "a")

    for epoch in range(epoch_num):
        running_loss = 0
        # file_labelweight is a tuple containing the file basename and a weight label
        for i, (image, target, file_labelweight) in enumerate(trainloader):
            image, target = image.to(device), target.to(device)
            output = model(image)
            loss = criterion(output, target)
            step = i + epoch * len(trainloader)

            if isinstance(file_labelweight, list) and sample_weights:
                _, label = file_labelweight
                sample_weight = torch.tensor([sample_weights[l] for l in label]).cuda()
                loss = custom_mse_loss(output, target, sample_weight)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print(i)
            if i % len(trainloader) == len(trainloader) - 1:
                print('[Epoch: {}, i: {}] loss: {:.5f}'.format(epoch + 1, i + 1, running_loss / len(trainloader)))
                # Tensorboard
                writer.add_scalar('train', running_loss / len(trainloader), step)

                if stop_condition:
                    stop_condition.evaluate_stop(running_loss)
                    if stop_condition.stop:
                        print('Early stop at [Epoch: {}, i: {}] loss: {:.5f}'.format(epoch + 1, i + 1, running_loss / len(trainloader)))

                running_loss = 0

            if step % plot_steps == 0:  # Generate training progress reconstruction figures every # steps
                fig = plt.figure()
                randint = np.random.randint(0, trainloader.batch_size - 1)
                input_im = image.detach().cpu().numpy()[randint]
                output_im = output.detach().cpu().numpy()[randint]
                input_im, output_im = np.transpose(input_im, (1, 2, 0)), np.transpose(output_im, (1, 2, 0))
                plt.title("Step " + str(step), y=0.9)
                plt.axis('off')
                fig.add_subplot(1, 2, 1)
                plt.imshow(input_im)
                plt.axis('off')
                fig.add_subplot(1, 2, 2)
                plt.imshow(output_im)
                plt.axis('off')
                # plt.show()
                writer.add_figure('train', fig, step)

        torch.save(model.state_dict(), 'checkpoint.pth')
        epoch_log.write('Epoch: ' + str(epoch))
        scheduler.step(loss)
    writer.close()
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
            losses.append(criterion(output, target))
            if isinstance(file_labelweight, list):
                filenames.append(file_labelweight[0])  # appends file_name
            else:
                filenames.append(file_labelweight)

    return outputs, losses, filenames


# Convert output to RGB images
def tensors_to_images(tensors, filenames, valid_data_path):
    quality_val = 90
    transform = transforms.ToPILImage()
    for i in range(len(tensors)):
        for volume in tensors[i]:
            volume = volume.cpu().numpy().transpose((1, 2, 0))
            for j in range(volume.shape[2]):
                channel = volume[:, :, j]
                minimum = np.min(channel)
                maximum = np.max(channel)
                volume[:, :, j] = 255 * (channel - minimum) / (maximum - minimum)
                print(volume[:, :, j])
            image = transform(np.uint8(volume))
            file_name = filenames[i][0].split('.')
            image.save(os.path.join(valid_data_path, file_name[0] + '_valid.jpg'), 'JPEG',
                       quality=quality_val)
