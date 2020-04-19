import numpy as np
import torch.nn as nn
from torchvision.transforms import transforms

import hdf5storage
import skimage.io
from loss_utils import *
from autoencoder_traintest_functions import StopCondition
from matplotlib import pyplot as plt
plt.switch_backend('agg')


def train(model, model2, trainloader, epoch_num, criterion, optimizer, scheduler, device, writer, output_path,
          plot_steps=1000, stop_condition=4000):
    model.train()
    model2.train()
    if stop_condition:
        stop_condition = StopCondition(stop_condition)
    epoch_log = open("log.txt", "a")

    for epoch in range(epoch_num):
        running_loss = 0
        for i, (image, image2, target, target2, image_files) in enumerate(trainloader):
            optimizer.zero_grad()
            image, target = image.to(device), target.to(device)
            features, output = model(image)
            if isinstance(criterion, SSIM_Loss) or isinstance(criterion, MS_SSIM_Loss):
                output, target = denormalize(output), denormalize(target)
            loss = criterion(output, target)

            image2, target2 = image2.to(device), target2.to(device)
            features2, output2 = model2(image2)
            loss2 = criterion(output2, target2)

            features = features.view(1, 8192)
            features2 = features2.view(1, 8192)

            latent_corr = nn.CosineSimilarity(dim=1)  # Resize into 1 dimensional array to get a single value?
            cos_loss = latent_corr(features, features2)
            total_loss = loss + loss2 + cos_loss

            step = i + epoch * len(trainloader)

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
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
        for i, (image, image2, target, target2, image_files) in enumerate(testloader):
            image, target = image.to(device), target.to(device)
            output = model(image)
            outputs.append(output)

            if isinstance(criterion, SSIM_Loss) or isinstance(criterion, MS_SSIM_Loss):
                output, target = denormalize(output), denormalize(target)
            losses.append(criterion(output, target))

            filenames.append(image_files[0])
            # if isinstance(file_labelweight, list):
            #     filenames.append(file_labelweight[0])  # appends file_name
            # else:
            #     filenames.append(file_labelweight)

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
            image = transform(np.uint8(volume))
            file_name = filenames[i][0].split('.')
            image.save(os.path.join(valid_data_path, file_name[0] + '_valid.jpg'), 'JPEG',
                       quality=quality_val)
