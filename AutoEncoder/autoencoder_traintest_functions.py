import os

import numpy as np
import torch
from torchvision.transforms import transforms


def custom_mse_loss(output, target, sample_weight):
    loss = 0
    errors = (output - target) ** 2
    for w, error in zip(sample_weight, errors):
        loss += torch.mean(w * error)
    return loss


# If need scheduler, pass it in as a parameter
def train(model, trainloader, epoch_num, criterion, optimizer, scheduler, device, sample_weights=None):
    model.train()
    for epoch in range(epoch_num):
        print('epoch: {}'.format(epoch))
        running_loss = 0
        # file_labelweight is a tuple containing the file basename and a weight label
        for i, (image, target, file_labelweight) in enumerate(trainloader):
            image, target = image.to(device), target.to(device)
            output = model(image)
            loss = criterion(output, target)
            if isinstance(file_labelweight, list) and sample_weights:
                _, label = file_labelweight
                sample_weight = torch.tensor([sample_weights[l] for l in label]).cuda()
                loss = custom_mse_loss(output, target, sample_weight)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print(i)
            if i % trainloader.batch_size == trainloader.batch_size - 1:
                print('[Epoch: {}, i: {}] loss: {:.5f}'.format(epoch + 1, i + 1, running_loss / trainloader.batch_size))
                running_loss = 0
        scheduler.step(loss)


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
