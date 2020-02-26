import torch
import torch.nn as nn
import os
import numpy as np
import sys

from PIL import Image, ImageFile
import albumentations as A

import torch.nn.functional as F

from autoencoder_aux_functions import *

import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
from torchvision import transforms

from autoencoder import AutoEncoder


ImageFile.LOAD_TRUNCATED_IMAGES = True

data_path = "../../STARE_Data/Test_Crop/train"
save_model_path = "../../FLIO-Thesis-Project/AutoEncoder/AutoEncoder_Results"
valid_data_path = "../../STARE_Data/Test_Crop/validation"
overwrite = True

epoch_num = 150

device = torch.device('cuda')  # cpu or cuda

print(torch.cuda.is_available())
print(torch.backends.cudnn.enabled)

# seed = 42
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)

transform_pipeline = transforms.Compose([transforms.ToTensor()])
augmentation_pipeline = A.Compose(
    [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5)]
)


class UnlabeledDataset(Dataset):  # Use Dataset or TensorDataset?

    def __init__(self, data_path, transformations=None, augmentations=None):
        self.data_path = data_path
        self.data, self.file_names = self.images_to_tensors(data_path)
        self.transformations = transformations
        self.augmentations = augmentations
        self.size = self.__getitem__(0)[0].size()

    def __getitem__(self, index):
        image = None
        if self.augmentations:
            raw_image = np.array(self.data[index])
            augmented = self.augmentations(image=raw_image)
            image = self.transformations(augmented['image'])
        else:
            image = self.transformations(self.data[index])
        label = image
        return image, label, self.file_names[index]

    def __len__(self):
        return len(self.data)

    def images_to_tensors(self, data_path):
        files = sorted(os.listdir(data_path))
        data = []
        file_names = []
        for file in files:
            if 'jpg' in file:
                data.append(Image.open(os.path.join(data_path, file)))
                file_names.append(file)
        return data, file_names


unlabeled_dataset = UnlabeledDataset(data_path, transform_pipeline, augmentation_pipeline)
dataloader = DataLoader(unlabeled_dataset, batch_size=4, shuffle=True, num_workers=4)

# initialize model parameters with normal distribution
model = AutoEncoder(n_channels=unlabeled_dataset.size[0],
                    n_encoder_filters=[32, 64, 128, 256, 256],
                    n_decoder_filters=[256, 128, 64, 32, 32]).to(device)
model.apply(AutoEncoder.init_weights)
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
# optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,  'min', factor=0.8)


def train(model, trainloader, epoch_num, criterion, optimizer):
    model.train()
    for epoch in range(epoch_num):
        print('epoch: {}'.format(epoch))
        running_loss = 0
        for i, (image, label, _) in enumerate(trainloader):
            image, label = image.to(device), label.to(device)
            output = model(image)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 4 == 3:
                print('[{}, {}] loss: {:.5f}'.format(epoch + 1, i + 1, running_loss / 4))
                running_loss = 0
        # scheduler.step(loss)


save_model_path_filename = os.path.join(save_model_path, 'test_autoencoder.pth')
if not os.path.isfile(save_model_path_filename) or overwrite:
    train(model, dataloader, epoch_num, criterion, optimizer)
    torch.save(model.state_dict(), save_model_path_filename)
else:
    model.load_state_dict(torch.load(save_model_path_filename))

print(next(iter(model.parameters())))


validation_dataset = UnlabeledDataset(valid_data_path, transform_pipeline)
valid_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=True, num_workers=4)


def test(model, testloader, criterion):
    model.eval()
    outputs = []
    losses = []
    filenames = []
    with torch.no_grad():
        for image, label, file_name in testloader:
            image, label = image.to(device), label.to(device)
            output = model(image)
            outputs.append(output)
            losses.append(criterion(output, label))
            filenames.append(file_name)

    return outputs, losses, filenames


valid_outputs, valid_losses, valid_filenames = test(model, valid_dataloader, criterion)
print("Validation outputs")
print(valid_outputs[0])
print("Validation losses")
print(valid_losses[0])

quality_val = 90


def tensors_to_images(tensors, filenames):
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
            image.save(os.path.join(valid_data_path, 'Results', file_name[0] + '_valid.jpg'), 'JPEG', quality=quality_val)


tensors_to_images(valid_outputs, valid_filenames)
