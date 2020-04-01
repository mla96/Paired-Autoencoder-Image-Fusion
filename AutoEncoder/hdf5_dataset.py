import os

import h5py
import numpy as np
import skimage.io
import torch
from torch.utils.data.dataset import Dataset


class ImbalancedHDF5Dataset(Dataset):

    def __init__(self, file_names, save_path, augmentations=None):
        self.file_names = file_names  # Contains labels
        self.save_path = save_path
        self.augmentations = augmentations

    def __getitem__(self, index):
        file_name, label = self.file_names[index]
        with h5py.File(self.save_path, 'r') as f:
            image_group = f['images']
            image = image_group[file_name].value
            if self.augmentations:
                augmented = self.augmentations(image=image)
                image = augmented['image']
            image = np.transpose(image, (2, 1, 0))
            image = torch.from_numpy(image / 255).float()
            target = image
            return image, target, (file_name, label)

    def __len__(self):
        return len(self.file_names)


def save_dataset(data_paths_1, data_paths_2, save_path, overwrite):
    if not isinstance(data_paths_2, list):
        raise TypeError('Must input a list for data_paths_2')

    labels = [0, 1]
    data_paths = [(data_path, labels[0]) for data_path in data_paths_1]
    data_paths.extend([(data_path, labels[1]) for data_path in data_paths_2])

    file_names = []
    if not os.path.exists(save_path) or overwrite:
        with h5py.File(save_path, 'w') as f:
            i = 0
            image_group = f.create_group('images')
            for data_path, label in data_paths:
                for file_name in os.listdir(data_path):
                    file_names.append((file_name, label))
                    file_path = os.path.join(data_path, file_name)
                    data = skimage.io.imread(file_path)
                    image_group.create_dataset(file_name, data=data)

                    i += 1
                    print(i)
    return file_names
