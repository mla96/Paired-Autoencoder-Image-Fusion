import os

import h5py
import numpy as np
import skimage.io
import torch
from torch.utils.data.dataset import Dataset

import PIL.Image
import collections

class ImbalancedHDF5Dataset(Dataset):

    def __init__(self, data_dictionary, data_specs, save_path, augmentations=None):
        self.data_dictionary = data_dictionary  # Contains labels
        self.data_specs = data_specs
        self.save_path = save_path
        self.augmentations = augmentations

        self.num_image_list = list(data_specs.values())
        self.cumulative_lengths = self.get_cumulative_lengths()

    def __getitem__(self, index):
        dataset_num, position = self.find_position(index)
        data_shape = list(self.data_specs.keys())[dataset_num]
        file_path, label = self.data_dictionary.get(data_shape)[position]

        with h5py.File(self.save_path, 'r') as f:
            image_group = f['images']
            image = image_group[f'data_size {data_shape}'].value[position]
            if self.augmentations:
                augmented = self.augmentations(image=image)
                image = augmented['image']
            image = np.transpose(image, (2, 1, 0))
            image = torch.from_numpy(image / 255).float()
            target = image
            return image, target, (file_path, label)

    def __len__(self):
        return self.cumulative_lengths[-1]

    def get_cumulative_lengths(self):
        cumulative_image_list = []
        start = 0
        for num in self.num_image_list:
            start += num
            cumulative_image_list.append(start)
        return cumulative_image_list

    def find_position(self, index):
        i, n = 0, len(self.cumulative_lengths)
        while index >= self.cumulative_lengths[i] and i < n:
            i += 1  # finds which dataset to go into
        delta = self.cumulative_lengths[i] - index
        position = self.num_image_list[i] - delta
        return i, position


def save_dataset(data_dictionary, save_path, overwrite):
    if not os.path.exists(save_path) or overwrite:
        print('Saving dataset...')
        file_counter = 0
        with h5py.File(save_path, 'w') as f:
            image_group = f.create_group('images')
            for image_shape, values in data_dictionary.items():
                num_images = len(values)  # number of images with same dimensions
                data_dim = (num_images, *image_shape)

                image_counter = 0
                same_shape_data = np.zeros(data_dim)
                for file_path, labels in values:
                    data = skimage.io.imread(file_path)
                    same_shape_data[image_counter, :, :, :] = data
                    image_counter += 1

                    file_counter += 1
                    if file_counter % 10 == 0:
                        print(f'Processed {file_counter} files...')
                print(image_shape)
                image_group.create_dataset(f'data_size {image_shape}', data=same_shape_data)


def get_data_dictionary(data_paths_1, data_paths_2):
    if not isinstance(data_paths_2, list):
        raise TypeError('Must input a list for data_paths_2')

    # Assumes that all images have 3 channels
    n_channels = 3

    labels = [0, 1]
    data_paths = [(data_path, labels[0]) for data_path in data_paths_1]
    data_paths.extend([(data_path, labels[1]) for data_path in data_paths_2])

    # Keys are data shapes and values are list of file paths attached with labels
    print('Collecting terms for data dictionary...')
    data_dictionary = collections.defaultdict(list)
    count_data_path = 0
    for data_path, label in data_paths:
        count_files_collected = 0
        for file_name in os.listdir(data_path):
            file_path = os.path.join(data_path, file_name)
            data = PIL.Image.open(file_path)  # Opening for size so PIL is much faster with this
            data_shape = (*data.size, n_channels)
            data_dictionary[data_shape].append((file_path, label))

            count_files_collected += 1
            if count_files_collected % 100 == 0:
                print(f'Collected {count_files_collected} files...')

        count_data_path += 1
        print(f'Finished data path {count_data_path}!')
    return data_dictionary


def get_default_dict_specs(default_dict):
    shapes_to_num = collections.OrderedDict()
    for k, v in default_dict.items():
        shapes_to_num[k] = len(v)
    return shapes_to_num


# class ImbalancedHDF5Dataset(Dataset):
#
#     def __init__(self, file_names, save_path, augmentations=None):
#         self.file_names = file_names  # Contains labels
#         self.save_path = save_path
#         self.augmentations = augmentations
#
#     def __getitem__(self, index):
#         file_name, label = self.file_names[index]
#         with h5py.File(self.save_path, 'r') as f:
#             image_group = f['images']
#             image = image_group[file_name].value
#             if self.augmentations:
#                 augmented = self.augmentations(image=image)
#                 image = augmented['image']
#             image = np.transpose(image, (2, 1, 0))
#             image = torch.from_numpy(image / 255).float()
#             target = image
#             return image, target, (file_name, label)
#
#     def __len__(self):
#         return len(self.file_names)
#
#
# def save_dataset(data_paths_1, data_paths_2, save_path, overwrite):
#     if not isinstance(data_paths_2, list):
#         raise TypeError('Must input a list for data_paths_2')
#
#     labels = [0, 1]
#     data_paths = [(data_path, labels[0]) for data_path in data_paths_1]
#     data_paths.extend([(data_path, labels[1]) for data_path in data_paths_2])
#
#     file_names = []
#     if not os.path.exists(save_path) or overwrite:
#         with h5py.File(save_path, 'w') as f:
#             i = 0
#             image_group = f.create_group('images')
#             for data_path, label in data_paths:
#                 for file_name in os.listdir(data_path):
#                     file_names.append((file_name, label))
#                     file_path = os.path.join(data_path, file_name)
#                     data = skimage.io.imread(file_path)
#                     image_group.create_dataset(file_name, data=data)
#
#                     i += 1
#                     print(i)
#     return file_names
