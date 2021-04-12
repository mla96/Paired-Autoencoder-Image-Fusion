#!/usr/bin/env python3
"""
This file contains classes and functions to instantiate custom PyTorch Datasets.

Contents
---
    UnlabeledDataset : load unlabeled images for AutoEncoder model
    PairedUnlabeledDataset : load unlabeled RGB fundus images paired with FLIO data as .mat parameters
"""


import os

import hdf5storage
import numpy as np
import skimage.io
import skimage.transform
import torch
from torch.utils.data.dataset import Dataset
import albumentations as A


class UnlabeledDataset(Dataset):
    """
    UnlabeledDataset inherits from the PyTorch Dataset to create an object storing images and paths to be input into the PyTorch DataLoader.

    UnlabeledDataset historically had no labels because only a single data set/data type at a time is used for AutoEncoder reconstruction.
    It now incorporates optional labeling of a second sub-data set, from previously-defined now-defunct class ImbalancedDataset.
    Defining labels for imbalanced sample weights allows us to weight AMD and non-AMD data differently during training/transfer learning.
    
    Attributes:
        file_paths: directories containing image files, optionally labeled to apply different weights later
        transformations: optional albumentations pipeline (normalize) as seen in autoencoder_main.py
        augmentations: optional albumentations pipeline (flips and rotations) as seen in autoencoder_main.py
    """

    def __init__(self, data_paths_1, data_paths_2=None, transformations=None, augmentations=None):
        self.file_paths = []
        labels = (0, 1)  # Keep track of labels
        for data_path in data_paths_1:  # Label 1 samples
            self.file_paths.extend(self.get_file_paths(data_path, labels[0]))

        if data_paths_2:  # Label 2 samples if they are available
            for data_path in data_paths_2:
                self.file_paths.extend(self.get_file_paths(data_path, labels[1]))

        self.transformations = transformations
        self.augmentations = augmentations

    def __getitem__(self, index):
        """
        :param index: index of image path called by PyTorch DataLoader
        :return: identical image and target for AutoEncoder reconstruction, basename as data point identifier, and label
        """
        file_path, label = self.file_paths[index]
        image = skimage.io.imread(file_path)
        if self.augmentations:
            image = self.augmentations(image=image).get('image')
        image = self.transformations(image=image).get('image')  # Normalizes to [-1, 1]
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image).float()
        target = image.clone()
        return image, target, (os.path.basename(file_path), label)

    def __len__(self):
        return len(self.file_paths)

    def get_file_paths(self, data_path, label):
        """
        :param data_path: directory containing image files
        :param label: label distinguishing sub-data set for sample weighting later
        :return: full paths leading to .jpg or .jpeg files, attached to corresponding label
        """
        files = os.listdir(data_path)  # Extracts filenames
        return [(os.path.join(data_path, file), label) for file in files
                if '.jpg' in file or '.jpeg' in file]


class PairedUnlabeledDataset:
    """
    PairedUnlabeledDataset creates an object storing matched fundus and FLIO data to be input into the PyTorch DataLoader.

    FLIO data may be formatted as parameter maps or single-channel Tau_mean images.

    Attributes:
        fundus_ext, flio_ext: string-formatted file extensions
        data: collects only matched data (one file of each type)
        n_fundus_channels, n_flio_channels: number of fundus and FLIO channels, respectively, to be normalized
        fundus_transformations, flio_transformations: appropriate normalization for each data type
        augmentations: optional albumentations pipeline (flips and rotations) as seen in autoencoder_main.py
    """

    def __init__(self, data_path, subdirectories_map, spectral_channel, n_channel_tuple=(3, 3), augmentations=None):
        """
        :param data_path: Leads to subject directories that contain both RGB fundus images and FLIO parameter maps
        :param subdirectories_map: String tuple identifying matched file keywords as named in directories; i.e. ("jpg", ["rgb"]), ("jpg", ["taumean_minscaled"])
        :param spectral_channel: desired channel of FLIO data
        :param n_channel_tuple: number of fundus and FLIO channels, respectively, to be normalized
        """
        self.fundus_ext, self.flio_ext = self.get_extensions(subdirectories_map)

        self.data = []
        data_paths = sorted(os.listdir(data_path)) # "/OakData/FLIO_Data/"
        for data_dir in data_paths:
            if "AMD" in data_dir:  # Exclude "Test"
                target_data = []
                for subdirectory, (ext, keywords) in subdirectories_map.items():
                    target_dir = os.path.join(data_path, data_dir, subdirectory)
                    if os.path.exists(target_dir):  # Remove this when all subjects are registered
                        for file in os.listdir(target_dir):
                            if ext in file and spectral_channel in file and all(k in file for k in keywords):
                                target_data.append(os.path.join(target_dir, file))
                if len(target_data) == 2:
                    self.data.append(tuple(target_data))

        self.n_fundus_channels, self.n_flio_channels = n_channel_tuple
        fundus_normal_params = tuple(0.5 for _ in range(self.n_fundus_channels))
        self.fundus_transformations = A.Compose(
            [A.Normalize(fundus_normal_params, fundus_normal_params)]
        )
        flio_normal_params = tuple(0.5 for _ in range(self.n_flio_channels))
        self.flio_transformations = A.Compose(
            [A.Normalize(flio_normal_params, flio_normal_params)]
        )
        self.augmentations = augmentations

    def __getitem__(self, index):
        """
        :param index: index of image path called by PyTorch DataLoader
        :return: identical images and targets for AutoEncoder reconstruction, and basenames as data point identifier
        """
        image_files = self.data[index]  # Two files in one list
        #for image in raw_image:  # For the fundus image and FLIO map

        fundus_image = skimage.io.imread(image_files[0])
        # fundus_image = skimage.transform.resize(fundus_image, (512, 512))

        if self.flio_ext == 'mat':
            mat_image = channel_Mat2Array(image_files[1])
            flio_image = np.dstack((mat_image['Amplitude1'], mat_image['Amplitude2'], mat_image['Amplitude3'],
                                    mat_image['Tau1'], mat_image['Tau2'], mat_image['Tau3']))
            flio_image = np.flipud(flio_image).copy()
        else:
            flio_image = skimage.io.imread(image_files[1])

        # # Show fundus image
        # fundus_imshow = PIL.Image.fromarray(np.uint8(fundus_image))
        # fundus_imshow.show()
        # # Show amplitude 0:3 of FLIO image
        # flio_imshow = PIL.Image.fromarray(np.uint8(flio_image[:, :, 0:3]))
        # flio_imshow.show()

        if self.augmentations:
            # Implement applying the same transform - for image may not apply
            # Return target properly
            augmented = self.augmentations(image=fundus_image, image2=flio_image)
            fundus_image, flio_image = augmented.get('image'), augmented.get('image2')
        fundus_image = self.fundus_transformations(image=fundus_image).get('image')  # Normalizes to [-1, 1]
        flio_image = self.flio_transformations(image=flio_image).get('image')
        if self.n_flio_channels == 1:
            flio_image = flio_image[:, :, np.newaxis]
        fundus_image, flio_image = np.transpose(fundus_image, (2, 0, 1)), np.transpose(flio_image, (2, 0, 1))
        fundus_image, flio_image = torch.from_numpy(fundus_image).float(), torch.from_numpy(flio_image).float()
        fundus_target, flio_target = fundus_image.clone(), flio_image.clone()
        return fundus_image, flio_image, fundus_target, flio_target, image_files

    def __len__(self):
        return len(self.data)

    def get_extensions(self, subdirectories_map):
        """
        :param subdirectories_map: String tuple identifying matched file keywords as named in directories; i.e. ("jpg", ["rgb"]), ("jpg", ["taumean_minscaled"])
        :return: string-formatted file extensions
        """
        fundus_ext = None
        flio_ext = None
        for subdirectory, (ext, _) in subdirectories_map.items():
            if "fundus" in subdirectory:
                fundus_ext = ext
            elif "FLIO" in subdirectory:
                flio_ext = ext
        return fundus_ext, flio_ext


class FLIOUnlabeledDataset:
    # data_path leads to subject directories that contain both RGB fundus images and FLIO parameter maps
    def __init__(self, data_path, spectral_channel, transformations=None, augmentations=None, taumean=False):
        self.data_path = data_path  # "/OakData/FLIO_Data/"
        self.subdirectory = "FLIO_parameters"

        self.data = []
        # data_paths = sorted(os.listdir(data_path))
        # for data_dir in data_paths:
        #     if "AMD" in data_dir:  # Exclude "Test"
        #         ext = "mat"
        #         if os.path.exists(os.path.join(data_path, data_dir, self.subdirectory)):  # Remove this when all subjects are registered
        #             for file in os.listdir(os.path.join(data_path, data_dir, self.subdirectory)):
        #                 if ext in file and spectral_channel in file:
        #                     self.data.append(os.path.join(data_path, data_dir, self.subdirectory, file))
        f = open(data_path, 'r')
        data_paths = f.readlines()
        f.close()
        for data_dir in data_paths:
            data_dir = data_dir.strip()
            ext = "mat"
            if os.path.exists(os.path.join("../../../../OakData/FLIO_Data", data_dir, self.subdirectory)):  # Remove this when all subjects are registered
                for file in os.listdir(os.path.join("../../../../OakData/FLIO_Data", data_dir, self.subdirectory)):
                    if ext in file and spectral_channel in file:
                        self.data.append(os.path.join("../../../../OakData/FLIO_Data", data_dir, self.subdirectory, file))

        self.transformations = transformations
        self.augmentations = augmentations
        self.taumean = taumean

    def __getitem__(self, index):
        image_files = self.data[index]
        mat_image = hdf5storage.loadmat(image_files)
        mat_image = mat_image['result'][0][0]['results'][0][0]['pixel'][0][0]
        flio_image = np.flipud(np.dstack((mat_image['Amplitude1'], mat_image['Amplitude2'], mat_image['Amplitude3'],
                               mat_image['Tau1'], mat_image['Tau2'], mat_image['Tau3']))).copy()


        # print(np.min(flio_image[:, :, 0]), np.max(flio_image[:, :, 0]))
        # print(np.min(flio_image[:, :, 1]), np.max(flio_image[:, :, 1]))
        # print(np.min(flio_image[:, :, 2]), np.max(flio_image[:, :, 2]))
        # print(np.min(flio_image[:, :, 3]), np.max(flio_image[:, :, 3]))
        # print(np.min(flio_image[:, :, 4]), np.max(flio_image[:, :, 4]))
        # print(np.min(flio_image[:, :, 5]), np.max(flio_image[:, :, 5]))
        # print('\n')

        if self.taumean:
            amplitude_FLIO_image = flio_image[:, :, 0:3]
            tau_FLIO_image = flio_image[:, :, 3:6]
            taumean_FLIO_image = np.sum(amplitude_FLIO_image * tau_FLIO_image, axis=2) / (
                        np.sum(amplitude_FLIO_image, axis=2) + 1e-6)
            taumean_FLIO_image[taumean_FLIO_image <= 300] = 300
            taumean_FLIO_image[taumean_FLIO_image >= 500] = 500
            flio_image = taumean_FLIO_image


        if self.augmentations:
            # Implement applying the same transform - for image may not apply
            # Return target properly
            flio_image = self.augmentations(image=flio_image).get('image')
        # Should the FLIO image be normalized here? There is instance normalization
        if len(flio_image.shape) == 2:
            flio_image = flio_image[:, :, np.newaxis]
            flio_image = flio_image / 255
            flio_image = (flio_image - 0.5) / 0.5
        else:
            flio_image[:, :, 0:3] = self.transformations(image=flio_image[:, :, 0:3]).get('image')
            flio_image[:, :, 3:6] = self.transformations(image=flio_image[:, :, 3:6]).get('image')
        flio_image = np.transpose(flio_image, (2, 0, 1))
        flio_image = torch.from_numpy(flio_image).float()
        flio_target = flio_image.clone()
        return flio_image, flio_target, image_files

    def __len__(self):
        return len(self.data)


# Convert .mat file to array
def channel_Mat2Array(path):
    channel = hdf5storage.loadmat(path)
    return channel['result'][0, 0]['results'][0, 0]['pixel'][0, 0]
    # channel_raw = channel['rawData']
    # channel_rawflat = channel['rawDataFlat']
    # return channel_raw
