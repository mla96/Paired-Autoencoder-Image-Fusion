#!/usr/bin/env python3
"""
This file contains classes and functions to instantiate custom PyTorch Datasets.

Contents
---
    UnlabeledDataset : load unlabeled images for AutoEncoder model
    ImbalancedDataset : almost the same thing as UnlabeledDataset but accommodates sample weights; will likely combine
    PairedUnlabeledDataset : load unlabeled RGB fundus images paired with FLIO data as .mat parameters
"""


import os

import PIL.Image
import hdf5storage
import numpy as np
import skimage.io
import skimage.transform
import torch
from torch.utils.data.dataset import Dataset


# UnlabeledDataset takes care of imbalanced sample weights by defining labels
# Labels for AMD (or mixed sets) and non-AMD data for sample weighting during transfer learning
class UnlabeledDataset(Dataset):

    def __init__(self, data_paths_1, data_paths_2=None, transformations=None, augmentations=None):
        self.data_paths_1 = data_paths_1  # Label 1 samples
        self.labels = (0, 1)  # Keep track of labels
        self.file_paths = []
        for data_path in self.data_paths_1:
            self.file_paths.extend(self.get_file_paths(data_path, self.labels[0]))

        # Add to file_paths if data_paths_2 variable exists
        if data_paths_2:
            self.data_paths_2 = data_paths_2  # Label 2 samples
            for data_path in self.data_paths_2:
                self.file_paths.extend(self.get_file_paths(data_path, self.labels[1]))

        self.transformations = transformations
        self.augmentations = augmentations

    def __getitem__(self, index):
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
        files = os.listdir(data_path)  # Extracts the file names technically
        return [(os.path.join(data_path, file), label) for file in files
                if '.jpg' in file or '.jpeg' in file]


class PairedUnlabeledDataset(UnlabeledDataset):
    # data_path leads to subject directories that contain both RGB fundus images and FLIO parameter maps
    def __init__(self, data_path, subdirectories, filetype, spectral_channel, transformations=None, augmentations=None):
        self.data_path = data_path  # "/OakData/FLIO_Data/"
        self.subdirectories = subdirectories  # ["fundus_registered", "FLIO_parameters"]
        self.filetype = filetype  # [["tiff", "fullsize"], "mat"]  If item is list, then index in for "keyword"
        self.spectral_channel = spectral_channel

        self.data = []
        data_paths = sorted(os.listdir(data_path))
        for data_dir in data_paths:
            if "AMD" in data_dir:  # Exclude "Test"
                data_append = []
                for i in range(len(subdirectories)):
                    keyword = []
                    ftype = filetype[i]
                    if isinstance(ftype, list):
                        ext = ftype[0]
                        for word in ftype[1:]:
                            keyword.append(word)
                    else:
                        ext = ftype
                    if os.path.exists(os.path.join(data_path, data_dir, subdirectories[i])):  # Remove this when all subjects are registered
                        for file in os.listdir(os.path.join(data_path, data_dir, subdirectories[i])):
                            if ext in file and spectral_channel in file and all(k in file for k in keyword):
                                data_append.append(os.path.join(data_path, data_dir, subdirectories[i], file))
                if len(data_append) == 2:
                    self.data.append(data_append)


        # self.data, self.file_names = self.images_to_tensors(data_path, subdirectories)
        # self.data = [[[[1, 2], [3, 4]]], [[[2, 3], [4, 5]]]]
        self.transformations = transformations
        self.augmentations = augmentations


    def __getitem__(self, index):
        image_files = self.data[index]  # Two files in one list
        #for image in raw_image:  # For the fundus image and FLIO map

        fundus_image = skimage.io.imread(image_files[0])
        # fundus_image = skimage.transform.resize(fundus_image, (512, 512))
        mat_image = hdf5storage.loadmat(image_files[1])
        mat_image = mat_image['result'][0][0]['results'][0][0]['pixel'][0][0]
        flio_image = np.dstack((mat_image['Amplitude1'], mat_image['Amplitude2'], mat_image['Amplitude3'],
                               mat_image['Tau1'], mat_image['Tau2'], mat_image['Tau3']))

        if self.augmentations:
            # Implement applying the same transform - for image may not apply
            # Return target properly
            augmented = self.augmentations(image=fundus_image, image2=flio_image)
            fundus_image, flio_image = augmented['image'], augmented['image2']
        # Should the FLIO image be normalized here? There is instance normalization
        fundus_image = self.transformations(image=fundus_image).get('image')  # Normalizes to [-1, 1]
        fundus_image, flio_image = np.transpose(fundus_image, (2, 0, 1)), np.transpose(flio_image, (2, 0, 1))
        fundus_image, flio_image = torch.from_numpy(fundus_image).float(), torch.from_numpy(flio_image).float()
        fundus_target, flio_target = fundus_image.clone(), flio_image.clone()
        return fundus_image, flio_image, fundus_target, flio_target, image_files

    def __len__(self):
        return len(self.data)

    # Clean this up
    def images_to_tensors(self, data_path, subdirectories):  # Does not convert to tensors right now, assembles data
        subjects = sorted(os.listdir(data_path))
        data = []
        for subject in subjects:
            if 'AMD_01' in subject and os.path.isdir(os.path.join(data_path, subject)):  # Ignore other directories
                temp_subject = []
                # Go into subdirectories
                for file in os.listdir(os.path.join(data_path, subject, subdirectories[0])):  # RGB fundus images
                    if self.filetype[0] in file and 'registered_fullsize' in file and self.spectral_channel in file:  # If a particular identifier string is in this file, in this directory
                        temp_subject.append(PIL.Image.open(os.path.join(data_path, subject, subdirectories[0], file)))
                for file in os.listdir(os.path.join(data_path, subject, subdirectories[1])):  # FLIO parameter maps
                    if self.filetype[1] in file and 'result' in file and self.spectral_channel in file:  # Channel 2 (LSC) first, as changes are more obvious
                        # Create array from complete result.mat file
                        mat_file = channel_Mat2Array(os.path.join(data_path, subject, subdirectories[1], file))
                        # Create arrays from first 3 parameters and second 3 parameters
                        # image_012 = np.array([mat_file['Amplitude1'], mat_file['Amplitude2'], mat_file['Amplitude3']])
                        # image_345 = np.array([mat_file['Tau1'], mat_file['Tau2'], mat_file['Tau3']])
                        param_tensor = np.asarray([mat_file['Amplitude1'], mat_file['Amplitude2'], mat_file['Amplitude3'],
                                        mat_file['Tau1'], mat_file['Tau2'], mat_file['Tau3']])

                        # Output 3 images
                        # temp_subject.append(image_012)
                        # temp_subject.append(image_345)
                        temp_subject.append(param_tensor)
                if len(temp_subject) == 2:  # If both files are present in directory and contained in temporary variable
                    data.append([temp_subject[0], temp_subject[1]])

        return data, subjects


# Convert .mat file to array
def channel_Mat2Array(path):
    channel = hdf5storage.loadmat(path)
    return channel['result'][0, 0]['results'][0, 0]['pixel'][0, 0]
    # channel_raw = channel['rawData']
    # channel_rawflat = channel['rawDataFlat']
    # return channel_raw
