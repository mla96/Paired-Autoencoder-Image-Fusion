import os

import PIL.Image
import hdf5storage
import numpy as np
import skimage.io
import torch
from torch.utils.data.dataset import Dataset


class UnlabeledDataset(Dataset):  # Use Dataset or TensorDataset?

    def __init__(self, data_paths, valid_file_types=('jpg', 'jpeg', 'tiff'), augmentations=None):
        self.data_paths = data_paths
        self.valid_file_types = valid_file_types
        self.file_paths = []
        for data_path in data_paths:
            self.file_paths.extend(self.get_file_paths(data_path))

        # self.data = []
        # self.file_names = []

        # for data_path in data_paths:
        #     temp_data, temp_file_names = self.images_to_tensors(data_path)
        #     self.data.extend(temp_data)
        #     self.file_names.extend(temp_file_names)

        self.augmentations = augmentations
        # self.size = self.get_size()

    def __getitem__(self, index):
        file_path = self.file_paths[index]
        image = skimage.io.imread(file_path)
        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented['image']
        image = np.transpose(image, (2, 1, 0))
        image = torch.from_numpy(image / 255).float()
        target = image
        return image, target, os.path.basename(file_path)

    def __len__(self):
        return len(self.file_paths)

    # def get_size(self):
    #     file_path = self.file_paths[0]
    #     image = PIL.Image.open(file_path)
    #     tensor_image = self.transformations(np.array(image))
    #     return tensor_image.size()

    # def __getitem__(self, index):
    #     image = None
    #     if self.augmentations:
    #         raw_image = np.array(self.data[index])
    #         augmented = self.augmentations(image=raw_image)
    #         image = self.transformations(augmented['image'])
    #     else:
    #         image = self.transformations(self.data[index])
    #     label = image
    #     return image, label, self.file_names[index]
    #
    # def __len__(self):
    #     return len(self.data)

    def get_file_paths(self, data_path):
        files = os.listdir(data_path)  # Extracts the file names technically
        return [os.path.join(data_path, file) for file in files if 'jpg' in file or 'jpeg' in file]

    # Too memory intensive because it opens image every time to store in array
    # def images_to_tensors(self, data_path):
    #     files = sorted(os.listdir(data_path))
    #     data = []
    #     file_names = []
    #     for file in files:
    #         if 'jpg' in file or 'jpeg' in file:
    #             img = PIL.Image.open(os.path.join(data_path, file))
    #             copy_img = img.copy()
    #             data.append(copy_img)
    #             file_names.append(file)
    #     return data, file_names


# Labels for AMD (or mixed sets) and non-AMD data for sample weighting during transfer learning
class ImbalancedDataset(Dataset):

    def __init__(self, data_paths_1, data_paths_2=[], # sample_weights=[0.005, 0.995],
                 augmentations=None):
        self.data_paths_1 = data_paths_1  # Label 1 samples
        self.data_paths_2 = data_paths_2  # Label 2 samples
        # self.sample_weights = sample_weights  # Weighting each sample for contributions to the loss
        self.labels = [0, 1]  # In case I want different labels
        self.file_paths = []
        for data_path in self.data_paths_1:
            self.file_paths.extend(self.get_file_paths(data_path, self.labels[0]))
        for data_path in self.data_paths_2:
            self.file_paths.extend(self.get_file_paths(data_path, self.labels[1]))

        self.augmentations = augmentations
        # self.size = self.get_size()

    def __getitem__(self, index):
        file_path, label = self.file_paths[index]
        image = skimage.io.imread(file_path)
        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented['image']
        image = np.transpose(image, (2, 1, 0))
        image = torch.from_numpy(image / 255).float()
        target = image
        return image, target, (os.path.basename(file_path), label)

    def __len__(self):
        return len(self.file_paths)

    def get_file_paths(self, data_path, label):
        files = os.listdir(data_path)  # Extracts the file names technically
        return [(os.path.join(data_path, file), label) for file in files if 'jpg' in file or 'jpeg' in file]


# For paired RGB fundus images and FLIO parameter map data
class PairedUnlabeledDataset(UnlabeledDataset):
    # data_path leads to subject directories that contain both RGB fundus images and FLIO parameter maps
    def __init__(self, data_path, subdirectories, filetype, spectral_channel, augmentations=None):
        # self.data_paths = [data_path] if data_path is not isinstance(data_path, list) else data_path
        self.data_path = data_path  # "/OakData/FLIO_Data/"
        self.subdirectories = subdirectories  # ["fundus_registered", "FLIO_parameters"]
        # self.data_path = self.data_paths[0]
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
        self.size = 0
        self.augmentations = augmentations


    def __getitem__(self, index):
        image_files = self.data[index]  # Two files in one list
        #for image in raw_image:  # For the fundus image and FLIO map

        fundus_image = skimage.io.imread(image_files[0])
        mat_image = hdf5storage.loadmat(image_files[1])
        mat_image = mat_image['result'][0][0]['results'][0][0]['pixel'][0][0]
        flio_image = np.dstack((mat_image['Amplitude1'], mat_image['Amplitude2'], mat_image['Amplitude3'],
                               mat_image['Tau1'], mat_image['Tau2'], mat_image['Tau3']))
        if self.augmentations:
            # Implement applying the same transform - for image may not apply
            # Return target properly
            augmented = self.augmentations(image=fundus_image, image2=flio_image)
            image, image2 = augmented['image'], augmented['image2']
        image, image2 = np.transpose(image, (2, 1, 0)), np.transpose(image2, (2, 1, 0))
        image, image2 = torch.from_numpy(image / 255).float(), torch.from_numpy(image2 / 255).float()
        target, target2 = image, image2
        return image, image2, target, target2, image_files

    def __len__(self):
        return len(self.data)

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
