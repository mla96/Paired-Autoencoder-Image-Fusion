import os

import PIL.Image
import numpy as np
from torch.utils.data.dataset import Dataset


filetype = ["jpg", "mat"]  # String identifiers for the desired files in each subdirectory


class UnlabeledDataset(Dataset):  # Use Dataset or TensorDataset?

    def __init__(self, data_paths, transformations=None, augmentations=None):
        self.data_paths = data_paths
        self.file_paths = []
        for data_path in data_paths:
            self.file_paths.extend(self.get_file_paths(data_path))

        # self.data = []
        # self.file_names = []

        # for data_path in data_paths:
        #     temp_data, temp_file_names = self.images_to_tensors(data_path)
        #     self.data.extend(temp_data)
        #     self.file_names.extend(temp_file_names)

        self.transformations = transformations
        self.augmentations = augmentations
        self.size = self.__getitem__(0)[0].size()

    def __getitem__(self, index):
        file_path = self.file_paths[index]
        image = PIL.Image.open(file_path)
        if self.augmentations:
            raw_image = np.array(image)
            augmented = self.augmentations(image=raw_image)
            image = self.transformations(augmented['image'])
        else:
            image = self.transformations(image)
        label = image
        return image, label, os.path.basename(file_path)

    def __len__(self):
        return len(self.file_paths)

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


    def images_to_tensors(self, data_path):
        files = sorted(os.listdir(data_path))
        data = []
        file_names = []
        for file in files:
            if 'jpg' in file or 'jpeg' in file:
                img = PIL.Image.open(os.path.join(data_path, file))
                copy_img = img.copy()
                data.append(copy_img)
                file_names.append(file)

        return data, file_names


# For paired RGB fundus images and FLIO paramater map data
class PairedUnlabeledDataset(UnlabeledDataset):
    # data_path leads to subject directories that contain both RGB fundus images and FLIO parameter maps
    def __init__(self, data_path, subdirectories, filetype, transformations=None, augmentations=None):
        super().__init__(data_path, transformations=transformations, augmentations=augmentations)
        self.subdirectories = subdirectories
        self.filetype = filetype
        self.data, self.file_names = self.images_to_tensors(data_path, subdirectories)
        # self.size = self.__getitem__(0)[0].size()

    def __getitem__(self, index):
        image = None
        raw_image = np.array(self.data[index])
        for image in raw_image:
            # IN PROGRESS.....................................................
            if self.augmentations:
                augmented = self.augmentations(image=raw_image)
                image = self.transformations(augmented['image'])
            else:
                image = self.transformations(self.data[index])
        label = image
        return image, label, self.file_names[index]

    def __len__(self):
        return len(self.data)

    def images_to_tensors(self, data_path, subdirectories):
        subjects = sorted(os.listdir(data_path))  # Deal with "TEST SUBJECT"
        data = []
        for subject in subjects:
            temp_subject = []
            # Go into subdirectories
            for file in os.listdir(os.path.join(data_path, subdirectories[0])):  # RGB fundus images
                if filetype[0] in file:  # If a particular identifier string is in this file, in this directory
                    temp_subject.append(PIL.Image.open(os.path.join(data_path, subdirectories[0], file)))
            for file in os.listdir(os.path.join(data_path, subdirectories[1])):  # FLIO parameter maps
                if filetype[1] in file:
                    temp_subject.append(os.path.join(data_path, subdirectories[0], file))
                    # THIS IS A MAT FILE, SHOULD BE READ APPROPRIATELY...
            data.append([temp_subject[0], temp_subject[1]])

        return data, subjects
