import os

import PIL.Image
import h5py
import numpy as np

# Convert large training datasets to hdf5 files for (goal) more efficient DataLoader use


transfer_data_path = "../../../../OakData/Transfer_learning_datasets"
data_paths = [os.path.join(transfer_data_path, "train"),
              os.path.join(transfer_data_path, "KaggleDR_crop")]
data_paths_AMD = [os.path.join(transfer_data_path, "train_AMD")]
save_path = os.path.join(transfer_data_path, "train.hdf5")
save_path_AMD = os.path.join(transfer_data_path, "train_AMD.hdf5")

names_AMD = []
for data_path in data_paths_AMD:
    for file_name in os.listdir(data_path):
        names_AMD.append(os.path.splitext(file_name)[0])


def get_data_specs(data_path):
    file_name = os.listdir(data_path)[0]
    file_path = os.path.join(data_path, file_name)
    data = np.array(PIL.Image.open(file_path))
    return data.shape, type(data[0][0][0])


# check what type these images are when loaded into numpy? Generally go for native type
def save_dataset(data_paths, save_path, overwrite=False):
    if not os.path.exists(save_path) or overwrite:
        with h5py.File(save_path, 'w') as f:
            # d_shape, d_type = get_data_specs(data_paths[0])
            i = 0
            for data_path in data_paths:
                for file_name in os.listdir(data_path):
                    file_path = os.path.join(data_path, file_name)
                    data = np.asarray(PIL.Image.open(file_path))
                    f.create_dataset(file_name, data=data, chunks=True,
                                     compression="gzip", compression_opts=9,
                                     shuffle=True, fletcher32=True)
                    i += 1

# for hdf5 1imageperdataset, get index and index into list of filenames to get key.
# if index is greater than length of first hdf5 file, then call second one


def save_dataset_dim(data_paths, save_path, overwrite=False):
    if not os.path.exists(save_path) or overwrite:
        with h5py.File(save_path, 'w') as f:
            d_shape, d_type = get_data_specs(data_paths[0])
            i = 0
            dim_set = set()
            for data_path in data_paths:
                for file_name in os.listdir(data_path):
                    # Make a dict of existing dimensions, if new dimensions match in the dict then add to that dataset

                    file_path = os.path.join(data_path, file_name)
                    data = np.array(PIL.Image.open(file_path))
                    dimension = data.shape

                    if dimension not in dict:
                        dataset = f.create_dataset(str(dimension), data=data, chunks=True,
                                                   compression="gzip", compression_opts=9,
                                                   shuffle=True, fletcher32=True)
                        # Add dimension to set
                        dim_set.add(dimension)
                    else:
                        # f[str(dimension)][i] = data
                        # append/resize??
                        f[str(dimension)][i] = data
                    i += 1

# def save_dataset(data_paths, save_path, overwrite=False):
#     if not os.path.exists(save_path) or overwrite:
#         with h5py.File(save_path, 'w') as f:
#             d_shape, d_type = get_data_specs(data_paths[0])
#             dataset = f.create_dataset('cool', (100, *d_shape), d_type)
#             i = 0
#             for data_path in data_paths:
#                 for file_name in os.listdir(data_path):
#                     file_path = os.path.join(data_path, file_name)
#                     data = np.array(PIL.Image.open(file_path))
#                     f['cool'][i] = data
#                     i += 1


save_dataset(data_paths, save_path, overwrite=True)
# save_dataset(data_paths_AMD, save_path_AMD, overwrite=True)


with h5py.File(save_path_AMD, 'r') as f:
    for name in names_AMD:
        image = f.get(name)
        print(image, type(image))

    # List all groups
    print("Keys: %s" % f.keys())
    group_keys = list(f.keys())
    print(group_keys)
    print(len(group_keys))

    # Get the data
    for key in group_keys:
        data = list(f[key])


# def store_single_hdf5(image, image_id, label):
#     """ Stores a single image to an HDF5 file.
#         Parameters:
#         ---------------
#         image       image array, (32, 32, 3) to be stored
#         image_id    integer unique ID for image
#         label       image label
#     """
#     # Create a new HDF5 file
#     file = h5py.File(hdf5_dir / f"{image_id}.h5", "w")
#
#     # Create a dataset in the file
#     dataset = file.create_dataset(
#         "image", np.shape(image), h5py.h5t.STD_U8BE, data=image
#     )
#     meta_set = file.create_dataset(
#         "meta", np.shape(label), h5py.h5t.STD_U8BE, data=label
#     )
#     file.close()
#
#
# def store_many_disk(images, labels):
#     """ Stores an array of images to disk
#         Parameters:
#         ---------------
#         images       images array, (N, 32, 32, 3) to be stored
#         labels       labels array, (N, 1) to be stored
#     """
#     num_images = len(images)
#
#     # Save all the images one by one
#     for i, image in enumerate(images):
#         Image.fromarray(image).save(disk_dir / f"{i}.png")
#
#     # Save all the labels to the csv file
#     with open(disk_dir / f"{num_images}.csv", "w") as csvfile:
#         writer = csv.writer(
#             csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
#         )
#         for label in labels:
#             # This typically would be more than just one value per row
#             writer.writerow([label])
#
# def store_many_hdf5(images, labels):
#     """ Stores an array of images to HDF5.
#         Parameters:
#         ---------------
#         images       images array, (N, 32, 32, 3) to be stored
#         labels       labels array, (N, 1) to be stored
#     """
#     num_images = len(images)
#
#     # Create a new HDF5 file
#     file = h5py.File(hdf5_dir / f"{num_images}_many.h5", "w")
#
#     # Create a dataset in the file
#     dataset = file.create_dataset(
#         "images", np.shape(images), h5py.h5t.STD_U8BE, data=images
#     )
#     meta_set = file.create_dataset(
#         "meta", np.shape(labels), h5py.h5t.STD_U8BE, data=labels
#     )
#     file.close()