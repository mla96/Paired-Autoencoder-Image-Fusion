import os

from hdf5_dataset import save_dataset, get_data_dictionary, get_default_dict_specs

transfer_data_path = "../../../../OakData/Transfer_learning_datasets"
data_paths = [os.path.join(transfer_data_path, "train"),
              os.path.join(transfer_data_path, "KaggleDR_crop")]
data_paths_AMD = [os.path.join(transfer_data_path, "train_AMD")]
save_path = os.path.join(transfer_data_path, "train.hdf5")

data_dictionary = get_data_dictionary(data_paths, data_paths_AMD)
print(get_default_dict_specs(data_dictionary))
save_dataset(data_dictionary, save_path, overwrite=True)