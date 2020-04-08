#!/usr/bin/env python3
"""
This file is the main script to train an AutoEncoder model on a square image set with PyTorcb.
The training set is a collection of publicly available RGB fundus images that have been appropriately resized and
cropped.
This file includes calling the AutoEncoder with hyperparameters, training, and validation.

"""


import albumentations as A
from PIL import ImageFile

from autoencoder import *
from autoencoder_datasets import ImbalancedDataset, UnlabeledDataset
from autoencoder_traintest_functions import *

import torch.multiprocessing
from torch.utils.tensorboard import SummaryWriter


# Paths to source data, validation data, and output saves
transfer_data_path = "../../../../OakData/Transfer_learning_datasets"
data_paths = [os.path.join(transfer_data_path, "train"),
              os.path.join(transfer_data_path, "KaggleDR_crop")]
data_paths_AMD = [os.path.join(transfer_data_path, "train_AMD")]
valid_data_path = [os.path.join(transfer_data_path, "validation")]  # Contains small data set for validation
save_model_path = "../../FLIO-Thesis-Project/AutoEncoder/AutoEncoder_Results"
model_base_name = "autoencoder"

# Training parameters
model_architecture = "noRes"  # Options: noRes, Res34
epoch_num = 300
train_data_type = "AMDonly"  # Options: None, AMDonly
loss_type = "L1"  # Options: L1, MSE
batch_size = 20
num_workers = 12
plot_steps = 200  # Number of steps between getting random input/output to show training progress
stop_condition = 10000  # Number of steps without improvement for early stopping

overwrite = False  # Overwrite existing model to train again


ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.multiprocessing.set_sharing_strategy('file_system')

model_name_pt1 = model_base_name + "_" + model_architecture + "_" + str(epoch_num) + "e_"
if train_data_type:
    model_name = model_name_pt1 + train_data_type + "_" + loss_type + "_batch" + str(batch_size) + "_workers" + str(num_workers)
else:
    model_name = model_name_pt1 + "_" + loss_type +  "batch" + str(batch_size) + "_workers" + str(num_workers)

n_channels = 3

device = torch.device('cuda')  # cpu or cuda
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.backends.cudnn.enabled)

# Create AutoEncoder model path directory if it doesn't exist
model_output_path = os.path.join(save_model_path, model_name)
if not os.path.exists(model_output_path):
    os.mkdir(model_output_path)

# Create validation path directory if it doesn't exist
valid_output_path = os.path.join(valid_data_path[0], model_name)
if not os.path.exists(valid_output_path):
    os.mkdir(valid_output_path)

w = 0.995  # weight of rare event
sample_weights = [1 - w, w]
augmentation_pipeline = A.Compose(
    [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
     A.RandomRotate90(p=0.5),
     A.RandomCrop(128, 128, p=1.0)]
)

unlabeled_dataset = ImbalancedDataset(data_paths_AMD, augmentations=augmentation_pipeline)
dataloader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)


# Define model with hyperparameters
if model_architecture == "Res34":
    model = AutoEncoder_ResEncoder(n_channels=n_channels,
                        n_decoder_filters=[128, 64, 32, 32],
                        trainable=True).to(device)
    model.apply(AutoEncoder_ResEncoder.init_weights)  # initialize model parameters with normal distribution
else:  # noRes
    model = AutoEncoder(n_channels=n_channels,
                        n_encoder_filters=[32, 64, 128, 256, 256],
                        n_decoder_filters=[128, 64, 32, 32],
                        trainable=True).to(device)
    model.apply(AutoEncoder.init_weights)  # Initialize weights


if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
print("Now using", torch.cuda.device_count(), "GPU(s) \n")
model.to(device)

# Select loss
if loss_type == "MSE":
    criterion = nn.MSELoss().to(device)
else:  # L1
    criterion = nn.L1Loss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
# optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8)

# Training
save_model_path_filename = os.path.join(model_output_path, model_name + '.pth')
if not os.path.isfile(save_model_path_filename) or overwrite:
    tensorboard_writer = SummaryWriter(model_output_path)
    dummy = torch.zeros([20, 3, 256, 256], dtype=torch.float)
    tensorboard_writer.add_graph(model, input_to_model=(dummy.to(device), ), verbose=True)
    # tensorboard_writer.add_graph(model, input_to_model=(image,), verbose=True)

    train(model, dataloader, epoch_num, criterion, optimizer, scheduler, device, tensorboard_writer,
          os.path.join(save_model_path, model_name, 'Figures'),
          plot_steps=plot_steps, stop_condition=stop_condition, sample_weights=sample_weights)
    torch.save(model.state_dict(), save_model_path_filename)

    tensorboard_writer.flush()
    tensorboard_writer.close()
else:
    model.load_state_dict(torch.load(save_model_path_filename))

# Testing
validation_dataset = ImbalancedDataset(valid_data_path)
valid_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=True, num_workers=12)

valid_outputs, valid_losses, valid_filenames = test(model, valid_dataloader, criterion, device)
print("Validation outputs")
print(valid_outputs[0])
print("Validation losses")
print(valid_losses[0])

# Save outputs as JPEG images
tensors_to_images(valid_outputs, valid_filenames, valid_output_path)
