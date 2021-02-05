#!/usr/bin/env python3
"""
This file is the main script to train an AutoEncoder model on a square image set with PyTorcb.
The training set is a collection of publicly available RGB fundus images that have been appropriately resized and
cropped.
This file includes calling the AutoEncoder with hyperparameters, training, and validation.
"""


import albumentations as A
from fastai.vision import DataLoader
import os
from PIL import ImageFile
import numpy as np

import torch.multiprocessing
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from autoencoder import AutoEncoder, AutoEncoder_ResEncoder
from autoencoder_datasets import UnlabeledDataset
from autoencoder_traintest_functions import train, test, tensors_to_images, get_tsne
from loss_utils import get_loss_type, SSIM_Loss, MS_SSIM_Loss, MS_SSIM_L1_Loss
from torchsummary import summary


# Paths to source data, validation data, and output saves
transfer_data_path = "../../../../OakData/Transfer_learning_datasets"
data_paths = [os.path.join(transfer_data_path, "train")]
              # os.path.join(transfer_data_path, "KaggleDR_crop512")]
data_paths_AMD = [os.path.join(transfer_data_path, "train_AMD")]
valid_data_path = [os.path.join(transfer_data_path, "validation")]  # Contains small data set for validation
save_model_path = "../../FLIO-Thesis-Project/AutoEncoder/AutoEncoder_Results"
model_base_name = "sepcoder_autoencoder_32-64-64-16-64-64-32_tanh_31_lr0035_sched8_MSorig"

# Training parameters
model_architecture = "noRes"  # Options: noRes, Res34
epoch_num = 10
train_data_type = None  # Options: None, AMDonly
loss_type = "MS-SSIM-L1"  # Options: L1, MSE, SSIM, MS-SSIM, MS-SSIM-L1
batch_size = 16
num_workers = 12
plot_steps = 500  # Number of steps between getting random input/output to plot training progress in TensorBoard
stop_condition = 10000  # Number of steps without improvement for early stopping
w = 0.995  # weight of rare event
sample_weights = [1 - w, w]  # Options: None, [1 - w, w]

overwrite = False  # Overwrite existing model to train again
t_sne = True


# Create model name based on model_base_name and training parameters
model_name_pt1 = model_base_name + "_" + model_architecture + "_" + str(epoch_num) + "e_"
if train_data_type:
    model_name = model_name_pt1 + train_data_type + "_" + loss_type + "_batch" + str(batch_size) + "_workers" + str(num_workers)
else:
    model_name = model_name_pt1 + loss_type + "_batch" + str(batch_size) + "_workers" + str(num_workers)

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.multiprocessing.set_sharing_strategy('file_system')

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

n_channels = 3
transformation_pipeline = A.Compose(
    [A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

# Isometric augmentatations
augmentations = [A.HorizontalFlip(p=0.5),
                 A.VerticalFlip(p=0.5),
                 A.RandomRotate90(p=0.5)]
# MS-SSIM requires images larger than 160 x 160 to calculate loss
if loss_type == "MS-SSIM" or loss_type == "MS-SSIM-L1":
    augmentations.append(A.RandomCrop(256, 256, p=1.0))
else:
    augmentations.append(A.RandomCrop(128, 128, p=1.0))
# Color augmentations must be done after cropping
# augmentations.extend([
#     A.HueSaturationValue(p=0.3),  # Use color augmentations? Some results seem to get worse
#     A.RandomBrightnessContrast(p=0.3)
#     # FancyPCA(p=0.5)
# ])
augmentation_pipeline = A.Compose(augmentations)


# Load training set
unlabeled_dataset = UnlabeledDataset(data_paths, data_paths_AMD,
                                     transformations=transformation_pipeline, augmentations=augmentation_pipeline)
dataloader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)


# Define model with hyperparameters
if model_architecture == "Res34":
    model = AutoEncoder_ResEncoder(n_channels=n_channels,
                        n_decoder_filters=[256, 128, 64, 32, 16],
                        trainable=True).to(device)
    model.apply(AutoEncoder_ResEncoder.init_weights)  # initialize model parameters with normal distribution
else:  # noRes
    model = AutoEncoder(n_channels=n_channels,
                        n_encoder_filters=[32, 64, 64, 16],
                        n_decoder_filters=[64, 64, 32],
                        trainable=False).to(device)
    model.apply(AutoEncoder.init_weights)  # Initialize weights

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
print("Now using", torch.cuda.device_count(), "GPU(s) \n")
model.to(device)

# summary(model, input_size=(3, 256, 256))

# Select loss
criterion = get_loss_type(loss_type).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0035)
# optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8)

# TRAINING
save_model_path_filename = os.path.join(model_output_path, model_name + '.pth')
if not os.path.isfile(save_model_path_filename) or overwrite:  # If training a new model
    # TensorBoard
    tensorboard_writer = SummaryWriter(model_output_path)
    dummy = torch.zeros([20, 3, 128, 128], dtype=torch.float)
    tensorboard_writer.add_graph(model, input_to_model=(dummy.to(device), ), verbose=True)
    # tensorboard_writer.add_graph(model, input_to_model=(image,), verbose=True)

    train(model, dataloader, epoch_num, criterion, optimizer, scheduler, device, tensorboard_writer,
          os.path.join(save_model_path, model_name, 'Figures'),
          plot_steps=plot_steps, stop_condition=stop_condition, sample_weights=sample_weights)
    torch.save(model.state_dict(), save_model_path_filename)

    tensorboard_writer.flush()
    tensorboard_writer.close()
else:  # Load old model with the given name
    model.load_state_dict(torch.load(save_model_path_filename), strict=False)

# TESTING
validation_dataset = UnlabeledDataset(valid_data_path, transformations=transformation_pipeline)
valid_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

valid_outputs, valid_losses, valid_filenames = test(model, valid_dataloader, criterion, device)
print("Validation outputs")
print(valid_outputs[0])
print("Validation losses")
print(valid_losses[0])

# Save outputs as JPEG images
tensors_to_images(valid_outputs, valid_filenames, valid_output_path)

# t-SNE
if t_sne:
    tsne_dataset = UnlabeledDataset(data_paths, data_paths_AMD, transformations=transformation_pipeline)
    tsne_dataloader = DataLoader(tsne_dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
    tsne_fitted, sw = get_tsne(model, tsne_dataloader, device)
    tx, ty = tsne_fitted[:, 0], tsne_fitted[:, 1]
    tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))
    sw = np.array(sw)
    plt.scatter(tx, ty, c=sw)
    plt.savefig(model_output_path)
