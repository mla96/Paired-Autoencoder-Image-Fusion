#!/usr/bin/env python3
"""
This file is the main script to train paired AutoEncoder models on RGB fundus and FLIO parameter datasets with PyTorcb.
This file includes calling the AutoEncoders with hyperparameters, training, and validation.
"""


import albumentations as A
from fastai.vision import DataLoader
import os
from PIL import ImageFile

import torch.multiprocessing
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from paired_autoencoder import AutoEncoder_fundus, AutoEncoder_FLIO, AutoEncoder_ResEncoder_FLIO, AutoEncoder_ResEncoder_Fundus
from autoencoder_datasets import PairedUnlabeledDataset
from paired_autoencoder_traintest_functions import train, test, tensors_to_images
from loss_utils import get_loss_type, SSIM_Loss, MS_SSIM_Loss, MS_SSIM_L1_Loss


# Paths to source data, validation data, and output saves
data_path = "../../../../OakData/FLIO_Data"
subdirectories_map = {"fundus_registered": ("jpg", ["rgb"]), "FLIO_parameters": ("jpg", ["taumean_minscaled"])}  # Nested in each subject folder
# ("jpg", ["rgb"]), ("mat", [])  # Tuple identifiers for the desired files in each subdirectory
# ("jpg", ["rgb"]), ("jpg", ["taumean_minscaled"])
save_model_path = "../../FLIO-Thesis-Project/AutoEncoder/Paired_AutoEncoder_Results"
model_base_name = "autoencoder_32-64-64-16-64-64-32_lossfundus0pt01flio0pt01_freeze_taumean"
valid_data_path = [os.path.join(data_path, "validation")]  # Contains small data set for validation
# valid_data_path = os.path.join(transfer_data_path, "STARE/Test_Crop/validation")
transfer_learning_model_path = "../../FLIO-Thesis-Project/AutoEncoder/AutoEncoder_Results/" \
                               "sepcoder_autoencoder_32-64-64-16-64-64-32_tanh_31_lr0035_sched8_MSorig_noRes_512e_MS-SSIM-L1_batch16_workers12/" \
                               "sepcoder_autoencoder_32-64-64-16-64-64-32_tanh_31_lr0035_sched8_MSorig_noRes_512e_MS-SSIM-L1_batch16_workers12.pth"

# Training parameters
model_architecture = "noRes"  # Options: noRes, Res34
epoch_num = 512
loss_type = "MS-SSIM-L1"  # Options: L1, MSE, SSIM, MS-SSIM, MS-SSIM-L1
loss_type_latent = "MEF-MSSSIM"  # Options: MEFSSIM, MEF-MSSSIM, MEF-MSSSIM-L1
batch_size = 1
num_workers = 12
plot_steps = 500  # Number of steps between getting random input/output to plot training progress in TensorBoard
stop_condition = 10000  # Number of steps without improvement for early stopping

overwrite = True  # Overwrite existing model to train again
load_pretrained = True  # Load pretrained model for RGB fundus AutoEncoder
freeze = True

n_channels = 3
n_channels2 = 1
spectral_channel = "02"

# Create model name based on model_base_name and training parameters
model_name = model_base_name + "_" + model_architecture + "_" + str(epoch_num) + "e_" + loss_type_latent + "_batch" + str(batch_size) + "_workers" + str(num_workers)

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

# Isometric augmentatations
augmentations = [A.HorizontalFlip(p=0.5),
                 A.VerticalFlip(p=0.5),
                 A.RandomRotate90(p=0.5),
                 A.RandomCrop(256, 256, p=1.0)]
augmentation_pipeline = A.Compose(augmentations, additional_targets={"image2": "image"})


# Load training set
unlabeled_dataset = PairedUnlabeledDataset(data_path, subdirectories_map, spectral_channel,
                                           n_channel_tuple=(n_channels, n_channels2),
                                           augmentations=augmentation_pipeline)
dataloader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

# Define model with hyperparameters
if model_architecture == "Res34":
    model = AutoEncoder_ResEncoder_FLIO(n_channels=n_channels,
                        n_decoder_filters=[256, 128, 64, 32, 16],
                        trainable=True).to(device)
    model.apply(AutoEncoder_ResEncoder_FLIO.init_weights)  # initialize model parameters with normal distribution
else:  # noRes
    model = AutoEncoder_fundus(n_channels=n_channels,
                        n_encoder_filters=[32, 64, 64, 16],
                        n_decoder_filters=[64, 64, 32],
                        trainable=False).to(device)
    model.apply(AutoEncoder_fundus.init_weights)  # Initialize weights

# Load in pre-trained model
if load_pretrained:
    model.load_state_dict(torch.load(transfer_learning_model_path))
    # This will only work with noRes!!!
    if freeze:
        for param in model.double_conv_block.parameters():
            param.requires_grad = False
        for param in model.down_blocks.parameters():
            param.requires_grad = False

model2 = AutoEncoder_FLIO(n_channels=n_channels2,
                    n_encoder_filters=[32, 64, 64, 16],
                    n_decoder_filters=[64, 64, 32],
                    trainable=False).to(device)
model.apply(AutoEncoder_FLIO.init_weights)  # Initialize weights

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
print("Now using", torch.cuda.device_count(), "GPU(s) \n")
model.to(device)


# Select loss
criterion = get_loss_type(loss_type).to(device)
criterion2 = get_loss_type(loss_type, channel=n_channels2).to(device)
criterion_latent = get_loss_type(loss_type_latent)
optimizer = optim.Adam(model.parameters(), lr=0.0035)
# optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.7)

# Training
save_model_path_filename = os.path.join(model_output_path, model_name + "_fundus.pth")
save_model_path_filename2 = os.path.join(model_output_path, model_name + "_FLIO.pth")

# if not os.path.isfile(save_model_path_filename) or overwrite:  # If training a new model#
if overwrite:  # If training a new model
    # TensorBoard
    tensorboard_writer = SummaryWriter(model_output_path)
    dummy = torch.zeros([20, 3, 256, 256], dtype=torch.float)
    tensorboard_writer.add_graph(model, input_to_model=(dummy.to(device), ), verbose=True)
    # tensorboard_writer.add_graph(model, input_to_model=(image,), verbose=True)

    train(model, model2, dataloader, epoch_num, criterion, criterion2, criterion_latent, optimizer, scheduler, device,
          tensorboard_writer, os.path.join(save_model_path, model_name, 'Figures'),
          plot_steps=plot_steps, stop_condition=stop_condition)
    torch.save(model.state_dict(), save_model_path_filename)
    torch.save(model2.state_dict(), save_model_path_filename2)

    tensorboard_writer.flush()
    tensorboard_writer.close()
else:  # Load old model with the given name
    model.load_state_dict(torch.load(save_model_path_filename))


# Testing
validation_dataset = PairedUnlabeledDataset(data_path, subdirectories_map, spectral_channel,
                                            n_channel_tuple=(n_channels, n_channels2))
valid_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

valid_fundus_outputs, valid_flio_outputs, valid_losses, valid_fundus_filenames, valid_flio_filenames = \
    test(model, model2, valid_dataloader, criterion, criterion2, criterion_latent, device)
print("Validation fundus outputs")
print(valid_fundus_outputs[0])
print("Validation flio outputs")
print(valid_flio_outputs[0])
print("Validation losses")
print(valid_losses[0])

# Save outputs as JPEG images
tensors_to_images(valid_fundus_outputs, valid_fundus_filenames, valid_output_path)
tensors_to_images(valid_flio_outputs, valid_flio_filenames, valid_output_path)
