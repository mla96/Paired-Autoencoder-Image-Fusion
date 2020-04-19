import albumentations as A
from fastai.vision import DataLoader
import os
from PIL import ImageFile

import torch.multiprocessing
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from paired_autoencoder import AutoEncoder, AutoEncoder_ResEncoder_FLIO, AutoEncoder_ResEncoder_Fundus
from autoencoder_datasets import PairedUnlabeledDataset
from paired_autoencoder_traintest_functions import train, test, tensors_to_images
from loss_utils import SSIM_Loss, MS_SSIM_Loss


# Paths to source data, validation data, and output saves
data_path = "../../../../OakData/FLIO_Data"
subdirectories = ["fundus_registered", "FLIO_parameters"]  # Nested in each subject folder
save_model_path = "../../FLIO-Thesis-Project/AutoEncoder/Paired_AutoEncoder_Results"
model_base_name = "autoencoder_32-64-64-16-64-64-32"
valid_data_path = [os.path.join(data_path, "validation")]  # Contains small data set for validation
# valid_data_path = os.path.join(transfer_data_path, "STARE/Test_Crop/validation")
# valid_data_path = "../../STARE_Data/Test_Crop/validation"
transfer_learning_model_path = "../../FLIO-Thesis-Project/AutoEncoder/AutoEncoder_Results/" \
                               "fixednew_autoencoder_32-64-64-16-64-64-32_jupyterparams_tanh_31_noRes_512e_AMDonly_MS-SSIM_batch16_workers12/" \
                               "fixednew_autoencoder_32-64-64-16-64-64-32_jupyterparams_tanh_31_noRes_512e_AMDonly_MS-SSIM_batch16_workers12.pth"

# Training parameters
model_architecture = "noRes"  # Options: noRes, Res34
epoch_num = 1
loss_type = "MS-SSIM"  # Options: L1, MSE, SSIM, MS-SSIM
batch_size = 8
num_workers = 12
plot_steps = 500  # Number of steps between getting random input/output to plot training progress in TensorBoard
stop_condition = 10000  # Number of steps without improvement for early stopping

overwrite = True  # Overwrite existing model to train again
load_pretrained = True  # Load pretrained model for RGB fundus AutoEncoder

filetype = [["jpg", "rgb"], "mat"]  # String identifiers for the desired files in each subdirectory
spectral_channel = "02"


# Create model name based on model_base_name and training parameters
model_name = model_base_name + "_" + model_architecture + "_" + str(epoch_num) + "e_batch" + str(batch_size) + "_workers" + str(num_workers)

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
n_channels2 = 6
transformation_pipeline = A.Compose(
    [A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

# Isometric augmentatations
augmentations = [A.HorizontalFlip(p=0.5),
                 A.VerticalFlip(p=0.5),
                 A.RandomRotate90(p=0.5)]
# MS-SSIM requires images larger than 160 x 160 to calculate loss
if loss_type == "MS-SSIM":
    augmentations.append(A.RandomCrop(256, 256, p=1.0))
else:
    augmentations.append(A.RandomCrop(128, 128, p=1.0))
augmentation_pipeline = A.Compose(augmentations, additional_targets={"image2": "image"})


# Load training set
unlabeled_dataset = PairedUnlabeledDataset(data_path, subdirectories, filetype, spectral_channel,
                                           transformations=transformation_pipeline, augmentations=augmentation_pipeline)
dataloader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

# Define model with hyperparameters
if model_architecture == "Res34":
    model = AutoEncoder_ResEncoder_FLIO(n_channels=n_channels,
                        n_decoder_filters=[256, 128, 64, 32, 16],
                        trainable=True).to(device)
    model.apply(AutoEncoder_ResEncoder_FLIO.init_weights)  # initialize model parameters with normal distribution
else:  # noRes
    model = AutoEncoder(n_channels=n_channels,
                        n_encoder_filters=[32, 64, 64, 16],
                        n_decoder_filters=[64, 64, 32],
                        trainable=False).to(device)
    model.apply(AutoEncoder.init_weights)  # Initialize weights

# Load in pre-trained model
if load_pretrained:
    model.load_state_dict(torch.load(transfer_learning_model_path))

model2 = AutoEncoder(n_channels=n_channels,
                    n_encoder_filters=[32, 64, 64, 16],
                    n_decoder_filters=[64, 64, 32],
                    trainable=False).to(device)
model.apply(AutoEncoder.init_weights)  # Initialize weights

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
print("Now using", torch.cuda.device_count(), "GPU(s) \n")
model.to(device)

# Select loss
loss_types = {'MSE': nn.MSELoss(),
              'L1': nn.L1Loss(),
              'SSIM': SSIM_Loss(data_range=1.0),
              'MS-SSIM': MS_SSIM_Loss(data_range=1.0, nonnegative_ssim=True)}
criterion = loss_types.get(loss_type).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0003)
# optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8)


# Training
save_model_path_filename = os.path.join(model_output_path, model_name + "_fundus.pth")
save_model_path_filename2 = os.path.join(model_output_path, model_name + "_FLIO.pth")

if not os.path.isfile(save_model_path_filename) or overwrite:  # If training a new model
    # TensorBoard
    tensorboard_writer = SummaryWriter(model_output_path)
    dummy = torch.zeros([20, 3, 128, 256], dtype=torch.float)
    tensorboard_writer.add_graph(model, input_to_model=(dummy.to(device), ), verbose=True)
    # tensorboard_writer.add_graph(model, input_to_model=(image,), verbose=True)

    train(model, model2, dataloader, epoch_num, criterion, optimizer, scheduler, device, tensorboard_writer,
          os.path.join(save_model_path, model_name, 'Figures'),
          plot_steps=plot_steps, stop_condition=stop_condition)
    torch.save(model.state_dict(), save_model_path_filename)
    torch.save(model2.state_dict(), save_model_path_filename2)

    tensorboard_writer.flush()
    tensorboard_writer.close()
else:  # Load old model with the given name
    model.load_state_dict(torch.load(save_model_path_filename))


# Testing
validation_dataset = PairedUnlabeledDataset(data_path, subdirectories, filetype, spectral_channel)
valid_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

valid_outputs, valid_losses, valid_filenames = test(model, valid_dataloader, criterion, device)
print("Validation outputs")
print(valid_outputs[0])
print("Validation losses")
print(valid_losses[0])

# Save outputs as JPEG images
tensors_to_images(valid_outputs, valid_filenames, valid_output_path)
