import albumentations as A
from fastai.vision import DataLoader
import os
from PIL import ImageFile

import torch.multiprocessing
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from autoencoder import AutoEncoder, AutoEncoder_ResEncoder
from autoencoder_datasets import UnlabeledDataset
from autoencoder_traintest_functions import train, test, tensors_to_images
from loss_utils import SSIM_Loss


ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.multiprocessing.set_sharing_strategy('file_system')

transfer_data_path = "../../../../OakData/Transfer_learning_datasets"
data_paths = [os.path.join(transfer_data_path, "cifar-10-batches-py_jpeg")]
valid_data_path = [os.path.join(transfer_data_path, "validation_CIFARjpeg")]
save_model_path = "../../FLIO-Thesis-Project/AutoEncoder/AutoEncoder_Results"
model_base_name = "fixednew_autoencoder_32-64-64-16-64-64-32_jupyterparams_tanh_norm_coloraug_31"

# Training parameters
model_architecture = "noRes"  # Options: noRes, Res34
epoch_num = 30
train_data_type = None  # Options: None, AMDonly
loss_type = "SSIM"  # Options: L1, MSE, SSIM
batch_size = 32
num_workers = 12
plot_steps = 5000  # Number of steps between getting random input/output to show training progress
stop_condition = 10000  # Number of steps without improvement for early stopping
w = 0.995  # weight of rare event
sample_weights = None  # Options: None, [1 - w, w]

overwrite = True  # Overwrite existing model to train again


# Create model name based on model_base_name and training parameters
model_name_pt1 = model_base_name + "_" + model_architecture + "_" + str(epoch_num) + "e_"
if train_data_type:
    model_name = model_name_pt1 + train_data_type + "_" + loss_type + "_batch" + str(batch_size) + "_workers" + str(num_workers)
else:
    model_name = model_name_pt1 + loss_type + "_batch" + str(batch_size) + "_workers" + str(num_workers)

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

transformation_pipeline = A.Compose(
    [A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
augmentation_pipeline = A.Compose(
    [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
     A.RandomRotate90(p=0.5),
     A.HueSaturationValue(p=0.5),
     A.RandomBrightnessContrast(p=0.5)]
     # A.RandomCrop(128, 128, p=1.0)]
)

# Load training set
unlabeled_dataset = UnlabeledDataset(data_paths, transformations=transformation_pipeline, augmentations=augmentation_pipeline)
dataloader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)


# Define model with hyperparameters
if model_architecture == "Res34":
    model = AutoEncoder_ResEncoder(n_channels=n_channels,
                        n_decoder_filters=[64, 32, 16],
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

# Select loss
if loss_type == "MSE":
    criterion = nn.MSELoss().to(device)
elif loss_type == "L1":
    criterion = nn.L1Loss().to(device)
else:  # loss_type == "SSIM":
    criterion = SSIM_Loss(data_range=1.0).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0003)
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
    model.load_state_dict(torch.load(save_model_path_filename))

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
