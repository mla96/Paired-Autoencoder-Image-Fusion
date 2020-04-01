import albumentations as A
import torch.multiprocessing
from PIL import ImageFile

from autoencoder import *
from autoencoder_datasets import ImbalancedDataset, UnlabeledDataset
from autoencoder_traintest_functions import train, test, tensors_to_images

from torch.utils.tensorboard import SummaryWriter


ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.multiprocessing.set_sharing_strategy('file_system')

transfer_data_path = "../../../../OakData/Transfer_learning_datasets"
data_paths = [os.path.join(transfer_data_path, "train"),
              os.path.join(transfer_data_path, "KaggleDR_crop")]
data_paths_AMD = [os.path.join(transfer_data_path, "train_AMD")]
valid_data_path = [os.path.join(transfer_data_path, "validation")]
save_model_path = "../../FLIO-Thesis-Project/AutoEncoder/AutoEncoder_Results"
model_base_name = "autoencoder_imbweight"

# Training parameters
model_architecture = "Res34"  # Options: noRes, Res34
epoch_num = 50
train_data_type = "AMDonly"  # Options: None, AMDonly
batch_size = 20
num_workers = 12
plot_steps = 50  # Number of steps between getting random input/output to show training progress
stop_condition = 10000  # Number of steps without improvement for early stopping


overwrite = True

model_name_pt1 = model_base_name + "_" + model_architecture + "_" + str(epoch_num) + "e_"
if train_data_type:
    model_name = model_name_pt1 + train_data_type + "_batch" + str(batch_size) + "_workers" + str(num_workers)
else:
    model_name = model_name_pt1 + "batch" + str(batch_size) + "_workers" + str(num_workers)

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
    [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5), A.RandomCrop(128, 128, p=1.0)]
)

unlabeled_dataset = ImbalancedDataset(data_paths_AMD, augmentations=augmentation_pipeline)
dataloader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)


if model_architecture == "Res34":
    model = AutoEncoder_ResEncoder(n_channels=n_channels,
                        n_decoder_filters=[128, 64, 32, 32],
                        trainable=False).to(device)
    model.apply(AutoEncoder_ResEncoder.init_weights)  # initialize model parameters with normal distribution
else:  # noRes
    model = AutoEncoder(n_channels=n_channels,
                        n_encoder_filters=[32, 64, 128, 256, 256],
                        n_decoder_filters=[128, 64, 32, 32],
                        trainable=False)  #.to(device)
    model.apply(AutoEncoder.init_weights)  # Initialize weights


if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
print("Now using", torch.cuda.device_count(), "GPU(s) \n")
model.to(device)

criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
# optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8)

# TRAINING
save_model_path_filename = os.path.join(model_output_path, model_name + '.pth')
if not os.path.isfile(save_model_path_filename) or overwrite:
    tensorboard_writer = SummaryWriter(model_output_path)
    dummy = torch.zeros([20, 3, 256, 256], dtype=torch.float)
    tensorboard_writer.add_graph(model, input_to_model=(dummy.to(device), ), verbose=True)
    # tensorboard_writer.add_graph(model, input_to_model=(image,), verbose=True)

    train(model, dataloader, epoch_num, criterion, optimizer, scheduler, device, tensorboard_writer,
          plot_steps=plot_steps, stop_condition=stop_condition, sample_weights=sample_weights)
    torch.save(model.state_dict(), save_model_path_filename)

    tensorboard_writer.flush()
    tensorboard_writer.close()
else:
    model.load_state_dict(torch.load(save_model_path_filename))

# TESTING
validation_dataset = ImbalancedDataset(valid_data_path)
valid_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=True, num_workers=12)

valid_outputs, valid_losses, valid_filenames = test(model, valid_dataloader, criterion, device)
print("Validation outputs")
print(valid_outputs[0])
print("Validation losses")
print(valid_losses[0])

tensors_to_images(valid_outputs, valid_filenames, valid_output_path)
