import albumentations as A
import torch.multiprocessing
from PIL import ImageFile

from autoencoder import *
from autoencoder_datasets import ImbalancedDataset, UnlabeledDataset
from autoencoder_traintest_functions import train, test, tensors_to_images

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.multiprocessing.set_sharing_strategy('file_system')

transfer_data_path = "../../../../OakData/Transfer_learning_datasets"
data_paths = [os.path.join(transfer_data_path, "train"),
              os.path.join(transfer_data_path, "KaggleDR_crop")]
data_paths_AMD = [os.path.join(transfer_data_path, "train_AMD")]
model_name = "autoencoder_imbweight_Res34"
valid_data_path = [os.path.join(transfer_data_path, "validation")]
save_model_path = "../../FLIO-Thesis-Project/AutoEncoder/AutoEncoder_Results"

overwrite = True

n_channels = 3
epoch_num = 150

device = torch.device('cuda')  # cpu or cuda

print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.backends.cudnn.enabled)


valid_output_path = os.path.join(valid_data_path[0], model_name)
if not os.path.exists(valid_output_path):
    os.mkdir(valid_output_path)

w = 0.995  # weight of rare event
sample_weights = [1 - w, w]
augmentation_pipeline = A.Compose(
    [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5), A.RandomCrop(128, 128, p=1.0)]
)

unlabeled_dataset = ImbalancedDataset(data_paths, data_paths_AMD, augmentations=augmentation_pipeline)
dataloader = DataLoader(unlabeled_dataset, batch_size=20, shuffle=True, num_workers=12, pin_memory=True)

# initialize model parameters with normal distribution
model = AutoEncoder_ResEncoder(n_channels=n_channels,
                    n_decoder_filters=[128, 64, 32, 32],
                    trainable=False).to(device)
model.apply(AutoEncoder_ResEncoder.init_weights)  # Initialize weights

# model = AutoEncoder(n_channels=n_channels,
#                     n_encoder_filters=[32, 64, 128, 256, 256],
#                     n_decoder_filters=[128, 64, 32, 32],
#                     trainable=False)  #.to(device)
# model.apply(AutoEncoder.init_weights)  # Initialize weights

print(torch.cuda.device_count())
if torch.cuda.device_count() > 1:
    print("Now using", torch.cuda.device_count(), "GPUs")
    model = nn.DataParallel(model)
model.to(device)

criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
# optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8)

# TRAINING
save_model_path_filename = os.path.join(save_model_path, model_name + '.pth')
if not os.path.isfile(save_model_path_filename) or overwrite:
    train(model, dataloader, epoch_num, criterion, optimizer, scheduler, device, sample_weights=sample_weights)
    torch.save(model.state_dict(), save_model_path_filename)
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
