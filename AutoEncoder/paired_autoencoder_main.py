import albumentations as A
import torch.multiprocessing
from PIL import ImageFile

from autoencoder import *
from autoencoder_datasets import PairedUnlabeledDataset
from autoencoder_traintest_functions import train, test, tensors_to_images


ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.multiprocessing.set_sharing_strategy('file_system')

filetype = [["tiff", "fullsize"], "mat"]  # String identifiers for the desired files in each subdirectory
spectral_channel = "02"

data_path = "../../../../OakData/FLIO_Data"
# valid_data_path = os.path.join(transfer_data_path, "STARE/Test_Crop/validation")
save_model_path = "../../FLIO-Thesis-Project/AutoEncoder/AutoEncoder_Results"
# valid_data_path = "../../STARE_Data/Test_Crop/validation"
subdirectories = ["fundus_registered", "FLIO_parameters"]  # Nested in each subject folder

# Load in pre-trained model

overwrite = True

epoch_num = 150

device = torch.device('cuda')  # cpu or cuda

print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.backends.cudnn.enabled)


augmentation_pipeline = A.Compose(
    [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5), A.RandomCrop(128, 128, p=1.0)],
    additional_targets={"image2": "image"}
)

unlabeled_dataset = PairedUnlabeledDataset(data_path, subdirectories, filetype, spectral_channel, augmentations=augmentation_pipeline)
dataloader = DataLoader(unlabeled_dataset, batch_size=20, shuffle=True, num_workers=12, pin_memory=True)

model = AutoEncoder(n_channels=3,
                    n_encoder_filters=[32, 64, 128, 256, 256],
                    n_decoder_filters=[128, 64, 32, 32],
                    trainable=False)  #.to(device)
model.apply(AutoEncoder.init_weights)  # Initialize weights

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
save_model_path_filename = os.path.join(save_model_path, 'wubs' + '.pth')
if not os.path.isfile(save_model_path_filename) or overwrite:
    train(model, dataloader, epoch_num, criterion, optimizer, scheduler, device)
    torch.save(model.state_dict(), save_model_path_filename)
else:
    model.load_state_dict(torch.load(save_model_path_filename))

