import albumentations as A
import torch.multiprocessing
from PIL import ImageFile
from torchvision import transforms

from autoencoder import *
from autoencoder_datasets import UnlabeledDataset
from autoencoder_traintest_functions import train, test, tensors_to_images

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.multiprocessing.set_sharing_strategy('file_system')

transfer_data_path = "../../../../OakFundus/Transfer_learning_datasets"
data_paths = [os.path.join(transfer_data_path, "STARE/Test_Crop/train"),
              os.path.join(transfer_data_path, "Kaggle_diabetic-retinopathy-detection/train"),
              os.path.join(transfer_data_path, "Kaggle_diabetic-retinopathy-detection/test")]
valid_data_path = os.path.join(transfer_data_path, "STARE/Test_Crop/validation")
# data_path = "../../STARE_Data/Test_Crop/train"
save_model_path = "../../FLIO-Thesis-Project/AutoEncoder/AutoEncoder_Results"
# valid_data_path = "../../STARE_Data/Test_Crop/validation"
subdirectories = ["fundus_registered", "FLIO_parameters/ch01"]  # Nested in each subject folder

overwrite = True

epoch_num = 150

device = torch.device('cuda')  # cpu or cuda

print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.backends.cudnn.enabled)

# seed = 42
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)

transform_pipeline = transforms.Compose([transforms.ToTensor()])
augmentation_pipeline = A.Compose(
    [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5), A.RandomCrop(128, 128, p=1.0)]
)

unlabeled_dataset = UnlabeledDataset(data_paths, transform_pipeline, augmentation_pipeline)
dataloader = DataLoader(unlabeled_dataset, batch_size=20, shuffle=True, num_workers=4)

# initialize model parameters with normal distribution
model = AutoEncoder(n_channels=unlabeled_dataset.size[0],
                    n_encoder_filters=[32, 64, 128, 256, 256],
                    n_decoder_filters=[128, 64, 32, 32],
                    trainable=False).to(device)
model.apply(AutoEncoder.init_weights)  # Initialize weights
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
# optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8)

# TRAINING
save_model_path_filename = os.path.join(save_model_path, 'test_autoencoder.pth')
if not os.path.isfile(save_model_path_filename) or overwrite:
    train(model, dataloader, epoch_num, criterion, optimizer, device)
    torch.save(model.state_dict(), save_model_path_filename)
else:
    model.load_state_dict(torch.load(save_model_path_filename))

# TESTING
validation_dataset = UnlabeledDataset(valid_data_path, transform_pipeline)
valid_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=True, num_workers=4)

valid_outputs, valid_losses, valid_filenames = test(model, valid_dataloader, criterion, device)
print("Validation outputs")
print(valid_outputs[0])
print("Validation losses")
print(valid_losses[0])

tensors_to_images(valid_outputs, valid_filenames. valid_data_path)
