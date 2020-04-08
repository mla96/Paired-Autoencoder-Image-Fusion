import albumentations as A
import torch.multiprocessing
from PIL import ImageFile

from autoencoder import *
from autoencoder_DALI_datasets import *
from autoencoder_DALI_traintest_functions import train, test, tensors_to_images
from nvidia.dali.plugin.pytorch import DALIGenericIterator


ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.multiprocessing.set_sharing_strategy('file_system')

transfer_data_path = "../../../../OakData/Transfer_learning_datasets"
data_paths = [os.path.join(transfer_data_path, "train"),
              os.path.join(transfer_data_path, "KaggleDR_crop")]
data_paths_AMD = [os.path.join(transfer_data_path, "train_AMD")]
valid_data_path = [os.path.join(transfer_data_path, "validation")]
save_model_path = "../../FLIO-Thesis-Project/AutoEncoder/AutoEncoder_DALI_Results"
model_base_name = "autoencoder_DALI_imbweight"

# Training parameters
model_architecture = "Res34"  # Options: noRes, Res34
epoch_num = 300
train_data_type = "AMDonly"  # Options: None, AMDonly
batch_size = 20
num_workers = 12
plot_steps = 200  # Number of steps between getting random input/output to show training progress
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

# print('0: here')
# pipeline = DaliSimplePipeline(data_paths[1], batch_size, num_workers, 0)
# pipeline.build()
# out_comes_bru = pipeline.run()
unlabeled_dataset = DaliImbalancedDataset(data_paths_AMD, batch_size=batch_size)
iterator = iter(unlabeled_dataset)
dali_pipeline = DaliImbalancedPipeline(iterator, batch_size, num_workers, 0)
dali_pipeline.build()
out_comes_bru = dali_pipeline.run()
dali_gen_iterator = DALIGenericIterator([dali_pipeline], ['images', 'labels'], len(unlabeled_dataset))
print('test')


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
    train(model, dataloader, epoch_num, criterion, optimizer, scheduler, device, model_output_path,
          plot_steps=plot_steps, stop_condition=stop_condition, sample_weights=sample_weights)
    torch.save(model.state_dict(), save_model_path_filename)
else:
    model.load_state_dict(torch.load(save_model_path_filename))

# TESTING
# validation_dataset = ImbalancedDataset(valid_data_path)
# valid_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=True, num_workers=12)

# valid_outputs, valid_losses, valid_filenames = test(model, valid_dataloader, criterion, device)
# print("Validation outputs")
# print(valid_outputs[0])
# print("Validation losses")
# print(valid_losses[0])

# tensors_to_images(valid_outputs, valid_filenames, valid_output_path)
