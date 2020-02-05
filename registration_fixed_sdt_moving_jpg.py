# Michelle La
# Jan. 30, 2020

# Main script to read .sdt files into .mat files; calculate registration transforms; and generate registered arrays,
# .gif, and images
# Fixed image: FLIO data
# Moving image: Fundus data (RGB)


import os
from dataformat_functions import *
from gif_animation import *
from registration import *


os.getcwd()
os.chdir("../")

num_channels = 2  # 2 spectral channels: SSC and LSC
path = 'Data/AMD_subj_list_TEST'  # List of all subject directories for registration

# crop_bound = 256
crop_bound_w = 256  # Pixel length of cropping for width and height boundaries on fundus images
crop_bound_h = 222

f = open(path, 'r')
lines = f.readlines()
f.close()

for subj in lines:
    subj = subj.strip()
    dir_path = os.path.join('Data', subj)

    moving_image = Image.open(os.path.join('Data', subj, subj + '.jpg'))  # Load image

    # Resize and format fixed image (fundus RGB image) into BW SITK image for registration
    w, h = moving_image.size
    moving_image = moving_image.crop((crop_bound_w, crop_bound_h, w - crop_bound_w, h - crop_bound_h))
    w_crop, h_crop = moving_image.size
    # moving_image.save(os.path.join(dir_path, "crop"), "TIFF")
    moving_image_resize = moving_image.resize((w_crop // 2, h_crop // 2), Image.LANCZOS)
    # moving_image_resize = moving_image

    moving_image = sitk.RescaleIntensity(sitk.GetImageFromArray(np.array(moving_image, dtype=np.float32), isVector=True))

    # Get RGB channels from original un-resized moving image to be reassembled later after registration
    RGB_select = sitk.VectorIndexSelectionCastImageFilter()
    channel1_movimage = RGB_select.Execute(moving_image, 0, sitk.sitkUInt8)
    channel2_movimage = RGB_select.Execute(moving_image, 1, sitk.sitkUInt8)
    channel3_movimage = RGB_select.Execute(moving_image, 2, sitk.sitkUInt8)

    # moving_image = moving_image.crop((crop_bound, crop_bound, int(w / 2) - crop_bound, int(h / 2) - crop_bound))
    w, h = moving_image_resize.size
    moving_image_resize = np.sum(moving_image_resize, axis=2)
    moving_image_resize = sitk.RescaleIntensity(sitk.GetImageFromArray(np.array(moving_image_resize, dtype=np.float32), isVector=False))

    for i in range(num_channels):
        channel_path = os.path.join(dir_path, 'measurement_ch0' + str(i + 1) + '.mat')
        if os.path.isfile(channel_path):
            fixed_raw = channel_Mat2NpArray(channel_path)  # Convert .mat to np array
            fixed_image_fullsize = np.flip(np.sum(fixed_raw, axis=2), 0).astype(np.float32)
            fixed_image_fullsize = Image.fromarray(fixed_image_fullsize).resize((w_crop, h_crop), Image.LANCZOS)
            fixed_image_fullsize = sitk.RescaleIntensity(sitk.GetImageFromArray(np.array(fixed_image_fullsize, dtype=np.float32), isVector=False))
            fixed_image = sitk.RescaleIntensity(channel_NpArray2SitkIm(fixed_raw, w, h))

            registered_image, init_transform, final_reg_transform = my_registration(fixed_image, moving_image_resize)
            registered_image = sitk.RescaleIntensity(registered_image)

            writer = sitk.ImageFileWriter()
            writer.SetImageIO("TIFFImageIO")
            writer.SetFileName(os.path.join(dir_path, 'registered_ch0' + str(i + 1) + '.tiff'))  # Save registered image
            writer.Execute(registered_image)

            # Save checkerboard of fixed and moving image
            checkerboard(fixed_image, moving_image_resize, os.path.join(dir_path, 'checker_fixed_moving_ch0' + str(i + 1)),
                         writer)
            # Save checkerboard of fixed and registered image
            checkerboard(fixed_image, registered_image, os.path.join(dir_path, 'checker_fixed_registered_ch0' + str(i + 1)),
                         writer)

            channel1_registered_image_fullsize = sitk.Resample(channel1_movimage, fixed_image_fullsize, final_reg_transform,
                                                               sitk.sitkLinear, 0.0,
                                                               channel1_movimage.GetPixelID())
            channel2_registered_image_fullsize = sitk.Resample(channel2_movimage, fixed_image_fullsize, final_reg_transform,
                                                               sitk.sitkLinear, 0.0,
                                                               channel2_movimage.GetPixelID())
            channel3_registered_image_fullsize = sitk.Resample(channel3_movimage, fixed_image_fullsize, final_reg_transform,
                                                               sitk.sitkLinear, 0.0,
                                                               channel3_movimage.GetPixelID())
            compose = sitk.ComposeImageFilter()
            registered_image_fullsize = compose.Execute(channel1_registered_image_fullsize, channel2_registered_image_fullsize, channel3_registered_image_fullsize)
            writer.SetFileName(os.path.join(dir_path, 'registered_fullsize_ch0' + str(i + 1) + '.tiff'))  # Save registered image
            writer.Execute(registered_image_fullsize)

            # Save np array of registered fundus data
            np.save(os.path.join(dir_path, 'registered_ch0' + str(i + 1)), np.array(registered_image_fullsize, dtype=np.float32))

            # Save .gif movie of moving and registered FLIO data
            # ani = voxel_movie(fixed_image_array, os.path.join(dir_path, 'raw_movie_ch0' + str(i + 1)))

        else:
            print('The file ' + channel_path + ' does not exist.')
