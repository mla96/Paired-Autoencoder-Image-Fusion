#!/usr/bin/env python3
"""
This script reads .sdt files into .mat files; calculate registration transforms; and generate registered arrays, .gifs,
and images.

Fixed image: Fundus data (RGB)
Moving image: FLIO data
"""


import os
from dataformat_functions import *
from gif_animation import *
from registration import *


num_channels = 2  # 2 spectral channels: SSC and LSC
data_path = '../../../../OakData/FLIO_Data'
path = os.path.join(data_path, 'AMD_subj_list_current')  # List of all subject directories for registration

fundus_dir_name = 'fundus_registered'

# crop_bound = 256
crop_bound_w = 256  # Pixel length of cropping for width and height boundaries on fundus images
crop_bound_h = 222

f = open(path, 'r')
lines = f.readlines()
f.close()


f = open(path, 'r')
lines = f.readlines()
f.close()

for subj in lines:
    subj = subj.strip()
    print(subj)
    dir_path = os.path.join(data_path, subj)

    fundus_output_path = os.path.join(dir_path, fundus_dir_name)
    if not os.path.exists(fundus_output_path):
        os.mkdir(fundus_output_path)

    manual_crop = os.path.join(data_path, subj, 'crop_manual.jpg')
    if os.path.isfile(manual_crop):
        print("Manual crop")
        manual_crop = Image.open(manual_crop)
        moving_image_resize = manual_crop.resize((806, 806), Image.LANCZOS)
    else:
        moving_image = Image.open(os.path.join(data_path, subj, subj + '.jpg'))  # Load image

        # Resize and format fixed image (fundus RGB image) into BW SITK image for registration
        w, h = moving_image.size  # 2124, 2056
        moving_image = moving_image.crop((crop_bound_w, crop_bound_h, w - crop_bound_w, h - crop_bound_h))
        w_crop, h_crop = moving_image.size  # 1612
        # moving_image.save(os.path.join(dir_path, "crop"), "TIFF")
        moving_image_resize = moving_image.resize((w_crop // 2, h_crop // 2), Image.LANCZOS)
        moving_image.save(os.path.join(dir_path, "crop"), "TIFF")

    moving_image = sitk.GetImageFromArray(np.array(moving_image_resize, dtype=np.float32), isVector=True)

    # Get RGB channels from original un-resized moving image to be reassembled later after registration
    RGB_select = sitk.VectorIndexSelectionCastImageFilter()
    channel1_movimage = RGB_select.Execute(moving_image, 0, sitk.sitkUInt8)
    channel2_movimage = RGB_select.Execute(moving_image, 1, sitk.sitkUInt8)
    channel3_movimage = RGB_select.Execute(moving_image, 2, sitk.sitkUInt8)

    w, h = moving_image_resize.size  # 806
    moving_image_resize = np.sum(moving_image_resize, axis=2)
    moving_image_resize = sitk.RescaleIntensity(sitk.GetImageFromArray(np.array(moving_image_resize, dtype=np.float32), isVector=False))
    #sitk.Show(moving_image_resize)

    for i in range(num_channels):
        channel_path = os.path.join(dir_path, 'measurement_ch0' + str(i + 1) + '.mat')
        if os.path.isfile(channel_path):
            if i == 0:
                fixed_raw = channel_Mat2NpArray(channel_path)  # Convert .mat to np array
                fixed_image = np.flip(np.transpose(np.sum(fixed_raw, axis=0).astype(np.float32)), 0)
                fixed_image = Image.fromarray(255 * fixed_image / np.max(fixed_image)).resize((w, h), Image.LANCZOS)  # 1612
                fixed_image = sitk.RescaleIntensity(sitk.GetImageFromArray(np.array(fixed_image, dtype=np.float32), isVector=False))
                # fixed_image = sitk.RescaleIntensity(channel_NpArray2SitkIm(fixed_raw, w, h))
                # sitk.Show(fixed_image)

                registered_image, init_transform, final_reg_transform = my_registration(moving_image_resize, fixed_image)  # Register with 806x806 images
                # registered_image = sitk.RescaleIntensity(registered_image)
                # init_transform = init_transform.GetInverse()
                final_reg_transform = final_reg_transform.GetInverse()

            registered_image = sitk.Resample(moving_image_resize, fixed_image, final_reg_transform, sitk.sitkLinear,
                                             0.0, moving_image_resize.GetPixelID())

            writer = sitk.ImageFileWriter()
            writer.SetImageIO("TIFFImageIO")
            writer.SetFileName(os.path.join(fundus_output_path, 'registered_ch0' + str(i + 1) + '.tiff'))  # Save registered image
            writer.Execute(registered_image)

            channel1_registered_image_fullsize = sitk.Resample(channel1_movimage, fixed_image, final_reg_transform,
                                                               sitk.sitkLinear, 0.0,
                                                               channel1_movimage.GetPixelID())
            channel2_registered_image_fullsize = sitk.Resample(channel2_movimage, fixed_image, final_reg_transform,
                                                               sitk.sitkLinear, 0.0,
                                                               channel2_movimage.GetPixelID())
            channel3_registered_image_fullsize = sitk.Resample(channel3_movimage, fixed_image, final_reg_transform,
                                                               sitk.sitkLinear, 0.0,
                                                               channel3_movimage.GetPixelID())


            compose = sitk.ComposeImageFilter()
            registered_image_fullsize = compose.Execute(channel1_registered_image_fullsize, channel2_registered_image_fullsize, channel3_registered_image_fullsize)
            writer.SetFileName(os.path.join(fundus_output_path, 'registered_rgb_ch0' + str(i + 1) + '.tiff'))  # Save registered image
            writer.Execute(registered_image_fullsize)

            # Save checkerboard of fixed and moving image
            checkerboard(fixed_image, moving_image_resize, os.path.join(dir_path, 'checker_fixed_moving_ch0' + str(i + 1)),
                         writer)
            # Save checkerboard of fixed and registered image
            checkerboard(fixed_image, registered_image, os.path.join(dir_path, 'checker_fixed_registered_ch0' + str(i + 1)),
                         writer)

            # Save np array of registered fundus data
            # np.save(os.path.join(dir_path, 'registered_ch0' + str(i + 1)), np.array(registered_image_fullsize, dtype=np.float32))

        else:
            print('The file ' + channel_path + ' does not exist.')
