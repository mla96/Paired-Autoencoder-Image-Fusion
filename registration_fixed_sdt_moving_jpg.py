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
    w, h = moving_image.size
    moving_image = moving_image.resize((w // 2, h // 2), Image.LANCZOS)
    moving_image.save(os.path.join(dir_path, "crop"), "TIFF")
    # moving_image = moving_image.crop((crop_bound, crop_bound, int(w / 2) - crop_bound, int(h / 2) - crop_bound))
    w, h = moving_image.size
    moving_image = np.sum(moving_image, axis=2)
    moving_image = sitk.RescaleIntensity(sitk.GetImageFromArray(np.array(moving_image, dtype=np.float32), isVector=False))

    for i in range(num_channels):
        channel_path = os.path.join(dir_path, 'measurement_ch0' + str(i + 1) + '.mat')
        if os.path.isfile(channel_path):
            fixed_raw = channel_Mat2NpArray(channel_path)  # Convert .mat to np array
            fixed_image = sitk.RescaleIntensity(channel_NpArray2SitkIm(fixed_raw, w, h))
            fixed_image_array = fixed_raw / np.max(fixed_raw) * 255
            registered_image, init_transform, final_reg_transform = my_registration(fixed_image, moving_image)
            registered_image = sitk.RescaleIntensity(registered_image)

            writer = sitk.ImageFileWriter()
            writer.SetImageIO("TIFFImageIO")
            writer.SetFileName(os.path.join(dir_path, 'registered_ch0' + str(i + 1) + '.tiff'))  # Save registered image
            writer.Execute(registered_image)

            # Save checkerboard of fixed and moving image
            checkerboard(fixed_image, moving_image, os.path.join(dir_path, 'checker_fixed_moving_ch0' + str(i + 1)),
                         writer)
            # Save checkerboard of fixed and registered image
            checkerboard(fixed_image, registered_image, os.path.join(dir_path, 'checker_fixed_registered_ch0' + str(i + 1)),
                         writer)

            # Save np array of registered fundus data
            #RESIZES
            # can you apply the saved transform to the full sized image? Try it
            np.save(os.path.join(dir_path, 'registered_ch0' + str(i + 1)), np.array(registered_image.resize((256, 256), Image.LANCZOS), dtype=np.float32))

            # Save .gif movie of moving and registered FLIO data
            # ani = voxel_movie(fixed_image_array, os.path.join(dir_path, 'raw_movie_ch0' + str(i + 1)))

        else:
            print('The file ' + channel_path + ' does not exist.')
