# Michelle La
# Jan. 30, 2020

# Main script to read .sdt files into .mat files; calculate registration transforms; and generate registered arrays,
# .gifs, and images


import os
from dataformat_functions import *
from gif_animation import *
from registration import *


os.getcwd()
os.chdir("../")

num_channels = 2
path = 'Data/AMD_subj_list'

# crop_bound = 256
crop_bound_w = 256
crop_bound_h = 222

f = open(path, 'r')
lines = f.readlines()
f.close()

for subj in lines:
    subj = subj.strip()
    dir_path = os.path.join('Data', subj)

    fixed_image = Image.open(os.path.join('Data', subj, subj + '.jpg'))
    w, h = fixed_image.size

    fixed_image = fixed_image.crop((crop_bound_w, crop_bound_h, w - crop_bound_w, h - crop_bound_h))
    w, h = fixed_image.size
    fixed_image = fixed_image.resize((w // 2, h // 2), Image.LANCZOS)
    fixed_image.save(os.path.join(dir_path, "crop"), "TIFF")

    # fixed_image = fixed_image.resize((int(w / 2), int(h / 2)), Image.LANCZOS)
    # fixed_image = fixed_image.crop((crop_bound, crop_bound, int(w / 2) - crop_bound, int(h / 2) - crop_bound))
    w, h = fixed_image.size
    # fixed_image = fixed_image.resize((256, 256))
    fixed_image = np.sum(fixed_image, axis=2)
    fixed_image = sitk.RescaleIntensity(sitk.GetImageFromArray(np.array(fixed_image, dtype=np.float32), isVector=False))

    for i in range(num_channels):
        channel_path = os.path.join(dir_path, 'measurement_ch0' + str(i + 1) + '.mat')
        if os.path.isfile(channel_path):
            moving_raw = channel_Mat2NpArray(channel_path)
            moving_image = sitk.RescaleIntensity(channel_NpArray2SitkIm(moving_raw, w, h))
            moving_image_array = moving_raw / np.max(moving_raw) * 255
            registered_image, init_transform, final_reg_transform = my_registration(fixed_image, moving_image)
            registered_image = sitk.RescaleIntensity(registered_image)

            writer = sitk.ImageFileWriter()
            writer.SetImageIO("TIFFImageIO")
            writer.SetFileName(os.path.join(dir_path, 'registered_ch0' + str(i + 1) + '.tiff'))
            writer.Execute(registered_image)

            checkerboard(fixed_image, moving_image, os.path.join(dir_path, 'checker_fixed_moving_ch0' + str(i + 1)), writer)
            checkerboard(fixed_image, registered_image, os.path.join(dir_path, 'checker_fixed_registered_ch0' + str(i + 1)),
                         writer)

            # Apply registration transform to slices
            registered_movie_array = transform_slices(fixed_image, final_reg_transform, moving_image_array, w, h)

            # registered_voxel_array, registered_movie_array = transform_slices(fixed_image, final_reg_transform,
            #                                                                   moving_image_array, w, h)

            np.save(os.path.join(dir_path, 'registered_ch0' + str(i + 1)), registered_movie_array)

            registered_movie_array = registered_movie_array / np.max(registered_movie_array) * 255

            ani = voxel_movie(moving_image_array, os.path.join(dir_path, 'raw_movie_ch0' + str(i + 1)))
            ani2 = voxel_movie(registered_movie_array, os.path.join(dir_path, 'registered_movie_ch0' + str(i + 1)))

        else:
            print('The file ' + channel_path + ' does not exist.')


# Define "show" boolean arg for each function with "show"


# fixed_image = sitk.ReadImage('Data/AMD/AMD59/Schuetze_Margit_19500827_20160314_(27926).jpg', sitk.sitkFloat32)
# dir_path = "Data/AMD59"

# fixed_image = Image.open('Data/' + subj + '.jpg')
# fixed_image = Image.open('Data/AMD_59.jpg')
# fixed_image = Image.open('Data/AMD/AMD59/Schuetze_Margit_19500827_20160314_(27926).jpg')

# sitk.Show(fixed_image)