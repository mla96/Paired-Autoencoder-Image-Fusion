import numpy as np
import scipy as sp
import scipy.io as sio
import SimpleITK as sitk
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import re
import shutil


from dataformat_functions import *
from gif_animation import *
from registration import *

# COMMENT AUXILIARY FILES


os.getcwd()
os.chdir("../")

# print(os.getcwd())
#
# path = 'OakHome/FLIO/Data/AMD_subj_list_TEST'
#
# f = open(path, 'r')
# lines = f.readlines()
# f.close()
#
# for line in lines:
#     line = line.strip()
#     files = os.listdir(os.path.join('Documents/FLIMX-master/studyData/AMD/', line.replace('_',  '')))
#     for file in files:
#         shutil.copy(os.path.join('Documents/FLIMX-master/studyData/AMD', line.replace('_',  ''), file), os.path.join('OakHome/FLIO/Data', line))


# Define "show" boolean arg for each function with "show"



# FOR SUBJECT IN LIST
# for subj in subj_list:

# Make results directory if it does not exist
# ## Copy the measurements data over/ then mkdir won't be necessary, just return an error if it doesn't exist

# fixed_image = sitk.ReadImage('Data/AMD/AMD59/Schuetze_Margit_19500827_20160314_(27926).jpg', sitk.sitkFloat32)
dir_path = "Data/AMD59"

# fixed_image = Image.open('Data/' + subj + '.jpg')
fixed_image = Image.open('Data/AMD_59.jpg')
# fixed_image = Image.open('Data/AMD/AMD59/Schuetze_Margit_19500827_20160314_(27926).jpg')

w, h = fixed_image.size
crop_bound = 128

fixed_image = fixed_image.resize((int(w/2), int(h/2)))
fixed_image = fixed_image.crop((crop_bound, crop_bound, int(w/2)-crop_bound, int(h/2)-crop_bound))
w, h = fixed_image.size
fixed_image = np.sum(fixed_image, axis=2)
fixed_image = sitk.RescaleIntensity(sitk.GetImageFromArray(np.array(fixed_image, dtype=np.float32), isVector=False))
# sitk.Show(fixed_image)


num_channels = 2
dir_path = "Data/AMD59"
for i in range(1):  # NUM_CHANNELS
    channel_path = os.path.join(dir_path, 'measurement_ch0' + str(i + 1) + '.mat')
    if os.path.isfile(channel_path):
        moving_raw = channel_Mat2NpArray(channel_path)
        moving_image = sitk.RescaleIntensity(channel_NpArray2SitkIm(moving_raw, w, h))
        moving_image_array = moving_raw/np.max(moving_raw) * 255
        registered_image, init_transform, final_reg_transform = my_registration(fixed_image, moving_image)
        registered_image = sitk.RescaleIntensity(registered_image)

        writer = sitk.ImageFileWriter()
        writer.SetImageIO("TIFFImageIO")

        checkerboard(fixed_image, moving_image, os.path.join(dir_path,'fixed_moving_ch0' + str(i + 1)), writer)
        checkerboard(fixed_image, registered_image, os.path.join(dir_path,'fixed_registered_ch0' + str(i + 1)), writer)

        # Apply registration transform to slices
        registered_voxel_array, registered_movie_array = transform_slices(fixed_image, final_reg_transform, moving_image_array, w, h)
        registered_movie_array = registered_movie_array/np.max(registered_movie_array) * 255

        ani = voxel_movie(moving_image_array, 'raw_movie')
        ani2 = voxel_movie(registered_movie_array, 'registered_movie')

        np.save(os.path.join(dir_path, 'registered_ch0' + str(i + 1)), registered_voxel_array)
    else:
        print('The file ' + channel_path + ' does not exist.')
