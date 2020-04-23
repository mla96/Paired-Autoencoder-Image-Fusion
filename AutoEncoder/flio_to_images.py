import os

import PIL.Image
import hdf5storage
import numpy as np
import skimage.io
import skimage.transform
import torch
from torch.utils.data.dataset import Dataset

import matplotlib.pyplot as plt


num_channels = 2  # 2 spectral channels: SSC and LSC
data_path = '../../../../OakData/FLIO_Data'
path = os.path.join(data_path, 'AMD_subj_list')  # List of all subject directories for registration

flio_dir_name = 'FLIO_parameters'

f = open(path, 'r')
lines = f.readlines()
f.close()


def channel_normalize(image_array):
    for channel in range(image_array.shape[2]):
        image_array[:, :, channel] = 255 * image_array[:, :, channel] / np.max(image_array[:, :, channel])
    return image_array


for subj in lines:
    subj = subj.strip()
    print(subj)
    dir_path = os.path.join(data_path, subj)

    flio_output_path = os.path.join(dir_path, flio_dir_name)
    data_paths = sorted(os.listdir(flio_output_path))
    print(data_paths)

    for d_path in data_paths:
        if "result" in d_path and ".mat" in d_path:
            mat_image = hdf5storage.loadmat(os.path.join(flio_output_path, d_path))
            mat_image = mat_image['result'][0][0]['results'][0][0]['pixel'][0][0]

            amplitude_FLIO_image = np.dstack((mat_image['Amplitude1'],mat_image['Amplitude2'], mat_image['Amplitude3']))
            tau_FLIO_image = np.dstack((mat_image['Tau1'], mat_image['Tau2'], mat_image['Tau3']))

            taumean_FLIO_image = np.sum(amplitude_FLIO_image * tau_FLIO_image, axis=2) / (np.sum(amplitude_FLIO_image, axis=2) + 1e-6)
            taumean_FLIO_image[taumean_FLIO_image <= 300] = 300
            taumean_FLIO_image[taumean_FLIO_image >= 500] = 500
            plt.imshow(taumean_FLIO_image.astype(np.uint8), cmap="jet_r")
            plt.gca().set_axis_off()
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.savefig(os.path.join(flio_output_path, d_path.split('.')[0] + '_taumean_color.jpg'), bbox_inches='tight', pad_inches=0)
            taumean_FLIO_image = PIL.Image.fromarray(np.uint8(255 * taumean_FLIO_image / np.max(taumean_FLIO_image)))
            taumean_FLIO_image.save(os.path.join(flio_output_path, d_path.split('.')[0] + '_taumean.jpg'), "JPEG")

            amplitude_FLIO_image = PIL.Image.fromarray(np.uint8(channel_normalize(amplitude_FLIO_image)))
            amplitude_FLIO_image.save(os.path.join(flio_output_path, d_path.split('.')[0] + '_amplitude.jpg'), "JPEG")

            tau_FLIO_image = PIL.Image.fromarray(np.uint8(channel_normalize(tau_FLIO_image)))
            tau_FLIO_image.save(os.path.join(flio_output_path, d_path.split('.')[0] + '_tau.jpg'), "JPEG")

            taumeanguess_FLIO_image = mat_image['TauMeanGuess']
            taumeanguess_FLIO_image = PIL.Image.fromarray(np.uint8(255 * taumeanguess_FLIO_image / np.max(taumeanguess_FLIO_image)))
            taumeanguess_FLIO_image.save(os.path.join(flio_output_path, d_path.split('.')[0] + '_taumeanguess.jpg'), "JPEG")

            rauc_FLIO_image = np.dstack((mat_image['RAUC1'], mat_image['RAUC2'], mat_image['RAUC3']))
            rauc_FLIO_image = PIL.Image.fromarray(np.uint8(channel_normalize(rauc_FLIO_image)))
            rauc_FLIO_image.save(os.path.join(flio_output_path, d_path.split('.')[0] + '_RAUC.jpg'), "JPEG")

            raucis_FLIO_image = np.dstack((mat_image['RAUCIS1'], mat_image['RAUCIS2'], mat_image['RAUCIS3']))
            raucis_FLIO_image = PIL.Image.fromarray(np.uint8(channel_normalize(raucis_FLIO_image)))
            raucis_FLIO_image.save(os.path.join(flio_output_path, d_path.split('.')[0] + '_RAUCIS.jpg'), "JPEG")
