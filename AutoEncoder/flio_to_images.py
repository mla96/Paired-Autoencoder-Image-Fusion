#!/usr/bin/env python3
"""
This file generates images from various scaled dimensions of FLIO data, including amplitude, tau, taumean, taumeanguess,
 RAUC, and RAUCIS.
"""

import os

import PIL.Image
import hdf5storage
import numpy as np

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
        image_array[:, :, channel] = 255 * ((image_array[:, :, channel] - np.min(image_array[:, :, channel])) /
                                            (np.max(image_array[:, :, channel]) - np.min(image_array[:, :, channel])))
    return image_array


def image_normalize(image_array):  # Normalize across whole image
    image_array = 255 * (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
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

            amplitude_FLIO_image = np.flipud(np.dstack((mat_image['Amplitude1'],mat_image['Amplitude2'], mat_image['Amplitude3']))).copy()
            tau_FLIO_image = np.flipud(np.dstack((mat_image['Tau1'], mat_image['Tau2'], mat_image['Tau3']))).copy()

            taumean_FLIO_rgb_image = PIL.Image.fromarray(np.uint8(image_normalize(amplitude_FLIO_image * tau_FLIO_image)))
            taumean_FLIO_rgb_image.save(os.path.join(flio_output_path, d_path.split('.')[0] + '_taumean_rgb.jpg'), "JPEG")

            eq = amplitude_FLIO_image * np.exp(-400 / (tau_FLIO_image + 1e-6))
            taumean_FLIO_deconv_intensity_rgb_image = PIL.Image.fromarray(np.uint8(image_normalize(eq)))
            taumean_FLIO_deconv_intensity_rgb_image.save(os.path.join(flio_output_path, d_path.split('.')[0] + '_deconv_intensity.jpg'), "JPEG")


            taumean_FLIO_image = np.sum(amplitude_FLIO_image * tau_FLIO_image, axis=2) / (np.sum(amplitude_FLIO_image, axis=2) + 1e-6)
            taumean_FLIO_image_minscaled = PIL.Image.fromarray(np.uint8(255 * (taumean_FLIO_image - np.min(taumean_FLIO_image)) /
                                                              (np.max(taumean_FLIO_image) - np.min(taumean_FLIO_image))))
            taumean_FLIO_image_minscaled.save(os.path.join(flio_output_path, d_path.split('.')[0] + '_taumean_minscaled.jpg'), "JPEG")


            taumean_FLIO_image_unclipped = PIL.Image.fromarray(np.uint8(image_normalize(taumean_FLIO_image)))
            taumean_FLIO_image_unclipped.save(os.path.join(flio_output_path, d_path.split('.')[0] + '_taumean_unclipped.jpg'), "JPEG")

            taumean_FLIO_image[taumean_FLIO_image <= 300] = 300
            taumean_FLIO_image[taumean_FLIO_image >= 500] = 500

            plt.imshow(taumean_FLIO_image.astype(np.uint8), cmap="jet_r")
            plt.gca().set_axis_off()
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.savefig(os.path.join(flio_output_path, d_path.split('.')[0] + '_taumean_color.jpg'), bbox_inches='tight', pad_inches=0)

            taumean_FLIO_image = PIL.Image.fromarray(np.uint8(image_normalize(taumean_FLIO_image)))
            taumean_FLIO_image.save(os.path.join(flio_output_path, d_path.split('.')[0] + '_taumean.jpg'), "JPEG")

            amplitude_FLIO_image = PIL.Image.fromarray(np.uint8(channel_normalize(amplitude_FLIO_image)))
            amplitude_FLIO_image.save(os.path.join(flio_output_path, d_path.split('.')[0] + '_amplitude_channelNorm.jpg'), "JPEG")

            tau_FLIO_image = PIL.Image.fromarray(np.uint8(channel_normalize(tau_FLIO_image)))
            tau_FLIO_image.save(os.path.join(flio_output_path, d_path.split('.')[0] + '_tau_channelNorm.jpg'), "JPEG")

            taumeanguess_FLIO_image = mat_image['TauMeanGuess']
            taumeanguess_FLIO_image = PIL.Image.fromarray(np.uint8(np.flipud(image_normalize(taumeanguess_FLIO_image)).copy()))
            taumeanguess_FLIO_image.save(os.path.join(flio_output_path, d_path.split('.')[0] + '_taumeanguess.jpg'), "JPEG")

            rauc_FLIO_image = np.flipud(np.dstack((mat_image['RAUC1'], mat_image['RAUC2'], mat_image['RAUC3']))).copy()
            rauc_FLIO_image = PIL.Image.fromarray(np.uint8(channel_normalize(rauc_FLIO_image)))
            rauc_FLIO_image.save(os.path.join(flio_output_path, d_path.split('.')[0] + '_RAUC_channelNorm.jpg'), "JPEG")

            raucis_FLIO_image = np.flipud(np.dstack((mat_image['RAUCIS1'], mat_image['RAUCIS2'], mat_image['RAUCIS3']))).copy()
            raucis_FLIO_image = PIL.Image.fromarray(np.uint8(channel_normalize(raucis_FLIO_image)))
            raucis_FLIO_image.save(os.path.join(flio_output_path, d_path.split('.')[0] + '_RAUCIS_channelNorm.jpg'), "JPEG")
