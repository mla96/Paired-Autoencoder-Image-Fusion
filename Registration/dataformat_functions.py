# Michelle La
# Jan. 30, 2020


import numpy as np
import SimpleITK as sitk
from PIL import Image
import hdf5storage


# Get .mat files from FLIMX study manager directory
def channel_Mat2NpArray(path):
    channel = hdf5storage.loadmat(path)
    channel_raw = channel['rawData']
    # channel_rawflat = channel['rawDataFlat']
    return channel_raw


# Get resized 2D moving image in sitk format from rawData time series in channel .mat file
# Specify image parameters as global variable?
def channel_NpArray2SitkIm(channel_raw, w, h):
    disp_im_array = np.flip(np.sum(channel_raw, axis=2), 0).astype(np.float32)
    moving_im = Image.fromarray(disp_im_array).resize((w, h), Image.LANCZOS)
    moving_im = sitk.GetImageFromArray(np.array(moving_im, dtype=np.float32), isVector=False)
    # sitk.Show(moving_im)

    # Resize
    # resample = sitk.ResampleImageFilter()
    # resample.SetInterpolator = sitk.sitkLinear
    # resample.SetOutputDirection = moving_im.GetDirection()
    # resample.SetOutputOrigin = moving_im.GetOrigin()
    #
    # orig_size = np.array(moving_im.GetSize(), dtype=np.int)
    # print(orig_size)
    # new_size = copy.deepcopy(orig_size)
    # new_size[0] = 772
    # new_size[1] = 772
    # resample.SetSize = new_size
    #
    # orig_spacing = moving_im.GetSpacing()
    # new_spacing = (orig_spacing[0] * new_size[0]/orig_size[0], orig_spacing[1] * new_size[1]/orig_size[1])
    # resample.SetOutputSpacing = new_spacing
    #
    # # new_size = orig_size * (orig_spacing / new_spacing)
    # # new_size = np.ceil(new_size).astype(np.int)  # Image dimensions are in integers
    # # new_size = [int(s) for s in new_size]
    #
    # moving_im_final = resample.Execute(moving_im)
    #
    # # reference_origin = np.zeros(dimension)
    # # reference_direction = np.identity(dimension).flatten()
    # # reference_size = [128] * dimension  # Arbitrary sizes, smallest size that yields desired results.
    # # reference_spacing = [phys_sz / (sz - 1) for sz, phys_sz in zip(reference_size, reference_physical_size)]
    # #
    # # reference_image = sitk.Image(reference_size, data[0].GetPixelIDValue())
    # # reference_image.SetOrigin(reference_origin)
    # # reference_image.SetSpacing(reference_spacing)
    # # reference_image.SetDirection(reference_direction)

    # sitk.Show(moving_im_final)
    return moving_im
