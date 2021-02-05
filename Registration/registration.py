#!/usr/bin/env python3
"""
This file contains functions to register a moving image to a fixed image.
"""


import numpy as np
import SimpleITK as sitk
from PIL import Image


# Print registration metrics
def command_iteration(method):
    print("{0:3} = {1:10.5f} : {2}".format(method.GetOptimizerIteration(),
                                     method.GetMetricValue(),
                                     method.GetOptimizerPosition()))


# Register moving image to fixed image
def my_registration(fixed_im, moving_im):
    initial_transform = sitk.CenteredTransformInitializer(fixed_im,
                                                          moving_im,
                                                          sitk.AffineTransform(2),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)

    moving_resampled = sitk.Resample(moving_im, fixed_im, initial_transform, sitk.sitkLinear, 0.0,
                                     moving_im.GetPixelID())

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsANTSNeighborhoodCorrelation(50)  # sampling?
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.4)
    # R.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    # R.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    # R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    # R.SetOptimizerAsRegularStepGradientDescent(1.0, 0.0001, 500, estimateLearningRate=R.Once)
    R.SetOptimizerAsGradientDescent(1.0, numberOfIterations=900, convergenceMinimumValue=1e-14)
    # R.SetOptimizerAsGradientDescentLineSearch(1.0, numberOfIterations=300, convergenceMinimumValue=1e-12)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetInitialTransform(initial_transform, inPlace=False)
    R.SetInterpolator(sitk.sitkLinear)
    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

    final_transform = R.Execute(fixed_im, moving_resampled)

    print("-------")
    print(final_transform)
    print("Optimizer stop condition: {0}".format(R.GetOptimizerStopConditionDescription()))
    print(" Iteration: {0}".format(R.GetOptimizerIteration()))
    print(" Metric value: {0}".format(R.GetMetricValue()))

    moving_final = sitk.Resample(moving_resampled, fixed_im, final_transform, sitk.sitkLinear, 0.0,
                                 moving_im.GetPixelID())
    # sitk.Show(moving_final)

    return moving_final, initial_transform, final_transform


# Compares two images; useful for comparing un-registered and registered moving images with fixed images
def checkerboard(image1, image2, filename, writer):
    checker = sitk.CheckerBoard(image1, image2)
    # sitk.Show(checker)
    writer.SetFileName(filename + '.tiff')
    writer.Execute(checker)


# Register raw time series using a pre-calculated transform from flattened series
def transform_slices(fixed_im, final_transform, voxel_array, w, h):
    # transformed_voxel_array = np.zeros(shape=(h, w, 1024))  # Matches larger size
    transformed_movie_array = np.zeros(shape=(256, 256, 1024))
    for i in range(len(voxel_array[0][0])):
        slice = voxel_array[:, :, i]
        slice_im = Image.fromarray(slice)
        slice_im = sitk.GetImageFromArray(np.array(slice_im.resize((w, h), Image.LANCZOS), dtype=np.float32),
                                          isVector=False)
        transformed_slice = sitk.Resample(slice_im, fixed_im, final_transform, sitk.sitkLinear, 0.0,
                                          slice_im.GetPixelID())

        # transformed_voxel_array[:, :, i] = sitk.GetArrayFromImage(transformed_slice)
        transformed_movie_slice = Image.fromarray(sitk.GetArrayFromImage(transformed_slice))
        transformed_movie_slice = np.array(transformed_movie_slice.resize((256, 256), Image.LANCZOS), dtype=np.float32)
        transformed_movie_array[:, :, i] = transformed_movie_slice
    return transformed_movie_array
