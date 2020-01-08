'''

# dimension = 2
# reference_origin = np.zeros(dimension)
# reference_direction = np.identity(dimension).flatten()
# reference_size = [128]*dimension # Arbitrary sizes, smallest size that yields desired results.
# reference_spacing = [ phys_sz/(sz-1) for sz,phys_sz in zip(reference_size, reference_physical_size) ]
#
# reference_image = sitk.Image(reference_size, data[0].GetPixelIDValue())
# reference_image.SetOrigin(reference_origin)
# reference_image.SetSpacing(reference_spacing)
# reference_image.SetDirection(reference_direction)
# data_dict = sio.loadmat('SDT-File-Parsing/AMD59_flio.mat')
# channel1 = data_dict.get('channel1')
# channel2 = data_dict.get('channel2')


initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                      new_im,
                                                      sitk.AffineTransform(2),
                                                      sitk.CenteredTransformInitializerFilter.GEOMETRY)

moving_resampled = sitk.Resample(new_im, fixed_image, initial_transform, sitk.sitkLinear, 0.0, new_im.GetPixelID())

R = sitk.ImageRegistrationMethod()
R.SetMetricAsCorrelation()  # sampling?
R.SetMetricSamplingStrategy(R.RANDOM)
R.SetMetricSamplingPercentage(1.0)
# R.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
# R.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
# R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
R.SetOptimizerAsRegularStepGradientDescent(1.0, 0.01, 200, 0.1)
# R.SetOptimizerAsGradientDescent(1.0, numberOfIterations=300, convergenceMinimumValue=1e-6)
R.SetOptimizerScalesFromPhysicalShift()
R.SetInitialTransform(initial_transform, inPlace=False)
R.SetInterpolator(sitk.sitkLinear)

R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

final_transform = R.Execute(fixed_image, moving_resampled)

print("-------")
print(final_transform)
print("Optimizer stop condition: {0}".format(R.GetOptimizerStopConditionDescription()))
print(" Iteration: {0}".format(R.GetOptimizerIteration()))
print(" Metric value: {0}".format(R.GetMetricValue()))


moving_final = sitk.Resample(moving_resampled, fixed_image, final_transform, sitk.sitkLinear, 0.0, new_im.GetPixelID())
sitk.Show(moving_final)

disp_im_array = (disp_im_array/np.max(disp_im_array)) * 255
new_im_jpeg = Image.fromarray(disp_im_array.astype('uint8'))
new_im_jpeg = new_im_jpeg.save("new_im.tiff")

# def scale_array(array):
#     max_val = np.max(array)
#     for i in range(len(array)):
#         for j in range(len(array[i])):
#             array[i][j] = int((array[i][j] / max_val) * 255)
#     return array


# rgb_components = 3
#
# img_array = [[],[],[]]
# sum_img_array = np.zeros((2056, 2124))

# for component in range(rgb_components):
#     img_array[component] = sitk.GetArrayFromImage(sitk.VectorIndexSelectionCast(fixed_image, component))
#     for i in range(len(img_array[component])):
#         for j in range(len(img_array[component][i])):
#             sum_img_array[i][j] = sum_img_array[i][j] + img_array[component][i][j]
    # sitk.Show(sitk.VectorIndexSelectionCast(fixed_image, component))
#sum_img_array = scale_array(sum_img_array)
#fixed_image = sitk.VectorIndexSelectionCast(sitk.GetImageFromArray(sum_img_array, sitk.sitkFloat32), 0)
# why does this image still have 3 components...
# print(sitk.VectorIndexSelectionCast(fixed_image, 0))
# sitk.Show(sitk.VectorIndexSelectionCast(fixed_image, 0))
# print(sitk.VectorIndexSelectionCast(fixed_image, 1))
# sitk.Show(sitk.VectorIndexSelectionCast(fixed_image, 1))
# print(sitk.VectorIndexSelectionCast(fixed_image, 2))
# sitk.Show(sitk.VectorIndexSelectionCast(fixed_image, 2))

# import matplotlib.pyplot as plt
#
# from ipywidgets import interact, fixed
#
#
# # Callback invoked by the interact IPython method for scrolling through the image stacks of
# # the two images (moving and fixed).
# def display_images(fixed_image_z, moving_image_z, fixed_npa, moving_npa):
#     # Create a figure with two subplots and the specified size.
#     plt.subplots(1, 2, figsize=(10, 8))
#
#     # Draw the fixed image in the first subplot.
#     plt.subplot(1, 2, 1)
#     plt.imshow(fixed_npa[fixed_image_z, :, :], cmap=plt.cm.Greys_r);
#     plt.title('fixed image')
#     plt.axis('off')
#
#     # Draw the moving image in the second subplot.
#     plt.subplot(1, 2, 2)
#     plt.imshow(moving_npa[moving_image_z, :, :], cmap=plt.cm.Greys_r);
#     plt.title('moving image')
#     plt.axis('off')
#
#     plt.show()
#
#
# # Callback invoked by the IPython interact method for scrolling and modifying the alpha blending
# # of an image stack of two images that occupy the same physical space.
# def display_images_with_alpha(image_z, alpha, fixed, moving):
#     img = (1.0 - alpha) * fixed[:, :, image_z] + alpha * moving[:, :, image_z]
#     plt.imshow(sitk.GetArrayViewFromImage(img), cmap=plt.cm.Greys_r);
#     plt.axis('off')
#     plt.show()
#
#
# # Callback invoked when the StartEvent happens, sets up our new data.
# def start_plot():
#     global metric_values, multires_iterations
#
#     metric_values = []
#     multires_iterations = []
#
#
# # Callback invoked when the EndEvent happens, do cleanup of data and figure.
# def end_plot():
#     global metric_values, multires_iterations
#
#     del metric_values
#     del multires_iterations
#     # Close figure, we don't want to get a duplicate of the plot latter on.
#     plt.close()
#
#
# # Callback invoked when the IterationEvent happens, update our data and display new figure.
# def plot_values(registration_method):
#     global metric_values, multires_iterations
#
#     metric_values.append(registration_method.GetMetricValue())
#     # Clear the output area (wait=True, to reduce flickering), and plot current data
#     clear_output(wait=True)
#     # Plot the similarity metric values
#     plt.plot(metric_values, 'r')
#     plt.plot(multires_iterations, [metric_values[index] for index in multires_iterations], 'b*')
#     plt.xlabel('Iteration Number', fontsize=12)
#     plt.ylabel('Metric Value', fontsize=12)
#     plt.show()
#
#
# # Callback invoked when the sitkMultiResolutionIterationEvent happens, update the index into the
# # metric_values list.
# def update_multires_iterations():
#     global metric_values, multires_iterations
#     multires_iterations.append(len(metric_values))
#
#
# initial_transform = sitk.CenteredTransformInitializer(fixed_image,
#                                                       new_im,
#                                                       sitk.Euler3DTransform(),
#                                                       sitk.CenteredTransformInitializerFilter.GEOMETRY)
#
# moving_resampled = sitk.Resample(new_im, fixed_image, initial_transform, sitk.sitkLinear, 0.0, new_im.GetPixelID())
#
# interact(display_images_with_alpha, image_z=(0,fixed_image.GetSize()[2]-1), alpha=(0.0,1.0,0.05), fixed = fixed(fixed_image), moving=fixed(moving_resampled));
#
#
# registration_method = sitk.ImageRegistrationMethod()
#
# # Similarity metric settings.
# registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
# registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
# registration_method.SetMetricSamplingPercentage(0.01)
#
# registration_method.SetInterpolator(sitk.sitkLinear)
#
# # Optimizer settings.
# registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
# registration_method.SetOptimizerScalesFromPhysicalShift()
#
# # Setup for the multi-resolution framework.
# registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
# registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
# registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
#
# # Don't optimize in-place, we would possibly like to run this cell multiple times.
# registration_method.SetInitialTransform(initial_transform, inPlace=False)
#
# # Connect all of the observers so that we can perform plotting during registration.
# registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
# registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
# registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations)
# registration_method.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(registration_method))
#
# final_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32),
#                                                sitk.Cast(new_im, sitk.sitkFloat32))
'''