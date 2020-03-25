import os
import shutil
from PIL import Image


os.getcwd()
os.chdir("../../")

print(os.getcwd())

path = '../OakData/Transfer_learning_datasets/ARIA'
files = []

for root, _, file_list in os.walk(path):
    print(file_list)
    for file in file_list:
        if file.endswith(".tif"):  # JPEG
            files.append(os.path.join(root, file))

# f = open(path, 'r')
# lines = f.readlines()
# f.close()

# STARE
# crop_bound_w = 100  # Pixel length of cropping for width and height boundaries on fundus images
# crop_bound_h = 50
# final_dim = 1612

# BinRushed
# crop_bound_w = 646  # Pixel length of cropping for width and height boundaries on fundus images
# crop_bound_h = 250
# final_dim = 1084

# Magrabia
# crop_bound_w = 565  # Pixel length of cropping for width and height boundaries on fundus images
# crop_bound_h = 0
# final_dim = 1612

# MESSIDOR
# crop_bound_w = 591  # Pixel length of cropping for width and height boundaries on fundus images
# crop_bound_h = 215
# final_dim = 1058

# Kaggle
# crop_bound_w = 700  # Pixel length of cropping for width and height boundaries on fundus images
# crop_bound_h = 292
# final_dim = 1864


final_size = 256
quality_val = 90


for file in files:
    file = file.strip()

    file_image = Image.open(file)

    # file_image = Image.open(os.path.join(path, file))  # Load image

    # Resize and format fixed image (fundus RGB image) into BW SITK image for registration
    w, h = file_image.size
    crop_bound_w = (w - h)//2
    crop_bound_h = 0
    final_dim = crop_bound_w
    # crop_bound_w = int(0.2 * w)
    # crop_bound_h = int((h - (w - (2 * crop_bound_w)))//2)
    # final_dim = w - (2 * crop_bound_w)
    file_image = file_image.crop((crop_bound_w, crop_bound_h, w - crop_bound_w, h - crop_bound_h))
    w, h = file_image.size
    # file_image = file_image.resize((int(final_dim), int(final_dim * h / w)), Image.LANCZOS)
    w, h = file_image.size
    # file_image = file_image.crop((0, (h - w)/2, w, h - (h - w)/2))

    # file_image = file_image.resize((final_size, final_size), Image.LANCZOS)

    dir_name = os.path.basename(os.path.dirname(file))
    filename_split = os.path.basename(file).split('.')
    file_name = os.path.join(path, 'ARIA_' + dir_name + '_' + filename_split[0] + '_crop' + str(final_dim) + '.' + 'jpg')
    file_image.save(file_name, 'JPEG', quality=quality_val)
    print(file_name)
    # shutil.move(file_name, '../OakData/Transfer_learning_datasets/ARIA')


# crop_w = amount w to crop from original image, crop_h = amount h to crop from original image
# output_dim = w/h dimension of square output
def crop_images(file, crop_w, crop_h, output_dim):
    print(file)
