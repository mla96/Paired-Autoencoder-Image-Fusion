import os
import shutil
from PIL import Image


os.getcwd()
os.chdir("../../")

# print(os.getcwd())

path = 'STARE_Data/'

files = []

for file in os.listdir(path):
    if file.endswith(".jpg"):
        files.append(file)
print(files)
# f = open(path, 'r')
# lines = f.readlines()
# f.close()

crop_bound_w = 100  # Pixel length of cropping for width and height boundaries on fundus images
crop_bound_h = 50

final_dim = 1612

quality_val = 90


for file in files:
    file = file.strip()

    file_image = Image.open(os.path.join(path, file))  # Load image

    # Resize and format fixed image (fundus RGB image) into BW SITK image for registration
    w, h = file_image.size
    # file_image = file_image.crop((crop_bound_w, crop_bound_h, w - crop_bound_w, h - crop_bound_h))
    w, h = file_image.size
    file_image = file_image.resize((int(final_dim), int(final_dim * h / w)), Image.LANCZOS)
    w, h = file_image.size
    # file_image = file_image.crop((0, (h - w)/2, w, h - (h - w)/2))

    # TEST - small images
    file_image = file_image.resize((int(32), int(32)), Image.LANCZOS)

    filename_split = file.split('.')
    file_image.save(os.path.join(path, filename_split[0] + '_crop.' + filename_split[1]), 'JPEG', quality=quality_val)
    shutil.move(os.path.join(path, filename_split[0] + '_crop.' + filename_split[1]),
                os.path.join(path, 'Test_Crop/train', filename_split[0] + '_crop.' + filename_split[1]))
